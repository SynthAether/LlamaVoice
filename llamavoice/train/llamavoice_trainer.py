# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

# from utils.util import *

from llamavoice.utils.mel import mel_spectrogram_torch
from llamavoice.train.tts_trainer import TTSTrainer
from llamavoice.model import LlamaVoice, LlamaVoiceDiscriminator
from llamavoice.dataset.processor import Processor as P, LamaVoiceCollator
from llamavoice.dataset.dataset import Dataset
from llamavoice.tokenizer.tokenizer import get_tokenizer
from llamavoice.utils import get_segments as slice_segments
from llamavoice.utils.mel import extract_linear_features, extract_mel_features
from llamavoice.model.loss import (
    LamaVoiceLoss as GeneratorLoss,
    DiscriminatorAdversarialLoss as DiscriminatorLoss,
)

from torch.utils.data import DataLoader
from transformers.configuration_utils import PretrainedConfig


class LlamaVoiceTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)

    def _build_model(self):
        net_g = LlamaVoice(self.cfg)
        net_d = LlamaVoiceDiscriminator(self.cfg)
        model = {"generator": net_g, "discriminator": net_d}

        return model

    def _build_dataset(self):
        return ""

    def _build_dataloader(self):
        train_data = self.args.train_data_list
        cv_data = self.args.val_data_list
        C = self.cfg.dataset
        tokenizer = partial(
            get_tokenizer,
            multilingual=C.multilingual,
            num_languages=C.num_languages,
            language=C.language,
            task=C.task,
        )
        allowed_special = C.allowed_special
        tokenize = partial(
            P.tokenize, get_tokenizer=tokenizer, allowed_special=allowed_special
        )
        filter = partial(
            P.filter,
            max_length=C.max_length,
            min_length=C.min_length,
            token_max_length=C.token_max_length,
            token_min_length=C.token_min_length,
        )
        resample = partial(P.resample, resample_rate=C.sample_rate)
        compute_linear = partial(
            P.compute_linear, feat_extractor=extract_linear_features, cfg=C
        )
        compute_mel = partial(P.compute_mel, feat_extractor=extract_mel_features, cfg=C)
        shuffle = partial(P.shuffle, shuffle_size=C.shuffle_size)
        sort = partial(P.sort, sort_size=C.sort_size)
        batch = partial(
            P.batch,
            batch_type=C.batch_type,
            max_frames_in_batch=C.max_frames_in_batch,
            batch_size=C.batch_size,
        )
        padding = partial(P.padding, use_spk_embedding=False)

        data_pipeline = [
            P.parquet_opener,
            tokenize,
            filter,
            compute_linear,
            compute_mel,
            resample,
            shuffle,
            sort,
        ]

        train_dataset = Dataset(
            train_data,
            data_pipeline=data_pipeline,
            mode="train",
            shuffle=True,
            partition=True,
        )
        cv_dataset = Dataset(
            cv_data,
            data_pipeline=data_pipeline,
            mode="train",
            shuffle=False,
            partition=False,
        )
        collator = LamaVoiceCollator()
        # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=C.batch_size,
            pin_memory=self.cfg.train.dataloader.pin_memory,
            num_workers=self.cfg.train.dataloader.num_worker,
            prefetch_factor=C.prefetch,
            collate_fn=collator,
        )
        cv_data_loader = DataLoader(
            cv_dataset,
            batch_size=C.batch_size,
            pin_memory=self.cfg.train.dataloader.pin_memory,
            num_workers=self.cfg.train.dataloader.num_worker,
            prefetch_factor=C.prefetch,
            collate_fn=collator,
        )
        return train_data_loader, cv_data_loader

    def _build_optimizer(self):
        optimizer_g = torch.optim.AdamW(
            self.model["generator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer_d = torch.optim.AdamW(
            self.model["discriminator"].parameters(),
            self.cfg.train.learning_rate,
            betas=self.cfg.train.AdamW.betas,
            eps=self.cfg.train.AdamW.eps,
        )
        optimizer = {"optimizer_g": optimizer_g, "optimizer_d": optimizer_d}

        return optimizer

    def _build_scheduler(self):
        scheduler_g = ExponentialLR(
            self.optimizer["optimizer_g"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )
        scheduler_d = ExponentialLR(
            self.optimizer["optimizer_d"],
            gamma=self.cfg.train.lr_decay,
            last_epoch=self.epoch - 1,
        )

        scheduler = {"scheduler_g": scheduler_g, "scheduler_d": scheduler_d}
        return scheduler

    def _build_criterion(self):
        criterion = {
            "generator": GeneratorLoss(self.cfg.loss_config),
            "discriminator": DiscriminatorLoss(self.cfg),
        }
        return criterion

    def write_summary(
        self,
        losses,
        stats,
        images={},
        audios={},
        audio_sampling_rate=24000,
        tag="train",
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)
        self.sw.add_scalar(
            "learning_rate",
            self.optimizer["optimizer_g"].param_groups[0]["lr"],
            self.step,
        )

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

    def write_valid_summary(
        self, losses, stats, images={}, audios={}, audio_sampling_rate=24000, tag="val"
    ):
        for key, value in losses.items():
            self.sw.add_scalar(tag + "/" + key, value, self.step)

        if len(images) != 0:
            for key, value in images.items():
                self.sw.add_image(key, value, self.global_step, batchformats="HWC")
        if len(audios) != 0:
            for key, value in audios.items():
                self.sw.add_audio(key, value, self.global_step, audio_sampling_rate)

    def get_state_dict(self):
        state_dict = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
            "optimizer_g": self.optimizer["optimizer_g"].state_dict(),
            "optimizer_d": self.optimizer["optimizer_d"].state_dict(),
            "scheduler_g": self.scheduler["scheduler_g"].state_dict(),
            "scheduler_d": self.scheduler["scheduler_d"].state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.model["generator"].load_state_dict(checkpoint["generator"])
        self.model["discriminator"].load_state_dict(checkpoint["discriminator"])
        self.optimizer["optimizer_g"].load_state_dict(checkpoint["optimizer_g"])
        self.optimizer["optimizer_d"].load_state_dict(checkpoint["optimizer_d"])
        self.scheduler["scheduler_g"].load_state_dict(checkpoint["scheduler_g"])
        self.scheduler["scheduler_d"].load_state_dict(checkpoint["scheduler_d"])

    @torch.inference_mode()
    def _valid_step(self, batch):
        r"""Testing forward step. Should return average loss of a sample over
        one batch. Provoke ``_forward_step`` is recommended except for special case.
        See ``_test_epoch`` for usage.
        """

        return self._train_step(batch, no_grad=True)

    def _train_step(self, batch, no_grad=False):
        r"""Forward step for training and inference. This function is called
        in ``_train_step`` & ``_test_step`` function.
        """

        train_losses = {}
        total_loss = 0
        training_stats = {}

        batch["target_feats"] = batch["speech_feat"]
        batch["target_feats_len"] = batch["speech_feat_len"]

        # Train Discriminator
        # Generator output
        outputs_g = self.model["generator"](batch)

        y_mel = slice_segments(
            batch["mel"],
            outputs_g["ids_slice"],
            self.cfg.decoder_config["segment_size"],
        )
        y_hat_mel = mel_spectrogram_torch(
            outputs_g["predicted_audio"].squeeze(1), self.cfg.dataset
        )

        y = slice_segments(
            batch["speech"],
            outputs_g["ids_slice"] * self.cfg.dataset.hop_size,
            self.cfg.decoder_config["segment_size"] * self.cfg.dataset.hop_size,
        )

        # Discriminator output
        outputs_d_hat = self.model["discriminator"](
            outputs_g["predicted_audio"].detach()
        )
        outputs_d = self.model["discriminator"](y)

        ##  Discriminator loss

        real_loss, fake_loss = self.criterion["discriminator"](outputs_d_hat, outputs_d)
        loss_d = {
            "loss_disc_all": real_loss + fake_loss,
            "real": real_loss,
            "fake": fake_loss,
        }
        train_losses.update(loss_d)

        # BP and Grad Updated
        if not no_grad:
            self.optimizer["optimizer_d"].zero_grad()
            self.accelerator.backward(loss_d["loss_disc_all"])
            self.optimizer["optimizer_d"].step()

        ## Train Generator
        outputs_d_hat = self.model["discriminator"](outputs_g["predicted_audio"])
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            outputs_d = self.model["discriminator"](y)

        loss_g = self.criterion["generator"](
            outputs_g, outputs_d, outputs_d_hat, y_mel, y_hat_mel
        )
        train_losses.update(loss_g)

        # BP and Grad Updated
        if not no_grad:
            self.optimizer["optimizer_g"].zero_grad()
            self.accelerator.backward(loss_g["loss_gen_all"])
            self.optimizer["optimizer_g"].step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        total_loss = loss_g["loss_gen_all"] + loss_d["loss_disc_all"]

        return (
            total_loss.item(),
            train_losses,
            training_stats,
        )

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1

            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Generator Loss": train_losses["loss_gen_all"],
                        "Step/Discriminator Loss": train_losses["loss_disc_all"],
                        "Step/Generator Learning Rate": self.optimizer[
                            "optimizer_d"
                        ].param_groups[0]["lr"],
                        "Step/Discriminator Learning Rate": self.optimizer[
                            "optimizer_g"
                        ].param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        epoch_sum_loss = (
            epoch_sum_loss / epoch_step * self.cfg.train.gradient_accumulation_step
        )

        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / epoch_step
                * self.cfg.train.gradient_accumulation_step
            )

        return epoch_sum_loss, epoch_losses

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss: float = 0.0
        epoch_step: int = 0
        epoch_losses = dict()
        for batch in tqdm(
            self.valid_dataloader,
            desc=f"Validating Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss += total_loss
            epoch_step += 1
            if isinstance(valid_losses, dict):
                for key, value in valid_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

        epoch_sum_loss = epoch_sum_loss / epoch_step
        for key in epoch_losses.keys():
            epoch_losses[key] = epoch_losses[key] / epoch_step

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses


def test():
    from llamavoice.model import LlamaVoiceConfig

    class args:
        exp_name = "llamavoice"
        log_level = "debug"
        resume = ""  # The model name to restore
        train_stage = 0
        checkpoint_path = ""  # Checkpoint for resume training or finetuning.
        ar_model_ckpt_dir = ""
        resume_type = "resume"  # [resume, finetune]. Resume training or finetuning
        # data
        train_data_list = "dump/parquet/train/data.list"
        val_data_list = "dump/parquet/dev/data.list"

    c = LlamaVoiceConfig()
    # ------------------- debug only --------------------
    c.dataset.batch_size = 2
    c.train.dataloader.num_worker = 2
    # ------------------- debug only END ---------------------
    print(c)
    c.log_dir = "logs"
    trainer = LlamaVoiceTrainer(args, c)
    trainer.train_loop()


if __name__ == "__main__":
    test()
