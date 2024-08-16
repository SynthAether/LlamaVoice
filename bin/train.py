# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

import torch

from llamavoice.model import LlamaVoiceConfig
from llamavoice.train.llamavoice_trainer import LlamaVoiceTrainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="json files for configurations. If not specified, use default values",
    )
    parser.add_argument(
        "--train_data_list",
        type=str,
        default="dump/parquet/train/data.list",
        help="The training data list",
    )
    parser.add_argument(
        "--val_data_list",
        type=str,
        default="dump/parquet/dev/data.list",
        help="The validation data list",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="llamavoice",
        help="A specific name to note the experiment",
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="resume",
        help="Resume training or finetuning.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        default="",
        help="Checkpoint for resume training or finetuning.",
    )

    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        cfg = LlamaVoiceConfig.from_json_file(args.config)
    else:
        # use default
        cfg = LlamaVoiceConfig()

    # ------------------- debug only --------------------
    cfg.dataset.batch_size = 2
    cfg.train.dataloader.num_worker = 2
    # ------------------- debug only END ---------------------

    print("experiment name: ", args.exp_name)
    # CUDA settings
    cuda_relevant()

    # Build trainer
    cfg.log_dir = args.log_dir
    print(f"Building {cfg.model_type} trainer")
    trainer = LlamaVoiceTrainer(args, cfg)
    print(f"Start training {cfg.model_type} model")
    trainer.train_loop()


if __name__ == "__main__":
    main()
