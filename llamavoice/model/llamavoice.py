from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from transformers import LlamaModel, LlamaConfig, LogitsWarper, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available
from transformers.modeling_outputs import ModelOutput

from llamavoice.config import Config as DefultConfig
from llamavoice.encoder import PosteriorEncoder
from llamavoice.flow import ResidualAffineCouplingBlock
from llamavoice.decoder import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)
from llamavoice.utils import (
    pad_unpad_sequence,
    build_aligned_inputs_and_targets,
    split_hidden_states,
    get_random_segments,
    get_segments,
)


class LlamaVoiceConfig(PretrainedConfig):
    model_type = "llamavoice"

    def __init__(self, **kwargs):
        self.use_flash_attn = kwargs.get("use_flash_attn", True)
        self.gpt_config = kwargs.get("gpt_config", asdict(DefultConfig.gpt))
        self.num_text_tokens = kwargs.get(
            "num_text_tokens", 51866 + 3
        )  # 3 special tokens
        self.audio_encoder_config = kwargs.get(
            "audio_encoder_config", asdict(DefultConfig.audio_encoder)
        )
        self.flow_config = kwargs.get("flow_config", asdict(DefultConfig.flow))
        self.decoder_config = kwargs.get("decoder_config", asdict(DefultConfig.decoder))
        self.discriminator_config = kwargs.get(
            "discriminator_config", asdict(DefultConfig.discriminator)
        )
        self.dataset = PretrainedConfig(
            **kwargs.get("dataset", asdict(DefultConfig.dataset))
        )
        self.train = PretrainedConfig(**kwargs.get("train", asdict(DefultConfig.train)))
        self.train.dataloader = PretrainedConfig(**self.train.dataloader)
        self.train.AdamW = PretrainedConfig(**self.train.AdamW)
        self.loss_config = kwargs.get("loss_config", asdict(DefultConfig.loss))
        self.stop_threshold = kwargs.get("stop_threshold", 0.5)
        # for amphion compatibility
        self.preprocess = PretrainedConfig(**{"use_phone": False, "use_spkid": False})
        # text
        self.bos_token_id = self.num_text_tokens - 1
        self.pad_token_id = 0
        self.eos_token_id = self.num_text_tokens - 2
        # speech prompt
        self.speech_prompt_segment_size = kwargs.get("speech_prompt_segment_size", 64)
        super().__init__(**kwargs)


class LlamaVoiceDiscriminator(PreTrainedModel):
    def __init__(self, config: LlamaVoiceConfig):
        super().__init__(config)
        self.discriminator = self._build_discriminator(config.discriminator_config)

    def _build_discriminator(self, config: dict):
        return HiFiGANMultiScaleMultiPeriodDiscriminator(**config)

    def forward(self, x):
        return self.discriminator(x)


class LlamaVoice(PreTrainedModel):
    def __init__(self, config: LlamaVoiceConfig):
        super().__init__(config)
        self.use_flash_attn = config.use_flash_attn

        self.gpt, self.llama_config = self._build_llama(config.gpt_config)
        self.model_dim = int(self.gpt.config.hidden_size)

        self.text_embedding = torch.nn.Embedding(config.num_text_tokens, self.model_dim)
        self.posterior_encoder = self._build_posterior_encoder(
            config.audio_encoder_config
        )
        self.text_head = nn.Linear(self.model_dim, config.num_text_tokens)
        self.dist_head = nn.Linear(self.model_dim, 2 * self.model_dim)
        self.stop_proj = nn.Linear(self.model_dim, 1)

        self.flow = self._build_flow(config.flow_config)
        self.decoder = self._build_decoder(config.decoder_config)
        self.config = config

    def _build_llama(
        self,
        config: dict,
    ) -> Tuple[LlamaModel, LlamaConfig]:

        if self.use_flash_attn and is_flash_attn_2_available():
            llama_config = LlamaConfig(
                **config,
                attn_implementation="flash_attention_2",
            )
            self.logger.warning(
                "enabling flash_attention_2 may make gpt be even slower"
            )
        else:
            llama_config = LlamaConfig(**config)

        model = LlamaModel(llama_config)
        del model.embed_tokens

        return model, llama_config

    def _build_posterior_encoder(self, config: dict):
        config = PretrainedConfig(**config)
        return PosteriorEncoder(
            in_channels=config.aux_channels,
            out_channels=self.model_dim,
            hidden_channels=config.hidden_channels,
            kernel_size=config.posterior_encoder_kernel_size,
            layers=config.posterior_encoder_layers,
            stacks=config.posterior_encoder_stacks,
            base_dilation=config.posterior_encoder_base_dilation,
            global_channels=config.global_channels,
            dropout_rate=config.posterior_encoder_dropout_rate,
            use_weight_norm=config.use_weight_norm_in_posterior_encoder,
        )

    def _build_flow(self, config: dict):
        config = PretrainedConfig(**config)
        return ResidualAffineCouplingBlock(
            in_channels=self.model_dim,
            hidden_channels=config.hidden_channels,
            flows=config.flow_flows,
            kernel_size=config.flow_kernel_size,
            base_dilation=config.flow_base_dilation,
            layers=config.flow_layers,
            global_channels=config.global_channels,
            dropout_rate=config.flow_dropout_rate,
            use_weight_norm=config.use_weight_norm_in_flow,
            use_only_mean=config.use_only_mean_in_flow,
        )

    def _build_decoder(self, config: dict):
        config = PretrainedConfig(**config)
        self.upsample_factor = int(np.prod(config.decoder_upsample_scales))
        return HiFiGANGenerator(
            in_channels=self.model_dim,
            out_channels=1,
            channels=config.decoder_channels,
            global_channels=config.global_channels,
            kernel_size=config.decoder_kernel_size,
            upsample_scales=config.decoder_upsample_scales,
            upsample_kernel_sizes=config.decoder_upsample_kernel_sizes,
            resblock_kernel_sizes=config.decoder_resblock_kernel_sizes,
            resblock_dilations=config.decoder_resblock_dilations,
            use_weight_norm=config.use_weight_norm_in_decoder,
        )

    def dist_sampling(self, x):
        stats = self.dist_head(x)  # (b, t, c)
        m, logs = stats.split(stats.size(2) // 2, dim=2)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs

    def forward(self, batch: dict) -> Dict[str, Optional[torch.Tensor]]:
        # length in the last dim
        """
        # ("speech", torch.Size([16, 1, 404160])),
        # ("speech_len", torch.Size([16])),
        # ("text_token", torch.Size([16, 55])),
        # ("text_token_len", torch.Size([16])),
        # ("speech_feat", torch.Size([16, 513, 1578])),
        # ("speech_feat_len", torch.Size([16])),
        """

        # 1. parse inputs
        text_token = batch["text_token"]
        text_token_len = batch["text_token_len"]
        target_feats = batch["target_feats"]
        target_feats_len = batch["target_feats_len"]

        if "prompt_feats" in batch and "prompt_feats_len" in batch:
            prompt_feats = batch["prompt_feats"]
            prompt_feats_len = batch["prompt_feats_len"]
        else:
            # generate a random segment as prompt
            speech_prompt_segment_size = self.config.speech_prompt_segment_size
            prompt_feats, start_idxs = get_random_segments(
                target_feats,
                target_feats_len,
                segment_size=speech_prompt_segment_size,
            )
            prompt_feats_len = (
                torch.clamp(
                    start_idxs + speech_prompt_segment_size, max=target_feats_len
                )
                - start_idxs
            )

        # 2. vae encoder
        vae_z, vae_m, vae_logs, vae_mask = self.posterior_encoder(
            target_feats, target_feats_len
        )
        with torch.no_grad():
            prompt_z, prompt_m, prompt_logs, prompt_mask = self.posterior_encoder(
                prompt_feats, prompt_feats_len
            )

        # 3. flow
        flow_z = self.flow(vae_z, vae_mask)  # (B, H, T_feats)
        with torch.no_grad():
            prompt_flow_z = self.flow(prompt_z, prompt_mask)

        # 4. prepare llm input and target
        vae_mask_target = torch.nn.functional.pad(vae_mask, (1, 0), value=1)
        text_token, text_targets, text_token_len = build_aligned_inputs_and_targets(
            text_token,
            text_token_len,
            self.config.bos_token_id,
            self.config.eos_token_id,
        )
        text_embed = self.text_embedding(text_token)
        prompt_flow_z, prompt_z_target, prompt_len = build_aligned_inputs_and_targets(
            prompt_flow_z, prompt_feats_len
        )
        _, prompt_logs_target, _ = build_aligned_inputs_and_targets(
            prompt_logs, prompt_feats_len
        )
        _, vae_m_target, _ = build_aligned_inputs_and_targets(vae_m, target_feats_len)
        flow_z, flow_z_target, z_len = build_aligned_inputs_and_targets(
            flow_z, target_feats_len
        )
        _, vae_logs_target, _ = build_aligned_inputs_and_targets(
            vae_logs, target_feats_len
        )

        lm_input, lm_input_len = pad_unpad_sequence(
            text_embed,
            text_token_len,
            prompt_flow_z.transpose(1, 2),
            prompt_len,
            flow_z.transpose(1, 2),
            z_len,
            IGNORE_ID=0,
        )  # (B, T, C), (B, )

        # 5. run lm forward
        outputs: BaseModelOutputWithPast = self.gpt(
            inputs_embeds=lm_input, use_cache=False, output_attentions=False
        )
        # 6. parse llm output
        text_logits, prompt_logits, dist_logits = split_hidden_states(
            outputs.last_hidden_state,
            text_token_len,
            prompt_len,
            z_len,
        )

        # 7. multi-head prediction
        text_logits = self.text_head(text_logits)
        lm_z, lm_m, lm_logs = self.dist_sampling(dist_logits)
        plm_z, plm_m, plm_logs = self.dist_sampling(prompt_logits)
        # stop token prediction
        stop = self.stop_proj(dist_logits)
        stop = torch.sigmoid(stop)

        # 8. decoder forward
        z_segments, start_idxs = get_random_segments(
            vae_z,
            target_feats_len,
            self.config.decoder_config["segment_size"],
        )

        # forward decoder with random segments
        gen_wav = self.decoder(z_segments)

        output = ModelOutput(
            lm_m=lm_m.transpose(1, 2),
            lm_logs=lm_logs.transpose(1, 2),
            flow_z=flow_z,
            vae_m=vae_m_target,
            vae_logs=vae_logs_target,
            vae_mask=vae_mask_target,
            stop_predict=stop,
            target_feats_len=z_len,
            text_logits=text_logits,
            text_targets=text_targets,
            prompt_m=prompt_z_target,
            prompt_logs=prompt_logs_target,
            plm_m=plm_m.transpose(1, 2),
            plm_logs=plm_logs.transpose(1, 2),
            predicted_audio=gen_wav,
            z_segments=z_segments,
            ids_slice=start_idxs,
        )
        return output


def test():
    c = LlamaVoiceConfig()

    model = LlamaVoice(c)
    print(model)

    # generate test data
    batch_size = 2
    text_token_len = torch.randint(5, 32, (batch_size,))
    text_token = torch.randint(
        0, c.num_text_tokens, (batch_size, text_token_len.max())
    )  # start and end is added
    target_feats_len = torch.randint(500, 1000, (batch_size,))
    target_feats = torch.randn(
        batch_size, c.audio_encoder_config["aux_channels"], target_feats_len.max()
    )
    hop_size = int(np.prod(c.decoder_config["decoder_upsample_scales"]))
    target_audio = torch.randn(batch_size, 1, target_feats_len.max() * hop_size)

    prompt_feats_len = torch.randint(100, 300, (batch_size,))
    prompt_feats = torch.randn(
        batch_size, c.audio_encoder_config["aux_channels"], prompt_feats_len.max()
    )

    batch = {
        "text_token": text_token,
        "text_token_len": text_token_len,
        "target_feats": target_feats,
        "target_feats_len": target_feats_len,
        "prompt_feats": prompt_feats,
        "prompt_feats_len": prompt_feats_len,
        "target_audio": target_audio,
    }

    for k, v in batch.items():
        print("--- input: ", k, v.shape)
        if k.endswith("_len"):
            print("value:", v)

    out = model(batch)

    for k, v in out.items():
        print("--- output: ", k, v.shape)
        if k.endswith("_len"):
            print("value:", v)


if __name__ == "__main__":
    test()
