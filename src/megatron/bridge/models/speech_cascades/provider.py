# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provider for the native Cascades speech model (Qwen3.5 GDN language model + audio encoder)."""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn.functional as F
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel


@dataclass
class SpeechCascadesModelProvider(GPTModelProvider):
    """Build a :class:`SpeechCascadesModel`: Qwen3.5 (GDN hybrid) LM plus the Cascades audio encoder.

    The Qwen3.5 language-model fields (GDN hybrid, QK layernorm, RMSNorm) are expected to be set by
    the bridge from the HF config; sensible Qwen3.5 defaults are provided so the provider is usable
    standalone. Audio fields configure the vendored ConformerEncoder and its projection.
    """

    # --- Cascades audio encoder ---
    audio_encoder_config: dict[str, Any] = field(default_factory=dict)
    audio_projection_cls: str = "mlp"
    audio_compression_rate: int = 8
    audio_downsample_rate: int = 1
    retain_audio_attention_maps: bool = True
    audio_token_id: int = 151646

    freeze_language_model: bool = False
    freeze_audio_model: bool = False
    # Freeze only the audio encoder (Conformer), leaving the projector trainable. Used by the
    # stage-2 KD recipe, which trains the projector + MTP heads with the encoder + LLM frozen.
    freeze_audio_encoder: bool = False

    # --- Knowledge distillation (stage-2 S2T) ---
    # Off by default so the ASR path is unaffected; the KD forward in ``step.py`` is gated on the
    # presence of teacher tensors in the batch, and reads these hyperparameters off the config.
    knowledge_distillation: bool = False
    distillation_weight: float = 0.75
    distillation_temperature: float = 2.0
    distillation_kl_chunk_size: int = 128

    # --- Multi-token prediction (MTP) depth heads ---
    # All default off/0 so the ASR path is byte-identical when MTP is disabled.
    mtp_num_hidden_layers: int = 0
    mtp_loss_weight: float = 0.0
    mtp_loss_depth_weights: list[float] | None = None
    mtp_distillation_weight: float = 0.0
    mtp_init_extra_layers_from_first: bool = False
    # --- Qwen3.5 language-model defaults (GDN hybrid) ---
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    activation_func: Callable = F.silu
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = True
    hidden_dropout: float = 0.0
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gated_delta_net"
    linear_attention_freq: int = 4
    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rotary_percent: float = 0.25
    position_embedding_type: str = "rope"
    autocast_dtype: Any = None

    # GDN hybrid layers require the experimental attention-variant block spec.
    transformer_layer_spec: Any = get_transformer_block_with_experimental_attention_variant_spec

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> SpeechCascadesModel:
        """Build the composed audio-language model for the current pipeline stage."""
        model = SpeechCascadesModel(
            self,
            audio_token_id=self.audio_token_id,
            audio_encoder_config=self.audio_encoder_config,
            audio_projection_cls=self.audio_projection_cls,
            audio_compression_rate=self.audio_compression_rate,
            audio_downsample_rate=self.audio_downsample_rate,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )
        if self.freeze_language_model or self.freeze_audio_model or self.freeze_audio_encoder:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_audio_model=self.freeze_audio_model,
                freeze_audio_encoder=self.freeze_audio_encoder,
            )
        return model
