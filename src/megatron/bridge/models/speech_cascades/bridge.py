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

"""Bridge for the native Cascades speech model.

Maps a converted HuggingFace-format speech checkpoint (Qwen3.5/Qwen3.6 GDN language model plus the
Cascades audio encoder and projection) to :class:`SpeechCascadesModel`. Dense and MoE language-model
mappings are reused from the corresponding Qwen3.5 bridges under a ``language_model.`` prefix; the
audio encoder and projection are copied whole via :class:`ReplicatedMapping`.
"""

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.qwen.qwen35_bridge import (
    Qwen35Bridge,
    Qwen35MoEBridge,
    _apply_qwen35_common_config,
    _apply_qwen35_moe_config,
)
from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel
from megatron.bridge.models.speech_cascades.provider import SpeechCascadesModelProvider


_DEFAULT_AUDIO_TOKEN_ID = 151646


def _dtype_from_config(config):
    """Resolve the model dtype from either ``torch_dtype`` or ``dtype`` fields."""
    value = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(value, str):
        value = getattr(torch, value, None)
    return value if isinstance(value, torch.dtype) else torch.bfloat16




@MegatronModelBridge.register_bridge(source="Qwen3_5SpeechCascadesForConditionalGeneration", target=SpeechCascadesModel)
class SpeechCascadesBridge(MegatronModelBridge):
    """Bridge between the converted HF speech checkpoint and the native ``SpeechCascadesModel``."""

    def provider_bridge(self, hf_pretrained) -> SpeechCascadesModelProvider:
        """Build a :class:`SpeechCascadesModelProvider` from the flat speech config."""
        config = hf_pretrained.config
        audio_processor = config.audio_processor
        audio_config = audio_processor["config"] if isinstance(audio_processor, dict) else audio_processor.config
        # Disable the encoder's activation-checkpoint wrapper so param names match the HF keys
        # (the wrapper inserts `_checkpoint_wrapped_module.` and breaks the replicated mapping).
        audio_config = {**dict(audio_config), "activation_checkpointing": ""}
        embd_layer = getattr(config, "embd_layer", None) or {}
        rope_parameters = getattr(config, "rope_parameters", None) or {}

        model_dtype = _dtype_from_config(config)
        provider = SpeechCascadesModelProvider(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            ffn_hidden_size=getattr(config, "intermediate_size", getattr(config, "moe_intermediate_size", None)),
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            kv_channels=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            layernorm_epsilon=config.rms_norm_eps,
            init_method_std=config.initializer_range,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(config.vocab_size),
            vocab_size=config.vocab_size,
            seq_length=config.max_position_embeddings,
            share_embeddings_and_output_weights=getattr(config, "tie_word_embeddings", True),
            rotary_base=rope_parameters.get("rope_theta", 10000000),
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            audio_encoder_config=dict(audio_config),
            audio_projection_cls=embd_layer.get("projection_cls", "mlp"),
            audio_compression_rate=embd_layer.get("compression_rate", 1),
            audio_token_id=getattr(config, "audio_token_id", None) or _DEFAULT_AUDIO_TOKEN_ID,
        )
        # Fill in the Qwen3.5/Qwen3.6 GDN hybrid language-model settings from the real config.
        if getattr(config, "num_experts", None):
            _apply_qwen35_moe_config(provider, config)
        else:
            _apply_qwen35_common_config(provider, config)
        provider.position_embedding_type = "rope"
        provider.autocast_dtype = model_dtype
        provider.hetereogenous_dist_checkpoint = True
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Qwen3.5/Qwen3.6 language-model mappings plus replicated audio mappings."""
        hf_pretrained = getattr(self, "hf_pretrained", None)
        config = getattr(hf_pretrained, "config", None)
        if getattr(config, "num_experts", None):
            experts_packed = moe_experts_stored_packed(hf_pretrained, "model.layers.")
            mapping_list = Qwen35MoEBridge._get_moe_lm_mappings(
                hf_prefix="model.", megatron_prefix="language_model.", experts_packed=experts_packed
            )
        else:
            mapping_list = Qwen35Bridge._get_dense_lm_mappings(
                hf_prefix="model.", megatron_prefix="language_model."
            )
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="audio_model.encoder.**",
                    hf_param="model.embed_tokens_extend.encoder.**",
                ),
                ReplicatedMapping(
                    megatron_param="audio_model.audio_projection.**",
                    hf_param="model.embed_tokens_extend.audio_projection.**",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
