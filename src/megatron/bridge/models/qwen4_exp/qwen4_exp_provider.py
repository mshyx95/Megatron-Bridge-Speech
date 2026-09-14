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

from dataclasses import dataclass, field
from typing import Any

import torch.nn.functional as F

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider


def apply_qwen4_exp_config(provider: "Qwen4ExpModelProvider", text_config: Any) -> None:
    """Apply Qwen4Exp text configuration to a model provider.

    Args:
        provider: Provider to update.
        text_config: Hugging Face Qwen4Exp text configuration.
    """
    provider.normalization = "RMSNorm"
    provider.layernorm_zero_centered_gamma = True
    provider.gated_linear_unit = True
    provider.activation_func = F.silu
    provider.gdn_output_gate = getattr(text_config, "output_gate_type", None) or "silu"
    provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
    provider.add_bias_linear = False
    provider.qk_layernorm = True
    provider.attention_output_gate = True
    provider.hidden_dropout = 0.0
    provider.attention_dropout = getattr(text_config, "attention_dropout", 0.0)

    provider.experimental_attention_variant = "gdn"
    provider.linear_attention_freq = [
        1 if layer_type == "linear_attention" else 0 for layer_type in text_config.layer_types
    ]
    provider.linear_conv_kernel_dim = text_config.linear_conv_kernel_dim
    provider.linear_key_head_dim = text_config.linear_key_head_dim
    provider.linear_value_head_dim = text_config.linear_value_head_dim
    provider.linear_num_key_heads = text_config.linear_num_key_heads
    provider.linear_num_value_heads = text_config.linear_num_value_heads

    provider.ffn_hidden_size = text_config.moe_intermediate_size
    provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
    provider.num_moe_experts = text_config.num_experts
    provider.moe_router_topk = text_config.num_experts_per_tok
    provider.moe_shared_expert_intermediate_size = text_config.shared_expert_intermediate_size
    provider.moe_shared_expert_gate = True
    provider.moe_grouped_gemm = True
    provider.moe_router_load_balancing_type = "global_aux_loss"
    provider.moe_aux_loss_coeff = text_config.router_aux_loss_coef
    provider.moe_router_pre_softmax = False
    provider.moe_token_dispatcher_type = "alltoall"
    provider.moe_permute_fusion = True

    provider.hc_count = text_config.hc_count
    provider.hc_lowrank = text_config.hc_lowrank
    provider.indexer_budget = text_config.indexer_budget
    provider.indexer_compress_ratio = text_config.indexer_compress_ratio
    provider.indexer_head_dim = text_config.indexer_head_dim
    provider.indexer_kv_heads = text_config.indexer_kv_heads
    provider.indexer_n_heads = text_config.indexer_n_heads
    provider.ple_layer_ids = list(text_config.ple_layer_ids)
    provider.ple_embed_dim = text_config.ple_embed_dim
    provider.ple_conv_kernel_size = text_config.ple_conv_kernel_size
    provider.ngram_size = text_config.ngram_size
    provider.heads_per_ngram = text_config.heads_per_ngram
    provider.ngram_vocab_size_base = text_config.ngram_vocab_size_base
    provider.make_ngram_vocab_size_divisible_by = text_config.make_ngram_vocab_size_divisible_by
    provider.mtp_num_layers = text_config.mtp_num_hidden_layers


@dataclass
class Qwen4ExpModelProvider(Qwen35VLMoEModelProvider):
    """Model provider for the Qwen4Exp vision-language MoE architecture."""

    position_embedding_type: str = "rope"
    gdn_output_gate: str = "sigmoid"
    hc_count: int = 4
    hc_lowrank: int = 320

    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    indexer_head_dim: int = 128
    indexer_kv_heads: int = 1
    indexer_n_heads: int = 4

    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    hf_text_config: dict[str, Any] | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the text-only backbone; vision and MTP are not yet integrated."""
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

        from .modeling_qwen4_exp.model import Qwen4ExpGPTModel

        if pre_process is False or post_process is False or vp_stage is not None:
            raise NotImplementedError("Qwen4Exp pipeline partitioning is not yet supported.")
        if self.hf_text_config is None:
            raise ValueError("hf_text_config must retain the source PLE hashing configuration.")
        return Qwen4ExpGPTModel(
            self,
            Qwen4ExpTextConfig(**self.hf_text_config),
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_length,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            pg_collection=self._pg_collection,
        )

    def finalize(self) -> None:
        """Validate Qwen4Exp geometry before finalizing the MCore config."""
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4Exp requires hc_count > 1, got {self.hc_count}.")
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}.")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4Exp QSA requires indexer_kv_heads=1.")
        if self.indexer_budget % self.indexer_compress_ratio != 0:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
        if self.ple_embed_dim % ((self.ngram_size - 1) * self.heads_per_ngram) != 0:
            raise ValueError("ple_embed_dim must be divisible by the total number of n-gram heads.")
        invalid_ple_layers = [layer_id for layer_id in self.ple_layer_ids if not 1 <= layer_id <= self.num_layers]
        if invalid_ple_layers:
            raise ValueError(
                f"ple_layer_ids are 1-based and must refer to an existing layer, got {invalid_ple_layers}."
            )
        super().finalize()
