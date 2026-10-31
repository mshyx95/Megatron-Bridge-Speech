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

from typing import Any

import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from torch import Tensor, nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention

from .qsa_indexer import Qwen4ExpQSAIndexer


class Qwen4ExpDotProductAttention(nn.Module):
    """Mask-aware SDPA reference core for QSA, including grouped query heads."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.dropout = config.attention_dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        *,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Attend only to keys selected by the causal QSA mask."""
        if attention_bias is not None or packed_seq_params is not None:
            raise NotImplementedError("QSA SDPA does not support attention bias or packed sequences yet.")
        output = F.scaled_dot_product_attention(
            query.permute(1, 2, 0, 3),
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            attn_mask=~attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )
        return output.permute(2, 0, 1, 3).flatten(-2).contiguous()


class Qwen4ExpQSASelfAttention(Qwen3VLSelfAttention):
    """Qwen gated self-attention constrained by a QSA indexer mask."""

    def __init__(self, config, submodules, layer_number: int, **kwargs) -> None:
        if config.context_parallel_size > 1:
            raise NotImplementedError("Qwen4Exp QSA context parallelism is not yet supported.")
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        if not isinstance(self.core_attention, Qwen4ExpDotProductAttention):
            raise ValueError("Qwen4Exp QSA requires the mask-aware Qwen4ExpDotProductAttention core.")
        self.indexer = Qwen4ExpQSAIndexer(config, layer_number)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply QSA token selection before the regular Qwen attention path."""
        if packed_seq_params is not None:
            raise NotImplementedError("Qwen4Exp QSA does not yet support packed sequences.")
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Qwen4Exp QSA KV-cache inference is not yet supported.")
        indexer_rope = rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
        indexer_input = hidden_states
        if self.config.sequence_parallel:
            indexer_input = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.model_comm_pgs.tp
            )
        qsa_mask = self.indexer(indexer_input, indexer_rope, attention_mask)
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=qsa_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
        )
