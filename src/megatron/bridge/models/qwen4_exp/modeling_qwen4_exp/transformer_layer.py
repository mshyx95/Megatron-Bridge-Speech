# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import Tensor

from .hyper_connection import Qwen4ExpGatedResidual


class Qwen4ExpTransformerLayer(TransformerLayer):
    """Execute GDN/QSA and MoE branches with Qwen4Exp gated residual streams."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.input_layernorm, IdentityOp) or not isinstance(self.pre_mlp_layernorm, IdentityOp):
            raise ValueError("Qwen4Exp normalizes inside GR; standalone branch layernorms must be disabled.")
        if self.config.cuda_graph_impl != "none" or self.config.inference_fuse_tp_communication:
            raise NotImplementedError("Qwen4Exp GR does not support CUDA graphs or fused inference residuals yet.")
        self.attn_hyper_connection = Qwen4ExpGatedResidual(self.config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(self.config)

    def _run_input_layernorm(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor]]:
        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        normalized, _, _ = super()._run_input_layernorm(mixed)
        return normalized, residual, (injection,)

    def _apply_self_attn_bda_step(
        self,
        attention_output_with_bias: tuple[Tensor, Tensor | None],
        residual: Tensor,
        attn_state: tuple[Tensor, ...] = (),
    ) -> Tensor:
        if len(attn_state) != 1:
            raise ValueError("Qwen4Exp attention requires GR injection weights from its input hook.")
        branch_output = self._branch_output(attention_output_with_bias)
        output = self.attn_hyper_connection.combine(residual, branch_output, attn_state[0])
        output = self.attn_norm_manager.group_offload(output)
        self.attn_norm_manager = None
        return output

    def _pre_mlp_layernorm_and_residual(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor]]:
        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        normalized, _, _ = super()._pre_mlp_layernorm_and_residual(mixed)
        return normalized, residual, (injection,)

    def _apply_mlp_bda_step(
        self,
        mlp_output_with_bias: tuple[Tensor, Tensor | None],
        residual: Tensor,
        mlp_state: tuple[Tensor, ...] = (),
    ) -> Tensor:
        if len(mlp_state) != 1:
            raise ValueError("Qwen4Exp MoE requires GR injection weights from its input hook.")
        branch_output = self._branch_output(mlp_output_with_bias)
        output = self.mlp_hyper_connection.combine(residual, branch_output, mlp_state[0])
        if self.mlp_norm_manager is not None:
            output = self.mlp_norm_manager.group_offload(output)
            self.mlp_norm_manager = None
        return output

    def _branch_output(self, output_with_bias: tuple[Tensor, Tensor | None]) -> Tensor:
        output, bias = output_with_bias
        if bias is not None:
            output = output + bias
        return F.dropout(output, p=self.hidden_dropout, training=self.training)
