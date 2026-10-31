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

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Qwen4ExpGroupedRMSNorm(nn.Module):
    """Zero-centered RMS normalization over independent residual streams."""

    def __init__(self, hidden_size: int, group_size: int, eps: float) -> None:
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize each residual stream and apply the zero-centered affine weight."""
        input_dtype = hidden_states.dtype
        grouped = hidden_states.float().unflatten(-1, (-1, self.group_size))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + self.eps)
        return (normalized.flatten(-2) * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpGatedResidual(nn.Module):
    """Qwen4Exp low-rank gated residual mixer."""

    def __init__(
        self,
        config,
        layer_number: int | None = None,
        *,
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        del layer_number
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine
        hyper_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            hyper_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.input_mix_weight_down = nn.Linear(hyper_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hyper_hidden_size, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hyper_hidden_size, self.hc_count, bias=False)
        else:
            self.register_module("block_inject_weight", None)

        if config.sequence_parallel:
            for parameter in self.parameters():
                setattr(parameter, "sequence_parallel", True)

    def mix(self, hyper_input: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Normalize and mix residual streams into one transformer-block input."""
        expected_size = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected_size:
            raise ValueError(f"Expected {expected_size} hyper-connection features, got {hyper_input.shape[-1]}.")

        normalized = self.hc_norm(hyper_input)
        low_rank_gate = F.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        gate = torch.sigmoid(self.input_mix_weight_up(low_rank_gate))
        gate = gate.unflatten(-1, (self.hc_count, self.hidden_size))
        block_input = (gate * normalized.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)

        injection = None
        if self.block_inject_weight is not None:
            injection = 2.0 * torch.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return block_input.to(hyper_input.dtype), hyper_input, injection

    def combine(self, residual: Tensor, block_output: Tensor, injection: Tensor) -> Tensor:
        """Inject a single-stream block output into all residual streams."""
        residual_streams = residual.unflatten(-1, (self.hc_count, self.hidden_size))
        output = residual_streams + block_output.unsqueeze(-2) * injection.unsqueeze(-1)
        return output.flatten(-2).to(residual.dtype)

    def forward(self, hyper_input: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Mix streams, returning residual state when combine is enabled."""
        block_input, residual, injection = self.mix(hyper_input)
        if not self.use_combine:
            return block_input
        assert injection is not None
        return block_input, residual, injection
