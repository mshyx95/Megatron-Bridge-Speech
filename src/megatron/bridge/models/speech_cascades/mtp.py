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

"""Multi-token-prediction (MTP) depth heads for the Cascades speech ALM.

Faithful port of the production ``phyagi`` ``Qwen3_5MultiTokenPredictor`` depth-block construction
(``modeling_qwen3_5.py``): ``mtp_num_hidden_layers`` extra decoder layers predict tokens at depths
``k = 1..D``. Each depth ``k`` consumes the previous depth's hidden state (dropping its last
position) and the token embeddings shifted by ``k``, normalizes both, concatenates them, projects
``2H -> H`` through a shared ``fc``, runs one depth decoder layer, and applies a final RMSNorm to
produce the depth-``k`` hidden state. The depth blocks are built via a caller-supplied ``layer_builder``
so the composing model can build Megatron-Core ``TransformerLayer`` instances from its own decoder
spec while unit tests can substitute a lightweight CPU stub. When ``source_state_dict`` is given each
depth block is initialized from it (init-from-first), mirroring ``mtp_init_extra_layers_from_first``.

All tensors are sequence-first ``[S, B, H]`` (sbhd), matching the Megatron-Core decoder stream.
"""

from collections.abc import Callable

import torch
from torch import nn


class _RMSNorm(nn.Module):
    """Minimal RMSNorm mirroring the reference MTP norms (CPU-friendly, no Transformer-Engine dep)."""

    def __init__(self, hidden_size: int, *, eps: float = 1e-6, zero_centered_gamma: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        init = torch.zeros(hidden_size) if zero_centered_gamma else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight.float()
        if self.zero_centered_gamma:
            weight = weight + 1.0
        return (weight * hidden_states).to(input_dtype)


class SpeechCascadesMTP(nn.Module):
    """``num_depth_layers`` multi-token-prediction depth heads over a shared projection and norms.

    Args:
        hidden_size: Model hidden dimension ``H``.
        num_depth_layers: Number of MTP depths ``D`` (>= 1).
        layer_builder: Zero-arg callable that returns one depth decoder layer. In the composed model
            this builds a Megatron-Core ``TransformerLayer`` from the language-model decoder spec;
            in tests it may return a lightweight stub with the same call contract.
        source_state_dict: Optional ``state_dict`` copied into every depth block at construction to
            realize init-from-first. Must be load-compatible with the modules ``layer_builder`` returns.
        norm_eps: Epsilon for the shared RMSNorms.
        zero_centered_gamma: Whether the RMSNorm gain is ``1 + weight`` (Qwen3.5 convention).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_depth_layers: int,
        layer_builder: Callable[[], nn.Module],
        source_state_dict: dict[str, torch.Tensor] | None = None,
        norm_eps: float = 1e-6,
        zero_centered_gamma: bool = False,
    ) -> None:
        super().__init__()
        if num_depth_layers < 1:
            raise ValueError("num_depth_layers must be >= 1 to build MTP depth heads")
        self.num_depth_layers = num_depth_layers
        self.hidden_size = hidden_size

        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.enorm = _RMSNorm(hidden_size, eps=norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.hnorm = _RMSNorm(hidden_size, eps=norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.final_norm = _RMSNorm(hidden_size, eps=norm_eps, zero_centered_gamma=zero_centered_gamma)
        self.layers = nn.ModuleList(layer_builder() for _ in range(num_depth_layers))

        if source_state_dict is not None:
            self.init_layers_from_state_dict(source_state_dict)

    def init_layers_from_state_dict(self, source_state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize every depth block from ``source_state_dict`` (init-from-first)."""
        for layer in self.layers:
            layer.load_state_dict(source_state_dict)

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        rotary_pos_emb_fn: Callable[[int], torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Compute the ``D`` per-depth hidden states.

        Args:
            full_hidden_states: Main decoder output ``h^0``, sbhd ``[S, B, H]`` (post final norm).
            inputs_embeds: Token embeddings fed to the main decoder, sbhd ``[S, B, H]``.
            rotary_pos_emb_fn: Optional callable mapping a sequence length to the rotary embedding
                tensor for the depth decoder layer. ``None`` skips rotary (e.g. CPU stub layers).
            attention_mask: Optional attention mask forwarded to every depth decoder layer.

        Returns:
            List of ``D`` sbhd tensors; entry ``k-1`` (depth ``k``) has shape ``[S - k, B, H]``.
        """
        outputs: list[torch.Tensor] = []
        prev_hidden = full_hidden_states
        for depth_idx in range(self.num_depth_layers):
            depth = depth_idx + 1
            # Drop the previous depth's last position and shift the embeddings by `depth`; both
            # views then have sequence length S - depth so depth-k position i predicts token i+depth.
            hidden_in = prev_hidden[:-1].contiguous()
            embeds_in = inputs_embeds[depth:].contiguous()
            seq_len = embeds_in.shape[0]
            if seq_len == 0:
                break

            embeds_in = self.enorm(embeds_in)
            hidden_norm = self.hnorm(hidden_in)
            fused = torch.cat([embeds_in, hidden_norm], dim=-1)
            fused = self.fc(fused)

            rotary_pos_emb = rotary_pos_emb_fn(seq_len) if rotary_pos_emb_fn is not None else None
            layer_out = self.layers[depth_idx](
                fused, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
            )
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden = self.final_norm(hidden)

            outputs.append(hidden)
            prev_hidden = hidden
        return outputs
