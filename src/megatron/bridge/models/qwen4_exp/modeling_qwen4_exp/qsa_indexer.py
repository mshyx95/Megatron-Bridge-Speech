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

import math

import torch
from torch import Tensor, nn


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary(hidden_states: Tensor, rotary_pos_emb: Tensor) -> Tensor:
    rotary_dim = rotary_pos_emb.shape[-1]
    rotated, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cosine = rotary_pos_emb.cos().to(hidden_states.dtype)
    sine = rotary_pos_emb.sin().to(hidden_states.dtype)
    rotated = (rotated * cosine) + (_rotate_half(rotated) * sine)
    return torch.cat((rotated, passthrough), dim=-1)


class Qwen4ExpRMSNorm(nn.Module):
    """Qwen RMS normalization with a zero-centered affine weight."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the last dimension in float32."""
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpQSAIndexer(nn.Module):
    """Select causal key blocks for Qwen sparse attention."""

    def __init__(self, config, layer_number: int) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.layernorm_epsilon)
        self.k_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.layernorm_epsilon)
        if config.sequence_parallel:
            for parameter in self.parameters():
                setattr(parameter, "sequence_parallel", True)

    @staticmethod
    def _visible_tokens(attention_mask: Tensor | None, batch_size: int, sequence_length: int, device) -> Tensor:
        causal = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device).tril()
        causal = causal.unsqueeze(0).expand(batch_size, -1, -1)
        if attention_mask is None:
            return causal
        if attention_mask.ndim == 2:
            valid_keys = attention_mask.bool().unsqueeze(1)
            return causal & valid_keys
        if attention_mask.ndim != 4:
            raise ValueError(f"QSA expects a 2D or 4D attention mask, got shape {tuple(attention_mask.shape)}.")
        mask = attention_mask[:, 0]
        if mask.shape[0] == 1 and batch_size != 1:
            mask = mask.expand(batch_size, -1, -1)
        visible = ~mask if mask.dtype == torch.bool else mask == 0
        return causal & visible[..., -sequence_length:, -sequence_length:]

    def forward(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Return a boolean mask where ``True`` marks keys excluded by QSA."""
        sequence_length, batch_size, _ = hidden_states.shape
        qk = self.index_qk_proj(hidden_states)
        query_size = self.index_n_heads * self.index_head_dim
        query, raw_key = torch.split(qk, [query_size, self.index_head_dim], dim=-1)
        query = self.q_layernorm(query.unflatten(-1, (self.index_n_heads, self.index_head_dim)))

        if rotary_pos_emb is not None:
            query = _apply_rotary(query, rotary_pos_emb[-sequence_length:])

        visible = self._visible_tokens(attention_mask, batch_size, sequence_length, hidden_states.device)
        selected = torch.zeros_like(visible)
        for batch_index in range(batch_size):
            for query_index in range(sequence_length):
                visible_indices = torch.nonzero(visible[batch_index, query_index], as_tuple=False).flatten()
                complete_blocks = visible_indices.numel() // self.compress_ratio
                if complete_blocks:
                    block_indices = visible_indices[: complete_blocks * self.compress_ratio].view(
                        complete_blocks, self.compress_ratio
                    )
                    pooled_keys = raw_key[:, batch_index].index_select(0, block_indices.flatten())
                    pooled_keys = pooled_keys.view(complete_blocks, self.compress_ratio, self.index_head_dim)
                    pooled_keys = self.k_layernorm(pooled_keys.float().mean(dim=1).to(raw_key.dtype))
                    if rotary_pos_emb is not None:
                        block_positions = block_indices[:, 0]
                        rotary_batch = 0 if rotary_pos_emb.shape[1] == 1 else batch_index
                        block_rope = rotary_pos_emb.index_select(0, block_positions)[
                            :, rotary_batch : rotary_batch + 1
                        ]
                        pooled_keys = _apply_rotary(pooled_keys[:, None, None, :], block_rope).squeeze(1).squeeze(1)
                    scores = torch.matmul(
                        query[query_index, batch_index].float(), pooled_keys.float().transpose(0, 1)
                    ).transpose(0, 1)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                    chosen_blocks = scores.topk(min(self.block_topk, complete_blocks), dim=0).indices
                    selected_indices = block_indices.index_select(0, chosen_blocks).flatten()
                else:
                    selected_indices = visible_indices.new_empty(0)
                tail = visible_indices[complete_blocks * self.compress_ratio :]
                selected[batch_index, query_index, torch.cat((selected_indices, tail))] = True
        return (~selected).unsqueeze(1)
