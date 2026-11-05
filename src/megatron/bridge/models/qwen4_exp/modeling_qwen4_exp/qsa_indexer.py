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
from typing import Protocol

import torch
from torch import Tensor, nn


_QUERY_CHUNK_SIZE = 128


class _IndexerConfig(Protocol):
    """Geometry needed by the QSA indexer."""

    hidden_size: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    layernorm_epsilon: float
    sequence_parallel: bool


@torch.jit.script
def _reference_products(query: Tensor, keys: Tensor, rows: list[list[int]], max_blocks: int) -> Tensor:
    products = query.new_zeros((len(rows), query.shape[1], max_blocks))
    for index, row in enumerate(rows):
        query_index, count = row[0], row[1]
        # Preserve the reference GEMM shape; changing its geometry can change a top-k boundary.
        products[index, :, :count].copy_(torch.matmul(query[query_index], keys[:count].transpose(0, 1)))
    return products


def _reference_scores(query: Tensor, keys: Tensor, rows: list[list[int]], max_blocks: int, head_dim: int) -> Tensor:
    if query.is_cuda:
        return torch.relu(_reference_products(query, keys, rows, max_blocks)).sum(dim=1) / math.sqrt(head_dim)
    scores = query.new_full((len(rows), max_blocks), -float("inf"))
    # CPU reductions can change summation order when rows are batched.
    for index, (row, count) in enumerate(rows):
        products = torch.matmul(query[row], keys[:count].transpose(0, 1)).transpose(0, 1)
        scores[index, :count] = torch.relu(products).sum(dim=-1) / math.sqrt(head_dim)
    return scores


def _repair_causal_rows(
    query: Tensor,
    keys: Tensor,
    block_indices: Tensor,
    selected: Tensor,
    reference_rows: list[list[int]],
    block_topk: int,
    head_dim: int,
) -> None:
    if not reference_rows:
        return
    tokens = block_indices.flatten()
    ordinals = torch.full((selected.shape[-1],), tokens.numel(), device=query.device, dtype=torch.long)
    ordinals[tokens] = torch.arange(tokens.numel(), device=query.device)
    for start in range(0, len(reference_rows), _QUERY_CHUNK_SIZE):
        rows = reference_rows[start : start + _QUERY_CHUNK_SIZE]
        max_blocks = max(row[1] for row in rows)
        scores = _reference_scores(query, keys, rows, max_blocks, head_dim)
        row_indices = torch.tensor([row[0] for row in rows], device=query.device)
        counts = torch.tensor([row[1] for row in rows], device=query.device)
        scores.masked_fill_(torch.arange(max_blocks, device=query.device)[None] >= counts[:, None], -float("inf"))
        values, chosen = scores.topk(block_topk + 1, dim=-1)
        chosen = chosen[:, :-1].contiguous()
        tied = (values[:, -2] == values[:, -1]) | ~torch.isfinite(values[:, -2:]).all(dim=-1)
        # A batched top-k may order ties differently; retain the exact one-row tie rule.
        for index in torch.nonzero(tied, as_tuple=False).flatten().tolist():
            chosen[index].copy_(scores[index, : rows[index][1]].topk(block_topk, dim=0).indices)
        selection = selected.index_select(0, row_indices)
        selection &= ordinals[None] >= counts[:, None] * block_indices.shape[1]
        selection.scatter_(1, block_indices[chosen].flatten(1), True)
        selected.index_copy_(0, row_indices, selection)


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

    def __init__(self, config: _IndexerConfig, layer_number: int) -> None:
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
    def _visible_tokens(
        attention_mask: Tensor | None, batch_size: int, sequence_length: int, device: torch.device
    ) -> Tensor:
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

    def _pool_keys(
        self, raw_key: Tensor, block_indices: Tensor, rotary_pos_emb: Tensor | None, *, batch_index: int
    ) -> Tensor:
        pooled_keys = raw_key[:, batch_index].index_select(0, block_indices.flatten())
        pooled_keys = pooled_keys.view(-1, self.compress_ratio, self.index_head_dim)
        pooled_keys = self.k_layernorm(pooled_keys.float().mean(dim=1).to(raw_key.dtype))
        if rotary_pos_emb is not None:
            rotary_batch = 0 if rotary_pos_emb.shape[1] == 1 else batch_index
            block_rope = rotary_pos_emb.index_select(0, block_indices[:, 0])[:, rotary_batch : rotary_batch + 1]
            pooled_keys = _apply_rotary(pooled_keys[:, None, None, :], block_rope).squeeze(1).squeeze(1)
        return pooled_keys

    def _block_scores(self, query: Tensor, pooled_keys: Tensor) -> Tensor:
        scores = torch.matmul(query.float(), pooled_keys.float().transpose(0, 1))
        return torch.relu(scores.transpose(0, 1)).sum(dim=-1) / math.sqrt(self.index_head_dim)

    def _select_query(
        self,
        query: Tensor,
        raw_key: Tensor,
        rotary_pos_emb: Tensor | None,
        visible: Tensor,
        *,
        batch_index: int,
        query_index: int,
    ) -> Tensor:
        visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
        complete_blocks = visible_indices.numel() // self.compress_ratio
        if complete_blocks <= self.block_topk:
            return visible_indices
        block_indices = visible_indices[: complete_blocks * self.compress_ratio].view(
            complete_blocks, self.compress_ratio
        )
        pooled_keys = self._pool_keys(raw_key, block_indices, rotary_pos_emb, batch_index=batch_index)
        scores = self._block_scores(query[query_index, batch_index], pooled_keys)
        chosen_blocks = scores.topk(self.block_topk, dim=0).indices
        selected_indices = block_indices.index_select(0, chosen_blocks).flatten()
        tail = visible_indices[complete_blocks * self.compress_ratio :]
        return torch.cat((selected_indices, tail))

    def _select_causal_keys(
        self,
        query: Tensor,
        raw_key: Tensor,
        rotary_pos_emb: Tensor | None,
        visible: Tensor,
    ) -> Tensor:
        sequence_length, batch_size = query.shape[:2]
        selected = visible.clone()
        first_sparse_query = (self.block_topk + 1) * self.compress_ratio - 1
        for batch_index in range(batch_size):
            valid_keys = visible[batch_index, -1]
            key_indices = torch.nonzero(valid_keys, as_tuple=False).flatten()
            complete_blocks = key_indices.numel() // self.compress_ratio
            if complete_blocks <= self.block_topk:
                continue
            block_indices = key_indices[: complete_blocks * self.compress_ratio].view(
                complete_blocks, self.compress_ratio
            )
            pooled_keys = self._pool_keys(raw_key, block_indices, rotary_pos_emb, batch_index=batch_index)
            keys = pooled_keys.float()
            float_query = query[:, batch_index].float()
            key_ordinals = valid_keys.long().cumsum(dim=0) - 1
            block_counts = (key_ordinals + 1) // self.compress_ratio
            block_ids = torch.arange(complete_blocks, device=query.device)
            ambiguous = torch.zeros(sequence_length, dtype=torch.bool, device=query.device)
            # Bound both FP32 dot products, ReLU/head reductions and scaling. Using
            # sum(abs(q * k)) avoids the very loose max-key bound that forced most
            # queries back to the scalar reference even when their top-k was safe.
            rounding = (self.index_head_dim + self.index_n_heads + 4) * torch.finfo(torch.float32).eps
            gamma = rounding / (1 - rounding)
            difference_factor = 2 * gamma / (1 - gamma) / math.sqrt(self.index_head_dim)
            underflow_bound = (
                2
                * (self.index_head_dim + self.index_n_heads + 4)
                * self.index_n_heads
                * torch.finfo(torch.float32).tiny
                / math.sqrt(self.index_head_dim)
            )
            for start in range(first_sparse_query, sequence_length, _QUERY_CHUNK_SIZE):
                stop = min(start + _QUERY_CHUNK_SIZE, sequence_length)
                queries = float_query[start:stop]
                counts = block_counts[start:stop]
                sparse = counts > self.block_topk
                scores = torch.matmul(queries, keys.transpose(0, 1)).transpose(1, 2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                scores.masked_fill_(block_ids.unsqueeze(0) >= counts.unsqueeze(1), float("-inf"))
                top_scores, top_blocks = scores.topk(self.block_topk + 1, dim=-1)
                absolute_products = torch.matmul(queries.abs(), keys.abs().transpose(0, 1)).sum(dim=1)
                absolute_products.masked_fill_(block_ids.unsqueeze(0) >= counts.unsqueeze(1), 0)
                difference_bound = absolute_products.amax(dim=-1) * difference_factor + underflow_bound
                gap = top_scores[:, -2] - top_scores[:, -1]
                ambiguous[start:stop] = sparse & (
                    (gap <= 2 * difference_bound) | ~torch.isfinite(gap) | ~torch.isfinite(difference_bound)
                )
                if (
                    query.is_cuda
                    and query.dtype not in (torch.bfloat16, torch.float16)
                    and torch.backends.cuda.matmul.allow_tf32
                ):
                    ambiguous[start:stop] |= sparse
                chosen = block_indices[top_blocks[:, :-1]].flatten(1)
                tail = key_ordinals.unsqueeze(0) >= counts.unsqueeze(1) * self.compress_ratio
                selected_rows = visible[batch_index, start:stop] & (tail | ~sparse.unsqueeze(1))
                selected_rows.scatter_(1, chosen, True)
                selected[batch_index, start:stop] = selected_rows & visible[batch_index, start:stop]
            ambiguous_indices = torch.nonzero(ambiguous, as_tuple=False).flatten()
            reference_rows = torch.stack(
                (ambiguous_indices, block_counts.index_select(0, ambiguous_indices)), dim=-1
            ).tolist()
            _repair_causal_rows(
                float_query,
                keys,
                block_indices,
                selected[batch_index],
                reference_rows,
                self.block_topk,
                self.index_head_dim,
            )
        return selected

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Return the exact QSA exclusion mask with bounded causal query/repair batches."""
        sequence_length, batch_size, _ = hidden_states.shape
        if hidden_states.shape[-1] != self.index_qk_proj.in_features:
            raise ValueError(
                f"QSA expects hidden size {self.index_qk_proj.in_features}, got {hidden_states.shape[-1]}."
            )
        visible = self._visible_tokens(attention_mask, batch_size, sequence_length, hidden_states.device)
        if sequence_length < (self.block_topk + 1) * self.compress_ratio:
            return (~visible).unsqueeze(1)
        qk = self.index_qk_proj(hidden_states)
        query_size = self.index_n_heads * self.index_head_dim
        query, raw_key = torch.split(qk, [query_size, self.index_head_dim], dim=-1)
        query = self.q_layernorm(query.reshape(sequence_length, batch_size, self.index_n_heads, self.index_head_dim))

        if rotary_pos_emb is not None:
            query = _apply_rotary(query, rotary_pos_emb[-sequence_length:])

        causal = torch.ones_like(visible[:1]).tril()
        if self.block_topk > 0 and torch.equal(visible, causal & visible[:, -1:, :]):
            return (~self._select_causal_keys(query, raw_key, rotary_pos_emb, visible)).unsqueeze(1)

        # Query-dependent visibility can change block membership, so retain its reference path.
        selected = torch.zeros_like(visible)
        for batch_index in range(batch_size):
            for query_index in range(sequence_length):
                indices = self._select_query(
                    query,
                    raw_key,
                    rotary_pos_emb,
                    visible[batch_index, query_index],
                    batch_index=batch_index,
                    query_index=query_index,
                )
                selected[batch_index, query_index, indices] = True
        return (~selected).unsqueeze(1)
