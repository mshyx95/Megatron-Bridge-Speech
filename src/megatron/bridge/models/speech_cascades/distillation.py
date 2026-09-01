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

"""Token-aligned speech->text knowledge-distillation loss for the Cascades speech ALM.

Faithful port of the production ``phyagi`` KLD used in the stage-2 S2T recipes: the student is the
speech-conditioned pathway and the teacher is a text-only forward of the SAME frozen language model
with each audio span replaced by its aligned transcript. Both views keep identical response tokens,
so the per-token hidden states line up and the frozen output head maps them to a shared vocabulary
distribution. The KL is computed in vocab-token chunks (the vocabulary is ~248k wide) with per-chunk
gradient checkpointing so the full logits are never materialized at once.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_kl_divergence(
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """KL(teacher || student) over response tokens, computed in chunks of ``chunk_size`` tokens.

    Args:
        student_hidden_states: ``[N, H]`` hidden states of the speech pathway at the N response tokens.
        teacher_hidden_states: ``[N, H]`` hidden states of the frozen text pathway at the same tokens.
        logits_fn: Maps ``[chunk, H]`` hidden states to ``[chunk, V]`` logits via the (frozen, shared)
            output head. Kept as a callable so the caller controls tensor/model parallel projection.
        temperature: Softmax temperature applied to both distributions before the KL.
        chunk_size: Number of response tokens per chunk (bounds peak logits memory).

    Returns:
        Scalar KL loss averaged over the N response tokens.
    """
    if student_hidden_states.shape != teacher_hidden_states.shape:
        raise ValueError(
            "Student and teacher response hidden states must have the same shape, "
            f"got {student_hidden_states.shape} and {teacher_hidden_states.shape}"
        )
    if student_hidden_states.numel() == 0:
        raise ValueError("Knowledge distillation requires at least one response token")

    def kl_sum(student_chunk: torch.Tensor, teacher_chunk: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(logits_fn(student_chunk).float() / temperature, dim=-1)
        with torch.no_grad():
            teacher_probs = F.softmax(logits_fn(teacher_chunk).float() / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="sum")

    total_loss = student_hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden_states.shape[0], chunk_size):
        student_chunk = student_hidden_states[start : start + chunk_size]
        teacher_chunk = teacher_hidden_states[start : start + chunk_size]
        if torch.is_grad_enabled() and student_chunk.requires_grad:
            total_loss = total_loss + checkpoint(kl_sum, student_chunk, teacher_chunk, use_reentrant=False)
        else:
            total_loss = total_loss + kl_sum(student_chunk, teacher_chunk)
    return total_loss / student_hidden_states.shape[0]


def distillation_loss(
    kl_loss: torch.Tensor,
    ce_loss: torch.Tensor,
    distillation_weight: float,
) -> torch.Tensor:
    """Combine the KL and cross-entropy terms: ``w * KL + (1 - w) * CE`` (w == 1.0 -> pure KL)."""
    if distillation_weight >= 1.0:
        return kl_loss
    return distillation_weight * kl_loss + (1.0 - distillation_weight) * ce_loss
