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


def _weighted_depth_reduction(
    depth_losses: list[torch.Tensor], depth_weights: list[float] | None
) -> torch.Tensor:
    """Reduce per-depth losses to a scalar (normalized depth-weighted sum, or plain mean).

    Faithful port of the reference ``_weighted_mtp_loss``: with ``depth_weights`` the losses are
    combined as ``sum_k (w_k / sum(w)) * loss_k`` (deeper depths made quieter); without weights the
    reduction is the unweighted mean across depths.
    """
    stacked = torch.stack(depth_losses)
    if depth_weights is None:
        return stacked.mean()
    if len(depth_weights) < len(depth_losses):
        raise ValueError(
            f"mtp_loss_depth_weights has {len(depth_weights)} values, "
            f"but {len(depth_losses)} MTP depth losses were computed"
        )
    weights = torch.tensor(depth_weights[: len(depth_losses)], device=stacked.device, dtype=stacked.dtype)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("mtp_loss_depth_weights must sum to a positive value")
    return (stacked * (weights / weight_sum)).sum()


def mtp_depth_loss(
    mtp_hidden_states: list[torch.Tensor],
    main_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    temperature: float,
    chunk_size: int,
    mtp_distillation_weight: float,
    mtp_loss_weight: float,
    mtp_loss_depth_weights: list[float] | None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Per-depth multi-token-prediction KL+CE loss (faithful port of the reference MTP head loss).

    For each depth ``k = 1..D`` the depth head predicts the token ``k`` positions ahead. The loss at
    each depth mixes a cross-entropy against the shifted labels with a **self-distillation** KL whose
    teacher is the main model's own output hidden states (``main_hidden_states``, ``h^0``), detached
    and shifted by ``k`` so it scores the *same* target token. Matching the reference exactly:

    * depth label shift: ``labels[..., k + 1:]`` (here reconstructed from the already-causally-shifted
      collator labels as ``labels[..., k:-1]`` / ``loss_mask[..., k:-1]``);
    * teacher target: ``main_hidden_states[..., k:-1, :]`` detached;
    * ``depth_kl`` is scaled by ``T^2`` (unlike the main-path KL, which has no ``T^2``);
    * ``depth_loss_k = (1 - mtp_w) * depth_ce_k + mtp_w * depth_kl_k * T^2``.

    The depth losses are reduced by ``mtp_loss_depth_weights`` and scaled by ``mtp_loss_weight``, so
    the returned scalar already equals ``mtp_loss_weight * sum_k w_k * depth_loss_k`` (the caller adds
    it straight onto the main loss; no further ``mtp_loss_weight`` factor).

    Args:
        mtp_hidden_states: ``D`` per-depth student hidden states, sbhd; entry ``k-1`` (depth ``k``) is
            ``[S - k, B, H]`` and post-final-norm (ready for ``logits_fn``).
        main_hidden_states: Main decoder output ``h^0``, bshd ``[B, S, H]``; detached here to form the
            per-depth KL teacher target.
        labels: Causally-shifted, clamped next-token labels ``[B, S]`` (``labels[i]`` is the target at
            position ``i``); ``labels[..., k:-1]`` recovers the reference depth-``k`` targets.
        loss_mask: Float mask ``[B, S]`` (``1`` at supervised targets); its ``-100`` positions are the
            reference ``depth_label_mask``.
        logits_fn: Maps ``[N, H]`` hidden states to ``[N, V]`` logits via the shared frozen output head.
        temperature: Softmax temperature ``T`` for the KL (and the ``T^2`` scale on each depth KL).
        chunk_size: Response-token chunk size for :func:`chunked_kl_divergence`.
        mtp_distillation_weight: ``mtp_w`` mixing the depth CE and depth KL. ``0`` disables the KL.
        mtp_loss_weight: Global scale applied to the depth-weighted sum.
        mtp_loss_depth_weights: Optional per-depth weights (e.g. ``[0.50, 0.25, 0.15, 0.10]``).

    Returns:
        Tuple of the aggregated MTP loss scalar and a dict of per-depth components
        (``"depth_losses"``, ``"depth_ce"``, ``"depth_kl"``) as detached scalars for logging.
    """
    teacher_full = main_hidden_states.detach()
    depth_losses: list[torch.Tensor] = []
    depth_ce_list: list[torch.Tensor] = []
    depth_kl_list: list[torch.Tensor] = []

    for depth_idx, depth_hidden_sbhd in enumerate(mtp_hidden_states):
        depth = depth_idx + 1
        # Reconstruct the reference depth-`depth` slice from the already-shifted collator labels:
        # labels[..., depth + 1:] == shifted_labels[..., depth:-1]; the trailing `-1` drops the
        # always-ignored last shifted position.
        label_ids = labels[:, depth:-1]  # [B, S - depth - 1]
        if label_ids.shape[1] == 0:
            break
        depth_mask = loss_mask[:, depth:-1].bool()  # [B, S - depth - 1]

        student_seq = depth_hidden_sbhd.transpose(0, 1)[:, :-1, :]  # [B, S - depth - 1, H]
        teacher_seq = teacher_full[:, depth:-1, :]  # [B, S - depth - 1, H]

        # Reference drops the first supervised token across the flattened batch (`[1:]`).
        student_sel = student_seq[depth_mask][1:].contiguous()
        labels_sel = label_ids[depth_mask].reshape(-1)[1:]
        teacher_sel = teacher_seq[depth_mask][1:].contiguous()
        if student_sel.numel() == 0:
            continue

        depth_ce = F.cross_entropy(logits_fn(student_sel).float(), labels_sel)
        depth_ce_list.append(depth_ce.detach())

        if mtp_distillation_weight > 0:
            depth_kl = chunked_kl_divergence(student_sel, teacher_sel, logits_fn, temperature, chunk_size)
            depth_kl = depth_kl * (temperature**2)
            depth_kl_list.append(depth_kl.detach())
            depth_loss = (1.0 - mtp_distillation_weight) * depth_ce + mtp_distillation_weight * depth_kl
        else:
            depth_loss = depth_ce
        depth_losses.append(depth_loss)

    if not depth_losses:
        aggregated = main_hidden_states.new_zeros((), dtype=torch.float32)
    else:
        aggregated = mtp_loss_weight * _weighted_depth_reduction(depth_losses, mtp_loss_depth_weights)
    components = {
        "depth_losses": [loss.detach() for loss in depth_losses],
        "depth_ce": depth_ce_list,
        "depth_kl": depth_kl_list,
    }
    return aggregated, components
