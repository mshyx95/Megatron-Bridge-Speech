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

"""Teacher text-forward construction for speech->text knowledge distillation.

The KD teacher is a text-only forward of the same frozen language model with each audio span in
the student input replaced by its aligned transcript token ids. In the speech-cascades batch
schema the student ``input_ids`` marks the ``k``-th audio segment with the negative placeholder id
``-(k + 1)`` repeated ``audio_embed_sizes[k]`` times (see :mod:`.collator`). This module rewrites
those placeholder runs into transcript tokens so the response (answer) tokens stay byte-identical
between the two views and the per-token hidden states line up for the KL. This mirrors the
production ``phyagi`` teacher build, which performs the same substitution at the template level.
"""

from collections.abc import Sequence

import torch


IGNORE_INDEX = -100


def build_teacher_text_example(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    transcript_ids: Sequence[torch.Tensor],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single teacher (text-only) example from a student ALM example.

    Each contiguous run of an audio-placeholder id ``-(k + 1)`` in ``input_ids`` is replaced by the
    ``k``-th entry of ``transcript_ids``; every non-placeholder token (and its label) is copied
    verbatim, so the response segment is byte-identical to the student's. Placeholder positions carry
    ``ignore_index`` labels on the teacher side (they are never supervised).

    Args:
        input_ids: Student token ids, shape ``(L,)`` or ``(1, L)``. Audio segments are negative ids.
        labels: Student (unshifted) labels, same shape as ``input_ids``; ``ignore_index`` for
            non-response positions and the answer token id at each response position.
        transcript_ids: One 1D token-id tensor per audio segment, ordered by segment index ``k``.
        ignore_index: Label value marking ignored positions.

    Returns:
        A tuple ``(teacher_input_ids, teacher_labels)`` of 1D ``long`` tensors on the input device.
    """
    ids = input_ids.reshape(-1)
    labs = labels.reshape(-1)
    if ids.shape != labs.shape:
        raise ValueError(f"input_ids and labels must have the same length, got {ids.shape} and {labs.shape}")

    ids_list = ids.tolist()
    labs_list = labs.tolist()
    teacher_ids: list[int] = []
    teacher_labels: list[int] = []

    position = 0
    length = len(ids_list)
    while position < length:
        token = ids_list[position]
        if token < 0:
            segment = -token - 1
            if segment >= len(transcript_ids):
                raise ValueError(
                    f"audio placeholder id {token} maps to segment {segment} but only "
                    f"{len(transcript_ids)} transcript(s) were provided"
                )
            run_end = position
            while run_end < length and ids_list[run_end] == token:
                run_end += 1
            segment_ids = transcript_ids[segment].reshape(-1).tolist()
            teacher_ids.extend(segment_ids)
            teacher_labels.extend([ignore_index] * len(segment_ids))
            position = run_end
        else:
            teacher_ids.append(token)
            teacher_labels.append(labs_list[position])
            position += 1

    device = input_ids.device
    return (
        torch.tensor(teacher_ids, dtype=torch.long, device=device),
        torch.tensor(teacher_labels, dtype=torch.long, device=device),
    )


def collate_teacher_text_examples(
    per_example_input_ids: Sequence[torch.Tensor],
    per_example_labels: Sequence[torch.Tensor],
    pad_token_id: int,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, torch.Tensor]:
    """Right-pad and stack per-example teacher tensors into a batch.

    Args:
        per_example_input_ids: List of 1D teacher input-id tensors (from
            :func:`build_teacher_text_example`).
        per_example_labels: List of 1D teacher label tensors, aligned with ``per_example_input_ids``.
        pad_token_id: Pad id for ``teacher_input_ids``.
        ignore_index: Pad value for ``teacher_labels``.

    Returns:
        Dict with ``teacher_input_ids`` / ``teacher_labels`` (``long``, ``(B, T)``) and
        ``teacher_attention_mask`` (``bool``, ``(B, T)``, ``True`` on real tokens).
    """
    batch_size = len(per_example_input_ids)
    max_length = max(ids.numel() for ids in per_example_input_ids)

    teacher_input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    teacher_labels = torch.full((batch_size, max_length), ignore_index, dtype=torch.long)
    teacher_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, (ids, labs) in enumerate(zip(per_example_input_ids, per_example_labels)):
        valid = ids.numel()
        teacher_input_ids[index, :valid] = ids.reshape(-1)
        teacher_labels[index, :valid] = labs.reshape(-1)
        teacher_attention_mask[index, :valid] = True

    return {
        "teacher_input_ids": teacher_input_ids,
        "teacher_labels": teacher_labels,
        "teacher_attention_mask": teacher_attention_mask,
    }
