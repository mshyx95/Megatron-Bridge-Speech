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

"""Deterministic batch assembly for the speech-cascades ALM data pipeline.

The collator mirrors the batch schema of the Cascades ALM baseline so that a
captured reference batch can be reproduced bit-identically. Each example is a
dict with:

- ``input_ids``: ``(1, L)`` or ``(L,)`` token ids. Audio placeholders are
  encoded as negative ids (``-(k + 1)`` repeated ``audio_embed_sizes[k]`` times
  for the ``k``-th audio segment).
- ``labels``: ``(1, L)`` or ``(L,)`` label ids, ``-100`` for ignored positions.
- ``input_embeds``: list of per-audio log-mel feature tensors, each ``(T_k, F)``
  (un-padded). Empty / omitted for pure-text samples.
- ``audio_embed_sizes``: ``(num_audio,)`` per-segment placeholder counts.
- ``audio_frames``: ``(num_audio,)`` per-segment log-mel frame counts.
- ``is_text_sample``: bool; when ``True`` the ``input_embeds`` are not consumed.

The produced batch keys are ``input_ids``, ``labels``, ``inputs_embeds``,
``audio_attention_mask``, ``audio_embed_sizes``, ``audio_frames`` and,
optionally, ``attention_mask``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


IGNORE_INDEX = -100


def pad_features(
    inputs: list[torch.Tensor | np.ndarray],
    pad_value: float = 0.0,
    return_pt: bool = True,
) -> tuple[torch.Tensor | np.ndarray, list[int]]:
    """Right-pad a list of 2D audio feature arrays to a common time length.

    Args:
        inputs: List of ``(T_k, F)`` feature arrays (numpy or torch).
        pad_value: Constant value used for the time padding.
        return_pt: When ``True`` return a torch tensor, otherwise a numpy array.

    Returns:
        A tuple ``(features, nframe)`` where ``features`` has shape
        ``(N, T_max, F)`` and ``nframe`` is the list of original time lengths.
    """
    nframe = [int(i.shape[0]) for i in inputs]
    max_t = max(nframe)

    padded_inputs = [
        np.pad(
            np.asarray(feat),
            ((0, max_t - nframe[i]), (0, 0)),
            "constant",
            constant_values=pad_value,
        )
        for i, feat in enumerate(inputs)
    ]

    stacked = np.stack(padded_inputs)
    if return_pt:
        return torch.from_numpy(stacked), nframe
    return stacked, nframe


@dataclass
class ALMDataCollator:
    """Collate ALM examples into the model's batch schema.

    Args:
        pad_token_id: Token id used to right-pad ``input_ids`` to the batch max.
        audio_key: Batch key under which the padded audio features are stored.
        padding_mask: When ``True`` add a boolean ``attention_mask`` computed as
            ``input_ids != pad_token_id``.
    """

    pad_token_id: int
    audio_key: str = "inputs_embeds"
    padding_mask: bool = False

    def _as_2d(self, value: torch.Tensor) -> torch.Tensor:
        return value if value.dim() == 2 else value.unsqueeze(0)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(examples) == 0:
            raise ValueError("ALMDataCollator received an empty example list")

        audios: list[torch.Tensor | np.ndarray] = []
        for e in examples:
            if e.get("input_embeds") is not None and not e.get("is_text_sample", False):
                audios.extend(e["input_embeds"])

        if len(audios) > 0:
            input_embeds, nframe = pad_features(audios, return_pt=True)
            audio_attention_mask = input_embeds.new_ones(input_embeds.size()[:2], dtype=torch.long)
            for i in range(input_embeds.shape[0]):
                audio_attention_mask[i, nframe[i] :].fill_(0)
        else:
            input_embeds = torch.tensor([])
            audio_attention_mask = torch.tensor([])

        input_ids = [self._as_2d(e["input_ids"]) for e in examples]
        labels = [self._as_2d(e["labels"]) for e in examples]
        audio_embed_sizes = [e.get("audio_embed_sizes", torch.tensor([], dtype=torch.long)) for e in examples]
        audio_frames = [e.get("audio_frames", torch.tensor([], dtype=torch.long)) for e in examples]

        max_input_length = max(i.shape[1] for i in input_ids)
        input_ids = [
            torch.cat([i, i.new_full((i.size(0), max_input_length - i.size(1)), self.pad_token_id)], dim=1)
            for i in input_ids
        ]
        max_label_length = max(lab.shape[1] for lab in labels)
        labels = [
            torch.cat([lab, lab.new_full((lab.size(0), max_label_length - lab.size(1)), IGNORE_INDEX)], dim=1)
            for lab in labels
        ]

        batch: dict[str, Any] = {}
        batch["input_ids"] = torch.cat(input_ids)
        batch["labels"] = torch.cat(labels)
        if self.padding_mask:
            batch["attention_mask"] = batch["input_ids"] != self.pad_token_id
        batch[self.audio_key] = input_embeds
        batch["audio_attention_mask"] = audio_attention_mask
        batch["audio_embed_sizes"] = torch.cat(audio_embed_sizes).long()
        batch["audio_frames"] = torch.cat(audio_frames).long()
        return batch
