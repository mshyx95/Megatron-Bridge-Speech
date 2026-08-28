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

"""Map-style local ALM dataset and its :class:`DatasetProvider` for the training loop.

``ALMLocalDataset`` holds a flat list of pre-serialized ALM example records (loaded from ``.pt``
shards) and exposes a ``collate_fn`` that the bridge dataloader picks up automatically.
``ALMLocalDatasetProvider`` wires it into the training ``setup()`` via ``build_datasets``.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from megatron.bridge.data.speech_chunk.collator import ALMDataCollator
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


class ALMLocalDataset(Dataset):
    """Map-style dataset over pre-serialized ALM example records.

    Args:
        shard_paths: ``.pt`` files, each holding a ``list[dict]`` of ALM example records.
        collator: Collator exposed as ``collate_fn`` for the dataloader.
    """

    def __init__(self, shard_paths: list[str], collator: ALMDataCollator) -> None:
        self.records: list[dict[str, Any]] = []
        for path in shard_paths:
            records = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(records, list):
                raise ValueError(f"shard {path} must hold a list of example records")
            self.records.extend(records)
        if len(self.records) == 0:
            raise ValueError("ALMLocalDataset built with zero example records")
        self.collate_fn = collator

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index % len(self.records)]


@dataclass(kw_only=True)
class ALMLocalDatasetProvider(DatasetProvider):
    """Provider that builds map-style ALM datasets from local example-record shards.

    Args:
        seq_length: Sequence length (must match ``model.seq_length``).
        shard_paths: Train-split ``.pt`` shard paths.
        pad_token_id: Padding id for ``input_ids`` (see :class:`ALMDataCollator`).
        audio_key: Batch key for the padded audio features.
        padding_mask: Whether the collator emits a boolean ``attention_mask``.
        val_shard_paths / test_shard_paths: Optional validation / test shards.
    """

    seq_length: int
    shard_paths: list[str]
    pad_token_id: int = 0
    audio_key: str = "inputs_embeds"
    padding_mask: bool = True
    val_shard_paths: list[str] | None = None
    test_shard_paths: list[str] | None = None
    skip_getting_attention_mask_from_dataset: bool = True

    def _collator(self) -> ALMDataCollator:
        return ALMDataCollator(
            pad_token_id=self.pad_token_id, audio_key=self.audio_key, padding_mask=self.padding_mask
        )

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[ALMLocalDataset | None, ALMLocalDataset | None, ALMLocalDataset | None]:
        collator = self._collator()
        train = ALMLocalDataset(self.shard_paths, collator) if self.shard_paths else None
        valid = ALMLocalDataset(self.val_shard_paths, collator) if self.val_shard_paths else None
        test = ALMLocalDataset(self.test_shard_paths, collator) if self.test_shard_paths else None
        return train, valid, test
