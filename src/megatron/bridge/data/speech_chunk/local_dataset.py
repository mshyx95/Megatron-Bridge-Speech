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

"""Local iterable dataset for the speech-cascades ALM pipeline.

This is the offline, deterministic counterpart to the Azure blob streaming
dataset used in production. It reads pre-serialized example records from local
``.pt`` shards (each a ``list[dict]`` of ALM examples) and yields collated
batches via :class:`ALMDataCollator`. Shards are split across data-parallel
ranks and, within a rank, grouped into fixed-size micro-batches. The live
network streaming path (manifest + blob download) requires credentials and is
intentionally out of scope here; both paths share the identical
example-record -> collator contract, so batches are interchangeable.
"""

import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset

from megatron.bridge.data.speech_chunk.collator import ALMDataCollator


logger = logging.getLogger(__name__)


class ALMLocalChunkDataset(IterableDataset):
    """Iterable dataset over local ALM example-record shards.

    Args:
        shard_paths: Paths to ``.pt`` files, each holding a ``list[dict]`` of
            ALM example records (see :mod:`collator` for the example schema).
        collator: Collator that assembles a list of examples into a batch.
        batch_size: Number of examples per emitted batch.
        rank: Data-parallel rank; shards are assigned ``shard_paths[rank::world_size]``.
        world_size: Data-parallel world size.
        shuffle: When ``True`` shuffle shard order (and examples within a shard)
            deterministically per epoch using ``seed``.
        seed: Base RNG seed for shuffling.
        drop_last: When ``True`` drop a trailing partial micro-batch.
    """

    def __init__(
        self,
        shard_paths: list[str],
        collator: ALMDataCollator,
        *,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if world_size <= 0 or not (0 <= rank < world_size):
            raise ValueError(f"invalid rank {rank} for world_size {world_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(shard_paths) == 0:
            raise ValueError("shard_paths must be non-empty")

        self.shard_paths = list(shard_paths)
        self.collator = collator
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive deterministic shuffling."""
        self.epoch = epoch

    def _iter_examples(self) -> Iterator[dict[str, Any]]:
        order = list(range(len(self.shard_paths)))
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]

        local_order = order[self.rank :: self.world_size]
        for shard_idx in local_order:
            records = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
            if not isinstance(records, list):
                raise ValueError(f"shard {self.shard_paths[shard_idx]} must hold a list of examples")
            example_order = list(range(len(records)))
            if self.shuffle:
                g = torch.Generator().manual_seed(self.seed + self.epoch + shard_idx + 1)
                example_order = [example_order[i] for i in torch.randperm(len(example_order), generator=g).tolist()]
            for i in example_order:
                yield records[i]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        for example in self._iter_examples():
            buffer.append(example)
            if len(buffer) == self.batch_size:
                yield self.collator(buffer)
                buffer = []
        if buffer and not self.drop_last:
            yield self.collator(buffer)
