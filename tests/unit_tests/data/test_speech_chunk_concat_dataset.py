# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.data.speech_chunk_streaming.concat_dataset import WeightedConcatIterableDataset


class _RepeatingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.rank_and_world_size = None

    def __iter__(self):
        while True:
            yield {"dataset_id": self.dataset_id}

    def set_rank_and_world_size(self, rank, world_size):
        self.rank_and_world_size = (rank, world_size)


@pytest.mark.unit
def test_worker_specific_rng_preserves_child_rank_sharding(monkeypatch):
    child_datasets = [_RepeatingDataset(0), _RepeatingDataset(1)]
    dataset = WeightedConcatIterableDataset(child_datasets, [0.5, 0.5])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    dataset.set_rank_and_world_size(rank=3, world_size=8)

    assert [child.rank_and_world_size for child in child_datasets] == [(3, 8), (3, 8)]

    def selection_sequence(worker_id):
        worker_dataset = WeightedConcatIterableDataset(
            [_RepeatingDataset(0), _RepeatingDataset(1)],
            [0.5, 0.5],
        )
        worker_dataset.data_parallel_rank = 3
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=4),
        )
        iterator = iter(worker_dataset)
        next(iterator)
        next(iterator)
        return [next(iterator)["dataset_idx"] for _ in range(16)]

    assert selection_sequence(worker_id=0) != selection_sequence(worker_id=1)