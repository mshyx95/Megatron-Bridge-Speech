# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import bisect
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset

from megatron.bridge.data.speech_chunk_streaming._utils.file_utils import load_json_file, save_json_file
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)


class WeightedConcatDataset(ConcatDataset):
    """Weighted concatenation of datasets."""

    def __init__(self, datasets: Iterable[Dataset], weights: Iterable[float]) -> None:
        """Initialize the dataset by concatenating the given datasets according
        to the given weights.

        For a given dataset, the number of resulting samples is computed as
        ``ceil(len(dataset) * weight)``. If the resulting number of samples is
        greater than the length of the dataset, the dataset is concatenated
        multiple times to reach the desired number of samples.

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        datasets = list(datasets)
        weights = list(weights)
        assert len(datasets) == len(weights), "`datasets` and `weights` must have the same length."

        # Sequence length attribute is optional but useful for logging
        self.seq_len = getattr(datasets[0], "seq_len", None)

        weighted_datasets = []
        for dataset, weight in zip(datasets, weights):
            n_samples = math.ceil(len(dataset) * weight)

            if n_samples > len(dataset):
                n_concats = n_samples // len(dataset)
                n_remainder = n_samples % len(dataset)

                samples_idx = np.concatenate(
                    [np.concatenate([np.arange(len(dataset))] * n_concats), np.arange(n_remainder)]
                )
            else:
                samples_idx = np.arange(n_samples)

            weighted_datasets.append(torch.utils.data.Subset(dataset, samples_idx))

        super().__init__(weighted_datasets)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"Absolute index {idx} out of range for dataset with size of {len(self)}.")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample = self.datasets[dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["idx"] = idx
            sample["dataset_idx"] = dataset_idx

        return sample


class SequentialWeightedConcatDataset(WeightedConcatDataset):
    """Sequential weighted concatenation of datasets."""

    def __init__(self, datasets: Iterable[Dataset], weights: Iterable[float]) -> None:
        """Initialize the dataset by sequentially concatenating the given datasets according
        to the given weights.

        The concatenating order is determined by the order of the given datasets. If a given
        dataset has a weight greater than 1, it will be sequentially concatenated multiple times.

        Examples:
            >>> datasets = [A, B, C]
            >>> weights = [1.5, 0.3, 0.2]
            >>> dataset = SequentialWeightedConcatDataset(datasets, weights)
            >>> assert len(dataset[:len(A) * 1.5]) == len(A)

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        super().__init__(datasets, weights)

        self.dataset_idx = 0
        self.consumed_samples = 0
        self.cumulative_markers = [np.zeros(size) for size in self.cumulative_sizes]

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"Absolute index {idx} out of range for dataset with size of {len(self)}.")
            idx = len(self) + idx

        # When `consumed_samples` is greater than the length of current dataset,
        # we need to reset it and move to next dataset
        if self.consumed_samples == self.cumulative_sizes[self.dataset_idx]:
            self.dataset_idx += 1

        if idx >= self.cumulative_sizes[self.dataset_idx]:
            # When `idx` is equal or bigger than current dataset size,
            # we need to sample a valid index from current dataset
            logger.warning(
                f"idx {idx} exceeded the cumulative_size ({self.cumulative_sizes[self.dataset_idx]}) for "
                f"dataset_idx {self.dataset_idx}! random index will be used!"
            )
            sample_idx = np.random.choice(np.argwhere(self.cumulative_markers[self.dataset_idx] == 0)[0], 1)[0]
        else:
            # When `idx` is smaller than current dataset size,
            # we need to find the corresponding index in current dataset
            if self.dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[self.dataset_idx - 1]

        self.cumulative_markers[self.dataset_idx][sample_idx] = 1
        self.consumed_samples += 1

        sample = self.datasets[self.dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["idx"] = idx
            sample["dataset_idx"] = self.dataset_idx

        return sample


class WeightedConcatIterableDataset(IterableDataset):
    """Weighted concatenation of iterable datasets."""

    def __init__(self, datasets: Iterable[IterableDataset], weights: Iterable[float]) -> None:
        """Initialize the dataset by sequentially concatenating the given datasets according
        to the given weights.

        The weights are normalized and reflect the sampling rate of each dataset. However, if
        the datasets are `.npy` files, the weights are adjusted to reflect the number of times
        the full dataset is replicated.

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        self.datasets = datasets
        self.weights = np.array(weights)
        logger.info(f"Dataset weights: {self.weights}")

        # Sequence length attribute is optional but useful for logging
        self.seq_len = getattr(datasets[0], "seq_len", 1)

        self.datasets_iter: Optional[List[Iterator[Any]]] = None

        self.global_worker_id = 0

        # If all datasets are `.npy` files, adjust the weights to reflect the number of times
        self._adjust_npy_weights()

    def _adjust_npy_weights(self) -> None:
        is_original_size_available = all(
            [getattr(dataset, "original_size", None) is not None for dataset in self.datasets]
        )
        if not is_original_size_available:
            return

        sizes = np.array([dataset.original_size for dataset in self.datasets])
        sizes_weighted = sizes * self.weights

        self.weights = sizes_weighted / np.sum(sizes_weighted)
        logger.info(f"Total number of original size of the datasets: {np.sum(sizes_weighted)}.")
        logger.info(f"Adjusted weights to reflect the number of times the full dataset is replicated: {self.weights}.")

    def _get_seq_len(self, sample: Any) -> int:
        # If `sample` is a list or tuple, we assume the first element is the input
        if isinstance(sample, (list, tuple)):
            sample = sample[0]
        # If `sample` is a dictionary, we assume the first key is the input
        elif isinstance(sample, dict):
            sample = sample[list(sample.keys())[0]]

        # Check if `sample` is a tensor and return its sequence length
        if isinstance(sample, torch.Tensor):
            return sample.shape[0]

        return 0

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        if not torch.distributed.is_initialized():
            return

        rank = rank if rank is not None else torch.distributed.get_rank()
        world_size = world_size if world_size is not None else torch.distributed.get_world_size()

        for dataset in self.datasets:
            dataset.set_rank_and_world_size(rank, world_size)

        self.global_worker_id = rank
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.global_worker_id = rank * worker_info.num_workers + worker_info.id

    def get_state(self) -> List[Dict[str, Any]]:
        return [dataset.get_state() for dataset in self.datasets]

    def set_state(self, state: List[Dict[str, Any]]) -> None:
        for dataset_state in state:
            for dataset in self.datasets:
                if dataset_state["dataset_path"] == dataset.dataset_path:
                    dataset.set_state(dataset_state)

    def save_state(self, output_dir: str, step: int) -> None:
        file_path = os.path.join(output_dir, "dataset_state", str(step), "state.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_json_file(self.get_state(), file_path)

    def load_state(self, load_dir: str, step: Optional[str] = None) -> None:
        if step is None:
            latest_path = os.path.join(load_dir, "latest")
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    step = fd.read().strip()
            else:
                logger.warning(f"Unable to find dataset state file: {latest_path}.")
                return

        file_path = os.path.join(load_dir, "dataset_state", str(step), "state.json")
        try:
            state = load_json_file(file_path)
            self.set_state(state)
        except FileNotFoundError:
            logger.warning(f"Unable to load dataset state file: {file_path}.")

    def __iter__(self) -> Iterator[Any]:
        if self.datasets_iter is None:
            self.datasets_iter = [iter(dataset) for dataset in self.datasets]

        warmup_samples: list = [None] * len(self.datasets_iter)
        for i, it in enumerate(self.datasets_iter):
            warmup_samples[i] = next(it)
            logger.info(f"Pre-warmed dataset iterator {i}")

        # Create rng based on the rank so that each rank can choose a distinct dataset
        rng = np.random.default_rng(self.global_worker_id)
        dataset_idx = rng.choice(len(self.weights), p=self.weights)

        for i, sample in enumerate(warmup_samples):
            if sample is None:
                continue
            if isinstance(sample, dict):
                sample["idx"] = -1
                sample["dataset_idx"] = i
                if 'input_embeds' in sample and len(sample['input_embeds']) > 0:
                    sample['is_mm_sample'] = 1
                else:
                    sample['is_mm_sample'] = 0
            yield sample

        while True:
            # Yield a sample from the current dataset
            sample = next(self.datasets_iter[dataset_idx])
            if isinstance(sample, dict):
                sample["idx"] = -1
                sample["dataset_idx"] = dataset_idx
                if 'input_embeds' in sample and len(sample['input_embeds']) > 0:
                    # this sample has image 
                    sample['is_mm_sample'] = 1 
                else:
                    sample['is_mm_sample'] = 0
            yield sample

            dataset_idx = rng.choice(len(self.weights), p=self.weights)