# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Union

import numpy as np
from datasets import DatasetDict


def _create_split(
    dataset_dict: DatasetDict, split_keys: Tuple[str], split_size: float, shuffle: bool, seed: int
) -> DatasetDict:
    assert isinstance(split_keys, tuple) and len(split_keys) == 2, "`split_keys` must be a tuple of length 2."

    key_to_split, key_to_assign = split_keys
    are_keys_valid = key_to_split in dataset_dict and key_to_assign not in dataset_dict

    if are_keys_valid and split_size and split_size > 0.0:
        tmp_dataset_dict = dataset_dict[key_to_split].train_test_split(test_size=split_size, shuffle=shuffle, seed=seed)
        dataset_dict[key_to_split] = tmp_dataset_dict["train"]
        dataset_dict[key_to_assign] = tmp_dataset_dict["test"]

    return dataset_dict


def create_val_split(
    dataset_dict: DatasetDict,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> DatasetDict:
    """Create validation split from a dataset dictionary.

    Args:
        dataset_dict: Dataset dictionary.
        val_split: Fraction of the training dataset to be used for validation.
        shuffle: Whether to shuffle the dataset before splitting.
        seed: Random seed.

    Returns:
        Dataset dictionary with additional validation split.

    """

    return _create_split(dataset_dict, ("train", "validation"), val_split, shuffle, seed)


def split_ndarray(
    array: np.ndarray, axis: int = -1, split: Union[int, float] = 0.0, shuffle: bool = False, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a numpy array into two parts.

    Args:
        array: Numpy array to split.
        axis: Axis along which to split.
        split: Fraction of the array to be used for the second part.
        shuffle: Whether to shuffle the array before splitting.
        seed: Random seed.

    Returns:
        Tuple of the remaining and selected parts of the array.

    """

    assert array.ndim >= 1, "`array` should have at least 1 dimension."
    length = array.shape[axis]

    split = split or 0.0
    n_idx = int(split * length) if isinstance(split, float) else split
    assert 0 <= n_idx <= length, "`split` exceeds the array size when converted to an integer."

    if shuffle:
        # Set the random seed, which is used to persist shuffled indexes across multiple calls
        rng = np.random.RandomState(seed)
        idx = rng.permutation(length)

        selected_array = np.take(array, idx[:n_idx], axis=axis)
        remaining_array = np.take(array, idx[n_idx:], axis=axis)
    else:
        selected_array, remaining_array = np.split(array, [n_idx], axis=axis)

    return remaining_array, selected_array
