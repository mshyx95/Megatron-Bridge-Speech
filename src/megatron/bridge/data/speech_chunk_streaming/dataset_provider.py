# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union

from torch.utils.data import Dataset

from megatron.bridge.data.speech_chunk_streaming._utils.file_utils import get_full_path
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetProvider:
    """Abstract class for dataset providers.

    This class serves as a base for implementing dataset providers that can return
    training and validation datasets. It enforces implementation of two methods:
    :meth:`get_train_dataset` and :meth:`get_val_dataset`.

    Examples:
        >>> class MyDatasetProvider(DatasetProvider):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     def get_train_dataset(self) -> Dataset:
        >>>         return torchvision.datasets.MNIST(train=True)
        >>>
        >>>     def get_val_dataset(self) -> Dataset:
        >>>         return torchvision.datasets.MNIST(train=False)

    """

    def __init__(self) -> None:
        """Initialize the dataset provider."""

        pass

    @classmethod
    def from_config(cls: DatasetProvider, config: DefaultDatasetProviderConfig) -> DatasetProvider:
        """Load a dataset provider from a configuration.

        Args:
            config: Dataset provider configuration.

        Returns:
            Dataset provider.

        """

        return cls(**config.to_dict())

    def get_train_dataset(self) -> Dataset:
        """Get a training dataset.

        Returns:
            An instance of a training dataset.

        """

        pass

    def get_val_dataset(self) -> Dataset:
        """Get a validation dataset.

        Returns:
            An instance of a validation dataset, or the training dataset if
            validation dataset is not available.

        """

        pass


@dataclass
class DefaultDatasetProviderConfig:
    """Dataset provider configuration (used as default for language modeling datasets).

    Args:
        cache_dir: Path to the cached dataset directory.
        cache_path: Path to the cached dataset directory (deprecated, use `cache_dir`).
        train_file: Path to the cached `.npy` train file.
        validation_file: Path to the cached `.npy` validation file.
        validation_split: Part of the training dataset to be used for validation.
            If type is ``int``, will use number of tokens from the training dataset.
            If type is ``float``, will use fraction of the training dataset.
        tokenizer_file: Path to the cached `.pkl` tokenizer file.
        seq_len: Sequence length.
        shift_labels: Whether to shift labels.
        weight: Weight for the dataset.
        label: Label for the dataset.

    """

    cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to the cached dataset directory."})

    cache_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the cached dataset directory (deprecated, use `cache_dir`)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "Path to the cached `.npy` train file."})

    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the cached `.npy` validation file."}
    )

    validation_split: Optional[Union[int, float]] = field(
        default=None, metadata={"help": "Part of the training dataset to be used for validation."}
    )

    tokenizer_file: Optional[str] = field(default=None, metadata={"help": "Path to the cached `.pkl` tokenizer file."})

    seq_len: int = field(default=2048, metadata={"help": "Sequence length."})

    add_bos: bool = field(default=False, metadata={"help": "Whether to add the bos token"})

    shift_labels: bool = field(default=False, metadata={"help": "Whether to shift labels."})

    weight: float = field(default=1.0, metadata={"help": "Weight for the dataset."})

    label: str = field(default="", metadata={"help": "Label for the dataset."})

    def __post_init__(self) -> None:
        """Post initialization."""

        assert (
            self.cache_dir is None or self.cache_path is None
        ), "You cannot specify both `cache_dir` and `cache_path`."

        if self.cache_path:
            logger.warning(
                "The `cache_path` argument will soon be deprecated, please use the `train_file`"
                " and `validation_file` arguments to pass `.npy` files directly"
                " or use the `cache_dir` argument to pass directories."
            )
            self.cache_dir = self.cache_path

        if self.cache_dir:
            assert (
                self.train_file is None and self.validation_file is None
            ), "You cannot specify both `cache_dir` and `train_file` or `validation_file`."

            # If `cache_dir` is provided, we gather its complete path
            self.cache_dir = get_full_path(self.cache_dir)
            if not os.path.exists(self.cache_dir):
                raise FileNotFoundError(f"`cache_dir` ({self.cache_dir}) does not exist.")

            # If `cache_dir` is a directory, we set default names for the files
            if os.path.isdir(self.cache_dir):
                self.train_file = os.path.join(self.cache_dir, "train.npy")
                self.validation_file = os.path.join(self.cache_dir, "validation.npy")
            # If `cache_dir` is a file, we assume the folder is for training only
            else:
                self.train_file = self.cache_dir
                self.validation_file = None

        # After `cache_dir` is processed, `train_file` and `validation_file` must be validated
        # If a file is not valid, warn the user and set it to `None`
        if self.train_file is not None:
            if not os.path.exists(self.train_file):
                logger.warning(f"`train_file` ({self.train_file}) does not exist.")
                self.train_file = None

        if self.validation_split is not None:
            logger.warning("Setting `validation_file=None` since `validation_split` is provided.")
            self.validation_file = None

        if self.validation_file is not None:
            if not os.path.exists(self.validation_file):
                logger.debug(f"`validation_file` ({self.validation_file}) does not exist.")
                self.validation_file = None

        assert self.seq_len > 0, f"`seq_len` ({self.seq_len}) must be greater than 0."
        assert self.weight > 0.0, f"`weight` ({self.weight}) must be greater than 0.0."

    @classmethod
    def from_dict(cls: DefaultDatasetProviderConfig, config_dict: Dict[str, Any]) -> DefaultDatasetProviderConfig:
        """Load a configuration from a dictionary.

        Args:
            config_dict: Dictionary containing the configuration.

        Returns:
            Configuration.

        """

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dictionary containing the configuration.

        """

        config = asdict(self)

        # `cache_dir` or `cache_path` are only used to find the individual files
        config.pop("cache_dir", None)
        config.pop("cache_path", None)

        return config
