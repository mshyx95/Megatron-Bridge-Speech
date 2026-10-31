# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

from megatron.bridge.data.speech_chunk_streaming.dataset_provider import (
    DatasetProvider,
    DefaultDatasetProviderConfig,
)
from megatron.bridge.data.speech_chunk_streaming.dataset_provider_utils import split_ndarray
from megatron.bridge.data.speech_chunk_streaming.vlm.vlm_dataset import VLMDataset
from megatron.bridge.data.speech_chunk_streaming._utils.file_utils import validate_file_extension
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class VLMDatasetProviderConfig(DefaultDatasetProviderConfig):
    """Vision-language modeling dataset provider configuration.

    Args:
        img_file: Path to the `.tsv` or cached `.npy` image file.
        img_processor: Image processor name.
        n_img_tokens: Number of image tokens.

    """

    img_file: Optional[str] = field(default=None, metadata={"help": "Path to the cached `.tsv` or `.npy` image file."})

    img_processor: Optional[str] = field(default=None, metadata={"help": "Image processor name."})

    n_img_tokens: int = field(default=256, metadata={"help": "Number of image tokens."})

    num_crops: Optional[int] = field(default=4, metadata={"help": "Number of maximal crops for multicrop strategy."})

class VLMDatasetProvider(DatasetProvider):
    """Vision-language modeling dataset provider."""

    def __init__(
        self,
        train_file: Optional[Union[str, Path]] = None,
        validation_file: Optional[Union[str, Path]] = None,
        validation_split: Optional[Union[int, float]] = None,
        tokenizer_file: Optional[Union[str, Path]] = None,
        img_file: Optional[Union[str, Path]] = None,
        img_processor: Optional[str] = None,
        n_img_tokens: int = 256,
        seq_len: int = 2048,
        shift_labels: bool = False,
        weight: float = 1.0,
        anyres: bool = False, 
        num_crops: int = 4, 
        **kwargs,
    ) -> None:
        """Initialize the dataset provider.

        Args:
            train_file: Path to the ``.npy`` training array file.
            validation_file: Path to the ``.npy`` validation array file.
            validation_split: Part of the training dataset to be used for validation.
                If type is ``int``, will use number of tokens from the training dataset.
                If type is ``float``, will use fraction of the training dataset.
            tokenizer_file: Path to the ``.pkl`` tokenizer file.
            img_file: Path to the ``.tsv`` or ``.npy`` image file.
            img_processor: Image processor name.
            n_img_tokens: Number of image tokens.
            seq_len: Sequence length.
            shift_labels: Whether to shift labels.
            weight: Weight for the dataset.

        """

        super().__init__()
            
        self.train_file = train_file
        self.validation_file = validation_file
        self.validation_split = validation_split
        self.tokenizer_file = tokenizer_file

        self.img_file = img_file
        self.img_processor = img_processor
        self.n_img_tokens = n_img_tokens
        self.anyres = anyres
        self.num_crops = int(num_crops)

        self.seq_len = seq_len
        self.shift_labels = shift_labels
        self.weight = weight

        # `input_ids` attributes are used to avoid loading the files multiple times
        self.train_input_ids = None
        self.validation_input_ids = None

    @classmethod
    def from_config(cls: VLMDatasetProvider, config: VLMDatasetProviderConfig) -> VLMDatasetProvider:
        """Load a dataset provider from a configuration.

        Args:
            config: Dataset provider configuration.

        Returns:
            Dataset provider.

        """

        return cls(**config.to_dict())

    @classmethod
    def from_cache(cls: VLMDatasetProvider, cache_dir: Union[str, Path], **kwargs) -> VLMDatasetProvider:
        """Load a dataset provider from a cache directory.

        If a cached file (``train.npy``, ``validation.npy`` or ``tokenizer.pkl``)
        is missing, it will be set to ``None``.

        Args:
            cache_dir: Cached dataset directory.

        Returns:
            Dataset provider.

        """

        cache_dir = Path(cache_dir)
        cache_files = {}

        for split in ["train", "validation"]:
            cache_file = cache_dir / f"{split}.npy"
            cache_files[f"{split}_file"] = cache_file if cache_file.is_file() else None

        tokenizer_file = cache_dir / "tokenizer.pkl"
        cache_files["tokenizer_file"] = tokenizer_file if tokenizer_file.is_file() else None

        return VLMDatasetProvider(**cache_files, **kwargs)

    def get_train_dataset(self) -> VLMDataset:
        # is_train_file_valid = validate_file_extension(self.train_file, ".npy")
        # is_img_file_valid = validate_file_extension(self.img_file, [".tsv", ".npy"])
        is_train_file_valid = True 
        is_img_file_valid = True

        if is_train_file_valid and is_img_file_valid:
            if validate_file_extension(self.train_file, ".npy"):
                self.train_input_ids = np.load(self.train_file, mmap_mode="r")
            else:
                self.train_input_ids = np.zeros(1)
            if self.train_input_ids.ndim == 2:
                self.train_input_ids = self.train_input_ids.squeeze()

            # If we are creating a dynamic validation file, we need to ensure that
            # the `train_input_ids` do not overlap with the `validation_input_ids`
            if isinstance(self.validation_split, (int, float)):
                self.train_input_ids, self.validation_input_ids = split_ndarray(
                    self.train_input_ids, split=self.validation_split
                )

            return VLMDataset(
                self.train_input_ids,
                self.img_file,
                seq_len=self.seq_len,
                shift_labels=self.shift_labels,
                img_processor=self.img_processor,
                n_img_tokens=self.n_img_tokens,
            )

        return None

    def get_val_dataset(self) -> VLMDataset:
        is_img_file_valid = validate_file_extension(self.img_file, [".tsv", ".npy"])
        if is_img_file_valid:
            is_validation_file_valid = validate_file_extension(self.validation_file, ".npy")
            if is_validation_file_valid:
                self.validation_input_ids = np.load(self.validation_file, mmap_mode="r")
                if self.validation_input_ids.ndim == 2:
                    self.validation_input_ids = self.validation_input_ids.squeeze()

                return VLMDataset(
                    self.validation_input_ids,
                    self.img_file,
                    seq_len=self.seq_len,
                    shift_labels=self.shift_labels,
                    img_processor=self.img_processor,
                    n_img_tokens=self.n_img_tokens,
                )

            if self.validation_input_ids is not None:
                return VLMDataset(
                    self.validation_input_ids,
                    self.img_file,
                    seq_len=self.seq_len,
                    shift_labels=self.shift_labels,
                    img_processor=self.img_processor,
                    n_img_tokens=self.n_img_tokens,
                )

        return None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if validate_file_extension(self.tokenizer_file, ".pkl"):
            with open(self.tokenizer_file, "rb") as f:
                return pickle.load(f)

        return None
