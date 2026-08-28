# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import blobfile as bf
import pickle
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
from json import JSONDecodeError
import copy

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer 
import torch.distributed as dist

from megatron.bridge.data.speech_chunk_streaming.dataset_provider import (
    DatasetProvider,
    DefaultDatasetProviderConfig,
)
from megatron.bridge.data.speech_chunk_streaming.dataset_provider_utils import split_ndarray
from megatron.bridge.data.speech_chunk_streaming.vlm.vlm_dataset_provider import VLMDatasetProvider 
from megatron.bridge.data.speech_chunk_streaming.vlm.vlm_dataset import VLMDataset
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blob_chunk_alm_dataset import InfiniteBlobChunkALMDataset
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.s2s_blob_chunk_alm_dataset import (
    SpeechOutInfiniteBlobChunkALMDataset,
    SpeechOutRawAudioInfiniteBlobChunkALMDataset,
)
from megatron.bridge.data.speech_chunk_streaming._utils.file_utils import validate_file_extension
from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms import AudioTaskPhiTransform, AudioTextConversationTransform, SpeechOutAudioTextConversationTransform, SpeechOutRawAudioConversationTransform

logger = get_logger(__name__)


def _deep_merge_dict(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if base is None:
        base = {}
    if override is None:
        return copy.deepcopy(base)

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_data_spec_entry(data_spec_entry: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(data_spec_entry, str):
        return [{
            "spec_file": data_spec_entry,
            "ratio": 1,
            "sample_profile_id": None,
            "sample_overrides": {},
            "data_source_overrides": {},
            "sample_ratio": 1.0,
        }]

    if not isinstance(data_spec_entry, dict):
        raise TypeError(f"Unsupported dataset_spec entry type: {type(data_spec_entry)}")

    spec_files = data_spec_entry.get("file") or data_spec_entry.get("spec_file") or data_spec_entry.get("path")
    if spec_files is None:
        raise ValueError("dataset_spec dict entries must provide one of: file, spec_file, path")

    if isinstance(spec_files, str):
        spec_files = [spec_files]
    elif isinstance(spec_files, list):
        if not all(isinstance(spec_file, str) for spec_file in spec_files):
            raise TypeError("dataset_spec file lists must contain only strings")
    else:
        raise TypeError("dataset_spec file/spec_file/path must be a string or list of strings")

    normalized_entries = []
    for spec_file in spec_files:
        normalized_entries.append({
            "spec_file": spec_file,
            "ratio": int(data_spec_entry.get("ratio", data_spec_entry.get("repeat", 1))),
            "sample_profile_id": data_spec_entry.get("sample_profile") or data_spec_entry.get("sample_profile_id"),
            "sample_overrides": copy.deepcopy(data_spec_entry.get("sample_overrides", {})),
            "data_source_overrides": copy.deepcopy(data_spec_entry.get("data_source_overrides", {})),
            "sample_ratio": float(data_spec_entry.get("sample_ratio", 1.0)),
        })
    return normalized_entries


def _merge_sample_profile(sample_profiles: Optional[Dict[str, Dict[str, Any]]], sample_profile_id: Optional[str], sample_overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged_overrides = {}
    if sample_profile_id is not None:
        if not sample_profiles or sample_profile_id not in sample_profiles:
            raise KeyError(f"Unknown sample_profile '{sample_profile_id}' referenced by dataset_spec entry")
        merged_overrides = copy.deepcopy(sample_profiles[sample_profile_id])

    if sample_overrides:
        merged_overrides = _deep_merge_dict(merged_overrides, sample_overrides)

    return merged_overrides


def _load_spec_jsons(data_spec_files, sample_profiles=None, path_prefix_in_orange=None):
    dataset_spec = None
    assert isinstance(data_spec_files, list), 'data_spec_files should be a list'
    if isinstance(data_spec_files, list):
        dataset_spec = {
            'data_sources': [],
            'sample_profiles': copy.deepcopy(sample_profiles or {}),
        }

        normalized_entries = []
        for entry in data_spec_files:
            normalized_entries.extend(_normalize_data_spec_entry(entry))
        data_spec_ratio = [entry["ratio"] for entry in normalized_entries]

        for normalized_entry in normalized_entries:
            _spec_ratio = normalized_entry["ratio"]
            _spec_file_name = normalized_entry["spec_file"]
            logger.info(f'=> load data spec file: {_spec_file_name} with ratio: {_spec_ratio}')

        for _idx, normalized_entry in enumerate(normalized_entries):
            data_spec_file = normalized_entry["spec_file"]
            spec_file_handle = (
                bf.BlobFile(os.path.join(path_prefix_in_orange, *Path(data_spec_file).parts[2:]), 'r')
                if path_prefix_in_orange is not None
                else open(data_spec_file, 'r')
            )
            with spec_file_handle as f:
                try:
                    _spec = json.load(f)
                except JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse dataset spec JSON: {data_spec_file}") from exc

            sample_profile_id = normalized_entry["sample_profile_id"]
            merged_sample_overrides = _merge_sample_profile(
                sample_profiles,
                sample_profile_id,
                normalized_entry["sample_overrides"],
            )

            for data_source in _spec["data_sources"]:
                data_source = copy.deepcopy(data_source)
                chunk_type = _spec.get("chunk_type", None)
                if chunk_type is None:
                    data_source.update({"chunk_type": ["audio", "info", "transcription"]})
                else:
                    data_source.update({"chunk_type": chunk_type})

                if data_source.get("language", None) is None:
                    data_source.update({"language": _spec.get("language", None)})

                data_spec_root = Path(data_spec_file).parts[1]
                manifest_file_root = Path(data_source["manifest_file"]).parts[1]
                if data_spec_root != manifest_file_root:
                    # make manifest_file path relative to data_spec_file path
                    data_source["manifest_file"] = data_source["manifest_file"].replace(
                        manifest_file_root, data_spec_root
                    )

                if normalized_entry["data_source_overrides"]:
                    data_source.update(copy.deepcopy(normalized_entry["data_source_overrides"]))

                data_source["spec_file"] = data_spec_file
                data_source["sample_profile_id"] = sample_profile_id
                data_source["sample_ratio"] = normalized_entry["sample_ratio"]
                if merged_sample_overrides:
                    data_source["sample_overrides"] = copy.deepcopy(merged_sample_overrides)
                else:
                    data_source.pop("sample_overrides", None)

                dataset_spec['data_sources'] += [data_source] * data_spec_ratio[_idx]
            
    logger.debug(f'data spec: {dataset_spec}')

    return dataset_spec


@dataclass
class BlobChunkALMDatasetProviderConfig(DefaultDatasetProviderConfig):
    """Audio-language modeling dataset provider configuration.

    Args:
        audio_file: Path to the `.tsv` or cached `.npy` image file.
        audio_processor: Image processor name.
        tokenizer: Instance of tokenizer to use.
        dataset_spec: Dataset specification for blob chunk dataset.

    """

    audio_file: Optional[str] = field(default=None, metadata={"help": "Path to the cached `.tsv` or `.npy` image file."})

    audio_processor: Optional[str] = field(default=None, metadata={"help": "Audio processor name."})

    tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]] = field(
        default=None, metadata={"help": "Instance of tokenizer to use."}
    )

    compression_rate: int = field(default=1, metadata={"help": "Audio encoder compression ratio"})
    streaming_compression: int = field(default=False, metadata={"help": "Audio encoder streaming compression"})
    qformer_compression_rate: int = field(default=1, metadata={"help": "Qformer compression ration following audio encoder"})
    batcher_buffer_size: int = field(default=100, metadata={"help": "buffer size for dynamic batcher, group audio-text pairs into a sequence"})
    batcher_batch_size: int = field(default=7168, metadata={"help": "batch size for batcher buffer"})
    batcher_criterion: str = field(default="frame", metadata={"help": "batch criterion for batcher buffer"})
    batcher_input_key: str = field(default="audio", metadata={"help": "sample key used as raw audio/features for batcher length and audio extraction"})
    batcher_max_utterances: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional hard cap on # utterances emitted per batcher batch. "
                "Only takes effect when batcher_criterion='frame'. Prevents "
                "activation-memory blowup when the batcher's frame metric "
                "(audio_embed_sizes) underestimates the model's actual "
                "sequence length. Overflow utts go to the leftover buffer "
                "and are batched in the next round (no data is dropped). "
                "Default None preserves prior behaviour."
            )
        },
    )
    codec_cfg: Optional[dict] = field(default=None, metadata={"help": "codec configs for phi-mlora/S2ST data path"})
    num_chunks_for_shuffling: int = field(default=10, metadata={"help": "num_chunks for shuffling, buffer size in the blobchunk.py"})

    dataset_spec: Optional[List] = field(
        default=None, metadata={"help": "Dataset specification for blob chunk dataset."}
    )

    sample_profiles: Optional[Dict[str, Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "Reusable sample-level profiles referenced by dataset_spec entries."},
    )
    
    filter_options: Optional[dict] = field(default=None, metadata={"help": "options related to filtering out examples"})
    
    text_normalization: Optional[dict] = field(default=None, metadata={"help": "text normalization args for transcription"})
  
    transform_cls: Optional[str] = field(default=None, metadata={"help": "Transform class name."})

    num_workers: Optional[int] = field(default=1, metadata={"help": "Number of workers for data loading."})
    
    pad_to_context_length: Optional[bool] = field(default=True, metadata={"help": "Pad to context length."})

    allow_pure_text_return: Optional[bool] = field(default=False, metadata={"help": "Allow pure text return."})

    use_num_examples_as_durations: Optional[bool] = field(default=False, metadata={"help": "Use num examples for dataset size"})

    tsv_chunk_size: Optional[int] = field(default=256, metadata={"help": "chunk size for loading tsv data"})

    sp_transcription: Optional[bool] = field(default=False, metadata={"help": "notation of whether transcription is tokenized by sentencepiece"})

    retain_first_n_turns_in_conversation: Optional[int] = field(default=None, metadata={"help": "If specified, only keep the first n turns for training (to save memory)"})

    chunk_processing_pipe: Optional[List] = field(default=None, metadata={"help": "pipeline to process input chunks"})
    
    path_prefix_in_orange: Optional[str] = field(default=None, metadata={"help": "If not None it is path prefix in orange"})
    multi_threaded_download: bool = field(default=False, metadata={"help": "Whether to use multi-threaded download for blob chunk dataset"})
    prefetch_queue_size: int = field(default=64, metadata={"help": "Size of prefetch queue"})
    use_storage_api: Optional[bool] = field(default=False, metadata={"help": "If use azcopy instead of blob mount"})

class BlobChunkALMDatasetProvider(VLMDatasetProvider):
    """Blob Chunk Audio-language modeling dataset provider."""

    def __init__(
        self,
        train_file: Optional[Union[str, Path]] = None,
        validation_file: Optional[Union[str, Path]] = None,
        validation_split: Optional[Union[int, float]] = None,
        tokenizer_file: Optional[Union[str, Path]] = None,
        tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]] = None,
        audio_file: Optional[Union[str, Path]] = None,
        audio_processor: Optional[str] = None,
        dataset_spec: Optional[List] = None,
        sample_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        seq_len: int = 2048,
        shift_labels: bool = False,
        weight: float = 1.0,
        transform_cls: Optional[str] = None,
        num_workers: Optional[int] = 1,
        pad_to_context_length: Optional[bool] = True,
        filter_options: Optional[dict] = None,
        compression_rate: Optional[int] = 1,
        streaming_compression: bool = False,
        text_normalization: Optional[dict] = None,
        qformer_compression_rate: Optional[int] = 1,
        allow_pure_text_return: Optional[bool] = False,
        batcher_buffer_size: int = 10000,
        batcher_batch_size: int = 1000,
        batcher_criterion: str = "frame",
        batcher_input_key: str = "audio",
        batcher_max_utterances: Optional[int] = None,
        codec_cfg: Optional[dict] = None,
        num_chunks_for_shuffling: int = 25,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        sp_transcription: bool = False,
        retain_first_n_turns_in_conversation: Optional[int] = None,
        chunk_processing_pipe: Optional[List] = None,
        path_prefix_in_orange: Optional[str] = None,
        multi_threaded_download: bool = False,
        prefetch_queue_size: int = 64,
        use_storage_api: bool = False,
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
            audio_file: Path to the ``.tsv`` or ``.npy`` image file.
            audio_processor: Image processor name.
            dataset_spec: Dataset specification for blob chunk dataset.
            seq_len: Sequence length.
            shift_labels: Whether to shift labels.
            weight: Weight for the dataset.
            pad_to_context_length: Pad to context length.
        """            

        super().__init__(
            train_file=train_file,
            validation_file=validation_file,
            validation_split=validation_split,
            tokenizer_file=tokenizer_file,
            img_file=audio_file,
            img_processor=audio_processor,
            seq_len=seq_len,
            shift_labels=shift_labels,
            weight=weight,
            **kwargs,
        )
        tokenizer = _load_tokenizer(tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id 
        
        self.tokenizer = tokenizer

        self.audio_processor = audio_processor
        self.allow_pure_text_return = allow_pure_text_return
        self.compression_rate = compression_rate
        self.streaming_compression = streaming_compression
        self.qformer_compression_rate = qformer_compression_rate
        self.batcher_buffer_size = batcher_buffer_size
        self.batcher_batch_size = batcher_batch_size
        self.batcher_criterion = batcher_criterion
        self.batcher_input_key = batcher_input_key
        self.batcher_max_utterances = batcher_max_utterances
        self.codec_layer = codec_cfg.get("codec_layer", 3) if codec_cfg is not None else 3
        self.codec_cfg = codec_cfg
        self.num_chunks_for_shuffling = num_chunks_for_shuffling
        self.filter_options = filter_options
        self.text_normalization = text_normalization
        self.use_num_examples_as_durations = use_num_examples_as_durations
        self.tsv_chunk_size = tsv_chunk_size
        self.sp_transcription = sp_transcription
        self.conv_template = transform_cls.get("conv_template", "phi_3") if isinstance(transform_cls, dict) else "phi_3"
        self.retain_first_n_turns_in_conversation = retain_first_n_turns_in_conversation
        self.path_prefix_in_orange = path_prefix_in_orange

        if transform_cls is not None:
            transform_cls_kwargs = {}
            if isinstance(transform_cls, dict):
                transform_cls_kwargs = transform_cls.copy()
                transform_cls = transform_cls['name']
            if codec_cfg is not None and transform_cls == "AudioTextConversationTransform":
                transform_cls = "SpeechOutAudioTextConversationTransform"
            # forward_tokenize (raw-audio) path: emit seq_plan + raw answer_audio,
            # tokenize + build + pad inside model.forward.
            if codec_cfg is not None and codec_cfg.get("forward_tokenize", False):
                transform_cls = "SpeechOutRawAudioConversationTransform"

            if 'allow_pure_text_return' in transform_cls_kwargs and self.allow_pure_text_return != transform_cls_kwargs['allow_pure_text_return']:
                raise ValueError(f'allow_pure_text_return is not consistent with transform_cls_kwargs: {self.allow_pure_text_return} != {transform_cls_kwargs["allow_pure_text_return"]}')

            transform_cls_kwargs['allow_pure_text_return'] = self.allow_pure_text_return
            self.transform = eval(transform_cls)(
                { 
                    'context_length': self.seq_len,
                    'shift_labels': self.shift_labels,
                    'pad_to_context_length': pad_to_context_length,
                },
                **transform_cls_kwargs,
            )
        else:
            transform_cls_kwargs = {}
            transform_cls_kwargs['allow_pure_text_return'] = self.allow_pure_text_return
            self.transform = AudioTaskPhiTransform({
                'context_length': self.seq_len,
                'shift_labels': self.shift_labels,
                'pad_to_context_length': pad_to_context_length,
            }, **transform_cls_kwargs)

        if dataset_spec is not None:
            self.dataset_spec = _load_spec_jsons(
                dataset_spec,
                sample_profiles=sample_profiles,
                path_prefix_in_orange=self.path_prefix_in_orange,
            )
        else:
            self.dataset_spec = None
        
        self.num_workers = num_workers

        self.chunk_processing_pipe = chunk_processing_pipe
        self.multi_threaded_download = multi_threaded_download
        self.prefetch_queue_size = prefetch_queue_size
        self.use_storage_api = use_storage_api

    @classmethod
    def from_config(cls: BlobChunkALMDatasetProvider, config: BlobChunkALMDatasetProviderConfig) -> BlobChunkALMDatasetProvider:
        """Load a dataset provider from a configuration.

        Args:
            config: Dataset provider configuration.

        Returns:
            Dataset provider.

        """
        return cls(**config.to_dict())

    @classmethod
    def from_cache(cls: BlobChunAVLMDatasetProvider, cache_dir: Union[str, Path], **kwargs) -> BlobChunkALMDatasetProvider:
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

        return BlobChunkALMDatasetProvider(**cache_files, **kwargs)

    def get_train_dataset(self) -> Union[ALMDataset, InfiniteBlobChunkALMDataset]:
        if self.dataset_spec is None:
            # If the dataset specification is not provided, use the default VLMDataset
            return super().get_train_dataset()

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        try:
            num_replicas = dist.get_world_size()
        except:
            num_replicas = 1

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        try:
            rank = dist.get_rank()
        except:
            rank = 0

        dataset_cls = self._get_dataset_cls()
        dataset_kwargs = dict(
            dataset_spec=self.dataset_spec,
            num_chunks_for_shuffling=self.num_chunks_for_shuffling,
            transforms=self.transform,
            tokenize=self.tokenizer,
            context_length=self.seq_len,
            seed=0,
            rank=rank,
            num_replicas=num_replicas,
            drop_last=False,
            multi_threaded_download=self.multi_threaded_download,
            use_num_examples_as_durations=self.use_num_examples_as_durations,
            tsv_chunk_size=self.tsv_chunk_size,
            sp_transcription=self.sp_transcription,
            debug=False,
            tokenize_padding="max_length",
            path_prefix_in_orange=self.path_prefix_in_orange,
            shift_labels=self.shift_labels,
            audio_processor=self.audio_processor,
            seq_len=self.seq_len,
            filter_options=self.filter_options,
            text_normalization=self.text_normalization,
            compression_rate=self.compression_rate,
            streaming_compression=self.streaming_compression,
            qformer_compression_rate=self.qformer_compression_rate,
            batcher_buffer_size=self.batcher_buffer_size,
            batcher_batch_size=self.batcher_batch_size,
            batcher_criterion=self.batcher_criterion,
            batcher_input_key=self.batcher_input_key,
            conv_template=self.conv_template,
            retain_first_n_turns_in_conversation=self.retain_first_n_turns_in_conversation,
            allow_pure_text_return=self.allow_pure_text_return,
            chunk_processing_pipe=self.chunk_processing_pipe,
            prefetch_queue_size=self.prefetch_queue_size,
            use_storage_api=self.use_storage_api,
        )
        if dataset_cls in {SpeechOutInfiniteBlobChunkALMDataset, SpeechOutRawAudioInfiniteBlobChunkALMDataset}:
            dataset_kwargs["codec_layer"] = self.codec_layer
            dataset_kwargs["codec_cfg"] = self.codec_cfg
            dataset_kwargs["batcher_max_utterances"] = self.batcher_max_utterances
        dataset = dataset_cls(**dataset_kwargs)
        return dataset 

    def _get_dataset_cls(self):
        if self.codec_cfg is None:
            return InfiniteBlobChunkALMDataset
        if self.codec_cfg.get("forward_tokenize", False):
            return SpeechOutRawAudioInfiniteBlobChunkALMDataset
        if self.codec_cfg.get("dataset_cls") == "streaming" or self.codec_cfg.get("streaming_dataset", False):
            return SpeechOutInfiniteBlobChunkALMDataset
        return SpeechOutInfiniteBlobChunkALMDataset

    def get_val_dataset(self) -> ALMDataset:
        if self.dataset_spec is None:
            # If the dataset specification is not provided, use the default VLMDataset
            return super().get_val_dataset()
        
        # raise NotImplementedError("Validation dataset is not supported for BlobChunkVLMDatasetProvider")
        return None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if validate_file_extension(self.tokenizer_file, ".pkl"):
            with open(self.tokenizer_file, "rb") as f:
                return pickle.load(f)

        return None


def _load_tokenizer(tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]]) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)

    if isinstance(tokenizer, dict):
        return AutoTokenizer.from_pretrained(**tokenizer)

    return tokenizer
