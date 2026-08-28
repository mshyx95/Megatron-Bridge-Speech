# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import logging
import math
import os
import threading
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import blobfile as bf
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.blobchunk import (
    BlobChunkIterableDataset as BaseBlobChunkIterableDataset,
    ChunkSpeechTextSeq2SeqDataset as BaseChunkSpeechTextSeq2SeqDataset,
    DataParser as BaseDataParser,
    DatasetSpec as BaseDatasetSpec,
)


logger = logging.getLogger(__name__)


def safe_read(path, file_type="wav"):
    try:
        mode = "r" if file_type == "txt" else "rb"
        opener = bf.BlobFile if str(path).startswith("az://") else open
        with opener(path, mode) as f:
            if file_type == "txt":
                return f.read()
            return sf.read(f)
    except Exception:
        return None


class DataParser(BaseDataParser):
    def parse_data(self, data, data_type):
        data_type = data_type.lower()
        if data_type in ("audio", "answer_audio"):
            return self._parse_audio_data(data)
        if data_type in ("alignment", "answer_alignment"):
            return self._parse_json_data(data)
        return super().parse_data(data, data_type)

    def _parse_audio_data(self, data):
        if not data:
            return None
        try:
            return sf.read(io.BytesIO(data))
        except Exception:
            return None


class DatasetSpec(BaseDatasetSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = DataParser()
        self._s3tokenizer = None
        self._s3tokenizer_lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3tokenizer"] = None
        state["_s3tokenizer_lock"] = None
        state.pop("_s3tokenizer_device", None)
        state.pop("_s3tokenizer_module", None)
        state.pop("_s3tokenizer_stream", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._s3tokenizer = None
        self._s3tokenizer_lock = threading.Lock()

    def load_manifest_file(self, data_source, n_chunks=-1):
        full_path = data_source["manifest_file"]
        if self.is_tsv:
            col_names = data_source.get("col_names", ["id", "wav_paths", "messages"])
            with (bf.BlobFile(full_path, "rb") if full_path.startswith("az://") else open(full_path, "rb")) as file:
                if n_chunks == -1:
                    return pd.read_csv(file, names=col_names, sep="\t")
                file.seek(data_source["chunk_tsv_offsets"][n_chunks])
                return pd.read_csv(
                    file,
                    names=col_names,
                    nrows=data_source["local_chunk_sizes"][n_chunks],
                    sep="\t",
                )

        with (bf.BlobFile(full_path, "r") if full_path.startswith("az://") else open(full_path, "r", encoding="utf-8")) as f:
            try:
                return json.load(f)["fileInfo"]
            except JSONDecodeError as exc:
                raise ValueError(f"Failed to parse manifest JSON: {full_path}") from exc

    def load_chunk(self, global_chunk_index: int):
        assert 0 <= global_chunk_index < self.total_chunks
        data_source_index = np.searchsorted(self.accum_chunks, global_chunk_index, side="right")
        data_source = self.data_sources[data_source_index]
        local_chunk_index = global_chunk_index - (
            self.accum_chunks[data_source_index - 1] if data_source_index > 0 else 0
        )

        if self.is_tsv:
            all_chunk_data, chunk_type = self._load_tsv_chunk(data_source, local_chunk_index)
        else:
            all_chunk_data, chunk_type = self._load_binary_chunk(data_source, local_chunk_index)

        optional_chunk_type = data_source.get("optional_chunk_type", [])
        required_keys = [key for key in chunk_type if key not in optional_chunk_type]
        for i in range(len(all_chunk_data)):
            if any(all_chunk_data[i].get(key) is None for key in required_keys):
                all_chunk_data[i] = {}
            else:
                all_chunk_data[i]["language"] = data_source["language"]

        if self.chunk_processing_pipe:
            all_chunk_data = self.apply_chunk_processing_pipe(all_chunk_data)

        source_metadata = {
            "_data_source_index": data_source_index,
            "_spec_file": data_source.get("spec_file"),
            "_sample_profile_id": data_source.get("sample_profile_id"),
        }
        if data_source.get("sample_overrides"):
            source_metadata["_sample_overrides"] = data_source["sample_overrides"]
        for sample in all_chunk_data:
            if sample:
                sample.update(source_metadata)
        return all_chunk_data

    def _load_tsv_chunk(self, data_source, local_chunk_index):
        chunk_list = self.load_manifest_file(data_source, n_chunks=local_chunk_index)
        all_chunk_data = []
        for i in range(len(chunk_list["wav_paths"])):
            wav_paths = eval(chunk_list["wav_paths"][i])
            if self.path_prefix_in_orange:
                wav_paths = [os.path.join(self.path_prefix_in_orange, *Path(p).parts[2:]) for p in wav_paths]

            if self.allow_pure_text_return:
                audios = [("placeholder", 16000) for _ in wav_paths]
            else:
                audios = [
                    self.safe_read(wav_path, read_blob_file=self.path_prefix_in_orange is not None)
                    for wav_path in wav_paths
                ]
                if not any(audios):
                    audios = None

            chunk_data = {
                "audios": audios,
                "sft": eval(chunk_list["messages"][i]),
            }
            if "semantic_files" in chunk_list:
                semantic_paths = eval(chunk_list["semantic_files"][i])
                chunk_data["answer_cv_tokens"] = [
                    safe_read(semantic_path, file_type="txt")
                    for semantic_path in semantic_paths
                ]
            all_chunk_data.append(chunk_data)

        chunk_type = ["audios", "sft"]
        if data_source.get("col_names") and "semantic_files" in data_source["col_names"]:
            chunk_type.append("answer_cv_tokens")
        return all_chunk_data, chunk_type

    def _load_binary_chunk(self, data_source, local_chunk_index):
        chunk_list = self.load_manifest_file(data_source)
        actual_index = data_source["_sampled_chunk_indices"][local_chunk_index] if "_sampled_chunk_indices" in data_source else local_chunk_index
        chunk_name = chunk_list[actual_index]["name"]
        chunk_type = list(data_source["chunk_type"])
        chunk_path = data_source["chunk_path"]
        trans_path = data_source.get("trans_path", None)
        additional_data_path = data_source.get("additional_data_path", data_source.get("additional_path", {}))
        chunk_size = chunk_list[actual_index]["count"]
        all_chunk_data = [{} for _ in range(chunk_size)]

        for extension in chunk_type:
            if extension in ["transcription", "sft"]:
                trans_path = trans_path if trans_path is not None else chunk_path
                path = additional_data_path.get(extension, trans_path)
            else:
                path = additional_data_path.get(extension, chunk_path)
            if self.path_prefix_in_orange:
                path = os.path.join(self.path_prefix_in_orange, *Path(path).parts[2:])
            if self.use_storage_api:
                path = os.path.join(*Path(path).parts[2:])
            blob_name = os.path.join(path, chunk_name + "." + extension)

            try:
                chunk_data = self.load_chunk_with_chunkname(blob_name, extension, chunk_size)
                if len(chunk_data) != chunk_size:
                    chunk_data = [None for _ in range(chunk_size)]
            except Exception as e:
                logger.info(
                    f"decode chunk failed data source {data_source['manifest_file']}, "
                    f"blob name {blob_name}, return empty list {e}"
                )
                chunk_data = [None for _ in range(chunk_size)]

            for i in range(chunk_size):
                all_chunk_data[i][extension] = chunk_data[i]

        semantic_otf_config = data_source.get("semantic_otf")
        if semantic_otf_config is not None and "answer_audio" in chunk_type:
            all_chunk_data = self._run_semantic_otf(all_chunk_data, semantic_otf_config)
            chunk_type = [ct if ct != "answer_audio" else "answer_cv_token" for ct in chunk_type]
        return all_chunk_data, chunk_type

    def _run_semantic_otf(self, all_chunk_data, semantic_otf_config):
        if self._s3tokenizer is None:
            with self._s3tokenizer_lock:
                if self._s3tokenizer is None:
                    try:
                        import s3tokenizer
                    except ImportError as exc:
                        raise ImportError(
                            "s3tokenizer package required for semantic_otf mode. "
                            "Install with: pip install s3tokenizer"
                        ) from exc

                    model_path = semantic_otf_config.get(
                        "s3_tokenizer_path",
                        "/datablob1/users/yuxuanhu/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
                    )
                    effective_path = self._local_s3tokenizer_path(model_path)
                    device = self._s3tokenizer_device_from_config(semantic_otf_config)
                    logger.info(f"Loading S3Tokenizer from {effective_path} on {device}")
                    self._s3tokenizer = s3tokenizer.load_model(effective_path).to(device).eval()
                    self._s3tokenizer_device = device
                    self._s3tokenizer_module = s3tokenizer
                    self._s3tokenizer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        start_time = time.time()
        batch_size = semantic_otf_config.get("batch_size", 64)
        n_ok = 0
        n_fail = 0
        n_valid = 0

        # Memory-safe streaming tokenization.
        #
        # The previous implementation materialized the mel spectrogram of EVERY
        # sample in the chunk into a single `mels` list before quantizing, and
        # only freed the raw `answer_audio` waveforms after all batches finished.
        # Peak host RAM was therefore O(chunk_size) (raw waveforms + all mels held
        # at once). For large chunks (e.g. x2en ~2000 samples/chunk) this caused
        # host-RAM OOM (SIGKILL / exit 137) at 32-GPU scale.
        #
        # Source audio does NOT have this problem: its log-fbank features are
        # extracted lazily one sample at a time downstream in the sample processor
        # (alm_sample_processor -> LogFbankProcessor.extract_features), peak O(1).
        # We mirror that streaming behaviour here while keeping the GPU-efficient
        # batched `quantize` call: accumulate at most `batch_size` mels, flush the
        # batch, then immediately free its mels/device tensors. Each sample's raw
        # `answer_audio` is popped as soon as its mel is computed, so waveforms are
        # released incrementally. Peak host RAM is now O(batch_size). The produced
        # tokens are identical to the previous implementation (same batch order,
        # same padding, same quantize call).

        def _flush_batch(batch_mels, batch_indices):
            nonlocal n_ok, n_fail
            if not batch_mels:
                return
            try:
                padded_mels, mels_lens = self._s3tokenizer_module.padding(batch_mels)
                stream_ctx = torch.cuda.stream(self._s3tokenizer_stream) if self._s3tokenizer_stream else torch.no_grad()
                with torch.no_grad(), stream_ctx:
                    mels_device = padded_mels.to(self._s3tokenizer_device, dtype=torch.float32)
                    codes, codes_lens = self._s3tokenizer.quantize(
                        mels_device,
                        mels_lens.to(self._s3tokenizer_device),
                    )
                if self._s3tokenizer_stream:
                    self._s3tokenizer_stream.synchronize()
                codes_cpu = codes.cpu()
                codes_lens_cpu = codes_lens.cpu()
                for j, idx in enumerate(batch_indices):
                    all_chunk_data[idx]["answer_cv_token"] = str(
                        codes_cpu[j, : codes_lens_cpu[j].item()].tolist()
                    )
                    n_ok += 1
                del padded_mels, mels_device, codes, codes_lens, codes_cpu, codes_lens_cpu
            except Exception as e:
                logger.warning(f"OTF batch inference failed: {e}")
                for idx in batch_indices:
                    all_chunk_data[idx]["answer_cv_token"] = None
                n_fail += len(batch_indices)

        batch_mels, batch_indices = [], []
        for i, sample in enumerate(all_chunk_data):
            if not sample or sample.get("answer_audio") is None:
                continue
            try:
                audio_data, _ = sample["answer_audio"]
                if audio_data is None or len(audio_data) == 0:
                    sample["answer_cv_token"] = None
                    n_fail += 1
                    continue
                audio_tensor = torch.from_numpy(audio_data).float() if isinstance(audio_data, np.ndarray) else torch.tensor(audio_data, dtype=torch.float32)
                if audio_tensor.dim() > 1:
                    audio_tensor = audio_tensor.mean(dim=-1)
                batch_mels.append(self._s3tokenizer_module.log_mel_spectrogram(audio_tensor))
                batch_indices.append(i)
                n_valid += 1
            except Exception as e:
                logger.warning(f"OTF mel extraction failed for sample {i}: {e}")
                sample["answer_cv_token"] = None
                n_fail += 1
            finally:
                # Raw waveform is no longer needed once its mel is computed (or on
                # failure); drop it immediately to keep peak host RAM bounded.
                sample.pop("answer_audio", None)

            if len(batch_mels) >= batch_size:
                _flush_batch(batch_mels, batch_indices)
                batch_mels, batch_indices = [], []

        # Flush the final partial batch.
        _flush_batch(batch_mels, batch_indices)
        batch_mels, batch_indices = [], []

        for sample in all_chunk_data:
            if not sample:
                continue
            sample.pop("answer_audio", None)
            sample.setdefault("answer_cv_token", None)

        logger.debug(
            "OTF semantic tokenization: %d/%d OK, %d failed, %.2fs",
            n_ok,
            n_valid,
            n_fail,
            time.time() - start_time,
        )
        return all_chunk_data

    def _local_s3tokenizer_path(self, model_path):
        import fcntl
        import shutil

        cache_dir = os.environ.get("S3TOKENIZER_LOCAL_CACHE", "/tmp/s3tokenizer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, os.path.basename(model_path))
        if os.path.exists(local_path) or not os.path.exists(model_path):
            return local_path if os.path.exists(local_path) else model_path

        lock_fd = open(local_path + ".lock", "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if not os.path.exists(local_path):
                shutil.copy(model_path, local_path + ".tmp")
                os.replace(local_path + ".tmp", local_path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        return local_path

    def _s3tokenizer_device_from_config(self, semantic_otf_config):
        device_override = semantic_otf_config.get("device", "cuda")
        if device_override == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        if local_rank > 0:
            time.sleep(local_rank * 5)
        return torch.device(f"cuda:{local_rank}")


class BlobChunkIterableDataset(BaseBlobChunkIterableDataset):
    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        num_replicas: int = None,
        rank: int = None,
        seed: int = 0,
        multi_threaded_download: bool = False,
        dummy_data_mode: bool = False,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        allow_pure_text_return: bool = False,
        debug: bool = False,
        skip_blob_list=None,
        chunk_processing_pipe=None,
        path_prefix_in_orange: Optional[str] = None,
        use_storage_api: bool = False,
    ):
        IterableDataset.__init__(self)
        self.dataset_spec = DatasetSpec(
            dataset_spec,
            use_num_examples_as_durations,
            tsv_chunk_size,
            allow_pure_text_return,
            debug,
            skip_blob_list=skip_blob_list,
            chunk_processing_pipe=chunk_processing_pipe,
            path_prefix_in_orange=path_prefix_in_orange,
            use_storage_api=use_storage_api,
        )
        self.num_chunks_for_shuffling = num_chunks_for_shuffling
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode
        self.epoch = 0
        self.step_in_epoch = 0


class ChunkSpeechTextSeq2SeqDataset(BaseChunkSpeechTextSeq2SeqDataset):
    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        transforms: Callable = None,
        tokenize: Callable = None,
        context_length: int = 77,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,
        seed: int = 0,
        multi_threaded_download: bool = False,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        debug: bool = False,
        skip_blob_file_path=None,
        skip_missing_data=False,
        tokenize_padding="max_length",
        allow_pure_text_return=False,
        chunk_processing_pipe=None,
        path_prefix_in_orange: Optional[str] = None,
        use_storage_api: Optional[bool] = False,
    ):
        IterableDataset.__init__(self)
        self.transforms = transforms
        self.tokenize = tokenize
        self.tokenize_padding = "do_not_pad" if tokenize_padding is None else tokenize_padding
        self.context_length = context_length
        self.skip_missing_data = skip_missing_data
        skip_blob_list = None
        if skip_blob_file_path is not None and os.path.exists(skip_blob_file_path):
            with open(skip_blob_file_path, "r") as f:
                skip_blob_list = [line.strip() for line in f]

        self.use_num_examples_as_durations = use_num_examples_as_durations
        self.tsv_chunk_size = tsv_chunk_size
        self.allow_pure_text_return = allow_pure_text_return
        self.debug = debug
        self.audio_dataset = BlobChunkIterableDataset(
            dataset_spec=dataset_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            use_num_examples_as_durations=use_num_examples_as_durations,
            tsv_chunk_size=tsv_chunk_size,
            allow_pure_text_return=allow_pure_text_return,
            debug=debug,
            skip_blob_list=skip_blob_list,
            chunk_processing_pipe=chunk_processing_pipe,
            path_prefix_in_orange=path_prefix_in_orange,
            use_storage_api=use_storage_api,
        )
        self.num_samples = sum(self.audio_dataset.dataset_spec.chunk_sizes)
        self.total_durations = self.audio_dataset.dataset_spec.total_durations or 40000 * 3600
