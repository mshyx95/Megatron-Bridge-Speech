# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import backoff
import os
import io
import copy
import json
from json import JSONDecodeError
import logging
import math
from pathlib import Path
import struct
import random
import threading
import time
import torch
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from numpy import random
import pandas as pd
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import IterableDataset
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_vlm.blobchunk import KenslerPermutation
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.conversation_cropper import ConversationCropper
from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.transforms.word_speaker_label import GenWordSpeakerLabel

logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import blobfile as bf

_LARGE_FILE_TYPES = frozenset({"feature", "audio"})


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer value for %s=%r; using default %s", name, value, default)
        return default

class DataParser:
    def __init__(self, feat_dim=80):
        self.feat_dim = feat_dim

    def parse_data(self, data, data_type):
        if data_type.lower() == "audio":
            parsed_data = self._parse_audio_data(data)
        elif data_type.lower() in ["info", "sft"]:
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "feature":
            parsed_data = self._parse_feat_data(data)
        elif data_type.lower() == "label":
            parsed_data = self._parse_label_data(data)
        elif data_type.lower() == "alignment":
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "single_qa":
            parsed_data = self._parse_eval_data(data)
        else:
            parsed_data = self._parse_string_data(data)
        return parsed_data

    def _parse_audio_data(self, data):
        byte_stream = io.BytesIO(data)
        return sf.read(byte_stream)

    def _parse_label_data(self, data):  # one dimension numpy array
        label_pairs = []
        for i in range(int(len(data) / 4)):
            label = struct.unpack_from("<h", data, i * 4)[0]
            repeat_num = struct.unpack_from("<h", data, i * 4 + 2)[0]
            label_pairs.append((label, repeat_num))

        labels = []
        for label_pair in label_pairs:
            labels.extend([label_pair[0]] * label_pair[1])

        return " ".join(list(map(str, labels)))

    def _parse_json_data(self, data):
        str_data = str(data, "utf-8")
        json_data = json.loads(str_data)
        return json_data

    def _parse_string_data(self, data):
        str_data = str(data, "utf-8")
        return str_data

    def _parse_feat_data(self, data):
        feat = np.frombuffer(data, dtype=np.float32)
        feat = feat.reshape(-1, self.feat_dim)
        return feat

    def _parse_eval_data(self, data):
        str_data = str(data, "utf-8")
        json_data = eval(str_data)
        return json_data


class DatasetSpec:
    """
    Data set specification

    This class parses a data set specification dictionary
    and hosts the ContainerClients used to connect to the Azure Blob Storage Accounts.

    The given dictionary is expected to conform with the following example:
    {
        "data_sources": [
            {
                "manifest_file": "/datablob/am_data/en_us_asr.json",
            },
            {
                "manifest_file": "/datablob/am_data/en_uk_asr.json",
            }
        ],
        "chunk_type": ["info", "audio", "transcription"]
    }

    WARNING: The chunk_index_range list is inclusive. It does not match Python's range.
    """

    def __init__(
        self,
        dataset_spec: dict,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        allow_pure_text_return: bool = False,
        debug: bool = False,
        skip_blob_list: List = None,
        chunk_processing_pipe: List = None,     
        path_prefix_in_orange: Optional[str] = None,
        use_storage_api: bool = False,
    ):
        allowed_keys = {"data_sources", "sample_profiles"}
        unknown_keys = set(dataset_spec.keys()) - allowed_keys
        assert not unknown_keys, f"Unsupported dataset_spec keys: {unknown_keys}"
        self.__dict__.update(dataset_spec)
        self.sample_profiles = dataset_spec.get("sample_profiles", {})

        if len(self.data_sources[0]["chunk_type"]) == 1 and self.data_sources[0]["chunk_type"][0] == "tsv":
            self.is_tsv = True
        else:
            self.is_tsv = False

        self.debug = debug
        self.skip_blob_list = skip_blob_list
        self.allow_pure_text_return = allow_pure_text_return
        self.chunk_processing_pipe = self.init_chunk_processing_pipe(chunk_processing_pipe) if chunk_processing_pipe is not None else None
        self.path_prefix_in_orange = path_prefix_in_orange
        self.use_storage_api = use_storage_api
        if self.use_storage_api:
            self.create_downloader()

        self.chunk_sizes = []
        self.total_chunks = 0
        self.total_durations = 0
        self.accum_chunks = []
        for data_source in self.data_sources:
            if not os.path.isdir("/datablob1") and self.path_prefix_in_orange is not None:
                self.adapt_file_paths(data_source)
            
            chunk_list = self.load_manifest_file(data_source)
            if self.is_tsv:
                n_examples = [len(eval(m)) for m in chunk_list["messages"]]
                self.total_durations += sum(n_examples)
                assert use_num_examples_as_durations, "Please use examples as weight for tsv data!"
                mean_examples = sum(n_examples) / len(n_examples)
                in_tsv_chunk_size = int(tsv_chunk_size // mean_examples)

                # split on tsv files into chunks
                total_chunks = math.ceil(len(chunk_list) / in_tsv_chunk_size)
                data_source["total_chunks"] = total_chunks
                self.total_chunks += total_chunks
                last_chunk_size = len(chunk_list) - in_tsv_chunk_size * (total_chunks - 1)
                chunk_sizes = [in_tsv_chunk_size] * (total_chunks - 1) + [last_chunk_size]
                self.chunk_sizes.extend(chunk_sizes)

                data_source["local_chunk_sizes"] = chunk_sizes
                data_source["chunk_tsv_offsets"] = self.get_tsv_chunk_offset(data_source["manifest_file"], chunk_sizes)
            else:
                # Subsample chunks if sample_ratio is set (0 < ratio < 1)
                sample_ratio = data_source.get("sample_ratio", 1.0)
                if 0 < sample_ratio < 1.0:
                    n_total = len(chunk_list)
                    n_keep = max(1, int(n_total * sample_ratio))
                    rng = np.random.RandomState(hash(data_source.get("spec_file", "")) & 0xFFFFFFFF)
                    sampled_indices = sorted(rng.choice(n_total, size=n_keep, replace=False).tolist())
                    data_source["_sampled_chunk_indices"] = sampled_indices
                    chunk_list = [chunk_list[i] for i in sampled_indices]
                    logger.info(f"Subsampled {n_total} -> {n_keep} chunks (sample_ratio={sample_ratio}) for {data_source.get('spec_file', 'unknown')}")
                data_source["total_chunks"] = len(chunk_list)
                self.total_chunks += data_source["total_chunks"]
                for chunk in chunk_list:
                    self.get_chunk_size(chunk)
                    example_count = int(chunk.get("example_count", self.chunk_sizes[-1]))
                    self.total_durations += float(chunk["duration"]) if not use_num_examples_as_durations else example_count
            self.accum_chunks.append(self.total_chunks)
        assert len(self.chunk_sizes) == self.total_chunks

        self.parser = DataParser()

    def create_downloader(self):
        assert self.use_storage_api, "create_downloader is only used when use_storage_api is True"
        from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.az_downloader import AzureDownloaderConfig, AzureBlobDownloader

        AZ_DATA_ACCOUNT = os.environ.get("AZ_DATA_ACCOUNT", "https://highperf01safn.blob.core.windows.net/data/")
        
        self.container_name = Path(AZ_DATA_ACCOUNT).name
        self.account_url = AZ_DATA_ACCOUNT.replace(f"/{self.container_name}", "")

        self.az_cfg = AzureDownloaderConfig(
            account_url=self.account_url,
            container_name=self.container_name,
        )
        self.az_downloader = AzureBlobDownloader(cfg=self.az_cfg, lazy=False)
 
    def adapt_file_paths(self, data_source):
        if self.path_prefix_in_orange is not None:
            # replace the first part under root with the prefix
            data_source["manifest_file"] = os.path.join(self.path_prefix_in_orange, *Path(data_source["manifest_file"]).parts[2:])
        else:   # handle datablob1 and datablob inconsistency on different GPU clusters
            if "manifest_file" in data_source:
                if not os.path.isfile(data_source["manifest_file"]):
                    data_source["manifest_file"] = data_source["manifest_file"].replace("/datablob1/", "/datablob/")
            for key in ["manifest_file", "chunk_path", "trans_path"]:
                if key in data_source and not os.path.isdir(data_source[key]):
                    data_source[key] = data_source[key].replace("/datablob1/", "/datablob/")
            for key in ["additional_data_path", "additional_path"]:
                if key in data_source:
                    for sub_key in data_source[key]:
                        if not os.path.isdir(data_source[key][sub_key]):
                            data_source[key][sub_key] = data_source[key][sub_key].replace("/datablob1/", "/datablob/")
            for key in data_source:
                if "datablob1" in str(data_source[key]):
                    data_source[key] = data_source[key].replace("/datablob1/", "/datablob/")

    def get_chunk_size(self, chunk):
        if self.chunk_processing_pipe is not None:
            # some of the processing steps may change the chunk size, such as ConversationCropper
            for step in self.chunk_processing_pipe:
                if isinstance(step, ConversationCropper):
                    self.chunk_sizes.append(step.get_estimated_num_of_chunks(chunk["duration"]))
                    break
        else:
            self.chunk_sizes.append(int(chunk["count"]))

    def init_chunk_processing_pipe(self, chunk_processing_pipe):
        """ Initialize the chunk processing pipeline steps."""
        steps = []
        for step_cfg in chunk_processing_pipe:
            if step_cfg["type"] == "ConversationCropper":
                step = ConversationCropper(**step_cfg["params"])
            elif step_cfg["type"] == "GenWordSpeakerLabel":
                step = GenWordSpeakerLabel(**step_cfg["params"])
            else:
                raise ValueError(f"Unknown chunk processing step type: {step_cfg['type']}")
            steps.append(step)

        return steps

    def apply_chunk_processing_pipe(self, all_chunk_data):
        """Apply the chunk processing pipeline to the chunk data."""
        new_all_chunk_data = []
        for chunk_data in all_chunk_data:
            processed_chunk_data = chunk_data
            for step in self.chunk_processing_pipe:
                processed_chunk_data = step(processed_chunk_data)
            new_all_chunk_data += processed_chunk_data

        return new_all_chunk_data

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_manifest_file(self, data_source, n_chunks=-1):
        full_path = data_source["manifest_file"]
        
        if self.is_tsv:
            col_names = ["id", "wav_paths", "messages"]
            if n_chunks == -1:
                with (bf.BlobFile(full_path, "rb") if full_path.startswith("az://") else open(full_path, "rb")) as file:
                    return pd.read_csv(file, names=col_names, sep="\t")
            else:
                nrows = data_source["local_chunk_sizes"][n_chunks]
                # chunk_data =  pd.read_csv(full_path, names=col_names, skiprows=n_chunks * data_source["in_tsv_chunk_size"], nrows=nrows, sep="\t")
                # pandas reading may cause high cpu memory usage, we save the file offset for each chunk
                offset = data_source["chunk_tsv_offsets"][n_chunks]
                chunk_data = {"id": [], "wav_paths": [], "messages": []}

                with (bf.BlobFile(full_path, "rb") if full_path.startswith("az://") else open(full_path, "rb")) as file:
                    file.seek(offset)
                    return pd.read_csv(file, names=col_names, nrows=nrows, sep="\t")
                return chunk_data
        else:
            with (bf.BlobFile(full_path, "r") if full_path.startswith("az://") else open(full_path, "r", encoding="utf-8")) as f:
                try:
                    return json.load(f)["fileInfo"]
                except JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse manifest JSON: {full_path}") from exc

    def get_tsv_chunk_offset(self, tsv_file, chunk_sizes):
        offsets = []

        with (bf.BlobFile(tsv_file, "rb") if tsv_file.startswith("az://") else open(tsv_file, "rb")) as file:
            offset = 0
            for i in range(len(chunk_sizes)):
                offsets.append(offset)
                offset += sum(len(file.readline()) for _ in range(chunk_sizes[i]))
            assert file.readline() == b""
        return offsets

    # @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_chunk_with_chunkname(self, blob_name: str, chunk_type: str, chunk_size: int):
        if self.skip_blob_list is not None and blob_name in self.skip_blob_list:
            logging.error(f"chunk {blob_name} in missing file/skip list. Skip and return None.")
            return [None for _ in range(chunk_size)]
        ENDIAN = "little"
        example_list = []
        if self.use_storage_api:
            concurrency = None if chunk_type.lower() in _LARGE_FILE_TYPES else 1
            file_handle = self.az_downloader.download_to_buffer(blob_name, concurrency=concurrency)
        else:
            file_handle = bf.BlobFile(blob_name, "rb") if self.path_prefix_in_orange is not None else open(blob_name, "rb")
        
        with file_handle as f:
            target_type = f.read(len(chunk_type.encode())).decode()
            
            if chunk_type.lower() != target_type.lower():
                raise ValueError(f"Target type is not expected in {blob_name}, expected {chunk_type}, but got {target_type}")
            version_number = int.from_bytes(f.read(4), byteorder=ENDIAN)

            i = 0
            while True:
                buf = f.read(4)
                if buf == b"":
                    break
                example_index = int.from_bytes(buf, byteorder=ENDIAN)
                if example_index != i:
                    raise ValueError(f"The example index is corrupted in {blob_name}, expected {i}, but got {example_index}")
                if target_type.lower() == "audios":
                    parsed_data = []
                    n_audios = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    for j in range(n_audios):
                        data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                        data = f.read(data_size)
                        parsed_data.append(self.parser.parse_data(data, "audio"))
                else:
                    data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    if target_type.lower() == "label":
                        data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                    data = f.read(data_size)
                    parsed_data = self.parser.parse_data(data, chunk_type)
                example_list.append(parsed_data)
                i += 1

        return example_list
    
    def safe_read(self, wav_path, read_blob_file=False):
        """Read wav file, return None if failed"""
        try:
            if self.use_storage_api:
                wav_path = os.path.join(*Path(wav_path).parts[2:])
                byte_stream = self.az_downloader.download_to_buffer(wav_path, concurrency=None)
                return sf.read(byte_stream)
            
            if read_blob_file:
                with bf.BlobFile(wav_path, "rb") as f:
                    return sf.read(f)

            return sf.read(wav_path)
        except Exception as e:
            # show error message in logger, but do not raise exception since we want to be robust to single wav read failure
            # logger.info(f"read wav failed file name {wav_path}, return None " + str(e))
            return None

    def load_chunk(self, global_chunk_index: int):
        """
        download and return chunk data in form List[], each item in the list represents one sample

        Args:
            global_chunk_index (int): _description_

        Returns:
            _type_: _description_
        """

        assert 0 <= global_chunk_index < self.total_chunks
        # determine where the chunk should be fetched from
        data_source_index = np.searchsorted(self.accum_chunks, global_chunk_index, side="right")
        data_source = self.data_sources[data_source_index]
        local_chunk_index = global_chunk_index - (self.accum_chunks[data_source_index - 1] if data_source_index > 0 else 0)

        if self.is_tsv:
            chunk_list = self.load_manifest_file(data_source, n_chunks=local_chunk_index)
            all_chunk_data = []
            for i in range(len(chunk_list["wav_paths"])):
                chunk_data = {}
                wav_paths = eval(chunk_list["wav_paths"][i])
                if self.path_prefix_in_orange:
                    # replace the first part under root with the prefix
                    wav_paths = [os.path.join(self.path_prefix_in_orange, *Path(p).parts[2:]) for p in wav_paths]
                
                if self.allow_pure_text_return:
                    chunk_data["audios"] = [("placeholder", 16000) for _ in wav_paths]
                else:
                    chunk_data["audios"] = [self.safe_read(wav_path, read_blob_file=self.path_prefix_in_orange is not None) for wav_path in wav_paths]
                    if not any(chunk_data["audios"]):
                        logger.info(f"read wav failed file name {wav_paths[0]}")
                        chunk_data["audios"] = None
                chunk_data["sft"] = eval(chunk_list["messages"][i])
                all_chunk_data.append(chunk_data)
            chunk_type = ["audios", "sft"]
        else:
            chunk_list = self.load_manifest_file(data_source)
            actual_index = data_source["_sampled_chunk_indices"][local_chunk_index] if "_sampled_chunk_indices" in data_source else local_chunk_index
            chunk_name = chunk_list[actual_index]["name"]
            chunk_type = data_source["chunk_type"]
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
                    # replace the first part under root with the prefix
                    path = os.path.join(self.path_prefix_in_orange, *Path(path).parts[2:])
                
                if self.use_storage_api:
                    path_parts = list(Path(path).parts)[2:]
                    path = os.path.join(*path_parts)

                blob_name = os.path.join(path, chunk_name + "." + extension)

                try:
                    chunk_data = self.load_chunk_with_chunkname(blob_name, extension, chunk_size)
                    if len(chunk_data) != chunk_size:
                        logger.info(f"chunk size mismatch for chunk {blob_name} {extension}, expected {chunk_size}, got {len(chunk_data)}")
                        chunk_data = [None for _ in range(chunk_size)]
                except Exception as e:
                    logger.info(f"decode chunk failed data source {data_source['manifest_file']}, blob name {blob_name}, return empty list" + str(e))
                    chunk_data = [None for _ in range(chunk_size)]

                for i in range(chunk_size):
                    all_chunk_data[i][extension] = chunk_data[i]

        for i in range(len(all_chunk_data)):
            if any(all_chunk_data[i][key] is None for key in chunk_type):
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
            source_metadata["_sample_overrides"] = copy.deepcopy(data_source["sample_overrides"])

        for sample in all_chunk_data:
            if sample:
                sample.update(source_metadata)

        return all_chunk_data


class BlobChunkIterableDataset(IterableDataset):
    """
    An IterableDataset for efficient large-scale data loading

    Args:
        dataset_spec (dict):
            Dataset specification dictionary.
            See class `DatasetSpec` above for more information.
        decode_chunk_fn (Callable):
            A function used to decode the binary representation of a chunk.
            Typically, you would want to use one of the functions
            `decode_tsv_chunk` or `decode_binary_chunk` defined above.
        num_chunks_for_shuffling (int):
            Number of chunks that should be used for shuffling.
            Each worker is going to prefetch this number of chunks,
            shuffles the examples from all of these chunks together,
            and serves them out.
        decode_example_fn (Callable):
            An optional function used to decode an example.
            If None, no function is applied an example.
            Default: None
        num_replicas (int):
            Total number of GPUs / data set instances in distributed training.
            If None, use torch.distributed to determine num_replicas.
            Default: None
        rank (int):
            Rank of this GPU / data set instance in distributed training.
            If None, use torch.distributed to determine rank.
            Default: None
        seed (int):
            Seed for shuffling.
        multi_threaded_download (bool):
            If True, use threads to download chunks, one thread per chunk.
            Default: False
        dummy_data_mode (bool):
            Used for debugging and unit tests.
            If True, the data set does not connect to an Azure storage account
            but instead just returns integers.
            Default: False
        debug (bool):
            If True, append debug info when decoding examples.
            Default: False
        skip_blob_list (List[str]): A list of blob file name. Will skip reading these blob files.
    """

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
        super().__init__()
        self.dataset_spec = DatasetSpec(
            dataset_spec, use_num_examples_as_durations, tsv_chunk_size,
            allow_pure_text_return, debug, skip_blob_list=skip_blob_list,
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
            raise ValueError("Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode

        self.epoch = 0
        self.step_in_epoch = 0

    def set_epoch(self, epoch: int):
        """
        Set epoch

        The next time you iterate over the data set after calling this function,
        it will start at the given epoch.

        Args:
            epoch (int): epoch from which data loading should start
        """
        self.epoch = epoch
        self.step_in_epoch = 0

    def resume_from_checkpoint(self, epoch: int, step_in_epoch: int):
        """
        Resume data loading from a checkpoint

        The next time you iterate over the data set after calling this function,
        it will resume at the specified epoch and step_in_epoch.

        WARNING:
        When resuming a job that used a specific value for num_workers in the data loader,
        you must make sure the resuming job uses the same value both in the data loader
        and in the constructor of this class.

        Args:
            epoch (int): epoch from which data loading should resume
            step_in_epoch (int): step in epoch from which data loading should resume
        """
        self.epoch = epoch
        self.step_in_epoch = step_in_epoch

    def __iter__(self):
        assert self.rank is not None and self.num_replicas is not None, "Distributed config not set."
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # assert self.num_workers == 0, f"unexpected number of workers {self.num_workers} when worker_info is None"
            worker_step_in_epoch = self.step_in_epoch
            num_replicas = self.num_replicas
            rank = self.rank
        else:
            # assert self.num_workers == worker_info.num_workers, f"unexpected number of workers self.num_workers={self.num_workers} worker_info.num_workers={worker_info.num_workers}"
            self.num_workers = worker_info.num_workers

            # In order to correctly resume data loading from a specific step,
            # we have to make sure to correctly handle the case the the PyTorch data loader
            # uses multiple worker processes.
            #
            # Let's say we have 4 worker processes.
            # The PyTorch data loader consumes examples from these workers in a round-robin fashion.
            # Let's say we want to resume at the global step 6,
            # then worker 2 (= 6 % 4) has to supply the next example.
            # However, the PyTorch data loader will always get the first example from worker 0.
            # To fix this, we rotate the data worker ids so that, in this example, worker 2 becomes worker 0.
            # Additionally, we have to skip an additional example on some of the worker,
            # namely worker 0 and 1 in our example.
            #
            # The logic below achieves exactly that.
            num_workers = max(self.num_workers, 1)
            worker_id_offset = self.step_in_epoch % num_workers

            worker_id = (worker_info.id + worker_id_offset) % num_workers

            num_replicas = self.num_replicas * num_workers
            rank = self.rank * num_workers + worker_id

            worker_step_in_epoch = self.step_in_epoch // num_workers
            if worker_id < worker_id_offset:
                worker_step_in_epoch += 1

        it = _BlobChunkIterableDatasetIterator(
            self.dataset_spec,
            self.num_chunks_for_shuffling,
            self.epoch,
            worker_step_in_epoch,
            num_replicas,
            rank,
            self.seed,
            self.multi_threaded_download,
            self.dummy_data_mode,
        )

        # in the next epoch, we want to start at the beginning
        self.step_in_epoch = 0

        return it





class _ChunkIndexPermutation:
    """
    This helper class contains the logic necessary to:
        - shuffle chunks
        - distribute the chunks across GPU and / or data loader workers
        - translate between (local) worker chunk indices and global chunk indices
        - make sure that all workers receive the same number of chunks
    """

    def __init__(self, total_chunks, num_replicas, rank, seed, epoch):
        self.total_chunks = total_chunks
        self.num_replicas = num_replicas
        self.rank = rank

        self.chunk_permutation = KenslerPermutation(length=self.total_chunks, seed=seed + epoch)

    def __getitem__(self, worker_chunk_index):
        global_chunk_index = self.rank + worker_chunk_index * self.num_replicas
        return self.chunk_permutation[global_chunk_index]


class _BlobChunkIterableDatasetIterator:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        num_chunks_for_shuffling: int,
        epoch: int,
        worker_step_in_epoch: int,
        num_replicas: int,
        rank: int,
        seed: int,
        multi_threaded_download: bool,
        dummy_data_mode: bool,
    ):
        self.dataset_spec = dataset_spec
        self.num_chunks_for_shuffling = num_chunks_for_shuffling
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch
        self.worker_step_in_epoch = worker_step_in_epoch
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode

        self.resuming_from_checkpoint = self.worker_step_in_epoch != 0

        self.total_chunks = self.dataset_spec.total_chunks

        # TODO (Yi-Ling): Currently, this class requires that the number of chunks is greater or equal to the global number of replicas (#gpus * #workers_per_gpu). This might be a problem when training on a small data set with many GPUs. To address this, we could add a feature to "virtually duplicate" all chunks. Effectively, this would mean that we do multiple sweeps through the data in a single epoch. To implement this, we would have to modify `_ChunkIndexPermutation` and some of the logic in this class.
        assert self.total_chunks >= self.num_replicas, f"too few chunks ({self.total_chunks}) in data set for number of replicas ({self.num_replicas})"

        self.chunk_index_permutation = _ChunkIndexPermutation(self.total_chunks, self.num_replicas, self.rank, self.seed, self.epoch)

        worker_permuted_chunk_sizes = [self.dataset_spec.chunk_sizes[self.chunk_index_permutation.chunk_permutation[i]] for i in range(self.rank, self.total_chunks, self.num_replicas)]
        self.worker_accum_permuted_chunk_sizes = np.cumsum(worker_permuted_chunk_sizes)
        self.worker_total_num_examples = self.worker_accum_permuted_chunk_sizes[-1]
        self.worker_accum_buffered_chunk_sizes = [self.worker_accum_permuted_chunk_sizes[-1]] if len(worker_permuted_chunk_sizes) < self.num_chunks_for_shuffling else self.worker_accum_permuted_chunk_sizes[self.num_chunks_for_shuffling - 1 :: self.num_chunks_for_shuffling]
        if len(worker_permuted_chunk_sizes) > self.num_chunks_for_shuffling and len(worker_permuted_chunk_sizes) % self.num_chunks_for_shuffling != 0:
            assert self.worker_accum_buffered_chunk_sizes[-1] < self.worker_total_num_examples
            self.worker_accum_buffered_chunk_sizes = np.concatenate((self.worker_accum_buffered_chunk_sizes, [self.worker_total_num_examples]))

        self.example_buffer = []
        self.example_buffer_permutation = None

    def __next__(self):
        worker_step_this_sweep = self.worker_step_in_epoch % self.worker_total_num_examples
        search_index = np.searchsorted(self.worker_accum_buffered_chunk_sizes, worker_step_this_sweep)
        if worker_step_this_sweep == 0 or self.worker_accum_buffered_chunk_sizes[search_index] == worker_step_this_sweep:
            example_index = 0
        else:
            example_index = worker_step_this_sweep if search_index == 0 else worker_step_this_sweep - self.worker_accum_buffered_chunk_sizes[search_index - 1]

        data_exhausted = False
        if self.resuming_from_checkpoint or example_index == 0:
            data_exhausted = self._replenish_example_buffer()
            self.resuming_from_checkpoint = False

        if not data_exhausted and len(self.example_buffer) == 0:
            logging.warning(
                "Empty buffer after replenishment. Advance to the next chunks. \
                            Dangerous operations if you uses multiple iter-type data, when only one data is moved to next."
            )
            self.worker_step_in_epoch += block_size
            return next(self)

        if example_index >= len(self.example_buffer):
            logger.debug(f"Data set exhausted at rank {dist.get_rank()}")
            raise StopIteration  # data set exhausted

        shuffled_example_index = self.example_buffer_permutation[example_index]
        assert shuffled_example_index < len(self.example_buffer)

        self.worker_step_in_epoch += 1
        example = self.example_buffer[shuffled_example_index]

        return example

    def __iter__(self):
        return self

    def _replenish_example_buffer(self):
        worker_step_this_sweep = self.worker_step_in_epoch % self.worker_total_num_examples
        search_index = np.searchsorted(self.worker_accum_permuted_chunk_sizes, worker_step_this_sweep)
        assert self.resuming_from_checkpoint or worker_step_this_sweep == 0 or self.worker_accum_permuted_chunk_sizes[search_index] == worker_step_this_sweep
        if self.resuming_from_checkpoint:
            idx = np.searchsorted(self.worker_accum_buffered_chunk_sizes, worker_step_this_sweep)
            if self.worker_accum_buffered_chunk_sizes[idx] == worker_step_this_sweep:
                search_index = (idx + 1) * self.num_chunks_for_shuffling - 1
            else:
                search_index = idx * self.num_chunks_for_shuffling - 1
        worker_chunk_index_starts_from = 0 if worker_step_this_sweep == 0 else search_index + 1
        this_num_chunks_for_shuffling = min(
            self.num_chunks_for_shuffling,
            len(self.worker_accum_permuted_chunk_sizes) - worker_chunk_index_starts_from,
        )

        if self.multi_threaded_download:
            # Helper to submit chunk read jobs
            first_global_chunk_index = [0]

            def load_one_chunk(i: int):
                worker_chunk_index = worker_chunk_index_starts_from + i
                # global_chunk_index through permutation
                try:
                    global_chunk_index = self.chunk_index_permutation[worker_chunk_index]
                except:
                    global_chunk_index = None
                # Use dataset_spec to load the chunk
                if i == 0:
                    first_global_chunk_index[0] = global_chunk_index
                if global_chunk_index is None:  # chunks exhausted:
                    return []
                return self.dataset_spec.load_chunk(global_chunk_index)

            # Launch multi-threaded chunk reads
            max_workers = min(this_num_chunks_for_shuffling, _get_env_int("BLOBCHUNK_DOWNLOAD_MAX_WORKERS", 4))
            thread_results = [None] * this_num_chunks_for_shuffling
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(load_one_chunk, i): i for i in range(this_num_chunks_for_shuffling)
                }
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    try:
                        examples = fut.result()
                    except Exception as e:
                        logging.warning(f"Failed to load chunk {i}: {e}")
                        examples = []

                    thread_results[i] = examples

            # Aggregate results from threads into example_buffer
            self.example_buffer = []
            for result in thread_results:
                if result is not None:
                    self.example_buffer += result

            del thread_results
            if len(self.example_buffer) == 0:
                # data exhausted or all chunks failed
                return True
            
            # compute permutation used to shuffle examples in example buffer
            seed = self.seed + (self.worker_step_in_epoch // self.worker_total_num_examples) * self.total_chunks + first_global_chunk_index[0]
            self.example_buffer_permutation = KenslerPermutation(len(self.example_buffer), seed=seed)
            return False  # there are still remaining data
        else:
            self.example_buffer = []
            threads = []
            thread_results = [None] * this_num_chunks_for_shuffling
            for i in range(this_num_chunks_for_shuffling):
                worker_chunk_index = worker_chunk_index_starts_from + i
                global_chunk_index = self.chunk_index_permutation[worker_chunk_index]

                if i == 0:
                    first_global_chunk_index = global_chunk_index  # used to compute seed for example buffer permutation

                if global_chunk_index is None:  # chunks exhausted:
                    if i == 0:  # end of data set
                        return True
                    else:  # some remaining data to be processed
                        break

                self._read_chunk(global_chunk_index, i, thread_results)

            # collect results
            for result in thread_results:
                if result is not None:
                    self.example_buffer += result

            del thread_results
            # compute permutation used to shuffle examples in example buffer
            seed = self.seed + (self.worker_step_in_epoch // self.worker_total_num_examples) * self.total_chunks + first_global_chunk_index
        
            self.example_buffer_permutation = KenslerPermutation(len(self.example_buffer), seed=seed)
            return False  # there are still remaining data

        

    def _read_chunk(self, chunk_index: int, thread_id: int, thread_results: List):
        if self.dummy_data_mode:
            examples = list(
                range(
                    self.worker_accum_permuted_chunk_sizes[chunk_index],
                    self.worker_accum_permuted_chunk_sizes[chunk_index + 1],
                )
            )
        else:
            examples = self.dataset_spec.load_chunk(chunk_index)
        thread_results[thread_id] = examples


class ChunkSpeechTextSeq2SeqDataset(IterableDataset):
    """
    An IterableDataset for efficient large-scale data loading of image-text pairs for seq-to-seq modeling
    supporting multiple text for each image (e.g. caption, OD, face).

    Args:
        dataset_spec (dict):
            Dataset specification dictionary.
            See example below.
        num_chunks_for_shuffling (int):
            Number of chunks that should be used for shuffling.
            Each worker is going to prefetch this number of chunks,
            shuffles the examples from all of these chunks together,
            and serves them out.
        transforms (Callable):
            An optional function used transform the images.
            Default: None
        tokenize (Callable):
            An optional function used transform the text.
            Default: None
        context_length (int):
            The context length to use; We follow CLIP models to use 77 as the default context length.
            Default: 77
        num_replicas (int):
            Total number of GPUs / data set instances in distributed training.
            If None, use torch.distributed to determine num_replicas.
            Default: None
        rank (int):
            Rank of this GPU / data set instance in distributed training.
            If None, use torch.distributed to determine rank.
            Default: None
        seed (int):
            Seed for shuffling.
        multi_threaded_download (bool):
            If True, use threads to download chunks, one thread per chunk.
            Default: False
        debug (bool):
            If True, append debug info when decoding examples.
            Default: False
        skip_blob_list (List[str]): A list of blob file name. Will skip reading these blob files.
        skip_missing_data (bool): whether skip data if missing or wrongly decoded.

    Example dataset_spec:
    {
        "total_chunks": 100,
        "examples_per_chunk": 1000,
        "data_sources": [
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
            },
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
            },
        ]
    }
    """

    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        transforms: Callable = None,
        tokenize: Callable = None,
        context_length: int = 77,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,  # no use
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
        self.transforms = transforms
        self.tokenize = tokenize
        if tokenize_padding is None:
            tokenize_padding = "do_not_pad"
        self.tokenize_padding = tokenize_padding
        self.context_length = context_length
        self.skip_missing_data = skip_missing_data
        if skip_blob_file_path is not None and os.path.exists(skip_blob_file_path):
            skip_blob_list = []
            with open(skip_blob_file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                skip_blob_list.append(line.strip())
            logging.info("all skip blob list {}".format(skip_blob_list))
        else:
            skip_blob_list = None

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
        self.total_durations = self.audio_dataset.dataset_spec.total_durations
        if use_num_examples_as_durations:
            logger.info("Total Training Audio Text Pairs: {}, Total Training Examples {}".format(self.num_samples, self.total_durations))
        else:
            logger.info("Total Training Audio Text Pairs: {}, Total Duration: {} hours".format(self.num_samples, self.total_durations / 3600))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set epoch

        The next time you iterate over the data set after calling this function,
        it will start at the given epoch.

        Args:
            epoch (int): epoch from which data loading should start
        """
        random.seed(epoch)
        self.audio_dataset.set_epoch(epoch)

    def set_state(self, state: Dict[str, Any]):
        self.resume_from_checkpoint(state["epoch"], state["generator_state"])

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        if torch.distributed.is_initialized():
            rank = rank if rank is not None else torch.distributed.get_rank()
            world_size = world_size if world_size is not None else torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        self.audio_dataset.rank = rank
        self.audio_dataset.num_replicas = world_size

    def resume_from_checkpoint(self, epoch: int, step_in_epoch: int):
        """
        Resume data loading from a checkpoint

        The next time you iterate over the data set after calling this function,
        it will resume at the specified epoch and step_in_epoch.

        WARNING:
        When resuming a job that used a specific value for num_workers in the data loader,
        you must make sure the resuming job uses the same value both in the data loader
        and in the constructor of this class.

        Args:
            epoch (int): epoch from which data loading should resume
            step_in_epoch (int): step in epoch from which data loading should resume
        """

        logger.info(f"Resume data loader from checkpoint: epoch({epoch}), step_in_epoch({step_in_epoch})")
        self.audio_dataset.resume_from_checkpoint(epoch, step_in_epoch)

    def __iter__(self):
        raise NotImplementedError("This class is not meant to be used directly. Use one of the subclasses.")
