import ast
import os
import io
import csv
import copy
import json
import logging
import math
import random
import base64
import threading
import time
import torch
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Union


from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from torch.utils.data import IterableDataset

from PIL import Image
from typing import Callable, List, Union

logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger
# logger = get_logger(__name__)

import warnings

from numpy import uint64, random

class KenslerPermutation:
    """
    A generator that returns a pseudo-random permutation of the numbers in [0, length - 1].
    The generator only uses O(1) internal state and retrieving the next number requires O(1) time on expectation
    (under the assumption that the internal hash-function is random...).
    This is in contrast to the often-used Fisher-Yates based algorithm that requires O(n) internal state,
    see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

    The idea behind this class comes from Andrew Kensler's permute() function presented in
    Correlated Multi-Jittered Sampling
    Andrew Kensler
    Pixar Technical Memo 13-01
    see https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf

    For a good introduction to the ideas in this paper see
    https://andrew-helmer.github.io/permute/

    The hash function used in this class to support 64-bit integers was taken from
    https://github.com/camel-cdr/cauldron/blob/main/tools/random/permute/README.md
    """

    def __init__(self, length, seed=0):
        self.length = uint64(length)
        self.seed = random.default_rng(seed).integers(2**64, dtype=uint64)
        self.mask = KenslerPermutation.get_mask(self.length)

    def __iter__(self):
        return (self[i] for i in range(self.length))

    def __getitem__(self, i):
        return int(KenslerPermutation.permute(uint64(i), self.mask, self.seed, self.length))

    @staticmethod
    def get_mask(length):
        with warnings.catch_warnings():
            # ignore numpy overflow warnings as they are expected due to bit-wise operations
            warnings.simplefilter("ignore")
            mask = length - uint64(1)
            mask |= mask >> uint64(1)
            mask |= mask >> uint64(2)
            mask |= mask >> uint64(4)
            mask |= mask >> uint64(8)
            mask |= mask >> uint64(16)
            mask |= mask >> uint64(32)
            return mask

    @staticmethod
    def hash(index, mask, seed):
        # this hash function has been taken from
        # https://github.com/camel-cdr/cauldron/blob/main/tools/random/permute/README.md
        index ^= seed
        # splittable64
        index ^= (index & mask) >> uint64(30)
        index *= uint64(0xBF58476D1CE4E5B9)
        index ^= (index & mask) >> uint64(27)
        index *= uint64(0x94D049BB133111EB)
        index ^= (index & mask) >> uint64(31)
        index *= uint64(0xBF58476D1CE4E5B9)

        index ^= seed >> uint64(32)
        index &= mask
        index *= uint64(0xED5AD4BB)

        index ^= seed >> uint64(48)
        # hash16_xm3
        index ^= (index & mask) >> uint64(7)
        index *= uint64(0x2993)
        index ^= (index & mask) >> uint64(5)
        index *= uint64(0xE877)
        index ^= (index & mask) >> uint64(9)
        index *= uint64(0x0235)
        index ^= (index & mask) >> uint64(10)

        # From Andrew Kensler: "Correlated Multi-Jittered Sampling"
        index ^= seed
        index *= uint64(0xE170893D)
        index ^= seed >> uint64(16)
        index ^= (index & mask) >> uint64(4)
        index ^= seed >> uint64(8)
        index *= uint64(0x0929EB3F)
        index ^= seed >> uint64(23)
        index ^= (index & mask) >> uint64(1)
        index *= uint64(1) | seed >> uint64(27)
        index *= uint64(0x6935FA69)
        index ^= (index & mask) >> uint64(11)
        index *= uint64(0x74DCB303)
        index ^= (index & mask) >> uint64(2)
        index *= uint64(0x9E501CC3)
        index ^= (index & mask) >> uint64(2)
        index *= uint64(0xC860A3DF)
        index &= mask
        index ^= index >> uint64(5)

        return index

    @staticmethod
    def permute(index, mask, seed, length):
        with warnings.catch_warnings():
            # ignore numpy overflow warnings as they are expected due to bit-wise operations
            warnings.simplefilter("ignore")
            index = KenslerPermutation.hash(index, mask, seed)
            while index >= length:
                index = KenslerPermutation.hash(index, mask, seed)
            return (index + seed) % length

def decode_tsv_chunk(chunk_data: bytes, examples_per_chunk: int) -> List:
    """
    Extract examples from tsv chunk

    Args:
        data: bytes object downloaded from blob storage
    Returns:
        A list of data elements from each line of the original tsv.
        The data elements of each line are also represented as a list.
    """
    csv.field_size_limit(100_000_000)
    examples = list(csv.reader(chunk_data.decode().split("\n"), delimiter="\t"))
    # remove last example if it is empty
    if len(examples[-1]) == 0:
        examples = examples[:-1]
    if len(examples) == examples_per_chunk:
        return examples
    elif len(examples) > examples_per_chunk:
        logger.warning(f"Too many example in this chunk, expected {examples_per_chunk}, got {len(examples)}")
        return examples[:examples_per_chunk]
    else:
        logger.warning(f"Too few example in this chunk, use last item to fill in, expected {examples_per_chunk}, got {len(examples)}")
        return examples.extend([examples[-1]] * (examples_per_chunk - len(examples)))


def decode_tsv_chunk_debug(chunk_data: bytes, examples_per_chunk: int) -> List:
    """
    Extract examples from tsv chunk

    WARNING: THIS FUNCTION IS FOR DEBUGGING ONLY, DO NOT USE THIS FOR TRAINING!

    Instead of asserting that the chunk contains the correct number of examples,
    this function only prints a warning if this is not the case
    and changes the number of examples in the chunk in a naive way.

    Args:
        data: bytes object downloaded from blob storage
    Returns:
        A list of data elements from each line of the original tsv.
        The data elements of each line are also represented as a list.
    """
    csv.field_size_limit(10_000_000)
    examples = list(csv.reader(chunk_data.decode().split("\n"), delimiter="\t"))
    examples = examples[:examples_per_chunk]
    while len(examples) < examples_per_chunk:
        examples.append(examples[0])
    return examples


def decode_binary_chunk(chunk_data: bytes, examples_per_chunk: int) -> List:
    """
    Args:
        chunk_data (bytes): bytes object downloaded from blob storage
        examples_per_chunk (int): number of examples in a chunk
    Returns:
        A list of data elements from each line of the original tsv.
        The data elements of each line are also represented as a list.
    """
    endianness = "little"
    index_byte_len = 8  # 64 bit indices

    # TODO: evaluate whether we should use a memoryview here
    # chunk_data = memoryview(chunk_data)
    offsets = [
        int.from_bytes(chunk_data[i * index_byte_len : (i + 1) * index_byte_len], byteorder=endianness,)
        for i in range(examples_per_chunk)
    ] + [len(chunk_data)]
    examples = [chunk_data[offsets[i] : offsets[i + 1]] for i in range(examples_per_chunk)]
    return examples


def download_blob_with_retry_and_jitter(container_client: ContainerClient, blob_name: str, 
                                        skip_blob_list=None):
    max_jitter = 2
    retry_initial_sleep_time = 2
    retry_backoff_factor = 2

    full_blob_name = f"{container_client.account_name}/{container_client.container_name}/{blob_name}"

    if skip_blob_list is not None:
        if full_blob_name in skip_blob_list:
            logging.error(f"chunk {full_blob_name} in missing file/skip list. Skip and return None.")
            return None

    time.sleep(random.random() * max_jitter)  # jitter: sleep random interval to reduce load on storage account

    retry_sleep_time = retry_initial_sleep_time
    logging.debug(f" => fetching chunk {full_blob_name}")
    # Retry up to 3 times to download the specified blob and return None if the blob is missing.
    # If download failure is not due to temporary service unavailability, it doesn't break deterministic training course.
    for _ in range(3):
        try:
            # # if blob file size is too large, skip it
            # blob_properties = container_client.get_blob_client(blob_name).get_blob_properties()
            # # skip 5G blob file
            # if blob_properties.size > 100_000_000:
            #     logging.error(f"blob {full_blob_name} is too large, size is {blob_properties.size}. Skip and return None.")
            #     return None
            # download with timeout 
            # timeout: 10min
            start_time = time.time()
            chunk_data = container_client.download_blob(blob_name, timeout=600).readall()
            end_time = time.time()
            if end_time - start_time > 600:
                logging.warning(f"Download chunk {full_blob_name} takes {end_time - start_time} seconds, skip")
                return None
            assert isinstance(chunk_data, bytes) and len(chunk_data) > 0
            return chunk_data
        except ResourceNotFoundError:
            logging.error(f"{full_blob_name} does not exist!")
            break
        except Exception:
            logging.error(f"failed to download chunk {full_blob_name}, sleeping for {retry_sleep_time} seconds before retrying")
            time.sleep(retry_sleep_time)
            retry_sleep_time *= retry_backoff_factor  # exponential back-off
    logging.error(f"Failed to download chunk {full_blob_name}. Skip and return None.")
    return None


class DatasetSpec:
    """
    Data set specification

    This class parses a data set specification dictionary
    and hosts the ContainerClients used to connect to the Azure Blob Storage Accounts.

    The given dictionary is expected to conform with the following example:
    {
        "total_chunks": 100,
        "examples_per_chunk": 1000,
        "data_sources": [
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
                "chunk_data_type": "tsv",
                "chunk_blob_name_template": "folder-0/image-{:010d}.tsv",
                "with_image_key": True
            },
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
                "chunk_data_type": "tsv",
                "chunk_blob_name_template": "folder-1/image-{:010d}.tsv",
                "with_image_key": True
            },
            {
                "total_chunks": 50,
                "chunk_offset": 5,
                // chunk index for this data source is from 5 to 54
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
                "chunk_data_type": "tsv",
                "chunk_blob_name_template": "folder-1/image-{:010d}.tsv",
                "with_image_key": True
            }
        ]
    }

    WARNING: The chunk_index_range list is inclusive. It does not match Python's range.
    """

    def __init__(
        self,
        dataset_spec: dict,
        debug: bool = False,
        share_client: bool = False,
        skip_blob_list: List = None
    ):
        assert dataset_spec.keys() == {"total_chunks", "examples_per_chunk", "data_sources"}
        self.__dict__.update(dataset_spec)
        self.debug = debug
        self.share_client = share_client
        self.skip_blob_list = skip_blob_list 

        # check chunk index ranges
        assert isinstance(self.data_sources, list)
        # check the sum of chunk number of all data sources equal to total_chunks
        if len(self.data_sources) > 0:
            assert self.total_chunks == sum([d['total_chunks'] for d in self.data_sources]), \
                f"{self.total_chunks} vs {sum([d['total_chunks'] for d in self.data_sources])}"

        self.create_client()
    
    def create_client(self):
        self.clients = []
        # creating too many clients might cause connection error
        if self.share_client:
            self.shared_clients = {}

        for data_source in self.data_sources:
            if "clientid" in data_source:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential(managed_identity_client_id=data_source['clientid'])
            else:
                credential=data_source["SAS"]
                if len(credential) == 0:
                    # if SAS is empty, use env 
                    # storage account
                    storage_account_name = data_source['storage_account_url'].split('//')[1].split('.')[0]
                    credential = os.environ.get(f"AZURE_STORAGE_SAS_{storage_account_name}", None)
                    if credential is None:
                        raise ValueError(f"SAS for {storage_account_name} is not set in env or data source")

            if self.share_client:
                if (data_source["storage_account_url"], data_source["container_name"]) not in self.shared_clients:
                    client = ContainerClient(
                        account_url=data_source["storage_account_url"],
                        container_name=data_source["container_name"],
                        credential=credential,
                    )
                    self.shared_clients[(data_source["storage_account_url"], data_source["container_name"])] = client
            else:
                client = ContainerClient(
                    account_url=data_source["storage_account_url"],
                    container_name=data_source["container_name"],
                    credential=credential,
                )
                self.clients.append(client)
        
        if self.share_client:
            total_clients = len(self.shared_clients.keys())
        else:
            total_clients = len(self.clients)

        logger.warning("Total {} azure storage clients created. share client is set to {}".format(
            total_clients,
            self.share_client))

    def download_chunk_with_chunkname(self, blob_name: str, client, data_source):
        tic = time.time()
        chunk_data = download_blob_with_retry_and_jitter(
            client,
            blob_name,
            skip_blob_list=self.skip_blob_list)
        if chunk_data and data_source["chunk_data_type"] == "tsv":
            chunk_data = decode_tsv_chunk(chunk_data, self.examples_per_chunk)
        elif chunk_data and data_source["chunk_data_type"] == "binary":
            chunk_data = decode_binary_chunk(chunk_data, self.examples_per_chunk)
        if data_source["chunk_data_type"] == "binary" and data_source["with_image_key"]:
            image_key_data = download_blob_with_retry_and_jitter(
                client,
                blob_name + ".key",
                skip_blob_list=self.skip_blob_list)
            if image_key_data:
                image_keys = decode_tsv_chunk(image_key_data, self.examples_per_chunk)
                chunk_data = [[key[0], item] for key, item in zip(image_keys, chunk_data)]
        if chunk_data and self.debug:
            chunk_data = [
                (datum,
                    {"account": data_source["storage_account_url"],
                    "blob": blob_name}) for datum in chunk_data]
        if chunk_data is None:
            # chunk_data is None, must return data, otherwise resulting issues with key difference with other iter-data 
            chunk_data = [[] for _ in range(self.examples_per_chunk)]
        return chunk_data

    def download_chunk(self, global_chunk_index: int):
        """
        download and return chunk data in form List[], each item in the list represents one sample

        Args:
            global_chunk_index (int): _description_

        Returns:
            _type_: _description_
        """
        assert 0 <= global_chunk_index < self.total_chunks
        # determine where the chunk should be fetched from
        accum_chunks = 0
        for _data_source_idx, data_source in enumerate(self.data_sources):
            if self.share_client:
                client = self.shared_clients[(data_source["storage_account_url"], data_source["container_name"])]
            else:
                client = self.clients[_data_source_idx]
            # found correct data source, fetch chunk
            if accum_chunks <= global_chunk_index < accum_chunks + data_source["total_chunks"]:
                local_chunk_index = global_chunk_index - accum_chunks
                local_chunk_offset = data_source.get("chunk_offset", 0)
                local_chunk_index += local_chunk_offset
                if 'multi_text' in data_source:
                    # multi_text_keys is a list of text keys
                    # multi-text, multiple annotations per image 
                    examples_per_chunk = self.examples_per_chunk
                    all_chunk_data = [[] for _ in range(examples_per_chunk)]
                    for _text in data_source['multi_text_keys']:
                        blob_name = data_source["chunk_blob_name_template"].format(_text, local_chunk_index)
                        _chunk_data = self.download_chunk_with_chunkname(blob_name, client, data_source)
                        if _chunk_data is None:
                            # some chunks may be missing, due to failed api call or annotation generation, skip them
                            # TODO: check if this is safe to continue
                            continue
                        assert len(_chunk_data) == examples_per_chunk
                        for i in range(examples_per_chunk):
                            all_chunk_data[i].append(_chunk_data[i])
                    return all_chunk_data
                else:
                    if "multi-lingual-data" in data_source:
                        lang = random.choice(data_source["langs"])
                        blob_name = data_source["chunk_blob_name_template"].format(lang, local_chunk_index)
                    else:
                        blob_name = data_source["chunk_blob_name_template"].format(local_chunk_index)
                    try:
                        chunk_data = self.download_chunk_with_chunkname(blob_name, client, data_source)
                    except Exception:
                        logger.info(f"decode chunk failed data source {data_source}, blob name {blob_name}, return empty list")
                        chunk_data = [[] for _ in range(self.examples_per_chunk)]
                return chunk_data
            accum_chunks += data_source["total_chunks"]
        assert False, f"chunk_index {global_chunk_index} is not in chunk_index_range of any data source"


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
        drop_last (bool):
            If True, drop some chunks to make the total number of chunks divisible
            by the total number of workers (#gpus * #workers_per_gpu).
            If False, repeat some chunks to make the total number of chunks divisible
            by the total number of workers (#gpus * #workers_per_gpu).
            Default: False
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
        share_client (bool):
            If True, share azure storage blob client if the blobs are stored in the the same container 
        skip_blob_list (List[str]): A list of blob file name. Will skip reading these blob files.
    """

    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,
        seed: int = 0,
        multi_threaded_download: bool = False,
        dummy_data_mode: bool = False,
        debug: bool = False,
        share_client: bool = False,
        skip_blob_list=None,
    ):
        super().__init__()
        self.share_client = share_client
        self.dataset_spec = DatasetSpec(dataset_spec, debug, 
                                        share_client=self.share_client, 
                                        skip_blob_list=skip_blob_list)
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
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
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
            self.drop_last,
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

    def __init__(self, total_chunks, num_replicas, rank, seed, epoch, drop_last):
        self.total_chunks = total_chunks
        self.num_replicas = num_replicas
        self.rank = rank

        if drop_last:
            self.chunk_index_range_end = self.num_replicas * math.floor(self.total_chunks / self.num_replicas)
        else:
            self.chunk_index_range_end = self.num_replicas * math.ceil(self.total_chunks / self.num_replicas)

        self.chunk_permutation = KenslerPermutation(length=self.total_chunks, seed=seed + epoch)

    def __getitem__(self, worker_chunk_index):
        global_chunk_index = self.rank + worker_chunk_index * self.num_replicas
        if global_chunk_index >= self.chunk_index_range_end:
            return None  # outside of range
        return self.chunk_permutation[global_chunk_index % self.total_chunks]


class _BlobChunkIterableDatasetIterator:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        num_chunks_for_shuffling: int,
        epoch: int,
        worker_step_in_epoch: int,
        num_replicas: int,
        rank: int,
        drop_last: bool,
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
        self.drop_last = drop_last
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode

        self.resuming_from_checkpoint = self.worker_step_in_epoch != 0

        self.total_chunks = self.dataset_spec.total_chunks
        self.examples_per_chunk = self.dataset_spec.examples_per_chunk

        # TODO (Yi-Ling): Currently, this class requires that the number of chunks is greater or equal to the global number of replicas (#gpus * #workers_per_gpu). This might be a problem when training on a small data set with many GPUs. To address this, we could add a feature to "virtually duplicate" all chunks. Effectively, this would mean that we do multiple sweeps through the data in a single epoch. To implement this, we would have to modify `_ChunkIndexPermutation` and some of the logic in this class.
        assert (
            self.total_chunks >= self.num_replicas
        ), f"too few chunks ({self.total_chunks}) in data set for number of replicas ({self.num_replicas})"

        self.chunk_index_permutation = _ChunkIndexPermutation(
            self.total_chunks, self.num_replicas, self.rank, self.seed, self.epoch, self.drop_last,
        )

        self.example_buffer = []
        self.example_buffer_permutation = None

    def __next__(self):
        block_size = self.examples_per_chunk * self.num_chunks_for_shuffling
        example_index = self.worker_step_in_epoch % block_size

        data_exhausted = False
        if self.resuming_from_checkpoint or example_index == 0:
            data_exhausted = self._replenish_example_buffer()
            self.resuming_from_checkpoint = False

        if not data_exhausted and len(self.example_buffer) == 0:
            logging.warning("Empty buffer after replenishment. Advance to the next chunks. \
                            Dangerous operations if you uses multiple iter-type data, when only one data is moved to next.")
            self.worker_step_in_epoch += block_size
            return next(self)

        if example_index >= len(self.example_buffer):
            logger.debug(f'Data set exhausted at rank {dist.get_rank()}')
            raise StopIteration  # data set exhausted

        shuffled_example_index = self.example_buffer_permutation[example_index]
        assert shuffled_example_index < len(self.example_buffer)

        self.worker_step_in_epoch += 1
        example = self.example_buffer[shuffled_example_index]

        return example

    def __iter__(self):
        return self

    def _replenish_example_buffer(self):
        block_size = self.examples_per_chunk * self.num_chunks_for_shuffling
        assert self.resuming_from_checkpoint or self.worker_step_in_epoch % block_size == 0
        worker_block_index = self.worker_step_in_epoch // block_size

        self.example_buffer = []
        threads = []
        thread_results = [None] * self.num_chunks_for_shuffling
        for i in range(self.num_chunks_for_shuffling):
            worker_chunk_index = worker_block_index * self.num_chunks_for_shuffling + i
            global_chunk_index = self.chunk_index_permutation[worker_chunk_index]

            if i == 0:
                first_global_chunk_index = global_chunk_index  # used to compute seed for example buffer permutation

            if global_chunk_index is None:  # chunks exhausted:
                if i == 0:  # end of data set
                    return True
                else:  # some remaining data to be processed
                    break

            if self.multi_threaded_download:
                t = threading.Thread(target=self._read_chunk, args=(global_chunk_index, i, thread_results))
                t.start()
                threads.append(t)
            else:
                self._read_chunk(global_chunk_index, i, thread_results)

        # wait for threads to complete
        if self.multi_threaded_download:
            for t in threads:
                t.join()

        # collect results
        for result in thread_results:
            if result is not None:
                self.example_buffer += result

        # compute permutation used to shuffle examples in example buffer
        seed = self.seed + self.epoch * self.total_chunks + first_global_chunk_index
        self.example_buffer_permutation = KenslerPermutation(len(self.example_buffer), seed=seed)
        return False    # there are still remaining data

    def _read_chunk(self, chunk_index: int, thread_id: int, thread_results: List):
        if self.dummy_data_mode:
            examples = list(range(chunk_index * self.examples_per_chunk, (chunk_index + 1) * self.examples_per_chunk,))
        else:
            examples = self.dataset_spec.download_chunk(chunk_index)
        thread_results[thread_id] = examples


class ChunkImageTextSeq2SeqDataset(IterableDataset):
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
        drop_last (bool):
            If True, drop some chunks to make the total number of chunks divisible
            by the total number of workers (#gpus * #workers_per_gpu).
            If False, repeat some chunks to make the total number of chunks divisible
            by the total number of workers (#gpus * #workers_per_gpu).
            Default: False
        seed (int):
            Seed for shuffling.
        multi_threaded_download (bool):
            If True, use threads to download chunks, one thread per chunk.
            Default: False
        debug (bool):
            If True, append debug info when decoding examples.
            Default: False
        share_client (bool):
            If True, share azure storage blob client if the blobs are stored in the the same container.
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
                "image_blob_template": "split_test/image-tsv/image-{:010d}.tsv",
                "text_blob_template": "split_test/text-tsv/text-{:010d}.tsv",
                "image_data_type": "tsv",
                "text_data_type": "tsv",
                "with_image_key": True
            },
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
                "image_blob_template": "split_test/image-binary/image-{:010d}",
                "text_blob_template": "split_test/text-tsv/text-{:010d}.tsv",
                "image_data_type": "binary",
                "text_data_type": "tsv",
                "with_image_key": True
            },

            
            # multi-text config example: will read multiple text files for one image.
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
                "image_blob_template": "split_test/image-binary/image-{:010d}",
                "text_blob_template": "split_test/{}-{:010d}.tsv",
                "multi_text": True,
                "multi_text_keys": [
                    "text/en/text",
                    "facedet/facedet",
                    "faceattr/faceattr",
                    "od/dyhead-fl1.6.2-d5-plus/od3m"
                ]
                "image_data_type": "binary",
                "text_data_type": "tsv",
                "with_image_key": True
            }
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
        drop_last: bool = False,
        seed: int = 0,
        multi_threaded_download: bool = False,
        debug: bool = False,
        share_client: bool = False,
        skip_blob_file_path=None,
        skip_missing_data=False,
        tokenize_padding='max_length',
    ):
        self.transforms = transforms
        self.tokenize = tokenize
        if tokenize_padding is None:
            tokenize_padding = 'do_not_pad'
        self.tokenize_padding = tokenize_padding
        self.context_length = context_length
        self.share_client = share_client
        self.skip_missing_data = skip_missing_data 
        if skip_blob_file_path is not None and os.path.exists(skip_blob_file_path):
            skip_blob_list = []
            with open(skip_blob_file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                skip_blob_list.append(line.strip())
            logging.info('all skip blob list {}'.format(skip_blob_list))
        else:
            skip_blob_list = None

        self.debug = debug

        # create image data set
        image_spec = {
            "total_chunks": dataset_spec["total_chunks"],
            "examples_per_chunk": dataset_spec["examples_per_chunk"],
            "data_sources": []
        }
        for data_source in dataset_spec["data_sources"]:
            image_data_source = copy.deepcopy(data_source)
            image_data_source["chunk_blob_name_template"] = image_data_source["image_blob_template"]
            image_data_source.pop("image_blob_template")
            image_data_source["chunk_data_type"] = image_data_source["image_data_type"]
            image_data_source.pop("image_data_type")
            if "multi-lingual-data" in image_data_source:
                image_data_source.pop("multi-lingual-data")
            if 'multi_text' in image_data_source:
                image_data_source.pop('multi_text')
            image_spec["data_sources"].append(image_data_source)


        self.image_dataset = BlobChunkIterableDataset(
            dataset_spec=image_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=drop_last,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            debug=debug,
            share_client=share_client,
            skip_blob_list=skip_blob_list
        )

        # create text data set
        text_spec = {
            "total_chunks": dataset_spec["total_chunks"],
            "examples_per_chunk": dataset_spec["examples_per_chunk"],
            "data_sources": []
        }
        for data_source in dataset_spec["data_sources"]:
            text_data_source = copy.deepcopy(data_source)
            text_data_source["chunk_blob_name_template"] = text_data_source["text_blob_template"]
            text_data_source.pop("text_blob_template")
            text_data_source["chunk_data_type"] = text_data_source["text_data_type"]
            text_data_source.pop("text_data_type")
            text_spec["data_sources"].append(text_data_source)

        self.text_dataset = BlobChunkIterableDataset(
            dataset_spec=text_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            num_replicas=num_replicas,
            rank=rank,
            drop_last=drop_last,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            debug=debug,
            share_client=share_client,
            skip_blob_list=skip_blob_list
        )

        self.num_samples = dataset_spec["total_chunks"] * dataset_spec["examples_per_chunk"]

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
        self.image_dataset.set_epoch(epoch)
        self.text_dataset.set_epoch(epoch)

    def set_state(self, state: Dict[str, Any]):
        self.resume_from_checkpoint(state["epoch"], state["generator_state"])
    
    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        if torch.distributed.is_initialized():
            rank = rank if rank is not None else torch.distributed.get_rank()
            world_size = world_size if world_size is not None else torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        self.image_dataset.rank = rank
        self.text_dataset.rank = rank
        self.image_dataset.num_replicas = world_size
        self.text_dataset.num_replicas = world_size

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
        self.image_dataset.resume_from_checkpoint(epoch, step_in_epoch)
        self.text_dataset.resume_from_checkpoint(epoch, step_in_epoch)

    @staticmethod
    def decode_text_item(items: List[str]) -> dict:
        if isinstance(items[0], list):
            # multi-text
            raise NotImplementedError('multi-text data not supported yet')
        else:
            key = items[0]
            try:
                js = json.loads(items[1])
            except:
                js = ast.literal_eval(items[1])

        return key, js

    @staticmethod
    def decode_image_item(items: Union[bytes, List]):
        if isinstance(items, list) and isinstance(items[1], str):
            return ChunkImageTextSeq2SeqDataset.decode_image_item_base64(items)
        else:
            return ChunkImageTextSeq2SeqDataset.decode_image_item_binary(items)

    @staticmethod
    def decode_image_item_base64(items: List[str]):
        image = Image.open(io.BytesIO(base64.b64decode(items[1]))).convert("RGB")
        return image

    @staticmethod
    def decode_image_item_binary(item: Union[bytes, List]):
        if isinstance(item, list):
            if len(item[1]) == 0:
                return None
            image = Image.open(io.BytesIO(item[1])).convert("RGB")
        else:
            if len(item) == 0:
                return None
            image = Image.open(io.BytesIO(item)).convert("RGB")
        return image

    def __iter__(self):
        raise NotImplementedError("This class is not meant to be used directly. Use one of the subclasses.")


class InfiniteChunkImageTextSeq2SeqDataset(ChunkImageTextSeq2SeqDataset):
    # Ref: https://stackoverflow.com/questions/48376607/python-3-how-to-write-a-iter-method-for-derived-class-so-that-it-extends-on
    # Note: At the end of iter(self.image_dataset) and iter(self.text_dataset), self.step_in_epoch will be 
    #       set to 0. Thus, in the next iteration of the while loop below, we do not need to initialize 
    #       self.step_in_epoch to ensure the next example refers to the first one in the random permutation.
    def __iter__(self):
        while True:
            yield from super().__iter__()
