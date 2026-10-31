# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import io
import re
import csv
import copy
import json
import logging
import math
import numpy as np
import random
import threading
import torch
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional, Union
import soundfile
from torch.utils.data import IterableDataset


logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# from megatron.bridge.data.speech_chunk_streaming._utils.logging_utils import get_logger
# logger = get_logger(__name__)

import warnings

from numpy import uint64, random


class WavDataset():
    
    def __init__(
        self,dataset_spec: str, max_dur: float = 30.0, num_replicas: int = None, rank: int = None
    ):
        self.header_skipped = False
        self.epoch = 0 #dummy epoch count

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

        with open(dataset_spec, "r", encoding="utf-8") as f:
            self.audio_files = []
            for line in f:
                utt_id, wav_path = line.strip().split(None, 1)
                self.audio_files.append((utt_id, wav_path))
        
        self.audio_files = self.audio_files[self.rank::self.num_replicas]
        self.max_dur = max_dur
        self.num_samples = 0
        self.dur_per_frame = 0.025
        for _, wav in self.audio_files:
            info = soundfile.info(wav)
            discard_last = (info.duration % self.max_dur) < self.dur_per_frame  # less than one frame, we skip it 
            if discard_last: 
                self.num_samples += math.floor(info.frames / (info.samplerate * self.max_dur))
            else:
                self.num_samples += math.ceil(info.frames / (info.samplerate * self.max_dur))

    def __iter__(self):
        self._audio_iter = iter(self.audio_files)
        self._output_iter = iter(self._utterance_segment_iterator())
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        return next(self._output_iter)

    def _utterance_segment_iterator(self):
        while True:
            try:
                data = next(self._audio_iter)
            except StopIteration:
                return
            utt_id = data[0]
            wav_path = data[1]
            match = re.search(r'permanentmedia-([a-z]{2})-([a-z]{2})', wav_path)
            lang_code = match.group(1) if match else None

            waveform, fs = soundfile.read(wav_path)
            if len(waveform) / fs <= self.max_dur:
                example = {}
                example["utt_id"] = utt_id
                example["transcription"] = "dummy"
                example["audio"] = waveform
                example["fs"] = fs
                example["segment"] = (0, len(waveform) / fs)
                example["language"] = lang_code

                yield example
            else:
                segments = [(0, self.max_dur)]
                while segments[-1][1] < (len(waveform) / fs):
                    new_start = segments[-1][0] + self.max_dur
                    new_end = min(segments[-1][1] + self.max_dur, (len(waveform) / fs))
                    segments.append((new_start, new_end))

                for start, end in segments:
                    _start = int(start * fs)
                    _end = int(end * fs)

                    if (end - start) < self.dur_per_frame:
                        break 
                        
                    segment_wav = waveform[_start:_end]
                    example = {
                        "utt_id": utt_id,
                        "audio": segment_wav,
                        "transcription": "dummy",
                        "fs": fs,
                        "segment": (start, end),
                        "language": lang_code,
                    }

                    yield example

    def set_epoch(self, epoch):
        self.epoch = epoch


class SpeechTextSeq2SeqDataset(IterableDataset):
    """
    An IterableDataset for efficient large-scale data loading of image-text pairs for seq-to-seq modeling
    supporting multiple text for each image (e.g. caption, OD, face).

    Args:
        dataset_spec (dict):
            Dataset specification dictionary.
            See example below.
        transforms (Callable):
            An optional function used transform the images.
            Default: None
        tokenize (Callable):
            An optional function used transform the text.
            Default: None
        num_replicas (int):
            Total number of GPUs / data set instances in distributed training.
            If None, use torch.distributed to determine num_replicas.
            Default: None
        rank (int):
            Rank of this GPU / data set instance in distributed training.
            If None, use torch.distributed to determine rank.
            Default: None
    """

    def __init__(
        self,
        dataset_spec: str,
        transforms: Callable = None,
        tokenize: Callable = None,
        max_dur: float = 30.0,
        num_replicas: int = None,
        rank: int = None,
    ):
        
        self.audio_dataset = WavDataset(dataset_spec, max_dur=max_dur, num_replicas=num_replicas, rank=rank)
        self.transforms = transforms
        self.tokenize = tokenize

        self.num_samples = len(self.audio_dataset)

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
        self.audio_dataset.set_epoch(epoch)

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        if torch.distributed.is_initialized():
            rank = rank if rank is not None else torch.distributed.get_rank()
            world_size = world_size if world_size is not None else torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        self.audio_dataset.rank = rank
        self.audio_dataset.num_replicas = world_size

    def __iter__(self):
        raise NotImplementedError("This class is not meant to be used directly. Use one of the subclasses.")
