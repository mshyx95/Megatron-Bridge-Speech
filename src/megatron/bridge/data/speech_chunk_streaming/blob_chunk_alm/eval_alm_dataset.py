# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import math
import io
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from multiprocessing import Manager
import torch

from megatron.bridge.data.speech_chunk_streaming.blob_chunk_alm.dummy_blobchunk import SpeechTextSeq2SeqDataset


class EvalALMDataset(SpeechTextSeq2SeqDataset):
    def __init__(
        self,
        dataset_spec: str,
        num_replicas: int = None,
        rank: int = None,
        max_chunk_size: int = 3000,
        template: str = "phi_3",
    ):

        self._max_chunk_size = max_chunk_size
        self._template = template

        super().__init__(
            dataset_spec,
            transforms=None,
            tokenize=None,
            max_dur=self._max_chunk_size / 100,
            num_replicas=num_replicas,
            rank=rank,
        )

        # Create a shared dict so that we can get information from another
        # process (created by the dataloader) and save the state of the dataset
        self.dataset_spec_str = str(dataset_spec)
        self.shared_state = Manager().dict()
        self.shared_state["epoch"] = 0
        self.shared_state["generator_state"] = 0

    @property
    def dataset_path(self) -> str:
        return self.dataset_spec_str

    def get_state(self) -> Dict[str, Any]:
        return dict(self.shared_state)

    def __iter__(self):
        self.worker_id = 0
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self.worker_id = worker_info.id

        self.audio_iter = iter(self.audio_dataset)
        return self

    def __next__(self):
        sample = next(self.audio_iter)

        if self._template in ["phi_o", "phi_3"]:
            source = "<|user|>\n<audio>\nTranscribe the audio clip into text.<|end|>\n<|assistant|>\n"
        elif self._template == "phi_3_200k":
            source = "user:<audio>\nTranscribe the audio clip into text.<|endofprompt|>assistant:"
        elif self._template == "phi_o_long":
            source = "<|user|><audio>Transcribe the audio clip into text.<|end|><|assistant|>"
        else:
            source = ["user: <audio>\nTranscribe the audio clip into text. ", "assistant: "]

        return {
            "source": source,
            "audio": sample["audio"],
            "fs": sample["fs"],
            "segment": sample["segment"],
            "audio_id": sample["utt_id"],
            "language": sample["language"],
        }
