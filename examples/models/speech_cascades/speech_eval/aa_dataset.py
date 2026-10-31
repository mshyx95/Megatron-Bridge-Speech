# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pandas as pd
import soundfile as sf

from speech_eval import normalizer as data_utils
from speech_eval.audio_utils import apply_chat_template
from speech_eval.chunk.chunker import Chunker
from speech_eval.dataset_utils import DistributedDataset
from speech_eval.jsonl_dataset import get_asr_prompt


FRAME_LEN = 25e-3  # duration for 1 frame


class AADataset(DistributedDataset):
    """Dataset for ArtificialAnalysis short-form datasets (e.g., VoxPopuli-Cleaned-AA)."""

    def __init__(self, dataset_path, num_replicas=None, rank=None, template="phi_3", prompt=None):
        self.data = []
        self._template = template

        data_dir, dataset_name = dataset_path.split(" ")
        dataset_root = os.path.join(data_dir, dataset_name)

        # Find the JSONL file in the dataset directory
        jsonl_files = [f for f in os.listdir(dataset_root) if f.endswith(".jsonl") and "results" not in f]
        assert len(jsonl_files) == 1, f"Expected exactly one JSONL file in {dataset_root}, found: {jsonl_files}"
        jsonl_path = os.path.join(dataset_root, jsonl_files[0])

        if prompt:
            self.prompt = get_asr_prompt(prompt)
        else:
            self.prompt = "Transcribe the audio clip into text."

        df = pd.read_json(jsonl_path, orient="records", lines=True)

        for _, row in df.iterrows():
            audio_path = os.path.join(dataset_root, row["url"])
            if not os.path.exists(audio_path):
                continue

            array, fs = sf.read(audio_path)
            original_text = row["transcript"]
            norm_text = data_utils.normalizer(original_text)

            if not data_utils.is_target_text_in_range(norm_text):
                continue

            self.data.append(
                {
                    "audio_id": row["id"],
                    "audio": array,
                    "fs": fs,
                    "label": norm_text,
                }
            )

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        example.update(
            {
                "source": apply_chat_template(self.prompt, self._template),
            }
        )
        return example


class AADatasetLong(DistributedDataset):
    """Dataset for ArtificialAnalysis long-form datasets (e.g., Earnings22-Cleaned-AA)."""

    def __init__(
        self,
        dataset_path,
        max_dur=30,
        chunk_type="svad",
        num_replicas=None,
        rank=None,
        template="phi_3",
        vad_file=None,
        prompt=None,
    ):
        self.data = []
        self._template = template

        data_dir, dataset_name = dataset_path.split(" ")
        dataset_root = os.path.join(data_dir, dataset_name)

        # Find the JSONL file in the dataset directory
        jsonl_files = [f for f in os.listdir(dataset_root) if f.endswith(".jsonl") and "results" not in f]
        assert len(jsonl_files) == 1, f"Expected exactly one JSONL file in {dataset_root}, found: {jsonl_files}"
        jsonl_path = os.path.join(dataset_root, jsonl_files[0])

        if prompt:
            self.prompt = get_asr_prompt(prompt)
        else:
            self.prompt = "Transcribe the audio clip into text."

        self.chunker = Chunker(max_len=max_dur, type=chunk_type, vad_file=vad_file)

        df = pd.read_json(jsonl_path, orient="records", lines=True)

        for _, row in df.iterrows():
            audio_path = os.path.join(dataset_root, row["url"])
            if not os.path.exists(audio_path):
                continue

            original_text = row["transcript"]
            norm_text = data_utils.normalizer(original_text)

            if not data_utils.is_target_text_in_range(norm_text):
                continue

            self.data.append(
                {
                    "audio_id": row["id"],
                    "audio": audio_path,
                    "label": norm_text,
                    "locale": row.get("language", "en"),
                }
            )

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["audio"]
        data, fs = sf.read(audio_path)
        examples = []
        for st, et in self.chunker(audio_path):
            if (et - st) < FRAME_LEN:  # skip segments smaller than 1 frame
                continue

            seg = data[int(st * fs) : int(et * fs)]
            examples.append(
                {
                    "source": apply_chat_template(self.prompt, self._template),
                    "audio": seg,
                    "fs": fs,
                    "label": example["label"],
                    "locale": example["locale"],
                    "audio_id": example["audio_id"],
                    "start_time": st,
                    "end_time": et,
                }
            )
        return examples
