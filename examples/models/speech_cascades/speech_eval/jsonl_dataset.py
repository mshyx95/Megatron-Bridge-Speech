# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import pandas as pd
import soundfile as sf

from speech_eval.audio_utils import apply_chat_template, is_valid_audio
from speech_eval.chunk.chunker import Chunker
from speech_eval.dataset_utils import DistributedDataset


def read_jsonls(file_paths, fields=None):
    """Read multiple jsonl files"""
    dfs = []
    for file_path in file_paths:
        df = pd.read_json(file_path, orient="records", lines=True)
        if fields is not None:
            df = df[fields]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# fileds["CorpusName", "UUID", "WavPath", "Transcription", "Prompt"]

PROMPT_DICT = {
    "asr": "Transcribe the audio clip into text.",
    "asr_dia": "Transcribe the audio clip into text and add speaker labels.",
    "asr-lfg3": "Transcribe the speech exactly. Output only the transcript, with no commentary, labels, quotation marks, or special tags.",
    "asr-lexical": "Transcribe the audio clip into text. Output must be in lexical format.",
    "asr-verbatim": "Transcribe the audio clip into text. Transcribe verbatim, including all filler words and disfluencies.",
    "s2s": "",
    "s2st": "Translate the audio to English.",
    "s2t": "",
    "t2s": "Conduct a spoken conversation with the text input from user.",
}

PROMPT_ALIASES = {
    "default": "asr",
    "lfg3": "asr-lfg3",
    "lexical": "asr-lexical",
    "verbatim": "asr-verbatim",
}


def get_asr_prompt(prompt):
    prompt = PROMPT_ALIASES.get(prompt, prompt)
    if prompt not in PROMPT_DICT:
        raise ValueError(f"Unknown ASR prompt: {prompt}. Choices: {sorted(PROMPT_DICT)}")
    return PROMPT_DICT[prompt]


SAMPLE_PER_FRAME = 25  # 25 samples per frame
FRAME_LEN = 25e-3  # duration for 1 frame


class JsonlDataset(DistributedDataset):
    """dataset from jsonl files"""

    def __init__(
        self,
        data_path,
        template="phi_3",
        max_dur=None,
        prompt=None,
        chunk_type="svad",
        num_replicas=None,
        rank=None,
        vad_file=None,
        biasing_list=None,
    ):
        """init the dataset"""
        data_path = Path(data_path)
        self._template = template
        self._biasing_list = biasing_list
        jsonl_paths = list(data_path.rglob("*.jsonl")) if data_path.is_dir() else [data_path]
        df = read_jsonls(jsonl_paths)
        if prompt:
            df["Prompt"] = get_asr_prompt(prompt)
        # add biasing list to prompt if provided
        if biasing_list is not None and len(biasing_list) > 0:
            biasing_str = "*, *".join(biasing_list)
            biasing_str = f"*{biasing_str}*" if biasing_str else ""
            df["Prompt"] = df["Prompt"] + f" Pay extra attention to the following phrases/words: {biasing_str}."
        data = df.to_dict(orient="records")
        # data = [egs for egs in data if is_valid_audio(egs["WavPath"])]
        effective_chunk_type = "none" if max_dur is None else chunk_type
        self.chunker = Chunker(max_len=max_dur, type=effective_chunk_type, vad_file=vad_file)
        self.data = data
        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["WavPath"]
        data, fs = sf.read(audio_path)
        examples = []
        for st, et in self.chunker(audio_path):
            if (et - st) < FRAME_LEN:  # skip segments smaller than 1 frame
                continue

            seg = data[int(st * fs) : int(et * fs)]
            examples.append(
                {
                    "source": apply_chat_template(example["Prompt"], self._template),
                    "audio": seg,
                    "fs": fs,
                    "label": example["DisplayTranscription"]
                    if "DisplayTranscription" in example
                    else example["Transcription"],
                    "locale": example["locale"],
                    "audio_id": example["UUID"],
                    "start_time": st,
                    "end_time": et,
                }
            )
        return examples


class SpeechOutJsonlDataset(DistributedDataset):
    """dataset from jsonl files"""

    def __init__(
        self, data_path, template="phi_3", max_dur=None, prompt=None, chunk_type="svad", num_replicas=None, rank=None
    ):
        """init the dataset"""
        data_path = Path(data_path)
        self._template = template
        jsonl_paths = list(data_path.rglob("*.jsonl")) if data_path.is_dir() else [data_path]
        df = read_jsonls(jsonl_paths)
        if prompt and prompt in PROMPT_DICT:
            df["Prompt"] = PROMPT_DICT[prompt]
        data = df.to_dict(orient="records")
        original_len = len(data)
        data = [egs for egs in data if is_valid_audio(egs["WavPath"], min_seconds=0.2)]
        skipped = original_len - len(data)
        if skipped > 0:
            print(f"Skipped {skipped} invalid or <0.2s audio examples from {data_path}")
        # self.chunker = Chunker(max_len=max_dur, type=chunk_type)
        self.data = data

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["WavPath"]
        audio, fs = sf.read(audio_path)
        example = {
            "source": apply_chat_template(example["Prompt"], self._template),
            "audio": audio,
            "fs": fs,
            "label": example["Transcription"],
            "audio_id": example["UUID"],
            "audio_prompt_path": audio_path,
        }
        return example


class TextInSpeechOutJsonlDataset(DistributedDataset):
    """dataset from jsonl files"""

    def __init__(
        self, data_path, template="phi_3", max_dur=None, prompt="t2s", chunk_type="svad", num_replicas=None, rank=None
    ):
        """init the dataset"""
        data_path = Path(data_path)
        self._template = template
        jsonl_paths = list(data_path.rglob("*.jsonl")) if data_path.is_dir() else [data_path]
        df = read_jsonls(jsonl_paths)
        if prompt and prompt in PROMPT_DICT:
            df["Prompt"] = PROMPT_DICT[prompt]
        df["Prompt"] = df.apply(lambda x: x["Prompt"] + " " + x["Question"], axis=1)
        data = df.to_dict(orient="records")
        data = [egs for egs in data if is_valid_audio(egs["WavPath"])]
        # self.chunker = Chunker(max_len=max_dur, type=chunk_type)
        self.data = data

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["WavPath"]
        example = {
            "source": apply_chat_template(example["Prompt"], self._template, audio_tag=""),
            "label": example["Transcription"],
            "audio_id": example["UUID"],
            "audio_prompt_path": audio_path,
        }
        return example


class MultiTurnJsonlDataset(DistributedDataset):
    """dataset from jsonl files"""

    def __init__(
        self, data_path, template="phi_3", max_dur=None, prompt="s2s", chunk_type="svad", num_replicas=None, rank=None
    ):
        """init the dataset"""
        data_path = Path(data_path)
        self._template = template
        jsonl_paths = list(data_path.rglob("*.jsonl")) if data_path.is_dir() else [data_path]
        df = read_jsonls(jsonl_paths)
        if prompt and prompt in PROMPT_DICT:
            df["Prompt"] = PROMPT_DICT[prompt]
        data = df.to_dict(orient="records")
        data = [egs for egs in data if is_valid_audio(egs["WavPath"])]
        data = self.split_conversation(data)
        # self.chunker = Chunker(max_len=max_dur, type=chunk_type)
        self.data = data

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        conversation = self.data[idx]
        conv_list = []
        for conv_id, conv_examples in conversation.items():
            for example in conv_examples:
                audio_path = example["WavPath"]
                new_example = {
                    "source": apply_chat_template(example["Prompt"], self._template, audio_tag=""),
                    "question": example["Question"],
                    "audio_id": example["UUID"],
                    "audio_prompt_path": audio_path,
                }
                conv_list.append(new_example)
        return conv_list

    def split_conversation(self, data):
        # Split the conversation into turns
        conversations = []
        round_conv = None
        for item in data:
            conv_id = "-".join(item["UUID"].split("-")[:-1])
            if round_conv is None:
                round_conv = {conv_id: []}
            if conv_id not in round_conv:
                conversations.append(round_conv)
                round_conv = {conv_id: []}
            round_conv[conv_id].append(item)
        return conversations
