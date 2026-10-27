# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import tempfile

import soundfile
from datasets import Audio, load_dataset

from speech_eval import normalizer as data_utils
from speech_eval.audio_utils import apply_chat_template
from speech_eval.chunk.chunker import Chunker
from speech_eval.dataset_utils import DistributedDataset
from speech_eval.jsonl_dataset import get_asr_prompt


FRAME_LEN = 25e-3  # duration for 1 frame


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of {sample.keys()}. Ensure a text column name is present in the dataset."
        )


class HFDataset(DistributedDataset):
    def __init__(
        self,
        dataset_path,
        num_replicas=None,
        rank=None,
        template="phi_3",
        prompt=None,
        min_duration=None,
        max_samples=0,
    ):
        self.data = []
        self._template = template

        data_dir, dataset_name = dataset_path.split(" ")
        if max_samples and (dataset_name != "librispeech-clean" or not 1 <= max_samples <= 128):
            raise ValueError("Smoke sampling supports only 1..128 LibriSpeech-clean records")

        if dataset_name == "librispeech-clean":
            dataset = load_dataset(data_dir, "librispeech", split="test.clean")
        elif dataset_name == "librispeech-other":
            dataset = load_dataset(data_dir, "librispeech", split="test.other")
        else:
            dataset = load_dataset(data_dir, dataset_name, split="test")

        if prompt is not None:
            self.prompt = get_asr_prompt(prompt)
        elif dataset_name in ["librispeech-clean", "librispeech-other", "gigaspeech"]:
            self.prompt = "Transcribe the audio clip into text. Output must be in lexical format."
        else:
            self.prompt = "Transcribe the audio clip into text."

        try:
            sample = dataset[0]
        except:
            dataset = dataset.cast_column("audio", Audio(decode=False))

        n_filtered = 0
        for i in range(min(len(dataset), max_samples) if max_samples else len(dataset)):
            sample = dataset[i]
            if not isinstance(sample["audio"], dict):
                audio_buffer = io.BytesIO(sample["audio"]._hf_encoded["bytes"])
                array, fs = soundfile.read(audio_buffer)
            elif "array" in sample["audio"]:
                array, fs = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            else:
                audio_buffer = io.BytesIO(sample["audio"]["bytes"])
                array, fs = soundfile.read(audio_buffer)

            # Filter short audio (e.g., < 1 second) to match official leaderboard cleaned splits
            if min_duration is not None and len(array) / fs < min_duration:
                n_filtered += 1
                continue

            original_text = get_text(sample)
            norm_text = data_utils.normalizer(original_text)

            if not data_utils.is_target_text_in_range(norm_text):
                continue

            self.data.append(
                {
                    "audio_id": sample["id"],
                    "audio": array,
                    "fs": fs,
                    "label": norm_text,
                }
            )

        if min_duration is not None and n_filtered > 0:
            print(f"Filtered {n_filtered} samples shorter than {min_duration}s")

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = self.prompt
        example.update(
            {
                "source": apply_chat_template(prompt, self._template),
            }
        )
        return example


class HFDatasetML(DistributedDataset):
    def __init__(self, dataset_path, num_replicas=None, rank=None, template="phi_3", prompt=None):
        self.data = []
        self._template = template

        data_dir, dataset_name = dataset_path.split(" ")
        dataset = load_dataset(data_dir, dataset_name, split="test")

        if prompt is not None:
            self.prompt = get_asr_prompt(prompt)
        elif dataset_name.startswith("mls_it"):
            self.prompt = "Transcribe the audio clip into text. Output must be in lexical format."
        else:
            self.prompt = "Transcribe the audio clip into text."

        self.language = dataset_name.split("_")[1] if "_" in dataset_name else "en"
        print(f"Loading dataset {dataset_name} with language {self.language}")

        try:
            sample = dataset[0]
        except:
            dataset = dataset.cast_column("audio", Audio(decode=False))

        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample["audio"], dict):
                audio_buffer = io.BytesIO(sample["audio"]._hf_encoded["bytes"])
                array, fs = soundfile.read(audio_buffer)
            elif "array" in sample["audio"]:
                array, fs = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            else:
                audio_buffer = io.BytesIO(sample["audio"]["bytes"])
                array, fs = soundfile.read(audio_buffer)

            original_text = get_text(sample)
            norm_text = data_utils.ml_normalizer(original_text, lang=self.language)

            if not data_utils.is_target_text_in_range(norm_text):
                continue

            self.data.append(
                {
                    "audio_id": sample["file_name"],
                    "audio": array,
                    "fs": fs,
                    "label": norm_text,
                }
            )

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = self.prompt
        example.update(
            {
                "source": apply_chat_template(prompt, self._template),
            }
        )
        return example


class HFDatasetLong(DistributedDataset):
    def __init__(
        self,
        dataset_path,
        max_dur=None,
        chunk_type="savd",
        num_replicas=None,
        rank=None,
        template="phi_3",
        vad_file=None,
        prompt=None,
    ):
        self.data = []
        self._template = template

        data_dir, dataset_name = dataset_path.split(" ")
        if dataset_name == "coraal":
            data_dir = os.path.join(data_dir, "coraal")
            from datasets import concatenate_datasets

            subfolders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            datasets_list = []
            for subfolder in subfolders:
                subfolder_path = os.path.join(data_dir, subfolder)
                print(f"Loading subfolder {subfolder_path}...")
                try:
                    ds = load_dataset(subfolder_path, split="test")
                    datasets_list.append(ds)
                except Exception as e:
                    print(f"Failed to load subfolder {subfolder}: {e}")
            dataset = concatenate_datasets(datasets_list)
        else:
            dataset = load_dataset(data_dir, dataset_name, split="test")
        # Disable audio decoding to avoid torchcodec dependency;
        # we only need the file path and sampling rate metadata.
        dataset = dataset.cast_column("audio", Audio(decode=False))

        if prompt is not None:
            self.prompt = get_asr_prompt(prompt)
        else:
            self.prompt = "Transcribe the audio clip into text."

        self.chunker = Chunker(max_len=max_dur, type=chunk_type, vad_file=vad_file)

        for i in range(len(dataset)):
            sample = dataset[i]

            original_text = get_text(sample)
            norm_text = data_utils.normalizer(original_text)

            if not data_utils.is_target_text_in_range(norm_text):
                continue

            audio_id = f"{dataset_name}_{i}"
            audio_info = sample["audio"]
            self.data.append(
                {
                    "audio_id": audio_id,
                    "audio": audio_info["path"],
                    "audio_bytes": audio_info["bytes"],
                    "label": norm_text,
                    "locale": "en",
                }
            )

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["audio"]
        cleanup_tmp = False
        if audio_path is None or (not os.path.exists(audio_path)):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(example["audio_bytes"])
            tmp.close()
            audio_path = tmp.name
            cleanup_tmp = True
        data, fs = soundfile.read(audio_path)
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
        if cleanup_tmp:
            os.unlink(audio_path)
        return examples
