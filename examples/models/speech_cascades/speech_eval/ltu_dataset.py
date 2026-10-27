# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import json
import os

import blobfile as bf
import pandas as pd
import soundfile

from speech_eval.audio_utils import apply_chat_template
from speech_eval.dataset_utils import DistributedDataset


class SFTDataset(DistributedDataset):
    # The dataset is used to load our defined tsv format for stage-2 SFT training
    # Evaluation sets include: LTU, covost2, fleurs,....
    def __init__(self, dataset_path, num_replicas=None, rank=None, template="phi_3"):
        self.data = []
        col_names = ["id", "wav_paths", "messages"]
        self._template = template

        if dataset_path.startswith("az://"):
            data_list = pd.read_csv(bf.BlobFile(dataset_path, "r"), names=col_names, sep="\t")
        else:
            data_list = pd.read_csv(dataset_path, names=col_names, sep="\t")

        for i in range(len(data_list)):
            wav_paths = ast.literal_eval(data_list["wav_paths"][i])
            if dataset_path.startswith("az://"):
                for j in range(len(wav_paths)):
                    orange_path_prefix = "/".join(dataset_path.split("az://")[1].split("/")[:3])
                    wav_paths[j] = (
                        "az://" + orange_path_prefix + wav_paths[j].replace("/datablob1", "").replace("//", "/")
                    )
            try:
                messages = ast.literal_eval(data_list["messages"][i])
            except:
                continue

            id = data_list["id"][i]

            for j in range(len(messages)):
                message = messages[j]["messages"]

                user_prompt = message[0]["content"].replace("<audio>\n", "")
                label = message[1]["content"]

                self.data.append(
                    {
                        "utt_id": f"{id}_{j}",
                        "audio_path": wav_paths[0],
                        "user_prompt": user_prompt,
                        "label": label,
                    }
                )

        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["audio_path"]
        if audio_path.startswith("az://"):
            with bf.BlobFile(audio_path, "rb") as f:
                audio, fs = soundfile.read(f)
        else:
            audio, fs = soundfile.read(audio_path)

        example = {
            "source": apply_chat_template(example["user_prompt"], self._template),
            "audio": audio,
            "fs": fs,
            "label": example["label"],
            "audio_id": example["utt_id"],
            "audio_prompt_path": example["audio_path"],
        }
        return example


class LTUChoiceDataset(DistributedDataset):
    def __init__(self, dataset_path, num_replicas=None, rank=None, template="phi_3"):
        self.data = []
        self._template = template
        reader = json.load(open(dataset_path, "r"))
        for item in reader:
            wav_fn = item["audio_path"]
            question = item["question"]
            choice_a = item["choice_a"]
            choice_b = item["choice_b"]
            choice_c = item.get("choice_c", None)
            choice_d = item.get("choice_d", None)
            choices = f"A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}"
            instruction = question + "\n" + choices
            if os.path.exists(wav_fn) == False:
                print(f"lack wav {wav_fn}")
                continue
            self.data.append(
                {
                    "utt_id": item["utt_ids"],
                    "audio_path": wav_fn,
                    "user_prompt": instruction,
                    "label": item["label"],
                    "question": question,
                    "choice_a": choice_a,
                    "choice_b": choice_b,
                    "choice_c": choice_c,
                    "choice_d": choice_d,
                    "dataset_name": "LTU",
                    "task_name": item["task_name"],
                }
            )
        super().__init__(num_replicas, rank)

    def __getitem__(self, idx):
        example = self.data[idx]
        audio_path = example["audio_path"]
        audio, fs = soundfile.read(audio_path)
        user_prompt = example["user_prompt"]

        example = {
            "source": apply_chat_template(user_prompt, self._template),
            "audio": audio,
            "fs": fs,
            "label": example["label"],
            "audio_id": example["utt_id"],
            "task_name": example["task_name"],
            "question": example["question"],
            "dataset_name": example["dataset_name"],
            "answer_gt": example["label"],
            "path": example["audio_path"],
            "choice_a": example["choice_a"],
            "choice_b": example["choice_b"],
            "choice_c": example["choice_c"],
            "choice_d": example["choice_d"],
            "user_prompt": user_prompt,
        }
        return example
