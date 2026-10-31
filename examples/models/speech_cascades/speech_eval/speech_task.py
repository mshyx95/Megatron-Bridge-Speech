# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seven-suite HTTP evaluation using migrated reference loaders and scorers."""

import json
import logging
import os
import re
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from .aa_dataset import AADataset, AADatasetLong
from .hf_dataset import HFDataset, HFDatasetLong, HFDatasetML
from .jsonl_dataset import SpeechOutJsonlDataset
from .ltu_dataset import SFTDataset
from .pipeline_utils import batch_example_generator
from .prompts import asr_prompt_map, clean_asr_mode_tags
from .score_utils import HFWerScorer, SacrebleuScorer
from .speech_api import SpeechAPIPipeline


logger = logging.getLogger(__name__)
DATA_SET_DICT = {
    "openasr": HFDataset,
    "openasr_clean": partial(HFDataset, min_duration=1.0),
    "openasr_ml": HFDatasetML,
    "openasr_long": partial(HFDatasetLong, max_dur=30, chunk_type="svad"),
    "openasr_aa": AADataset,
    "openasr_aa_long": partial(AADatasetLong, max_dur=30, chunk_type="svad"),
    "fleurs": SFTDataset,
    "voice_bench_s2t": partial(SpeechOutJsonlDataset, prompt="s2t"),
    "bigbench_audio_s2t": partial(SpeechOutJsonlDataset, prompt="s2t"),
}
ASR_PROMPT_TASKS = {name for name in DATA_SET_DICT if name.startswith("openasr")}


def score_records(outputs: list[dict], *, task: str, sub_task: str) -> str | None:
    """Score the complete ordered corpus using the reference metric contract."""
    if sub_task.startswith("hfwer_"):
        scorer = HFWerScorer(sub_task, use_kaldialign="long" not in task)
    elif sub_task.startswith("bleu_"):
        scorer = SacrebleuScorer(sub_task, use_normalizer=True)
    elif sub_task == "s2t":
        return None
    else:
        raise ValueError(f"Unsupported scorer: {sub_task}")
    for output in outputs:
        scorer.add_string(output["label"][0], output["generated_texts"][0])
    return scorer.result_string()


class SpeechTask:
    """Run migrated reference evaluation semantics without loading a local model."""

    @staticmethod
    def run(*, tokenizer, client, task, sub_task, dataset_path, output_file_path, **kwargs):
        """Generate and score all loader samples, combining long-form segments by ID."""
        if task not in DATA_SET_DICT:
            raise ValueError(f"Unsupported task: {task}")
        template = kwargs["template"]
        started = time.monotonic()
        destination = Path(output_file_path)
        sample_limit = int(os.environ.get("SPEECH_LOG_SAMPLES", "2"))
        if not 0 <= sample_limit <= 10:
            raise ValueError("SPEECH_LOG_SAMPLES must be between 0 and 10")
        logger.info(
            "EVAL_LOAD task=%s dataset=%s output=%s audit=%s",
            task,
            dataset_path,
            destination,
            destination.with_suffix(".requests.jsonl"),
        )
        options = {"num_replicas": 1, "rank": 0, "template": template}
        smoke_samples = kwargs.get("smoke_samples", 0)
        if smoke_samples:
            if task != "openasr" or not 1 <= smoke_samples <= 128:
                raise ValueError("Smoke sampling is limited to OpenASR short")
            options["max_samples"] = smoke_samples
            logger.info("EVAL_SMOKE first_records=%d full_benchmark=false", smoke_samples)
        if task in ASR_PROMPT_TASKS:
            options["prompt"] = kwargs["asr_prompt"]
        dataset = DATA_SET_DICT[task](dataset_path, **options)
        if smoke_samples and len(dataset) != smoke_samples:
            raise ValueError("Smoke loader did not retain every requested record")
        sharded = kwargs.get("shard_world_size", 1) > 1
        if sharded:
            from .mpi import StridedDataset

            id_key = (
                "UUID"
                if task in {"voice_bench_s2t", "bigbench_audio_s2t"}
                else "utt_id"
                if task == "fleurs"
                else "audio_id"
            )
            dataset = StridedDataset(
                dataset,
                [row[id_key] for row in dataset.data],
                rank=kwargs["shard_rank"],
                world_size=kwargs["shard_world_size"],
            )
        logger.info(
            "EVAL_BEGIN task=%s shard_samples=%d rank=%d world_size=%d load_seconds=%.2f output=%s",
            task,
            len(dataset),
            kwargs.get("shard_rank", 0),
            kwargs.get("shard_world_size", 1),
            time.monotonic() - started,
            destination,
        )
        pipeline = SpeechAPIPipeline(None, tokenizer, client=client)
        generation = SimpleNamespace(
            stop_tokens=["<|im_end|>"],
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            max_new_tokens=kwargs["max_new_tokens"],
        )
        outputs = []
        batches = requests = 0
        next_progress = time.monotonic()
        lid_mode = kwargs.get("use_lid", "0")
        for inputs in batch_example_generator(
            dataset,
            kwargs["batch_size"],
            asr_prompt_map,
            sub_task=sub_task,
            lid_mode=lid_mode,
            template=template,
            instruction_first=kwargs["instruction_first"],
            bias_after_audio=kwargs.get("bias_after_audio", "false"),
        ):
            batch = pipeline(inputs, generation_config=generation)
            batches += 1
            requests += len(batch["audio_ids"])
            for index, audio_id in enumerate(batch["audio_ids"]):
                output = {key: [value[index]] for key, value in batch.items()}
                answer = output["generated_texts"][0]
                if lid_mode != "0":
                    matches = re.findall(r"<TXT>(.*?)</TXT>", answer, re.DOTALL)
                    answer = " ".join(part.strip() for part in matches) if matches else answer
                    answer = re.sub(r"Audio Language:\s*\w+\s*<ASR[^>]*><lang=\w+><TXT>", "", answer)
                    answer = re.sub(r"</ASR[^>]*>|</AST[^>]*>|</TXT>", " ", answer)
                    answer = re.sub(r"\s+", " ", answer).strip()
                if task in ASR_PROMPT_TASKS:
                    answer = clean_asr_mode_tags(answer)
                if task in {"openasr_long", "openasr_aa_long"}:
                    answer = answer.replace("<nonspeech>", "")
                    answer = re.sub(r"speaker\d+", " ", answer)
                    answer = re.sub(r"\s+", " ", answer).strip()
                    if outputs and outputs[-1]["audio_ids"][0] == audio_id:
                        outputs[-1]["generated_texts"][0] += " " + answer
                        continue
                output["generated_texts"] = [answer]
                outputs.append(output)
                if len(outputs) <= sample_limit:
                    logger.info(
                        "EVAL_SAMPLE task=%s id=%s reference=%s hypothesis=%s preview_chars=400",
                        task,
                        json.dumps(audio_id),
                        json.dumps(str(output.get("label", [""])[0])[:400]),
                        json.dumps(answer[:400]),
                    )
            if time.monotonic() >= next_progress or batches % 100 == 0 or len(outputs) == len(dataset):
                logger.info(
                    "EVAL_PROGRESS task=%s utterances=%d/%d requests=%d batches=%d elapsed_seconds=%.2f",
                    task,
                    len(outputs),
                    len(dataset),
                    requests,
                    batches,
                    time.monotonic() - started,
                )
                next_progress = time.monotonic() + 30
        if sharded:
            from .mpi import write_shard

            contract = {"task": task, "sub_task": sub_task, "dataset_path": dataset_path, **kwargs}
            contract.pop("shard_rank")
            write_shard(Path(output_file_path), outputs, dataset, contract)
            logger.info(
                "EVAL_DONE task=%s samples=%d requests=%d output=%s receipt=%s elapsed_seconds=%.2f",
                task,
                len(outputs),
                requests,
                destination,
                destination.with_suffix(destination.suffix + ".complete.json"),
                time.monotonic() - started,
            )
            return 0
        if not outputs:
            raise ValueError("No samples survived the reference loader")
        score = score_records(outputs, task=task, sub_task=sub_task)
        if score is not None:
            outputs.append({"final_score": score})
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination)
        logger.info(
            "EVAL_DONE task=%s samples=%d requests=%d score=%s output=%s elapsed_seconds=%.2f",
            task,
            len(outputs) - int(score is not None),
            requests,
            score,
            destination,
            time.monotonic() - started,
        )
        return outputs[-1].get("final_score", 0)
