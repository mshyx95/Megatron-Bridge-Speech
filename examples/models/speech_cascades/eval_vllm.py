# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the reference evaluation CLI while delegating inference only to vLLM."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech_eval.tokenizer import load_official_tokenizer


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    rank = os.environ.get("SPEECH_EVAL_RANK", "0")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [eval rank={rank}] %(levelname)s %(message)s")


def _reference(name: str):
    if name == "eval_native":
        from speech_eval import cli

        return cli
    from speech_eval.scripts.audio_eval import bfcl_runtime

    return bfcl_runtime


def _client(output: Path):
    from speech_eval.speech_api import SpeechAPI

    return SpeechAPI(
        json.loads(os.environ["SPEECH_BASE_URLS"])
        if os.environ.get("SPEECH_BASE_URLS")
        else os.environ["SPEECH_BASE_URL"],
        os.environ["SERVED_MODEL_NAME"],
        output.with_suffix(".requests.jsonl"),
        concurrency=int(os.environ["SPEECH_EVAL_CONCURRENCY"]) if os.environ.get("SPEECH_EVAL_CONCURRENCY") else None,
    )


def _completed(output: Path) -> bool:
    if not output.exists():
        return False
    audit_path = output.with_suffix(".requests.jsonl")
    if not audit_path.is_file():
        raise ValueError(f"Existing output has no vLLM request audit: {output}")
    predictions = json.loads(output.read_text())
    audit = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    predicted_ids = [audio_id for row in predictions for audio_id in row.get("audio_ids", [])]
    if not predicted_ids or not audit or any(row["status"] != "ok" for row in audit):
        raise ValueError(f"Existing vLLM results are incomplete: {output}")
    if set(predicted_ids) != {row["audio_id"] for row in audit}:
        raise ValueError(f"Existing predictions and request IDs disagree: {output}")
    return True


def _shard_options() -> dict:
    keys = ("SPEECH_EVAL_RANK", "SPEECH_EVAL_WORLD_SIZE", "SPEECH_EVAL_RUN_KEY")
    if not any(key in os.environ for key in keys):
        return {}
    if not all(os.environ.get(key) for key in keys):
        raise ValueError("Incomplete MPI evaluation shard context")
    rank, world_size = int(os.environ[keys[0]]), int(os.environ[keys[1]])
    if world_size != 8 or not 0 <= rank < world_size:
        raise ValueError("Expected eight disjoint evaluation ranks")
    return {"shard_rank": rank, "shard_world_size": world_size, "run_key": os.environ[keys[2]]}


def speech_main() -> None:
    """Run the original speech CLI with native datasets/scorers and an HTTP pipeline."""
    _configure_logging()
    from speech_eval import speech_task

    reference = _reference("eval_native")
    args, extra = reference.parse_args()
    if extra:
        raise ValueError(f"Unsupported extra arguments: {extra}")
    if args.use_mtp or args.use_mtp_speculative or args.token_healing or args.use_text_model.lower() != "false":
        raise ValueError(
            "This entrypoint requires audio input and server-owned decoding, without client MTP/token healing"
        )
    if args.task_name not in speech_task.DATA_SET_DICT:
        raise ValueError(f"Unsupported speech task: {args.task_name}")
    if args.num_beams != 1 or args.repetition_penalty != 1.0 or args.no_repeat_ngram_size != 0:
        raise ValueError("The vLLM speech client requires the native greedy decoding contract")
    if args.biasing_list_file or args.chunk_size != -1 or args.codec_decoder_path or args.output_wavdir:
        raise ValueError("Biasing, streaming and speech-output options are outside the seven-suite evaluator")
    output = Path(args.output_file_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_options = _shard_options()
    if shard_options and output.exists():
        raise ValueError("MPI evaluation requires a fresh rank output directory")
    if _completed(output):
        logger.info("Skipping completed vLLM dataset: %s", output)
        return
    if output.with_suffix(".requests.jsonl").exists():
        raise ValueError(f"Incomplete prior request audit requires a separate output path: {output}")
    tokenizer = load_official_tokenizer(args.pretrained_tokenizer_name_or_path)
    client = _client(output)
    logger.info("Inference backend=vllm, model=%s, task=%s", client.model, args.task_name)
    speech_task.SpeechTask.run(
        client=client,
        tokenizer=tokenizer,
        task=args.task_name,
        sub_task=args.sub_task,
        dataset_path=args.dataset_path,
        output_file_path=str(output),
        batch_size=max(args.batch_size, client.concurrency),
        template=args.chat_template,
        instruction_first=args.instruction_first,
        bias_after_audio=args.bias_after_audio,
        asr_prompt=args.asr_prompt,
        use_lid=args.use_lid,
        max_new_tokens=args.max_new_tokens,
        smoke_samples=int(os.environ.get("SPEECH_SMOKE_SAMPLES", "0")),
        **shard_options,
    )


def bfcl_main() -> None:
    """Reuse reference BFCL prompts, parsing and outputs with vLLM audio generation."""
    _configure_logging()
    reference = _reference("eval_bfcl_native")

    class Runtime:
        def __init__(self, args):
            if args.model_family != "qwen35-audio" or args.enable_thinking:
                raise ValueError("vLLM BFCL requires qwen35-audio with thinking disabled")
            self.args = args
            self.completed = 0
            self.sample_limit = int(os.environ.get("SPEECH_LOG_SAMPLES", "2"))
            if not 0 <= self.sample_limit <= 10:
                raise ValueError("SPEECH_LOG_SAMPLES must be between 0 and 10")
            self.next_progress = time.monotonic()
            logger.info(
                "BFCL_BEGIN model=%s data=%s output=%s audit=%s",
                args.model,
                args.data_dir,
                args.output,
                args.output.with_suffix(".requests.jsonl"),
            )
            self.tokenizer = load_official_tokenizer(args.tokenizer)
            self.client = _client(args.output)
            self.generation = SimpleNamespace(
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=args.max_new_tokens,
                stop_tokens=["<|im_end|>"],
                bad_words=["<think>"],
            )

        def generate(self, rows):
            def generate_row(row):
                audio, rate = reference.read_audio(row)
                tools, names = reference.make_tools(row["function"])
                prompt = reference.make_qwen_prompt(self.tokenizer, row, tools, self.args)
                started = time.monotonic()
                response = self.client.generate(
                    prompt,
                    audio,
                    rate,
                    row["id"],
                    self.generation,
                    self.tokenizer.eos_token,
                )
                return reference.GenerationResult(
                    response,
                    reference.qwen_response_to_bfcl(response, tools, names),
                    time.monotonic() - started,
                    {"backend": "vllm"},
                )

            with ThreadPoolExecutor(max_workers=self.client.concurrency) as workers:
                results = list(workers.map(generate_row, rows))
            for row, result in zip(rows, results):
                if self.completed < self.sample_limit:
                    logger.info(
                        "BFCL_SAMPLE id=%s hypothesis=%s",
                        row["id"],
                        json.dumps(result.raw_response[:400], ensure_ascii=True),
                    )
                self.completed += 1
            if time.monotonic() >= self.next_progress or self.completed % 100 == 0:
                logger.info("BFCL_PROGRESS completed=%d output=%s", self.completed, self.args.output)
                self.next_progress = time.monotonic() + 30
            return results

    shard_options = _shard_options()
    if not shard_options:
        with patch.object(reference, "load_runtime", Runtime):
            reference.main()
        return
    from speech_eval.mpi import StridedDataset, write_shard

    args = reference.parse_args()
    if args.output.exists() or args.output.with_suffix(".requests.jsonl").exists():
        raise ValueError("MPI BFCL requires fresh predictions and audit files")
    rows = reference.load_rows(args)
    dataset = [{"audio_id": row["id"], "row": row} for row in rows]
    shard = StridedDataset(
        dataset,
        [row["id"] for row in rows],
        rank=shard_options["shard_rank"],
        world_size=shard_options["shard_world_size"],
    )
    selected = [item["row"] for item in shard]
    with (
        patch.object(reference, "parse_args", return_value=args),
        patch.object(reference, "load_rows", return_value=selected),
        patch.object(reference, "load_runtime", Runtime),
    ):
        reference.main()
    records = json.loads(args.output.read_text()) if args.output.exists() else []
    write_shard(
        args.output,
        records,
        shard,
        {
            "task": "bfcl",
            "sub_task": "bfcl",
            "run_key": shard_options["run_key"],
            "categories": args.categories,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    logger.info(
        "BFCL_DONE rank=%d samples=%d output=%s receipt=%s",
        shard_options["shard_rank"],
        len(records),
        args.output,
        args.output.with_suffix(args.output.suffix + ".complete.json"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    if Path(__file__).name == "eval_bfcl.py":
        bfcl_main()
    else:
        speech_main()
