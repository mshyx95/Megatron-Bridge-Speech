# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score Hindi multiple-choice datasets with a fixed, auditable generation protocol."""

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


LOGGER = logging.getLogger(__name__)
DATASETS = ("global_mmlu_lite", "mmlu_prox", "indic_mmlu_pro")


def prompt(row: dict) -> str:
    """Render the question without exposing the gold answer or reasoning trace."""
    choices = row["choices"]
    if not 2 <= len(choices) <= 10:
        raise ValueError(f"Unsupported choice count in {row['id']}")
    labels = ", ".join(chr(65 + index) for index in range(len(choices)))
    return (
        f"निम्नलिखित बहुविकल्पीय प्रश्न का सही विकल्प चुनें। केवल एक अंग्रेज़ी अक्षर ({labels}) लिखें, और कुछ नहीं।\n"
        f"प्रश्न: {row['question']}\n"
        + "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
        + "\nउत्तर:"
    )


def extract_answer(text: str, size: int) -> int | None:
    """Accept one choice letter only, so ambiguous free-form answers remain incorrect."""
    match = re.fullmatch(r"\s*([A-J])\s*[.。)]?\s*", text, re.IGNORECASE)
    if not match:
        return None
    index = ord(match.group(1).upper()) - 65
    return index if index < size else None


def _load_existing(output: Path, rows: list[dict], model: str, *, resume: bool) -> list[dict]:
    if not output.exists():
        return []
    if not resume:
        raise FileExistsError(f"Use --resume or a new output path: {output}")
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if len(results) > len(rows):
        raise ValueError(f"More predictions than source rows in {output}")
    for index, result in enumerate(results):
        if (
            result["id"] != rows[index]["id"]
            or result["gold"] != rows[index]["answer_index"]
            or result["model"] != model
            or result["prompt_sha256"] != hashlib.sha256(prompt(rows[index]).encode()).hexdigest()
        ):
            raise ValueError(f"Prediction/source mismatch at offset {index}: {output}")
    return results


def score(rows: list[dict]) -> dict:
    """Return exact-letter accuracy with conflicting gold labels excluded."""
    scored = [row for row in rows if row["gold"] is not None]
    correct = sum(row["prediction"] == row["gold"] for row in scored)
    return {
        "completed": len(rows),
        "scored": len(scored),
        "label_conflicts": len(rows) - len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "unparsed": sum(row["prediction"] is None for row in scored),
    }


def run(
    source: Path,
    output: Path,
    endpoint: str,
    model: str,
    *,
    workers: int,
    limit: int,
    resume: bool,
    mirror: Path | None = None,
) -> dict:
    """Evaluate every input in source order and durably checkpoint each batch."""
    if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")) or workers not in (1, 2, 3, 4) or limit < 0:
        raise ValueError("Expected loopback endpoint, 1-4 workers and nonnegative limit")
    if not model or not source.is_file():
        raise ValueError("A served model and existing dataset file are required")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique, nonempty source rows")
    completed = _load_existing(output, rows, model, resume=resume)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)

    def infer(row: dict) -> dict:
        request = requests.post(
            endpoint.rstrip("/") + "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt(row)}],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=300,
        )
        request.raise_for_status()
        choices = request.json()["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"Unexpected completion for {row['id']}")
        text = choices[0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError(f"Missing text response for {row['id']}")
        return {
            "id": row["id"],
            "gold": row["answer_index"],
            "prediction": extract_answer(text, len(row["choices"])),
            "response": text,
            "finish_reason": choices[0]["finish_reason"],
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt(row).encode()).hexdigest(),
        }

    started = time.monotonic()
    with output.open("a", encoding="utf-8") as destination, ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map yields input order; no completed row can be silently skipped on failure.
        for result in pool.map(infer, rows[len(completed) :]):
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            completed.append(result)
            if len(completed) % 50 == 0 or len(completed) == len(rows):
                if mirror is not None:
                    shutil.copyfile(output, mirror)
                LOGGER.info(
                    "Progress %s %d/%d elapsed=%ds",
                    source.parent.name,
                    len(completed),
                    len(rows),
                    int(time.monotonic() - started),
                )
    if mirror is not None and not mirror.exists():
        shutil.copyfile(output, mirror)
    return score(completed)


def main() -> None:
    """Run a Hindi text suite against an existing local model endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, help="Persistent backup of completed predictions")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = args.root / args.dataset / "hindi_test.jsonl"
    metrics = run(
        source,
        args.output,
        args.endpoint,
        args.model,
        workers=args.workers,
        limit=args.limit,
        resume=args.resume,
        mirror=args.mirror,
    )
    LOGGER.info("RESULT %s", json.dumps({"dataset": args.dataset, **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
