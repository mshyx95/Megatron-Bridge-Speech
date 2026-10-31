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

"""GPT-4.1 CORRECT/INCORRECT judge harness for the BigBench-Audio (BBH) S2T eval.

Mirrors the reference ``bbh_gpt_judge.py`` + ``format_bigbench_audio_judge_input.py``: it turns the
speech-cascades model answers into GPT-judge inputs, asks a GPT-4.1 deployment whether each model
reply reaches the ground-truth answer, and reports the corpus CORRECT/total accuracy (also broken
down per BBH task). It is deliberately dependency-light (stdlib + ``urllib``, no torch/Megatron) so
it can run on a CPU judge box separate from the GPU decode in ``eval_bbh.py``.

The Azure/OpenAI client is configured **entirely from environment variables** — no secrets are ever
hardcoded. See ``JudgeClient`` for the exact variable names. With a tiny mocked ``judge_fn`` the
formatter + scorer are unit-testable without any network or GPU.

Standalone judge run (inputs already produced by ``eval_bbh.py``)::

    AZURE_OPENAI_ENDPOINT=https://<res>.cognitiveservices.azure.com/ \
    AZURE_OPENAI_API_KEY=... AZURE_OPENAI_DEPLOYMENT=gpt-4.1 \
    python examples/models/speech_cascades/bbh_gpt_judge.py \
        --input /out/eval_output/bbh/judge_input.json --cache /out/eval_output/bbh/judge_cache.jsonl
"""

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


SYSTEM_PROMPT = (
    "You are grading a model's answer to a reasoning question from BIG-Bench. "
    "You are given the correct answer and the model's full response. Decide "
    "whether the model's final answer matches the correct answer in meaning. "
    "Ignore formatting, explanations, and phrasing; judge only the final "
    "answer. Reply with exactly one word: CORRECT or INCORRECT."
)


def load_manifest(path: str) -> list[dict]:
    """Load a BigBench-Audio JSONL manifest (one JSON record per line)."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_predictions(path: str) -> dict[str, str]:
    """Load ``eval_bbh.py`` predictions (JSON list of {audio_ids, generated_texts}) keyed by UUID."""
    predictions: dict[str, str] = {}
    for record in json.loads(Path(path).read_text(encoding="utf-8")):
        if "audio_ids" not in record or "generated_texts" not in record:
            continue
        audio_id = record["audio_ids"][0]
        if audio_id in predictions:
            raise ValueError(f"Duplicate prediction for {audio_id}")
        predictions[audio_id] = record["generated_texts"][0]
    return predictions


def format_judge_inputs(manifest: list[dict], predictions: dict[str, str]) -> list[dict]:
    """Align model predictions to the manifest and build the GPT-judge input schema.

    Mirrors the reference ``format_bigbench_audio_judge_input.py``: each manifest record contributes
    one judge item ``{conversation_id, CorpusName, rounds:[{round, question, ground_truth,
    model_reply_text}]}``. Raises if the prediction/manifest UUID sets do not match exactly.
    """
    manifest_ids = {record["UUID"] for record in manifest}
    missing = sorted(manifest_ids - predictions.keys())
    unexpected = sorted(predictions.keys() - manifest_ids)
    if missing or unexpected:
        raise ValueError(
            f"Prediction alignment failed: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    items = []
    for record in manifest:
        audio_id = record["UUID"]
        items.append(
            {
                "conversation_id": audio_id,
                "CorpusName": record["CorpusName"],
                "rounds": [
                    {
                        "round": 1,
                        "question": record.get("Question", ""),
                        "ground_truth": record["Transcription"],
                        "model_reply_text": predictions[audio_id],
                    }
                ],
            }
        )
    return items


def build_user_prompt(item: dict) -> str:
    """Build the GPT-judge user turn (correct answer + model response) for one judge item."""
    round0 = item["rounds"][0]
    ground_truth = str(round0.get("ground_truth", "")).strip()
    reply = str(round0.get("model_reply_text", "")).strip()
    return (
        f"Correct answer: {ground_truth}\n\n"
        f"Model response:\n{reply}\n\n"
        "Does the model's final answer match the correct answer? "
        "Reply CORRECT or INCORRECT."
    )


def parse_verdict(text: str) -> str:
    """Map a raw GPT reply to CORRECT / INCORRECT / NONE (INCORRECT wins over the CORRECT substring)."""
    upper = text.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    if "CORRECT" in upper:
        return "CORRECT"
    return "NONE"


class JudgeClient:
    """Azure/OpenAI chat-completions judge client configured from environment variables.

    Environment variables (never hardcode secrets):
        AZURE_OPENAI_ENDPOINT      Required. e.g. ``https://<resource>.cognitiveservices.azure.com/``.
        AZURE_OPENAI_DEPLOYMENT    Deployment/model name (default ``gpt-4.1``).
        AZURE_OPENAI_API_VERSION   API version (default ``2025-01-01-preview``).
        AZURE_OPENAI_API_KEY       API key -> sent as the ``api-key`` header (key auth).
        AZURE_OPENAI_BEARER_TOKEN  AAD bearer token -> sent as ``Authorization: Bearer`` (used only
                                   when no API key is set).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        bearer_token: str | None = None,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set (pass --endpoint or export it).")
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.bearer_token = bearer_token or os.environ.get("AZURE_OPENAI_BEARER_TOKEN")
        if not self.api_key and not self.bearer_token:
            raise ValueError(
                "No credential: set AZURE_OPENAI_API_KEY or AZURE_OPENAI_BEARER_TOKEN in the env."
            )

    @property
    def url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def judge(self, ground_truth: str, reply: str, *, retries: int = 6, timeout: int = 60) -> str:
        """Return CORRECT / INCORRECT / NONE / ERROR for one (ground_truth, reply) pair."""
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        {"rounds": [{"ground_truth": ground_truth, "model_reply_text": reply}]}
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 0,
        }
        data = json.dumps(body).encode()
        for attempt in range(retries):
            try:
                request = urllib.request.Request(self.url, data=data, headers=self._headers())
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https endpoint)
                    content = json.loads(response.read())["choices"][0]["message"]["content"]
                return parse_verdict(content)
            except Exception:  # noqa: BLE001 (network judge is best-effort with backoff)
                time.sleep(2 * (attempt + 1))
        return "ERROR"


def run_judge(
    items: list[dict],
    judge_fn: Callable[[str, str], str],
    *,
    workers: int = 16,
    cache_path: str | None = None,
) -> dict[str, str]:
    """Judge every item with ``judge_fn(ground_truth, reply)`` and return {conversation_id: verdict}.

    Resumable: verdicts already present in ``cache_path`` (JSONL of {id, corpus, verdict}) are reused
    and only the remaining items are sent to the judge. ``judge_fn`` is injected so unit tests can
    pass a deterministic mock instead of the live GPT client.
    """
    verdicts: dict[str, str] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        for line in cache_file.open(encoding="utf-8"):
            line = line.strip()
            if line:
                record = json.loads(line)
                verdicts[record["id"]] = record["verdict"]

    todo = [item for item in items if item["conversation_id"] not in verdicts]

    def _judge_item(item: dict) -> dict:
        round0 = item["rounds"][0]
        verdict = judge_fn(str(round0.get("ground_truth", "")), str(round0.get("model_reply_text", "")))
        return {"id": item["conversation_id"], "corpus": item.get("CorpusName"), "verdict": verdict}

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    sink = cache_file.open("a", encoding="utf-8") if cache_file else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for record in pool.map(_judge_item, todo):
                verdicts[record["id"]] = record["verdict"]
                if sink:
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
    finally:
        if sink:
            sink.close()
    return verdicts


def score_verdicts(items: list[dict], verdicts: dict[str, str]) -> dict:
    """Compute overall + per-corpus CORRECT/total accuracy from a {conversation_id: verdict} map."""
    total = len(items)
    correct = none = errors = 0
    per_corpus: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # corpus -> [correct, total]
    for item in items:
        verdict = verdicts.get(item["conversation_id"], "ERROR")
        corpus = item.get("CorpusName", "unknown")
        per_corpus[corpus][1] += 1
        if verdict == "CORRECT":
            correct += 1
            per_corpus[corpus][0] += 1
        elif verdict == "NONE":
            none += 1
        elif verdict == "ERROR":
            errors += 1
    by_corpus = {
        corpus: {
            "correct": vals[0],
            "total": vals[1],
            "accuracy": round(100.0 * vals[0] / vals[1], 2) if vals[1] else 0.0,
        }
        for corpus, vals in sorted(per_corpus.items())
    }
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 2) if total else 0.0,
        "none": none,
        "errors": errors,
        "by_corpus": by_corpus,
    }


def main() -> None:
    """Judge BBH model answers and print/write the CORRECT/total accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Pre-formatted judge_input.json from eval_bbh.py.")
    parser.add_argument("--manifest", help="BBH JSONL manifest (with --predictions instead of --input).")
    parser.add_argument("--predictions", help="eval_bbh.py predictions.json (with --manifest).")
    parser.add_argument("--cache", default=None, help="Resumable JSONL verdict cache.")
    parser.add_argument("--output", default=None, help="Where to write the accuracy summary JSON.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--endpoint", default=None, help="Overrides AZURE_OPENAI_ENDPOINT.")
    parser.add_argument("--deployment", default=None, help="Overrides AZURE_OPENAI_DEPLOYMENT.")
    args = parser.parse_args()

    if args.input:
        items = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        if not args.manifest or not args.predictions:
            parser.error("--manifest and --predictions are required when --input is not given.")
        items = format_judge_inputs(load_manifest(args.manifest), load_predictions(args.predictions))

    client = JudgeClient(endpoint=args.endpoint, deployment=args.deployment)
    print(f"[bbh] judging {len(items)} items with deployment={client.deployment}", flush=True)
    verdicts = run_judge(items, client.judge, workers=args.workers, cache_path=args.cache)
    summary = score_verdicts(items, verdicts)
    print(json.dumps(summary, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
