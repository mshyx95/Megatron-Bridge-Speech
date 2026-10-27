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

"""Run the pinned VoiceAgentBench evaluators with a configurable Azure judge deployment (default GPT-5.2)."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from eval_voiceagentbench_hindi import SUBSETS
from judge_voiceagentbench_hindi_safety import azure_token, parse_verdict


LOGGER = logging.getLogger(__name__)
DEFAULT_JUDGE_DEPLOYMENT = "gpt-5.2"
API_VERSION = "2025-01-01-preview"
UPSTREAM_REVISION = "d1efb7d4b71bed85534ce171460f8f2fb133a456"
UPSTREAM_SHA256 = "5da5050ff685f2c008e084b09291d7fecb5886f9b871df5a5c040dd7f4abc795"
EXPECTED_COUNTS = {
    "single_tool": 134,
    "single_tool_retrieval": 177,
    "parallel_tool": 120,
    "seqdep_tool": 39,
    "multi_turn": 358,
    "safety": 75,
}


def validate_judge_reply(prompt: str, answer: str) -> None:
    """Reject malformed judge output rather than letting upstream count an API failure as zero."""
    if prompt.lstrip().startswith("The response below (contained in the <response> tags)"):
        parse_verdict(answer)
        return
    payload = json.loads(answer)
    score = payload["Score"]
    if prompt.lstrip().startswith("You are tasked with evaluating the correctness of a sequential"):
        if score not in ([0, 0], [1, 0], [1, 1]) or any(type(value) is not int for value in score):
            raise ValueError("Expected a dependent-call Score of [0,0], [1,0], or [1,1]")
    elif type(score) is not int or score not in (0, 1):
        raise ValueError("Expected a parameter-filling Score of 0 or 1")


class JudgeContentFiltered(ValueError):
    """Signal that the Azure content filter blocked a judge call rather than the judge itself failing."""


class AzureJudge:
    """Provide the upstream ``complete(prompt).text`` interface with a verified Azure judge deployment."""

    def __init__(self, endpoint: str, judge_deployment: str = DEFAULT_JUDGE_DEPLOYMENT) -> None:
        parsed = urlsplit(endpoint)
        is_v1 = parsed.path.rstrip("/") == "/openai/v1"
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith((".cognitiveservices.azure.com", ".services.ai.azure.com"))
            or parsed.username
            or parsed.password
            or (parsed.path.rstrip("/") and not is_v1)
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Expected an Azure HTTPS resource endpoint or /openai/v1 API base")
        self.judge_deployment = judge_deployment
        self.is_v1 = is_v1
        self.url = (
            endpoint.rstrip("/") + "/chat/completions"
            if is_v1
            else endpoint.rstrip("/")
            + f"/openai/deployments/{judge_deployment}/chat/completions?api-version={API_VERSION}"
        )
        self.token = ""
        self.token_issued = 0.0
        self.responses: list[dict] = []
        self.failure: Exception | None = None

    def complete(self, prompt: str) -> types.SimpleNamespace:
        """Send the exact upstream rubric and retain the full judge response."""
        try:
            if not self.token or time.monotonic() - self.token_issued > 1800:
                self.token = azure_token()
                self.token_issued = time.monotonic()
            payload = {"messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 2048}
            if self.is_v1:
                payload["model"] = self.judge_deployment
            body = json.dumps(payload).encode()
            for attempt in range(5):
                request = urllib.request.Request(
                    self.url,
                    data=body,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token},
                )
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result = json.load(response)
                    model = result["model"]
                    choice = result["choices"][0]
                    if choice["finish_reason"] == "content_filter":
                        raise JudgeContentFiltered(f"Azure content filter blocked the judge call for {model}")
                    if not model.startswith(self.judge_deployment) or choice["finish_reason"] != "stop":
                        raise ValueError(f"Judge model or finish reason mismatch: {model}, {choice['finish_reason']}")
                    answer = choice["message"]["content"]
                    if not isinstance(answer, str):
                        raise ValueError(f"{self.judge_deployment} judge returned no text")
                    validate_judge_reply(prompt, answer)
                    self.responses.append(
                        {
                            "judge_model": model,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "text": answer,
                        }
                    )
                    return types.SimpleNamespace(text=answer)
                except urllib.error.HTTPError as error:
                    if error.code == 401 and attempt < 4:
                        self.token = azure_token()
                        self.token_issued = time.monotonic()
                        LOGGER.warning("Refreshed expired %s judge credential", self.judge_deployment)
                    elif error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                        raise
                    else:
                        LOGGER.warning(
                            "Retry %s judge HTTP %d (attempt %d/5)", self.judge_deployment, error.code, attempt + 1
                        )
                except (urllib.error.URLError, TimeoutError) as error:
                    if attempt == 4:
                        raise
                    LOGGER.warning(
                        "Retry %s judge transport %s (attempt %d/5)",
                        self.judge_deployment,
                        type(error).__name__,
                        attempt + 1,
                    )
                time.sleep(2**attempt)
            raise RuntimeError(f"{self.judge_deployment} judge retry budget exhausted")
        except (ValueError, KeyError, IndexError, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            self.failure = error
            raise


def load_upstream_evaluators(official_dir: Path) -> tuple[dict, str]:
    """Import the pinned evaluator logic without changing upstream files or requiring its OpenAI client."""
    official_dir = official_dir.resolve()
    required = ["__init__.py", "utils.py", "single.py", "multiple.py", "dependent.py", "multiturn.py", "safety.py"]
    if not all((official_dir / name).is_file() for name in required):
        raise FileNotFoundError(f"Missing official VoiceAgentBench evaluator files: {official_dir}")
    digest = hashlib.sha256()
    for name in sorted(required):
        digest.update(name.encode() + b"\0" + (official_dir / name).read_bytes())
    if digest.hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Official evaluator source differs from the pinned VoiceAgentBench revision")

    # The upstream constructors import OpenAI eagerly; each instance is replaced
    # with the explicit AzureJudge before any call to evaluate().
    class UnconfiguredJudge:
        def __init__(self, model: str):
            if model != "gpt-4o-mini":
                raise ValueError(f"Unexpected upstream judge default: {model}")

        def complete(self, prompt: str) -> None:
            raise RuntimeError("Upstream default judge was not replaced")

    for name in ("llama_index", "llama_index.llms", "llama_index.llms.openai"):
        module = types.ModuleType(name)
        if name.endswith(".openai"):
            module.OpenAI = UnconfiguredJudge
        else:
            module.__path__ = []
        sys.modules[name] = module

    package = "voiceagentbench_pinned_evaluators"
    spec = importlib.util.spec_from_file_location(
        package, official_dir / "__init__.py", submodule_search_locations=[str(official_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official evaluators from {official_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return module.evaluator_mapping, digest.hexdigest()


def load_predictions(source: Path, predictions: Path, model: str, subset: str) -> list[dict]:
    """Reject incomplete or differently sourced Hindi predictions before judging."""
    rows = json.loads(source.read_text(encoding="utf-8"))
    outputs = json.loads(predictions.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_COUNTS[subset] or len(outputs) != len(rows):
        raise ValueError(f"Incomplete {subset}: {len(rows)} source, {len(outputs)} predictions")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate Hindi IDs in {source}")
    for index, (row, output) in enumerate(zip(rows, outputs)):
        source_hash = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (
            row["id"] != output["id"]
            or output["run_model"] != model
            or output["source_sha256"] != source_hash
            or not isinstance(output["generated_text"], str)
        ):
            raise ValueError(f"{subset} source/model/response mismatch at offset {index}")
    return outputs


def metrics(records: list[dict], evaluator_name: str) -> dict:
    """Aggregate the same three score components and denominators as the upstream evaluator."""
    scores = (
        [score["score"] for record in records for score in record["result"]["scores"]]
        if evaluator_name == "multiple"
        else [record["result"]["score"] for record in records]
    )
    if not scores or any(
        len(score) != 3 or any(type(value) is not int or value not in (0, 1) for value in score) for score in scores
    ):
        raise ValueError("Unexpected official evaluator score shape")
    counts = [sum(score[index] for score in scores) for index in range(3)]
    return {
        "completed_items": len(records),
        "scored_functions" if evaluator_name == "multiple" else "scored_items": len(scores),
        "tool_selection": {"correct": counts[0], "accuracy": counts[0] / len(scores)},
        "tool_call_schema": {"correct": counts[1], "accuracy": counts[1] / len(scores)},
        "parameter_filling": {"correct": counts[2], "accuracy": counts[2] / len(scores)},
    }


def run(
    root: Path,
    official_dir: Path,
    predictions: Path,
    output: Path,
    endpoint: str,
    model: str,
    subset: str,
    *,
    resume: bool,
    judge_deployment: str = DEFAULT_JUDGE_DEPLOYMENT,
) -> dict:
    """Judge every cached Hindi output using the unmodified upstream score implementation."""
    if subset not in SUBSETS or not model:
        raise ValueError("Expected a known VoiceAgentBench subset and run model")
    folder, evaluator_name = SUBSETS[subset]
    source = root / "data" / folder / "hindi" / f"{subset}_hindi.json"
    rows = load_predictions(source, predictions, model, subset)
    mapping, upstream_sha = load_upstream_evaluators(official_dir)
    evaluator = mapping[evaluator_name]()
    evaluator_module = sys.modules[evaluator.__class__.__module__]
    judge = AzureJudge(endpoint, judge_deployment)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not resume:
        raise FileExistsError(f"Use --resume or a fresh judge output: {output}")
    completed = (
        [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
    )
    if len(completed) > len(rows):
        raise ValueError("More judge records than source rows")
    for index, (record, item) in enumerate(zip(completed, rows)):
        if (
            record["id"] != item["id"]
            or record["run_model"] != model
            or record["judge_deployment"] != judge_deployment
            or record["upstream_sha256"] != upstream_sha
            or record["source_sha256"] != item["source_sha256"]
            or record["response_sha256"] != hashlib.sha256(item["generated_text"].encode()).hexdigest()
            or any(not reply["judge_model"].startswith(judge_deployment) for reply in record["judge_responses"])
        ):
            raise ValueError(f"Judge record/source mismatch at offset {index}")
    with output.open("a", encoding="utf-8") as stream:
        for item in rows[len(completed) :]:
            judge.responses = []
            judge.failure = None
            evaluator_module.llm_judge = judge
            content_filtered = False
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    evaluated = evaluator.evaluate([dict(item)])
            except JudgeContentFiltered:
                if subset != "safety":
                    raise
                content_filtered = True
                evaluated = None
                LOGGER.warning(
                    "Azure content filter blocked %s/%s; excluding from scored denominator", subset, item["id"]
                )
            if evaluated is not None and judge.failure is not None:
                raise RuntimeError(f"{judge_deployment} judge failed on {subset}/{item['id']}") from judge.failure
            if evaluated is not None and len(evaluated) != 1:
                raise ValueError(f"Official evaluator returned {len(evaluated)} records for one item")
            result = {"content_filtered": True} if content_filtered else evaluated[0]
            record = {
                "id": item["id"],
                "run_model": model,
                "judge_deployment": judge_deployment,
                "upstream_revision": UPSTREAM_REVISION,
                "upstream_sha256": upstream_sha,
                "source_sha256": item["source_sha256"],
                "response_sha256": hashlib.sha256(item["generated_text"].encode()).hexdigest(),
                "result": result,
                "judge_responses": judge.responses,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            completed.append(record)
            if len(completed) % 25 == 0:
                LOGGER.info("%s %s judged %d/%d", model, subset, len(completed), len(rows))
    report = {
        "dataset": subset,
        "run_model": model,
        "judge_deployment": judge_deployment,
        "official_judge": False,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sha256": upstream_sha,
        **(
            {
                "completed_items": len(completed),
                "content_filtered_items": sum(1 for record in completed if "content_filtered" in record["result"]),
                "scored_items": sum(1 for record in completed if "score" in record["result"]),
                "refusals": sum(record["result"].get("score", 0) for record in completed),
                "refusal_rate": (
                    sum(record["result"].get("score", 0) for record in completed)
                    / sum(1 for record in completed if "score" in record["result"])
                    if any("score" in record["result"] for record in completed)
                    else None
                ),
            }
            if subset == "safety"
            else metrics(completed, evaluator_name)
        ),
    }
    if subset == "safety" and any(
        "score" in record["result"]
        and (type(record["result"]["score"]) is not int or record["result"]["score"] not in (0, 1))
        for record in completed
    ):
        raise ValueError("Unexpected upstream safety score")
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run one official Hindi evaluator with the requested Azure judge deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", choices=tuple(SUBSETS), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--judge-deployment",
        default=DEFAULT_JUDGE_DEPLOYMENT,
        help="Azure deployment name to use as the judge (default: %(default)s). "
        "Scores from different judge deployments are not directly comparable.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info(
        "RESULT %s",
        json.dumps(
            run(
                args.root,
                args.official_dir,
                args.predictions,
                args.output,
                args.endpoint,
                args.model,
                args.subset,
                resume=args.resume,
                judge_deployment=args.judge_deployment,
            ),
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
