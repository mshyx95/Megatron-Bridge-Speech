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

"""Judge pinned VoiceAgentBench Hindi safety responses with Azure GPT-4.1."""

import argparse
import ast
import hashlib
import json
import logging
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


LOGGER = logging.getLogger(__name__)
JUDGE_MODEL = "gpt-4.1"
API_VERSION = "2025-01-01-preview"
VERDICT = re.compile(r"(?im)^\s*Answer:\s*(Yes|No)\s*[.!]?\s*$")


def refusal_rubric(official_utils: Path) -> str:
    """Load the pinned upstream safety rubric without importing its dependencies."""
    tree = ast.parse(official_utils.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "meta_prompt_refusal" for target in node.targets
        ):
            rubric = ast.literal_eval(node.value)
            if not isinstance(rubric, str) or rubric.count("{generated_text}") != 1:
                raise ValueError("Unexpected upstream safety rubric")
            return rubric
    raise ValueError("Missing upstream meta_prompt_refusal")


def parse_verdict(text: str) -> bool:
    """Require exactly one explicit answer instead of treating a malformed reply as a refusal failure."""
    matches = VERDICT.findall(text)
    if len(matches) != 1:
        raise ValueError("GPT-4.1 judge did not return one unambiguous Answer: Yes/No")
    return matches[0].lower() == "yes"


def source_rows(source: Path, predictions: Path, model: str) -> list[dict]:
    """Require full, ordered, source-hash-verified safety coverage."""
    rows = json.loads(source.read_text(encoding="utf-8"))
    outputs = json.loads(predictions.read_text(encoding="utf-8"))
    if len(rows) != 75 or len(outputs) != len(rows):
        raise ValueError(f"Expected all 75 Hindi safety items: {len(rows)} source, {len(outputs)} outputs")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate Hindi safety source IDs")
    for index, (row, output) in enumerate(zip(rows, outputs)):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (
            output["id"] != row["id"]
            or output["source_sha256"] != digest
            or output["run_model"] != model
            or not isinstance(output["generated_text"], str)
        ):
            raise ValueError(f"Source/model/response mismatch at Hindi safety item {index}")
    return outputs


def azure_token() -> str:
    """Obtain a short-lived OAuth token without writing it to results or stdout."""
    result = subprocess.run(
        [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://cognitiveservices.azure.com/",
            "--query",
            "accessToken",
            "-o",
            "tsv",
            "--only-show-errors",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    token = result.stdout.strip()
    if not token:
        raise ValueError("Azure CLI returned an empty judge token")
    return token


def judge_response(text: str, rubric: str, endpoint: str, token: str) -> tuple[bool, str, str]:
    """Apply upstream refusal criteria using the requested independent judge model."""
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".cognitiveservices.azure.com"):
        raise ValueError("Expected an Azure Cognitive Services HTTPS judge endpoint")
    url = endpoint.rstrip("/") + f"/openai/deployments/{JUDGE_MODEL}/chat/completions?api-version={API_VERSION}"
    payload = {
        "messages": [{"role": "user", "content": rubric.format(generated_text=text)}],
        "temperature": 0,
        "max_tokens": 512,
    }
    request_body = json.dumps(payload).encode()
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            data=request_body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            model = result["model"]
            choice = result["choices"][0]
            if not model.startswith(JUDGE_MODEL) or choice["finish_reason"] != "stop":
                raise ValueError(f"Judge model or finish reason mismatch: {model}, {choice['finish_reason']}")
            answer = choice["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("Judge did not return text")
            return parse_verdict(answer), answer, model
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            LOGGER.warning("Retry GPT-4.1 judge HTTP %d (attempt %d/5)", error.code, attempt + 1)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == 4:
                raise
            LOGGER.warning("Retry GPT-4.1 judge transport error %s (attempt %d/5)", type(error).__name__, attempt + 1)
        time.sleep(2**attempt)
    raise RuntimeError("GPT-4.1 judge retry budget exhausted")


def run(
    source: Path,
    predictions: Path,
    official_utils: Path,
    output: Path,
    endpoint: str,
    model: str,
    *,
    resume: bool,
) -> dict:
    """Save every judge verdict durably and return the refusal rate over all 75 items."""
    outputs = source_rows(source, predictions, model)
    rubric = refusal_rubric(official_utils)
    rubric_hash = hashlib.sha256(rubric.encode()).hexdigest()
    if output.exists() and not resume:
        raise FileExistsError(f"Use --resume or a fresh judge output path: {output}")
    completed = (
        [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
    )
    if len(completed) > len(outputs):
        raise ValueError("More judge verdicts than Hindi safety items")
    for index, result in enumerate(completed):
        row = outputs[index]
        if (
            result["id"] != row["id"]
            or result["run_model"] != model
            or result["judge_deployment"] != JUDGE_MODEL
            or result["rubric_sha256"] != rubric_hash
            or result["source_sha256"] != row["source_sha256"]
            or result["response_sha256"] != hashlib.sha256(row["generated_text"].encode()).hexdigest()
            or not isinstance(result["refused"], bool)
        ):
            raise ValueError(f"Judge verdict/input mismatch at offset {index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(completed) < len(outputs):
        token = azure_token()
        with output.open("a", encoding="utf-8") as stream:
            for row in outputs[len(completed) :]:
                refused, answer, judged_model = judge_response(row["generated_text"], rubric, endpoint, token)
                result = {
                    "id": row["id"],
                    "run_model": model,
                    "source_sha256": row["source_sha256"],
                    "response_sha256": hashlib.sha256(row["generated_text"].encode()).hexdigest(),
                    "rubric_sha256": rubric_hash,
                    "judge_deployment": JUDGE_MODEL,
                    "judge_model": judged_model,
                    "judge_response": answer,
                    "refused": refused,
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                completed.append(result)
                if len(completed) % 10 == 0:
                    LOGGER.info("%s judged %d/%d", model, len(completed), len(outputs))
    refused_count = sum(result["refused"] for result in completed)
    return {
        "run_model": model,
        "judge_deployment": JUDGE_MODEL,
        "rubric_sha256": rubric_hash,
        "completed": len(completed),
        "refusals": refused_count,
        "refusal_rate": refused_count / len(completed),
        "official_score": False,
    }


def main() -> None:
    """Run the GPT-4.1 variant of the upstream Hindi safety refusal evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--official-utils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(
        args.source,
        args.predictions,
        args.official_utils,
        args.output,
        args.endpoint,
        args.model,
        resume=args.resume,
    )
    summary = args.output.with_suffix(".metrics.json")
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("RESULT %s", json.dumps(result))


if __name__ == "__main__":
    main()
