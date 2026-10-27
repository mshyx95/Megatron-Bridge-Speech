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

"""Run Hindi VoiceAgentBench audio inference against an existing OpenAI-compatible server."""

import argparse
import base64
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import requests


LOGGER = logging.getLogger(__name__)
SUBSETS = {
    "single_tool": ("single_tool_data", "single"),
    "single_tool_retrieval": ("single_tool_retrieval_data", "single"),
    "parallel_tool": ("parallel_tool_data", "multiple"),
    "seqdep_tool": ("seqdep_tool_data", "dependent"),
    "multi_turn": ("multi_turn_data", "multiturn"),
    "safety": ("safety_data", "safety"),
}


def _audio(root: Path, relative: str) -> dict:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() != ".wav":
        raise ValueError(f"Invalid audio reference: {relative}")
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(path.read_bytes()).decode(), "format": "wav"},
    }


def messages(root: Path, row: dict) -> list[dict]:
    """Render benchmark instructions and actual recordings without using transcript hints."""
    values = {"functions": row["functions"]}
    if "user_info" in row:
        values["user_info"] = row["user_info"]
    prompt = row["instruction"][0][0]["content"].format(**values)
    if "user_request" in row:
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [_audio(root, row["path"])]},
        ]
    if row.get("chat_history"):
        output = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for turn in row["chat_history"]:
            if turn["role"] == "user":
                output.append({"role": "user", "content": [_audio(root, turn["path"])]})
            elif turn["role"] == "assistant":
                output.append({"role": "assistant", "content": turn["content"]})
            else:
                raise ValueError(f"Unexpected dialogue role: {turn['role']}")
        return output
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, _audio(root, row["path"])]}]


def run(
    root: Path,
    official_repo: Path,
    endpoint: str,
    model: str,
    subset: str,
    output: Path,
    limit: int,
    *,
    resume: bool = False,
    mirror: Path | None = None,
) -> int:
    """Produce official response-file shape; never imply unjudged generations are scored."""
    if subset not in SUBSETS or limit < 0 or not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("Expected a supported subset, nonnegative limit and loopback endpoint")
    if output.exists() and not resume:
        raise FileExistsError(f"Use --resume or a new output path to avoid mixing runs: {output}")
    sys.path.insert(0, str(official_repo / "voice_agent_bench"))
    from utils.parser import parse_response

    data_folder, evaluator = SUBSETS[subset]
    source = root / "data" / data_folder / "hindi" / f"{subset}_hindi.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected unique, nonempty source rows: {source}")
    responses = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    if len(responses) > len(rows):
        raise ValueError(f"Too many existing predictions: {output}")
    for index, existing in enumerate(responses):
        if (
            existing["id"] != rows[index]["id"]
            or existing["expected_tool_call"] != rows[index].get("expected_tool_call")
            or existing["run_model"] != model
        ):
            raise ValueError(f"Existing predictions/source/model disagree at {index}: {output}")
        expected_sha = hashlib.sha256(json.dumps(rows[index], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if existing["source_sha256"] != expected_sha:
            raise ValueError(f"Source row changed since earlier prediction at {index}: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)
    for row in rows[len(responses) :]:
        payload = {
            "model": model,
            "messages": messages(root, row),
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = requests.post(endpoint.rstrip("/") + "/chat/completions", json=payload, timeout=300)
        request.raise_for_status()
        choices = request.json()["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] not in {"stop", "length"}:
            raise ValueError(f"Unexpected completion shape for {row['id']}")
        text = choices[0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError(f"Missing plain-text completion for {row['id']}")
        response = parse_response(text, evaluator)
        if evaluator in {"single", "multiple", "multiturn"}:
            response = None if response is None else [item for item in response if item is not None]
        responses.append(
            {
                **row,
                "generated_text": text,
                "response": response,
                "run_model": model,
                "source_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
        )
        output.write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if mirror is not None and (len(responses) % 10 == 0 or len(responses) == len(rows)):
            shutil.copyfile(output, mirror)
        LOGGER.info("Completed %s %d/%d id=%s", subset, len(responses), len(rows), row["id"])
    if mirror is not None and not mirror.exists():
        shutil.copyfile(output, mirror)
    return len(responses)


def main() -> None:
    """Evaluate a single Hindi subset with a pre-existing endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded VoiceAgentBench dataset root")
    parser.add_argument("--official-repo", type=Path, required=True, help="Pinned upstream evaluator checkout")
    parser.add_argument("--endpoint", required=True, help="Existing local OpenAI API base, including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, help="Persistent backup of completed predictions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = run(
        args.root,
        args.official_repo,
        args.endpoint,
        args.model,
        args.subset,
        args.output,
        args.limit,
        resume=args.resume,
        mirror=args.mirror,
    )
    LOGGER.info("Generation complete (%d rows); run official evaluator separately with a configured judge", count)


if __name__ == "__main__":
    main()
