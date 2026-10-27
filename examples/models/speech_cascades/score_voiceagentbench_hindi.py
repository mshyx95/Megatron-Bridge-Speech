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

"""Compute conservative, deterministic Hindi voice diagnostics without an LLM judge."""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

from eval_voiceagentbench_hindi import SUBSETS


LOGGER = logging.getLogger(__name__)


def _normalized(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(Decimal(str(value)).normalize())
    if isinstance(value, str):
        normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", normalized):
            return str(Decimal(normalized).normalize())
        return normalized
    return value


def _calls(value: object) -> list[dict] | None:
    if not isinstance(value, list) or any(not isinstance(item, dict) or len(item) != 1 for item in value):
        return None
    return value


def score_subset(source: Path, predictions: Path, subset: str, model: str) -> dict:
    """Check exact source coverage and summarize name and exact-argument matches."""
    if subset not in SUBSETS or not model:
        raise ValueError("Expected a known benchmark subset and served model name")
    rows = json.loads(source.read_text(encoding="utf-8"))
    outputs = json.loads(predictions.read_text(encoding="utf-8"))
    if len(rows) != len(outputs) or not rows:
        raise ValueError(f"Incomplete {subset}: {len(outputs)}/{len(rows)}")
    name_correct = 0
    exact_correct = 0
    empty_responses = 0
    for index, (row, output) in enumerate(zip(rows, outputs)):
        source_sha = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if output["id"] != row["id"] or output["source_sha256"] != source_sha or output["run_model"] != model:
            raise ValueError(f"{subset} source or model mismatch at {index}")
        if not isinstance(output["generated_text"], str):
            raise ValueError(f"Missing prediction at {subset}:{index}")
        if not output["generated_text"]:
            if subset != "safety" and output.get("protocol") != "realtime-text-output-v1":
                raise ValueError(f"Missing prediction at {subset}:{index}")
            empty_responses += 1
        if subset == "safety":
            continue
        reference = _calls(row["expected_tool_call"])
        if reference is None:
            raise ValueError(f"Invalid {subset} gold tool calls at {index}")
        if subset == "seqdep_tool":
            expected_names = [next(iter(item)) for item in reference]
            if all(
                re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", output["generated_text"])
                for name in expected_names
            ):
                name_correct += 1
            continue
        predicted = _calls(output["response"])
        if predicted is None:
            continue
        expected_names = Counter(next(iter(call)) for call in reference)
        actual_names = Counter(next(iter(call)) for call in predicted)
        if actual_names == expected_names:
            name_correct += 1
            expected = Counter(json.dumps(_normalized(call), sort_keys=True, ensure_ascii=False) for call in reference)
            actual = Counter(json.dumps(_normalized(call), sort_keys=True, ensure_ascii=False) for call in predicted)
            if expected == actual:
                exact_correct += 1
    result = {
        "subset": subset,
        "model": model,
        "completed": len(outputs),
        "method": "offline-deterministic",
        "empty_responses": empty_responses,
    }
    if subset == "safety":
        result["note"] = "Refusal is unscored without the official LLM judge"
    else:
        result["all_function_names_correct"] = name_correct
        result["function_name_rate"] = name_correct / len(outputs)
        if subset != "seqdep_tool":
            result["strict_argument_match"] = exact_correct
            result["strict_argument_rate"] = exact_correct / len(outputs)
        else:
            result["note"] = "Dependent-call schema and parameters unscored without the official LLM judge"
    return result


def main() -> None:
    """Verify and score the full six-subset Hindi generation inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    scores = []
    for subset, (folder, _) in SUBSETS.items():
        source = args.root / "data" / folder / "hindi" / f"{subset}_hindi.json"
        scores.append(score_subset(source, args.results_dir / f"{subset}.json", subset, args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for result in scores:
        LOGGER.info("%s", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
