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

"""Download pinned Hindi benchmark subsets and verify their local inventory."""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


LOGGER = logging.getLogger(__name__)
DATASETS = {
    "global_mmlu_lite": ("CohereLabs/Global-MMLU-Lite", ("hi/*.parquet",), 400),
    "mmlu_prox": ("li-lab/MMLU-ProX", ("hi/*.parquet",), 11759),
    "indic_mmlu_pro": (
        "LinguaLift/IndicMMLU-Pro",
        ("data/indic_mmlu_pro/hindi/test/*", "data/indic_mmlu_pro/hindi/validation/*"),
        12032,
    ),
    "milu": ("ai4bharat/MILU", ("Hindi/*.parquet",), None),
    "voiceagentbench": (
        "krutrim-ai-labs/VoiceAgentBench",
        ("data/*_data/hindi/*.json", "data/*_audios/hindi/*.wav"),
        903,
    ),
}


def _count_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.read_metadata(path).num_rows
    if path.suffix == ".arrow":
        import pyarrow as arrow

        with arrow.memory_map(str(path), "r") as source:
            return arrow.ipc.open_stream(source).read_all().num_rows
    raise ValueError(f"Unsupported benchmark table: {path}")


def _export_test(name: str, paths: list[Path], destination: Path) -> tuple[int, int]:
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    count = 0
    conflicts = 0
    with destination.open("w", encoding="utf-8") as stream:
        for path in paths:
            if path.suffix == ".parquet":
                records = parquet.read_table(path).to_pylist()
            elif path.suffix == ".arrow":
                with arrow.memory_map(str(path), "r") as source:
                    records = arrow.ipc.open_stream(source).read_all().to_pylist()
            else:
                continue
            for row in records:
                if name == "global_mmlu_lite":
                    choices = [row[f"option_{letter}"] for letter in "abcd"]
                elif name == "mmlu_prox":
                    choices = [row[f"option_{index}"] for index in range(10)]
                    while choices and choices[-1] is None:
                        choices.pop()
                elif name == "indic_mmlu_pro":
                    choices = row["options"]
                else:
                    continue
                answer = row["answer"].upper()
                answer_index = ord(answer) - ord("A")
                if not choices or not all(isinstance(choice, str) and choice for choice in choices):
                    raise ValueError(f"Missing options in {name} test row {count}")
                if not 0 <= answer_index < len(choices):
                    raise ValueError(f"Invalid answer in {name} test row {count}")
                conflict = row.get("answer_index") is not None and row["answer_index"] != answer_index
                conflicts += conflict
                normalized = {
                    "id": f"{name}/hi/test/{count}",
                    "question": row["question"],
                    "choices": choices,
                    "answer_index": None if conflict else answer_index,
                    "subject": row.get("category", row.get("subject")),
                }
                if conflict:
                    normalized["label_conflict"] = {"answer": row["answer"], "answer_index": row["answer_index"]}
                stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                count += 1
    return count, conflicts


def _voice_files(root: Path) -> dict[str, int]:
    counts = {}
    for annotation in sorted(root.glob("data/*_data/hindi/*.json")):
        rows = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected annotation list: {annotation}")
        counts[annotation.stem] = len(rows)
        for row in rows:
            paths = [row["path"]] if row.get("path") else []
            paths.extend(turn["path"] for turn in row.get("chat_history", []) if turn.get("path"))
            if not paths:
                raise ValueError(f"No Hindi audio paths for {row.get('id')} in {annotation}")
            for relative in paths:
                audio = (root / relative).resolve()
                if not audio.is_relative_to(root.resolve()) or not audio.is_file() or audio.stat().st_size == 0:
                    raise ValueError(f"Missing or invalid referenced audio: {relative}")
    return counts


def prepare(root: Path, *, include_milu: bool, voice_audio: bool) -> dict:
    """Fetch version-pinned Hindi subsets without silently dropping unavailable data."""
    api = HfApi()
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (repo, patterns, expected_test) in DATASETS.items():
        if name == "milu" and not include_milu:
            result[name] = {"status": "blocked_gated_access", "repo": repo}
            continue
        if name == "voiceagentbench" and not voice_audio:
            patterns = ("data/*_data/hindi/*.json",)
        revision = api.dataset_info(repo, token=get_token() if name == "milu" else False).sha
        destination = root / name
        LOGGER.info("Downloading %s revision %s into %s", repo, revision, destination)
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=list(patterns),
                local_dir=destination,
                token=get_token() if name == "milu" else False,
                max_workers=8,
            )
        except (GatedRepoError, HfHubHTTPError) as error:
            if name != "milu":
                raise
            raise PermissionError(
                "MILU requires accepted gated access and a Hugging Face token on this node."
            ) from error
        files = sorted(path for path in destination.rglob("*") if path.is_file() and ".cache" not in path.parts)
        if not files:
            raise ValueError(f"No files downloaded for {repo}")
        entry = {
            "status": "ready" if name != "voiceagentbench" or voice_audio else "annotations_only",
            "repo": repo,
            "revision": revision,
            "files": [str(path.relative_to(destination)) for path in files],
        }
        if name == "voiceagentbench":
            if voice_audio:
                counts = _voice_files(destination)
            else:
                counts = {
                    path.stem: len(json.loads(path.read_text(encoding="utf-8")))
                    for path in destination.glob("data/*_data/hindi/*.json")
                }
            if sum(counts.values()) != expected_test or len(counts) != 6:
                raise ValueError(f"VoiceAgentBench Hindi count mismatch: {counts}")
            entry["counts"] = counts
        else:
            test_files = [path for path in files if path.name.startswith("test") or "/test/" in str(path)]
            test_count = sum(_count_rows(path) for path in test_files if path.suffix in {".parquet", ".arrow"})
            if not test_count or expected_test is not None and test_count != expected_test:
                raise ValueError(f"{repo} Hindi test count: {test_count}, expected {expected_test}")
            entry["test_count"] = test_count
            if name != "milu":
                normalized = destination / "hindi_test.jsonl"
                exported, conflicts = _export_test(name, test_files, normalized)
                if exported != test_count:
                    raise ValueError(f"{repo} normalized test count: {exported}, expected {test_count}")
                entry["normalized_test"] = normalized.name
                entry["label_conflicts"] = conflicts
        result[name] = entry
        (root / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    """Prepare the selected Hindi assets and record exact revisions and counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-milu", action="store_true", help="Requires approved gated access on this node")
    parser.add_argument(
        "--voice-audio", action="store_true", help="Download and verify all referenced Hindi WAV files"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare(args.root, include_milu=args.include_milu, voice_audio=args.voice_audio)
    for name, entry in manifest.items():
        LOGGER.info("%s: %s test=%s", name, entry["status"], entry.get("test_count", entry.get("counts")))


if __name__ == "__main__":
    main()
