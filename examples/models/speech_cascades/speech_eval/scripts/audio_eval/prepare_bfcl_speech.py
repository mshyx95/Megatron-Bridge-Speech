#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path

import soundfile as sf


DEFAULT_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: list[dict], limit: int, seed: int, category: str) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(f"{seed}:{category}")
    selected_ids = {row["id"] for row in rng.sample(rows, limit)}
    return [row for row in rows if row["id"] in selected_ids]


def stage_audio(
    row: dict,
    output_dir: Path,
    audio_mode: str,
) -> tuple[dict, list[dict]]:
    audio_paths = row.get("wav_path")
    if not isinstance(audio_paths, list) or len(audio_paths) != 1:
        raise ValueError(f"{row.get('id')}: expected exactly one wav_path")

    source_path = Path(audio_paths[0])
    if not source_path.is_file():
        raise FileNotFoundError(f"{row['id']}: missing audio {source_path}")

    info = sf.info(source_path)
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"{row['id']}: expected 16 kHz mono, got {info.samplerate} Hz/{info.channels} channels")

    staged_row = dict(row)
    staged_path = source_path
    if audio_mode != "reference":
        audio_dir = output_dir / "wavs"
        audio_dir.mkdir(parents=True, exist_ok=True)
        staged_path = audio_dir / f"{row['id']}.wav"
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        if audio_mode == "copy":
            shutil.copy2(source_path, staged_path)
        else:
            staged_path.symlink_to(source_path)
        staged_row["wav_path"] = [str(staged_path.resolve())]

    manifest_row = {
        "id": row["id"],
        "category": row["id"].rsplit("_", 1)[0],
        "audio_path": str(staged_path.resolve()),
        "duration_seconds": round(info.duration, 4),
        "question": row["question"][0],
        "functions": row["function"],
    }
    return staged_row, manifest_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a validated speech BFCL subset for local evaluation.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=10,
        help="Number of examples per category; use 0 for all examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audio-mode",
        choices=("reference", "symlink", "copy"),
        default="reference",
        help="How the prepared dataset should refer to source wav files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    possible_answer_dir = args.output_dir / "possible_answer"
    possible_answer_dir.mkdir(exist_ok=True)

    manifest = []
    category_counts = {}
    for category in args.categories:
        filename = f"BFCL_v3_{category}.json"
        source_path = args.source_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing BFCL category file: {source_path}")

        source_rows = load_jsonl(source_path)
        selected_rows = select_rows(source_rows, args.limit_per_category, args.seed, category)

        answer_path = args.source_dir / "possible_answer" / filename
        answers_by_id = {}
        if answer_path.is_file():
            answers_by_id = {row["id"]: row for row in load_jsonl(answer_path)}

        staged_rows = []
        selected_answers = []
        for row in selected_rows:
            if answers_by_id and row["id"] not in answers_by_id:
                raise ValueError(f"{row['id']}: no matching possible answer")
            staged_row, manifest_row = stage_audio(row, args.output_dir, args.audio_mode)
            staged_rows.append(staged_row)
            manifest.append(manifest_row)
            if answers_by_id:
                selected_answers.append(answers_by_id[row["id"]])

        write_jsonl(args.output_dir / filename, staged_rows)
        if selected_answers:
            write_jsonl(possible_answer_dir / filename, selected_answers)
        category_counts[category] = len(staged_rows)

    write_jsonl(args.output_dir / "manifest.jsonl", manifest)
    metadata = {
        "source_dir": str(args.source_dir.resolve()),
        "audio_mode": args.audio_mode,
        "seed": args.seed,
        "limit_per_category": args.limit_per_category,
        "category_counts": category_counts,
        "total_examples": len(manifest),
    }
    (args.output_dir / "dataset_info.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
