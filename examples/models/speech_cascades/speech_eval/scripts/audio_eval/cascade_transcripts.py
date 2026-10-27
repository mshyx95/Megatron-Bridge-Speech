import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = value if isinstance(value, list) else value.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid ASR transcript artifact: {path}")
    return rows


def _row_id(row: dict) -> str:
    row_id = row.get("id") or row.get("UUID") or row.get("audio_id")
    if row_id is None and row.get("audio_ids"):
        row_id = row["audio_ids"][0]
    if not isinstance(row_id, str):
        raise ValueError(f"Invalid ASR transcript row: {row}")
    return row_id


def _row_transcript(row: dict) -> str:
    transcript = row.get("asr_transcript") or row.get("hypothesis") or row.get("generated_text")
    if transcript is None and row.get("generated_texts"):
        transcript = row["generated_texts"][0]
    if not isinstance(transcript, str):
        raise ValueError(f"Invalid ASR transcript row: {row}")
    return transcript.strip()


def load_transcripts(path: str | Path) -> dict[str, str]:
    path = Path(path)
    transcripts = {}
    for row in _load_rows(path):
        row_id = _row_id(row)
        if row_id in transcripts:
            raise ValueError(f"Duplicate ASR transcript id: {row_id}")
        transcripts[row_id] = _row_transcript(row)
    if not transcripts:
        raise ValueError(f"No ASR transcripts found in {path}")
    return transcripts


def merge_shards(inputs: list[Path], output: Path, expected_count: int | None) -> None:
    rows = []
    seen_ids = set()
    for path in inputs:
        shard = _load_rows(path)
        for row in shard:
            audio_id = _row_id(row)
            if audio_id in seen_ids:
                raise ValueError(f"Duplicate transcript ID across shards: {audio_id}")
            _row_transcript(row)
            seen_ids.add(audio_id)
            rows.append(row)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} merged transcripts, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge cascade transcript shards.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    merge_shards(args.input, args.output, args.expected_count)
    print(f"Merged {len(load_transcripts(args.output))} transcripts into {args.output}")


if __name__ == "__main__":
    main()
