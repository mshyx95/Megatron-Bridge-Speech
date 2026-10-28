import argparse
import json
from pathlib import Path


def load_manifest(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_predictions(path):
    predictions = {}
    for record in json.loads(Path(path).read_text(encoding="utf-8")):
        if "audio_ids" not in record or "generated_texts" not in record:
            continue
        audio_id = record["audio_ids"][0]
        if audio_id in predictions:
            raise ValueError(f"Duplicate prediction for {audio_id}")
        predictions[audio_id] = record["generated_texts"][0]
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Format BigBench Audio S2T output for GPT judge")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    predictions = load_predictions(args.predictions)
    manifest_ids = {record["UUID"] for record in manifest}
    missing = sorted(manifest_ids - predictions.keys())
    unexpected = sorted(predictions.keys() - manifest_ids)
    if missing or unexpected:
        raise ValueError(f"Prediction alignment failed: missing={len(missing)}, unexpected={len(unexpected)}")

    output = []
    for record in manifest:
        audio_id = record["UUID"]
        output.append(
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(output)} judge items to {output_path}")


if __name__ == "__main__":
    main()
