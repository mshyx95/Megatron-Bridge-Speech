"""Validate bounded four-request text evaluation without a model server."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_four_workers_preserve_order_and_resume(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[3] / "examples/models/speech_cascades/eval_hindi_text.py"
    spec = importlib.util.spec_from_file_location("hindi_text_workers", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = [
        {"id": str(index), "question": "Question", "choices": ["one", "two"], "answer_index": 0} for index in range(12)
    ]
    source = tmp_path / "hindi_test.jsonl"
    output = tmp_path / "predictions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = []

    def post(endpoint, json, timeout):
        calls.append(json)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"finish_reason": "stop", "message": {"content": "A"}}]},
        )

    monkeypatch.setattr(runner.requests, "post", post)
    args = (source, output, "http://127.0.0.1:18201/v1", "verified-model")
    report = runner.run(*args, workers=4, limit=0, resume=True)
    assert report["correct"] == 12
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [row["id"] for row in rows]
    assert runner.run(*args, workers=4, limit=0, resume=True) == report
    assert len(calls) == 12
    with pytest.raises(ValueError, match="1-4 workers"):
        runner.run(*args, workers=5, limit=0, resume=True)
