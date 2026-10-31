"""Unit tests for the BigBench-Audio (BBH) judge harness: formatter schema + mocked accuracy math."""

import importlib.util
import sys
from pathlib import Path

import pytest


# Load bbh_gpt_judge.py from the examples tree (not an installed package) without GPU/torch deps.
_JUDGE_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "speech_cascades"
    / "bbh_gpt_judge.py"
)
_spec = importlib.util.spec_from_file_location("bbh_gpt_judge", _JUDGE_PATH)
bbh = importlib.util.module_from_spec(_spec)
sys.modules["bbh_gpt_judge"] = bbh
_spec.loader.exec_module(bbh)


def _manifest():
    return [
        {"UUID": "navigate-0", "CorpusName": "navigate", "Transcription": "Yes", "Question": "Q0"},
        {"UUID": "navigate-1", "CorpusName": "navigate", "Transcription": "No", "Question": "Q1"},
        {"UUID": "web_of_lies-0", "CorpusName": "web_of_lies", "Transcription": "Yes", "Question": "Q2"},
        {"UUID": "web_of_lies-1", "CorpusName": "web_of_lies", "Transcription": "No", "Question": "Q3"},
    ]


def test_format_judge_inputs_schema():
    manifest = _manifest()
    predictions = {
        "navigate-0": "The answer is Yes",
        "navigate-1": "No",
        "web_of_lies-0": "Yes",
        "web_of_lies-1": "I think Yes",
    }
    items = bbh.format_judge_inputs(manifest, predictions)

    assert len(items) == 4
    first = items[0]
    assert set(first.keys()) == {"conversation_id", "CorpusName", "rounds"}
    assert first["conversation_id"] == "navigate-0"
    assert first["CorpusName"] == "navigate"
    round0 = first["rounds"][0]
    assert set(round0.keys()) == {"round", "question", "ground_truth", "model_reply_text"}
    assert round0["round"] == 1
    assert round0["question"] == "Q0"
    assert round0["ground_truth"] == "Yes"
    assert round0["model_reply_text"] == "The answer is Yes"


def test_format_judge_inputs_alignment_error():
    manifest = _manifest()
    predictions = {"navigate-0": "Yes"}  # missing 3 UUIDs
    with pytest.raises(ValueError, match="alignment failed"):
        bbh.format_judge_inputs(manifest, predictions)


def test_parse_verdict():
    assert bbh.parse_verdict("CORRECT") == "CORRECT"
    assert bbh.parse_verdict("incorrect") == "INCORRECT"
    # "INCORRECT" contains the substring "CORRECT" -> INCORRECT must win.
    assert bbh.parse_verdict("The answer is INCORRECT") == "INCORRECT"
    assert bbh.parse_verdict("unsure") == "NONE"


def test_score_verdicts_accuracy_math():
    manifest = _manifest()
    predictions = {r["UUID"]: "reply" for r in manifest}
    items = bbh.format_judge_inputs(manifest, predictions)

    # Mock the GPT client: correct iff the ground-truth answer is "Yes" (2 of 4 items).
    def mock_judge(ground_truth, reply):
        return "CORRECT" if ground_truth == "Yes" else "INCORRECT"

    verdicts = bbh.run_judge(items, mock_judge, workers=2, cache_path=None)
    summary = bbh.score_verdicts(items, verdicts)

    assert summary["n"] == 4
    assert summary["correct"] == 2
    assert summary["accuracy"] == 50.0
    assert summary["none"] == 0
    assert summary["errors"] == 0
    # Per-corpus: each corpus has one "Yes" and one "No" -> 50% each.
    assert summary["by_corpus"]["navigate"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert summary["by_corpus"]["web_of_lies"] == {"correct": 1, "total": 2, "accuracy": 50.0}


def test_score_verdicts_counts_none_and_errors():
    manifest = _manifest()
    predictions = {r["UUID"]: "reply" for r in manifest}
    items = bbh.format_judge_inputs(manifest, predictions)
    verdicts = {
        "navigate-0": "CORRECT",
        "navigate-1": "NONE",
        "web_of_lies-0": "CORRECT",
        # web_of_lies-1 intentionally absent -> counted as ERROR.
    }
    summary = bbh.score_verdicts(items, verdicts)
    assert summary["correct"] == 2
    assert summary["none"] == 1
    assert summary["errors"] == 1
    assert summary["accuracy"] == 50.0


def test_run_judge_cache_resume(tmp_path):
    manifest = _manifest()
    predictions = {r["UUID"]: "reply" for r in manifest}
    items = bbh.format_judge_inputs(manifest, predictions)
    cache = tmp_path / "cache.jsonl"

    calls = {"n": 0}

    def counting_judge(ground_truth, reply):
        calls["n"] += 1
        return "CORRECT"

    bbh.run_judge(items, counting_judge, workers=1, cache_path=str(cache))
    assert calls["n"] == 4
    assert cache.exists()

    # Second pass must reuse the cache and issue zero new judge calls.
    calls["n"] = 0
    verdicts = bbh.run_judge(items, counting_judge, workers=1, cache_path=str(cache))
    assert calls["n"] == 0
    assert all(v == "CORRECT" for v in verdicts.values())
