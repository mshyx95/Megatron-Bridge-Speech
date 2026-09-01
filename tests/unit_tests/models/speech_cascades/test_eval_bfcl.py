"""Unit tests for the BFCL function-call micro-accuracy scorer (CPU, mocked, no GPU/model)."""

import importlib.util
from pathlib import Path

import pytest


_EVAL_BFCL = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "speech_cascades"
    / "eval_bfcl.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_bfcl", _EVAL_BFCL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bfcl = _load_module()


# --- decode_calls -------------------------------------------------------------------------------
def test_decode_calls_parses_single_and_multiple():
    assert bfcl.decode_calls("[get_weather(city='Paris')]") == [{"get_weather": {"city": "Paris"}}]
    assert bfcl.decode_calls("get_weather(city='Paris')") == [{"get_weather": {"city": "Paris"}}]
    assert bfcl.decode_calls("[a(x=1), b(y=2)]") == [{"a": {"x": 1}}, {"b": {"y": 2}}]


def test_decode_calls_returns_empty_for_non_calls():
    assert bfcl.decode_calls("I cannot help with that.") == []
    assert bfcl.decode_calls("[]") == []
    assert bfcl.decode_calls("") == []
    assert bfcl.decode_calls("get_weather(city=") == []  # syntax error -> no call


# --- call matching ------------------------------------------------------------------------------
def test_match_single_name_and_value():
    gold = {"get_weather": {"city": ["Paris", "paris"]}}
    assert bfcl._match_single({"get_weather": {"city": "Paris"}}, gold) is True
    # string standardization: case/punctuation-insensitive
    assert bfcl._match_single({"get_weather": {"city": "PARIS"}}, gold) is True
    # wrong function name
    assert bfcl._match_single({"get_forecast": {"city": "Paris"}}, gold) is False
    # value not in the possible-answer set
    assert bfcl._match_single({"get_weather": {"city": "London"}}, gold) is False


def test_match_single_optional_and_unexpected_params():
    gold = {"f": {"a": ["1"], "b": ["", "hi"]}}  # b is optional ("" allowed)
    assert bfcl._match_single({"f": {"a": "1"}}, gold) is True  # omit optional b
    assert bfcl._match_single({"f": {"a": "1", "b": "hi"}}, gold) is True
    # unexpected param not in gold
    assert bfcl._match_single({"f": {"a": "1", "c": "x"}}, gold) is False
    # missing required param a
    assert bfcl._match_single({"f": {"b": "hi"}}, gold) is False


def test_match_call_sets_is_order_independent():
    gold = [{"a": {"x": [1]}}, {"b": {"y": [2]}}]
    assert bfcl._match_call_sets([{"b": {"y": 2}}, {"a": {"x": 1}}], gold) is True
    assert bfcl._match_call_sets([{"a": {"x": 1}}], gold) is False  # wrong count
    assert bfcl._match_call_sets([{"a": {"x": 1}}, {"b": {"y": 99}}], gold) is False  # value mismatch


def test_int_float_leniency():
    gold = {"f": {"x": [1.0]}}
    assert bfcl._match_single({"f": {"x": 1}}, gold) is True
    # bool must not match numeric possible answers
    assert bfcl._match_single({"f": {"x": True}}, {"f": {"x": [1]}}) is False


# --- score_example per category -----------------------------------------------------------------
def test_score_example_irrelevance():
    # irrelevance is correct iff NO function call is produced
    assert bfcl.score_example("Sorry, I can't do that.", None, "irrelevance") is True
    assert bfcl.score_example("[do_thing(a=1)]", None, "irrelevance") is False


def test_score_example_simple_and_parallel():
    assert bfcl.score_example("[f(a=1)]", [{"f": {"a": [1]}}], "simple") is True
    assert bfcl.score_example("[f(a=2)]", [{"f": {"a": [1]}}], "simple") is False
    assert (
        bfcl.score_example("[f(a=1), g(b='x')]", [{"g": {"b": ["x"]}}, {"f": {"a": [1]}}], "parallel")
        is True
    )


# --- micro accuracy math ------------------------------------------------------------------------
def test_score_micro_accuracy_pools_examples():
    examples = [
        # simple: 1 correct, 1 wrong
        ("[f(a=1)]", [{"f": {"a": [1]}}], "simple"),
        ("[f(a=9)]", [{"f": {"a": [1]}}], "simple"),
        # parallel: 1 correct
        ("[a(x=1), b(y=2)]", [{"a": {"x": [1]}}, {"b": {"y": [2]}}], "parallel"),
        # irrelevance: 1 correct (no call), 1 wrong (emitted a call)
        ("no tool needed", None, "irrelevance"),
        ("[c(z=3)]", None, "irrelevance"),
    ]
    summary = bfcl.score_micro_accuracy(examples)

    assert summary["total"] == 5
    assert summary["correct"] == 3
    assert summary["micro_accuracy"] == pytest.approx(3 / 5)
    # per-category breakdown
    assert summary["per_category"]["simple"] == {"total": 2, "correct": 1, "accuracy": pytest.approx(0.5)}
    assert summary["per_category"]["parallel"]["accuracy"] == pytest.approx(1.0)
    assert summary["per_category"]["irrelevance"] == {
        "total": 2,
        "correct": 1,
        "accuracy": pytest.approx(0.5),
    }


def test_score_micro_accuracy_empty():
    summary = bfcl.score_micro_accuracy([])
    assert summary["total"] == 0
    assert summary["micro_accuracy"] == 0.0


# --- report end-to-end on mocked files ----------------------------------------------------------
def test_report_reads_responses_and_ground_truth(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "possible_answer").mkdir(parents=True)
    (data_dir / "possible_answer" / "BFCL_v3_simple.json").write_text(
        '{"id": "simple_0", "ground_truth": [{"f": {"a": [1]}}]}\n'
        '{"id": "simple_1", "ground_truth": [{"f": {"a": [2]}}]}\n',
        encoding="utf-8",
    )
    responses = tmp_path / "responses.json"
    responses.write_text(
        '[{"audio_ids": ["simple_0"], "generated_texts": ["[f(a=1)]"]},'
        ' {"audio_ids": ["simple_1"], "generated_texts": ["[f(a=99)]"]}]',
        encoding="utf-8",
    )
    summary = bfcl.report(responses, data_dir, tmp_path / "report")

    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["micro_accuracy"] == pytest.approx(0.5)
    written = (tmp_path / "report" / "bfcl_micro_accuracy.json").read_text(encoding="utf-8")
    assert '"micro_accuracy"' in written
