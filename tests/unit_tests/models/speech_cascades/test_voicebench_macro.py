"""Unit tests for the VoiceBench 9-task macro scorer (CPU, no model, no GPU)."""

import importlib.util
import math
import os

import pytest


# Load eval_voicebench.py directly (examples/ is not an importable package).
_EVAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples", "models", "speech_cascades", "eval_voicebench.py"
)
_spec = importlib.util.spec_from_file_location("eval_voicebench", os.path.abspath(_EVAL_PATH))
eval_voicebench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_voicebench)


# Hand-made per-subset metric dicts in the exact shape each official VoiceBench evaluator returns.
# Values are taken from the S2T_COMPARE_RESULTS.md "Qwen3.5-4B speech encoder + MLP / step 2000" row,
# whose published macro is 74.16.
_REFERENCE_METRICS = {
    "advbench": {"refusal_rate": 0.9962},
    "alpacaeval": {"gpt": 4.51},
    "bbh": {"acc": 65.10},
    "commoneval": {"gpt": 4.28},
    "ifeval": {"strict-prompt": 0.60, "loose-prompt": 0.62, "final": 0.6108},
    "mmsu": {"acc": 59.27, "fail": 1.0},
    "openbookqa": {"acc": 77.80, "fail": 0.0},
    "sd-qa": {"gpt": 44.12, "panda": None},
    "wildvoice": {"gpt": 4.24},
}


def test_open_subsets_scale_1_to_5_by_20():
    assert eval_voicebench.subset_headline_metric("alpacaeval", {"gpt": 4.51}) == pytest.approx(90.2)
    assert eval_voicebench.subset_headline_metric("commoneval", {"gpt": 4.28}) == pytest.approx(85.6)
    assert eval_voicebench.subset_headline_metric("wildvoice", {"gpt": 4.24}) == pytest.approx(84.8)


def test_ifeval_and_harm_scale_0_to_1_by_100():
    assert eval_voicebench.subset_headline_metric("ifeval", {"final": 0.6108}) == pytest.approx(61.08)
    assert eval_voicebench.subset_headline_metric("advbench", {"refusal_rate": 0.9962}) == pytest.approx(99.62)


def test_mcq_bbh_and_qa_pass_through_as_percentages():
    assert eval_voicebench.subset_headline_metric("mmsu", {"acc": 59.27, "fail": 1.0}) == pytest.approx(59.27)
    assert eval_voicebench.subset_headline_metric("bbh", {"acc": 65.10}) == pytest.approx(65.10)
    assert eval_voicebench.subset_headline_metric("sd-qa", {"gpt": 44.12}) == pytest.approx(44.12)


def test_macro_matches_reference_aggregation():
    summary = eval_voicebench.compute_macro(_REFERENCE_METRICS)
    # Unweighted mean of the nine normalized headline percentages.
    expected_headlines = {
        "advbench": 99.62,
        "alpacaeval": 90.2,
        "bbh": 65.10,
        "commoneval": 85.6,
        "ifeval": 61.08,
        "mmsu": 59.27,
        "openbookqa": 77.80,
        "sd-qa": 44.12,
        "wildvoice": 84.8,
    }
    for subset, value in expected_headlines.items():
        assert summary["per_subset"][subset] == pytest.approx(value)
    assert summary["num_subsets"] == 9
    assert summary["macro"] == pytest.approx(sum(expected_headlines.values()) / 9)
    # Matches the published macro (74.16) up to display rounding of the source table values.
    assert math.isclose(summary["macro"], 74.16, abs_tol=0.05)


def test_macro_is_unweighted_mean_of_equal_scores():
    metrics = {
        "alpacaeval": {"gpt": 5.0},
        "commoneval": {"gpt": 5.0},
        "wildvoice": {"gpt": 5.0},
        "sd-qa": {"gpt": 100.0},
        "mmsu": {"acc": 100.0},
        "openbookqa": {"acc": 100.0},
        "ifeval": {"final": 1.0},
        "advbench": {"refusal_rate": 1.0},
        "bbh": {"acc": 100.0},
    }
    summary = eval_voicebench.compute_macro(metrics)
    assert summary["macro"] == pytest.approx(100.0)


def test_macro_rejects_missing_or_extra_subsets():
    incomplete = {k: v for k, v in _REFERENCE_METRICS.items() if k != "bbh"}
    with pytest.raises(ValueError, match="missing"):
        eval_voicebench.compute_macro(incomplete)
    extra = dict(_REFERENCE_METRICS, extra_subset={"acc": 1.0})
    with pytest.raises(ValueError, match="unexpected"):
        eval_voicebench.compute_macro(extra)


def test_headline_extraction_rejects_missing_key():
    with pytest.raises(KeyError, match="headline key"):
        eval_voicebench.subset_headline_metric("mmsu", {"gpt": 4.0})


def test_all_nine_subsets_are_covered():
    assert set(eval_voicebench.SUBSET_EVALUATORS) == set(eval_voicebench.SUBSET_HEADLINE)
    assert set(eval_voicebench.CORPUS_TO_SUBSET.values()) == set(eval_voicebench.SUBSET_EVALUATORS)
    assert len(eval_voicebench.SUBSET_EVALUATORS) == 9
