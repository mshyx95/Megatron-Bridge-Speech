"""Unit tests for the Fleurs translation BLEU scorer (``eval_fleurs.FleursBleuScorer``).

Loads the ``examples/models/speech_cascades/eval_fleurs.py`` module by file path (its top level is
GPU-free: torch/megatron are imported lazily), then checks that ``FleursBleuScorer`` computes exactly
the same corpus BLEU/chrF as calling ``sacrebleu`` directly, and that the target-language tokenizer
selection matches the reference ``SacrebleuScorer`` (zh -> "zh", ja -> "ja-mecab", else "13a").
"""

import importlib.util
import pathlib

import pytest


sacrebleu = pytest.importorskip("sacrebleu")

_EVAL_FLEURS = (
    pathlib.Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "speech_cascades"
    / "eval_fleurs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_fleurs_under_test", _EVAL_FLEURS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_fleurs = _load_module()


# Tiny hand-made hypothesis/reference sample.
_REFS = [
    "the cat sat on the mat",
    "a quick brown fox jumps over the lazy dog",
    "hello world",
]
_HYPS = [
    "the cat sat on the mat",
    "a quick brown fox jumped over the lazy dog",
    "hello there world",
]


def test_bleu_matches_sacrebleu_reference():
    scorer = eval_fleurs.FleursBleuScorer("DE")  # non zh/ja -> "13a"
    for ref, hyp in zip(_REFS, _HYPS):
        scorer.add_string(ref, hyp)
    bleu, chrf = scorer.score()

    expected_bleu = sacrebleu.BLEU(tokenize="13a").corpus_score(_HYPS, [_REFS]).score
    expected_chrf = sacrebleu.CHRF().corpus_score(_HYPS, [_REFS]).score

    assert bleu == pytest.approx(expected_bleu, abs=1e-6)
    assert chrf == pytest.approx(expected_chrf, abs=1e-6)
    assert 0.0 < bleu < 100.0


def test_perfect_translation_is_100_bleu():
    scorer = eval_fleurs.FleursBleuScorer("FR")
    for ref in _REFS:
        scorer.add_string(ref, ref)
    bleu, _ = scorer.score()
    assert bleu == pytest.approx(100.0, abs=1e-6)


@pytest.mark.parametrize(
    "target,expected_tokenize",
    [("EN", "13a"), ("DE", "13a"), ("es", "13a"), ("ZH", "zh"), ("zh", "zh"), ("JA", "ja-mecab")],
)
def test_tokenizer_selection_matches_reference(target, expected_tokenize):
    assert eval_fleurs.FleursBleuScorer(target).tokenize == expected_tokenize
