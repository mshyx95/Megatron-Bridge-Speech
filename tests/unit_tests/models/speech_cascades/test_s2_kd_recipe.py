"""Unit tests for the stage-2 KD+MTP speech-cascades recipe (``speech_cascades_s2_kd_config``).

Patterned after the other recipe tests: load the recipe module without the recipes aggregator and
monkeypatch ``AutoBridge`` to a lightweight fake that returns a minimal provider recording attribute
writes. Asserts the freeze flags (encoder + LLM frozen, projector trainable), the KD + MTP keys, and
the two-phase (warm -> full) iteration / init-checkpoint wiring.
"""

import importlib
import os
import sys
import types

import pytest


torch = pytest.importorskip("torch")


class _FakeModelCfg:
    """Minimal provider stand-in that accepts the attribute writes the recipe performs."""

    def finalize(self):
        return None


class _FakeBridge:
    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


def _load_recipe_module():
    import megatron.bridge

    if "megatron.bridge.recipes" not in sys.modules:
        stub = types.ModuleType("megatron.bridge.recipes")
        stub.__path__ = [os.path.join(os.path.dirname(megatron.bridge.__file__), "recipes")]
        sys.modules["megatron.bridge.recipes"] = stub
    return importlib.import_module("megatron.bridge.recipes.speech_cascades.speech_cascades")


def _build(monkeypatch, **kwargs):
    module = _load_recipe_module()
    monkeypatch.setattr(module, "AutoBridge", _FakeBridge)
    return module.speech_cascades_s2_kd_config(
        hf_path="/fake/stage1_hf",
        streaming_config_yaml="/fake/data_en_s2_kd.yaml",
        **kwargs,
    )


def test_s2_kd_recipe_builds_config_and_sets_freeze_and_kd_keys(monkeypatch):
    cfg = _build(monkeypatch, phase="warm", load_hf_weights=True)

    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.optimizer is not None and cfg.scheduler is not None
    assert cfg.dataset is not None and cfg.dataset.seq_length == 8192

    model = cfg.model
    # Encoder + LLM frozen; the projector (audio_model, minus encoder) and MTP heads stay trainable.
    assert model.freeze_language_model is True
    assert model.freeze_audio_encoder is True
    assert model.freeze_audio_model is False

    # KD hyperparameters set explicitly (not left to step.py getattr defaults).
    assert model.knowledge_distillation is True
    assert model.distillation_weight == 0.75
    assert model.distillation_temperature == 2.0
    assert model.distillation_kl_chunk_size == 128

    # MTP depth heads + per-depth self-distillation loss.
    assert model.mtp_num_hidden_layers == 4
    assert model.mtp_loss_weight == 1.0
    assert model.mtp_loss_depth_weights == [0.50, 0.25, 0.15, 0.10]
    assert model.mtp_distillation_weight == 0.75
    assert model.mtp_init_extra_layers_from_first is True

    # Full recompute so the frozen LLM activations do not blow up peak memory.
    assert model.recompute_granularity == "full"


def test_s2_kd_warm_phase_iterations(monkeypatch):
    cfg = _build(monkeypatch, phase="warm", warm_iters=500, load_hf_weights=True)
    assert cfg.train.train_iters == 500
    # Warm phase cold-starts from the stage-1 HF weights, no Megatron pretrained checkpoint.
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_s2_kd_full_phase_inits_from_warm_checkpoint(monkeypatch):
    cfg = _build(
        monkeypatch,
        phase="full",
        full_iters=4000,
        init_checkpoint="/fake/warm/checkpoints/iter_0000500",
    )
    assert cfg.train.train_iters == 4000
    # Full phase initializes weights from the warm checkpoint (optimizer / step reset).
    assert cfg.checkpoint.pretrained_checkpoint == "/fake/warm/checkpoints/iter_0000500"


def test_s2_kd_train_iters_override(monkeypatch):
    cfg = _build(monkeypatch, phase="full", full_iters=4000, train_iters=2, load_hf_weights=True)
    assert cfg.train.train_iters == 2


def test_s2_kd_rejects_unknown_phase(monkeypatch):
    with pytest.raises(ValueError):
        _build(monkeypatch, phase="bogus", load_hf_weights=True)
