import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml
from safetensors.torch import load_file, save_file


pytestmark = pytest.mark.unit
ENTRY = Path(__file__).resolve().parents[4] / "examples/models/speech_cascades"


@pytest.fixture
def exporter():
    spec = importlib.util.spec_from_file_location("qwen4_audio_export", ENTRY / "export_qwen4_audio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    base, saved, output = (tmp_path / name for name in ("base", "iter_0002000", "output"))
    base.mkdir()
    text = {
        "hidden_size": 4,
        "model_type": "qwen4_exp_text",
        "mtp_num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "vocab_size": 16,
    }
    (base / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}))
    weight_map = {
        "lm_head.weight": "base.safetensors",
        "model.language_model.layers.0.weight": "base.safetensors",
        "model.visual.weight": "base.safetensors",
        "mtp.layers.0.weight": "base.safetensors",
    }
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "base.safetensors").write_bytes(b"immutable source")
    (base / "tokenizer.json").write_text("{}")
    state = {
        "audio_model.encoder.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "audio_model.audio_projection.0.weight": torch.ones(4, 2, dtype=torch.bfloat16),
        "language_model.weight": torch.zeros(4, 4),
    }
    dcp.save(state, checkpoint_id=saved, no_dist=True)
    run = {
        "model": {
            "_target_": "megatron.bridge.models.qwen4_exp.speech.Qwen4ExpSpeechProvider",
            "freeze_language_model": True,
            "freeze_audio_encoder": True,
            "freeze_audio_model": False,
            "mtp_num_layers": 0,
            "mtp_num_hidden_layers": 0,
            "mtp_enabled": False,
            "hf_text_config": {**text, "layer_types": ["qwen_sparse_attention"], "return_dict": True},
            "audio_encoder_config": {"time_reduction": 8},
            "audio_projection_cls": "mlp",
            "audio_compression_rate": 8,
            "audio_downsample_rate": 1,
        }
    }
    (saved / "run_config.yaml").write_text(yaml.safe_dump(run))
    data_config = tmp_path / "data.yaml"
    data_config.write_text("streaming_compression: true\ncompression_rate: 8\n")
    args = {
        "base": base,
        "checkpoint": saved,
        "data_config": data_config,
        "output": output,
        "base_config_sha256": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(),
        "base_index_sha256": hashlib.sha256((base / "model.safetensors.index.json").read_bytes()).hexdigest(),
    }
    return args, state, run


def test_audio_export_loads_trained_tensors_and_preserves_frozen_shards(exporter, checkpoint):
    args, state, _ = checkpoint
    exporter.export_checkpoint(**args)
    output = args["output"]
    audio = load_file(output / "speech.safetensors")
    assert set(audio) == {
        "model.embed_tokens_extend.encoder.weight",
        "model.embed_tokens_extend.audio_projection.0.weight",
    }
    for name, value in audio.items():
        assert torch.equal(value, state[name.replace("model.embed_tokens_extend.", "audio_model.")])
    assert (output / "base.safetensors").is_symlink()
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["Qwen4ExpAudioForCausalLM"]
    assert config["mtp_num_hidden_layers"] == config["text_config"]["mtp_num_hidden_layers"] == 0
    assert config["text_config"]["mtp"]["num_hidden_layers"] == 0
    assert config["speech_streaming_compression"] is True
    weights = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any("visual" in name or "mtp" in name for name in weights)
    assert not json.loads((output / "export_receipt.json").read_text())["mtp"]
    with pytest.raises(FileExistsError):
        exporter.export_checkpoint(**args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("freeze_language_model", False),
        ("freeze_audio_encoder", False),
        ("freeze_audio_model", True),
        ("mtp_num_layers", 4),
    ],
)
def test_audio_export_rejects_non_projector_only_training(exporter, checkpoint, field, value):
    args, _, run = checkpoint
    run["model"][field] = value
    (args["checkpoint"] / "run_config.yaml").write_text(yaml.safe_dump(run))
    with pytest.raises(ValueError):
        exporter.export_checkpoint(**args)
    assert not args["output"].exists()


def test_audio_export_rejects_wrong_base_identity(exporter, checkpoint):
    args, _, _ = checkpoint
    args["base_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        exporter.export_checkpoint(**args)


def test_audio_export_rejects_changed_backbone_configuration(exporter, checkpoint):
    args, _, run = checkpoint
    run["model"]["hf_text_config"]["hidden_size"] = 8
    (args["checkpoint"] / "run_config.yaml").write_text(yaml.safe_dump(run))
    with pytest.raises(ValueError, match="text configuration"):
        exporter.export_checkpoint(**args)


def test_audio_export_rejects_missing_encoder(exporter, tmp_path):
    dcp.save({"audio_model.audio_projection.weight": torch.ones(2)}, checkpoint_id=tmp_path, no_dist=True)
    with pytest.raises(ValueError, match="both"):
        exporter.load_audio_state(tmp_path)


@pytest.mark.parametrize("corruption", [None, "audio_sha256", "metadata_sha256", "shape", "dtype"])
def test_audio_export_checks_staged_extraction(exporter, checkpoint, tmp_path, corruption):
    args, state, _ = checkpoint
    audio = tmp_path / "audio.safetensors"
    tensors = {
        name.replace("audio_model.", "model.embed_tokens_extend.", 1): tensor
        for name, tensor in state.items()
        if name.startswith("audio_model.")
    }
    if corruption == "shape":
        tensors["model.embed_tokens_extend.encoder.weight"] = torch.zeros(4, 2)
    if corruption == "dtype":
        tensors["model.embed_tokens_extend.encoder.weight"] = tensors["model.embed_tokens_extend.encoder.weight"].to(
            torch.float16
        )
    save_file(tensors, audio)
    receipt = {
        "metadata_sha256": hashlib.sha256((args["checkpoint"] / ".metadata").read_bytes()).hexdigest(),
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
    }
    if corruption in receipt:
        receipt[corruption] = "incorrect"
    receipt_path = tmp_path / "extraction.json"
    receipt_path.write_text(json.dumps(receipt))
    if corruption is not None:
        with pytest.raises(ValueError, match="extraction receipt|keys, shapes or dtypes"):
            exporter.export_checkpoint(**args, audio_file=audio, audio_receipt=receipt_path)
        assert not args["output"].exists()
    else:
        exporter.export_checkpoint(**args, audio_file=audio, audio_receipt=receipt_path)
        exported = load_file(args["output"] / "speech.safetensors")
        for name, tensor in tensors.items():
            assert torch.equal(tensor, exported[name])


@pytest.mark.parametrize("missing", ["audio_file", "audio_receipt"])
def test_audio_export_requires_both_staged_inputs(exporter, checkpoint, tmp_path, missing):
    args, _, _ = checkpoint
    staged = {"audio_file": tmp_path / "audio", "audio_receipt": tmp_path / "receipt"}
    del staged[missing]
    with pytest.raises(ValueError, match="both"):
        exporter.export_checkpoint(**args, **staged)
    assert not args["output"].exists()


def test_audio_export_rejects_unsupported_encoder_before_writing(exporter, checkpoint):
    args, _, run = checkpoint
    run["model"]["audio_encoder_config"]["use_flashattention_t5bias"] = True
    (args["checkpoint"] / "run_config.yaml").write_text(yaml.safe_dump(run))
    with pytest.raises(ValueError, match="FlashAttention"):
        exporter.export_checkpoint(**args)
    assert not args["output"].exists()
