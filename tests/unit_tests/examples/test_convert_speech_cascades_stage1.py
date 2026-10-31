import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def converter():
    path = Path(__file__).resolve().parents[3] / "examples/conversion/convert_speech_cascades_stage1_to_hf.py"
    spec = importlib.util.spec_from_file_location("speech_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize("blocks", [23, 24])
def test_encoder_depth_and_cmvn(converter, tmp_path, blocks):
    checkpoint = tmp_path / "encoder.pt"
    weights = {f"encoder.encoders.{index}.weight": torch.tensor([index]) for index in range(25)}
    weights["decoder.weight"] = torch.ones(1)
    weights["encoder_embedding.global_mean"] = torch.zeros(80)
    weights["encoder_embedding.global_invstd"] = torch.ones(80)
    torch.save(weights, checkpoint)
    np.savetxt(tmp_path / "mean.vec", np.full(80, 2.0))
    np.savetxt(tmp_path / "invstd.vec", np.full(80, 3.0))
    mapped = converter._load_encoder(
        checkpoint, num_blocks=blocks, mean_file=tmp_path / "mean.vec", invstd_file=tmp_path / "invstd.vec",
    )
    prefix = "model.embed_tokens_extend.encoder."
    assert len([key for key in mapped if ".encoders." in key]) == blocks
    assert prefix + f"encoders.{blocks - 1}.weight" in mapped
    assert prefix + f"encoders.{blocks}.weight" not in mapped
    assert torch.equal(mapped[prefix + "encoder_embedding.global_mean"], torch.full((80,), 2.0).double())
    assert not any("decoder." in key for key in mapped)


@pytest.mark.unit
@pytest.mark.parametrize("tied", [False, True])
def test_config_preserves_backbone_and_encoder(converter, tmp_path, tied):
    base, output = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    output.mkdir()
    (base / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 2560}, "tie_word_embeddings": tied}))
    assert converter._write_config(base, output, num_blocks=24) == 2560
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["audio_processor"]["config"]["num_blocks"] == 24
    assert cfg["tie_word_embeddings"] is tied
    assert converter._STAGE1_AUDIO_CONFIG["num_blocks"] == 23