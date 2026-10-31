import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_qsa_module():
    path = Path(__file__).parents[4] / "src/megatron/bridge/models/qwen4_exp/modeling_qwen4_exp/qsa_indexer.py"
    spec = importlib.util.spec_from_file_location("qwen4_exp_qsa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_qsa_mask_is_causal_when_budget_covers_sequence() -> None:
    module = _load_qsa_module()
    config = SimpleNamespace(
        hidden_size=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=2,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )
    indexer = module.Qwen4ExpQSAIndexer(config, layer_number=1)

    mask = indexer(torch.randn(6, 1, 8), rotary_pos_emb=None, attention_mask=None)

    expected = ~torch.ones((6, 6), dtype=torch.bool).tril()
    assert torch.equal(mask[0, 0], expected)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qsa_beyond_budget_matches_hf(seed: int, dtype: torch.dtype) -> None:
    reference = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    module = _load_qsa_module()
    config = SimpleNamespace(
        hidden_size=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        layernorm_epsilon=1e-6,
        rms_norm_eps=1e-6,
        sequence_parallel=False,
    )
    torch.manual_seed(seed)
    indexer = module.Qwen4ExpQSAIndexer(config, layer_number=1).to(dtype)
    hf_indexer = reference.Qwen4ExpTextQSAIndexer(config, layer_idx=0).to(dtype)
    with torch.no_grad():
        indexer.q_layernorm.weight.uniform_(-0.4, 0.4)
        indexer.k_layernorm.weight.uniform_(-0.4, 0.4)
    hf_indexer.load_state_dict(indexer.state_dict())
    hidden = (torch.randn(2, 13, 8) * torch.arange(1, 14)[None, :, None]).to(dtype)
    frequencies = torch.randn(13, 2, 1, 2).repeat(1, 1, 1, 2)
    positions = frequencies[:, :, 0].transpose(0, 1)
    visible = torch.ones(2, 1, 13, 13, dtype=torch.bool).tril()
    visible[1, :, :, :3] = False

    with torch.no_grad():
        expected = hf_indexer(hidden, (positions.cos().to(dtype), positions.sin().to(dtype)), visible, None)
        actual = indexer(hidden.transpose(0, 1), frequencies, ~visible)

    assert torch.equal(actual, ~expected)
    assert ((~actual).sum(dim=-1) <= config.indexer_budget + config.indexer_compress_ratio - 1).all()


@pytest.mark.unit
def test_qsa_core_honors_selected_keys_and_backpropagates() -> None:
    from megatron.bridge.models.qwen4_exp.modeling_qwen4_exp.qsa import Qwen4ExpDotProductAttention

    torch.manual_seed(42)
    core = Qwen4ExpDotProductAttention(SimpleNamespace(attention_dropout=0.0))
    query = torch.randn(5, 2, 2, 4, requires_grad=True)
    key = torch.randn(5, 2, 1, 4, requires_grad=True)
    value = torch.randn(5, 2, 1, 4, requires_grad=True)
    excluded = ~torch.ones(2, 1, 5, 5, dtype=torch.bool).tril()
    excluded[:, :, 2:, 1] = True
    scores = torch.einsum("sbhd,tbhd->bhst", query, key.expand(-1, -1, 2, -1)) / 2
    probabilities = scores.masked_fill(excluded, float("-inf")).softmax(dim=-1)
    expected = torch.einsum("bhst,tbhd->sbhd", probabilities, value.expand(-1, -1, 2, -1)).flatten(-2)

    actual = core(query, key, value, excluded)

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
