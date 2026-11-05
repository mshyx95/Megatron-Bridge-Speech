import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_qsa import _load_qsa_module


pytestmark = pytest.mark.unit
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
]


@contextmanager
def _matmul_tf32(enabled: bool) -> Iterator[None]:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _indexers(device, dtype, budget):
    reference = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    module = _load_qsa_module()
    config = SimpleNamespace(
        hidden_size=24,
        indexer_n_heads=3,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=budget,
        indexer_compress_ratio=2,
        layernorm_epsilon=1e-6,
        rms_norm_eps=1e-6,
        sequence_parallel=False,
    )
    torch.manual_seed(423)
    indexer = module.Qwen4ExpQSAIndexer(config, 1).to(device=device, dtype=dtype)
    oracle = reference.Qwen4ExpTextQSAIndexer(config, layer_idx=0).to(device=device, dtype=dtype)
    with torch.no_grad():
        indexer.q_layernorm.weight.uniform_(-0.4, 0.4)
        indexer.k_layernorm.weight.uniform_(-0.4, 0.4)
    oracle.load_state_dict(indexer.state_dict(), strict=True)
    module._QUERY_CHUNK_SIZE = 7
    return module, indexer, oracle


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("seed", [17, 43])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_production_geometry_repair_preserves_exact_scores_and_masks(device, tf32, seed, source_dtype):
    module = _load_qsa_module()
    torch.manual_seed(seed)
    query = torch.randn(129, 4, 128, device=device).to(source_dtype).float()
    keys = torch.randn(2048, 128, device=device).to(source_dtype).float()
    rows = [[index, 513 + (997 * index) % 1536] for index in range(129)]
    blocks = torch.arange(2048 * 4, device=device).reshape(2048, 4)
    selected = torch.rand(129, 2048 * 4 + 3, device=device) > 0.3
    expected = selected.clone()
    with _matmul_tf32(tf32):
        scores = module._reference_scores(query, keys, rows, 2048, 128)
        for index, (row, count) in enumerate(rows):
            reference = (query[row] @ keys[:count].T).T.relu().sum(-1) / math.sqrt(128)
            torch.testing.assert_close(scores[index, :count], reference, rtol=0, atol=0)
        _scalar_repair(query, keys, blocks, expected, rows, 512, 128)
        module._repair_causal_rows(query, keys, blocks, selected, rows, 512, 128)
    assert torch.equal(selected, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["random", "near", "zero", "subnormal", "overflow"])
def test_absolute_product_guard_matches_scalar_selection_at_production_budget(device, dtype, kind):
    module = _load_qsa_module()
    config = SimpleNamespace(
        hidden_size=24,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
    )
    indexer = module.Qwen4ExpQSAIndexer(config, 1).to(device=device, dtype=dtype)
    length = 2183
    count = length // 4
    torch.manual_seed(173)
    query = torch.randn(length, 1, 4, 128, device=device, dtype=dtype)
    keys = torch.randn(count, 128, device=device, dtype=dtype)
    if kind == "near":
        keys = keys[:1].expand_as(keys).clone()
        keys *= 1 + torch.arange(count, device=device, dtype=dtype)[:, None] * 1e-7
    elif kind == "zero":
        keys.zero_()
    elif kind == "subnormal":
        query *= 1e-20
        keys *= 1e-20
    elif kind == "overflow":
        query *= 1e20
        keys *= 1e20
    visible = torch.ones(1, length, length, device=device, dtype=torch.bool).tril()
    expected = visible[0].clone()
    blocks = torch.arange(count * 4, device=device).reshape(count, 4)
    rows = [[row, (row + 1) // 4] for row in range((512 + 1) * 4 - 1, length)]
    with (
        _matmul_tf32(False),
        patch.object(indexer, "_pool_keys", return_value=keys),
    ):
        _scalar_repair(query[:, 0].float(), keys.float(), blocks, expected, rows, 512, 128)
        actual = indexer._select_causal_keys(
            query, torch.empty(length, 1, 128, device=device, dtype=dtype), None, visible
        )
    assert torch.equal(actual[0], expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", [0, 1, 5, 64])
@pytest.mark.parametrize(
    "mask_kind", ["none", "padding2d", "bool4d", "additive4d", "broadcast4d", "query_dependent", "all_masked"]
)
def test_batch_qsa_matches_hf_masks(device, dtype, budget, mask_kind):
    _, indexer, oracle = _indexers(device, dtype, budget)
    length, batch = 37, 2
    hidden = torch.randn(batch, length, 24, device=device, dtype=dtype)
    rope_batch = 1 if mask_kind == "broadcast4d" else batch
    rope = torch.randn(length, rope_batch, 1, 2, device=device).repeat(1, 1, 1, 2)
    positions = rope[:, :, 0].transpose(0, 1).expand(batch, -1, -1)
    visible = torch.ones(batch, 1, length, length, device=device, dtype=torch.bool).tril()
    mask = None
    if mask_kind != "none":
        valid = torch.ones(batch, length, device=device, dtype=torch.bool)
        valid[0, :3] = False
        valid[0, 11::5] = False
        valid[1, -6:] = False
        if mask_kind == "broadcast4d":
            valid = valid[:1].expand(batch, -1)
        visible &= valid[:, None, None, :]
        if mask_kind == "query_dependent":
            visible[:, :, 6::3, 4:9] = False
            visible[:, :, 12] = False
        elif mask_kind == "all_masked":
            visible.zero_()
        if mask_kind == "padding2d":
            mask = valid
        elif mask_kind == "additive4d":
            mask = torch.zeros_like(visible, dtype=dtype).masked_fill_(~visible, float("-inf"))
        else:
            mask = ~visible[:1] if mask_kind == "broadcast4d" else ~visible
    with torch.no_grad():
        expected = oracle(hidden, (positions.cos().to(dtype), positions.sin().to(dtype)), visible, None)
        actual = indexer(hidden.transpose(0, 1), rope, mask)
    assert actual.shape == (batch, 1, length, length)
    assert torch.equal(actual, ~expected)
    assert not ((~actual) & ~visible).any()
    assert ((~actual).sum(-1) <= (budget // 2) * 2 + 1).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied_scores", ["zero", "constant"])
def test_batch_qsa_preserves_reference_ties(device, dtype, tied_scores):
    _, indexer, oracle = _indexers(device, dtype, 5)
    if tied_scores == "zero":
        with torch.no_grad():
            indexer.index_qk_proj.weight.zero_()
        oracle.load_state_dict(indexer.state_dict(), strict=True)
    hidden = torch.ones(2, 35, 24, device=device, dtype=dtype)
    visible = torch.ones(2, 1, 35, 35, device=device, dtype=torch.bool).tril()
    visible[1, :, :, :3] = False
    cosine = torch.ones(2, 35, 4, device=device, dtype=dtype)
    with torch.no_grad():
        expected = oracle(hidden, (cosine, torch.zeros_like(cosine)), visible, None)
        actual = indexer(hidden.transpose(0, 1), None, ~visible)
    assert torch.equal(actual, ~expected)


@pytest.mark.parametrize("device", DEVICES)
def test_causal_batch_qsa_pools_once_per_sample(device):
    _, indexer, _ = _indexers(device, torch.float32, 5)
    hidden = torch.randn(128, 1, 24, device=device)
    with patch.object(indexer, "_pool_keys", wraps=indexer._pool_keys) as pool:
        indexer(hidden, None, None)
    assert pool.call_count == 1


def test_within_budget_batch_qsa_skips_projection():
    _, indexer, _ = _indexers("cpu", torch.float32, 64)
    with patch.object(indexer.index_qk_proj, "forward", side_effect=AssertionError("Unnecessary projection")):
        actual = indexer(torch.randn(16, 1, 24), None, None)
    assert torch.equal(actual[0, 0], ~torch.ones(16, 16, dtype=torch.bool).tril())
    with pytest.raises(ValueError, match="hidden size"):
        indexer(torch.randn(16, 1, 23), None, None)


def _scalar_repair(query, keys, blocks, selected, rows, topk, dim):
    for row, count in rows:
        scores = torch.matmul(query[row], keys[:count].transpose(0, 1)).transpose(0, 1)
        scores = torch.relu(scores).sum(-1) / math.sqrt(dim)
        indices = blocks[:count].index_select(0, scores.topk(topk, dim=0).indices).flatten()
        selected[row].index_fill_(0, blocks[:count].flatten(), False)
        selected[row].index_fill_(0, indices, True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads", [3, 4, 8])
@pytest.mark.parametrize("head_dim", [8, 128])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_reference_repair_preserves_exact_scores_and_masks(device, heads, head_dim, source_dtype):
    module = _load_qsa_module()
    torch.manual_seed(19)
    query = torch.randn(129, heads, head_dim, device=device).to(source_dtype).float()
    keys = torch.randn(83, head_dim, device=device).to(source_dtype).float()
    rows = [[index, 20 + index % 64] for index in range(129)]
    scores = module._reference_scores(query, keys, rows, 83, head_dim)
    for index, (row, count) in enumerate(rows):
        expected = (query[row] @ keys[:count].T).T.relu().sum(-1) / math.sqrt(head_dim)
        torch.testing.assert_close(scores[index, :count], expected, rtol=0, atol=0)
    blocks = (torch.arange(83 * 4, device=device) + 3).reshape(83, 4)
    selected = torch.rand(129, 83 * 4 + 8, device=device) > 0.3
    expected = selected.clone()
    _scalar_repair(query, keys, blocks, expected, rows, 15, head_dim)
    module._repair_causal_rows(query, keys, blocks, selected, rows, 15, head_dim)
    assert torch.equal(expected, selected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["zero", "constant", "near"])
def test_reference_repair_keeps_native_tie_rule_and_partial_tails(device, kind):
    module = _load_qsa_module()
    torch.manual_seed(17)
    query = torch.ones(35, 4, 128, device=device)
    keys = torch.ones(83, 128, device=device)
    if kind == "zero":
        keys.zero_()
    elif kind == "near":
        keys *= 1 + torch.arange(83, device=device)[:, None] * 1e-7
    blocks = (torch.arange(83 * 4, device=device) + 3).reshape(83, 4)
    rows = [[index, 20 + index] for index in range(35)]
    selected = torch.rand(35, 83 * 4 + 8, device=device) > 0.3
    expected = selected.clone()
    _scalar_repair(query, keys, blocks, expected, rows, 15, 128)
    module._repair_causal_rows(query, keys, blocks, selected, rows, 15, 128)
    assert torch.equal(selected, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_normal_repair_uses_one_batched_topk_not_one_per_query(device):
    module = _load_qsa_module()
    torch.manual_seed(43)
    query = torch.randn(89, 4, 128, device=device).bfloat16().float()
    keys = torch.randn(83, 128, device=device).bfloat16().float()
    blocks = torch.arange(83 * 4, device=device).reshape(83, 4)
    selected = torch.ones(89, 83 * 4 + 3, device=device, dtype=torch.bool)
    original = torch.Tensor.topk
    shapes = []

    def counted(self, *args, **kwargs):
        shapes.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    with patch.object(torch.Tensor, "topk", counted):
        module._repair_causal_rows(query, keys, blocks, selected, [[i, 83] for i in range(89)], 15, 128)
    assert shapes == [(89, 83)]


def test_empty_repair_does_not_launch_selection():
    module = _load_qsa_module()
    selected = torch.ones(3, 4, dtype=torch.bool)
    with patch.object(torch.Tensor, "topk", side_effect=AssertionError("Empty repair")):
        module._repair_causal_rows(
            torch.ones(3, 2, 4), torch.ones(2, 4), torch.arange(4).view(2, 2), selected, [], 1, 4
        )
    assert selected.all()
