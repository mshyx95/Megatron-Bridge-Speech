import importlib.util
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.speech_cascades.model import SpeechCascadesModel


@pytest.mark.unit
def test_gpu_restore_uses_production_base_then_adapter_load(tmp_path):
    source = Path(__file__).resolve().parents[3] / "functional_tests/models/speech_cascades/test_sequence_parallel.py"
    spec = importlib.util.spec_from_file_location("speech_sp_gpu_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 8)

        def cuda(self):
            return self

    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper

    base = TinyModel().to(torch.bfloat16)
    checkpoint = dict(base.state_dict())
    adapter = nn.Linear(4, 8, bias=False).to(torch.bfloat16)
    base.projection = AdapterWrapper(base.projection, adapter)
    saved_adapter = {key: value.clone() for key, value in base.state_dict().items() if ".adapter." in key}

    def load_base(path, models, *, fully_parallel_load, strict):
        assert fully_parallel_load is True
        assert strict is True
        assert not isinstance(models[0].projection, AdapterWrapper)
        models[0].load_state_dict(checkpoint, strict=strict)
        calls.append("base")

    def peft(model):
        assert calls == ["base"]
        model.projection = AdapterWrapper(model.projection, nn.Linear(4, 8, bias=False).to(torch.bfloat16))
        calls.append("inject")
        return model

    def load_adapters(models, policy, path):
        assert calls == ["base", "inject"]
        assert policy is peft
        incompatible = models[0].load_state_dict(saved_adapter, strict=False)
        assert set(incompatible.missing_keys) == {"projection.to_wrap.weight", "projection.to_wrap.bias"}
        assert not incompatible.unexpected_keys
        calls.append("adapters")

    with (
        patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint", side_effect=load_base),
        patch("examples.models.speech_cascades.lora_eval.load_lora_weights", side_effect=load_adapters),
    ):
        restored = module._restore_model(SimpleNamespace(provide=TinyModel), peft, tmp_path)
    assert calls == ["base", "inject", "adapters"]
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, dict(base.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.unit
def test_native_two_stage_checkpoint_roundtrip_on_cpu(tmp_path):
    import torch.distributed as dist
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    from megatron.bridge.models.speech_cascades.peft import SpeechCascadesLoRA
    from megatron.bridge.peft.adapter_wrapper import AdapterWrapper

    source = Path(__file__).resolve().parents[3] / "functional_tests/models/speech_cascades/test_sequence_parallel.py"
    spec = importlib.util.spec_from_file_location("speech_sp_gpu_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 8)
            self.audio_model = nn.Linear(8, 8)
            self.register_buffer("normalization", torch.randn(4))
            self.register_buffer("global_tokens_per_expert", torch.zeros(8), persistent=False)
            self.register_buffer("ga_steps", torch.tensor(0.0), persistent=False)
            self.projection.register_buffer("expert_bias", torch.randn(8))
            self.pg_collection = SimpleNamespace(dp_cp=dist.group.WORLD)

        def cuda(self):
            return self

        def sharded_state_dict(self, **kwargs):
            return {
                name: ShardedTensor.from_rank_offsets(name, tensor)
                for name, tensor in self.state_dict(keep_vars=True).items()
            }

    class TinyPEFT(SpeechCascadesLoRA):
        def __call__(self, model):
            model.projection.requires_grad_(False)
            model.projection = AdapterWrapper(model.projection, nn.Linear(4, 8, bias=False).to(torch.bfloat16))
            return model

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        peft = TinyPEFT()
        original = peft(TinyModel().to(torch.bfloat16))
        original.global_tokens_per_expert.fill_(65)
        original.ga_steps.fill_(2)
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        ):
            dist_checkpointing.save({"model": original.sharded_state_dict()}, str(checkpoint))
            restored = module._restore_model(SimpleNamespace(provide=TinyModel), peft, checkpoint)
        assert dict(restored.named_parameters()).keys() == dict(original.named_parameters()).keys()
        for name, parameter in restored.named_parameters():
            torch.testing.assert_close(parameter, dict(original.named_parameters())[name], rtol=0, atol=0)
        torch.testing.assert_close(restored.normalization, original.normalization, rtol=0, atol=0)
        assert not restored.global_tokens_per_expert.any()
        assert restored.ga_steps.item() == 0
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                restored.global_tokens_per_expert, original.global_tokens_per_expert, rtol=0, atol=0
            )
        module._assert_checkpoint_tensors_restored(original, restored)
        for name in ("normalization", "projection.to_wrap.expert_bias"):
            buffer = dict(restored.named_buffers())[name]
            expected = buffer.clone()
            buffer.add_(1)
            with pytest.raises(AssertionError, match=name):
                module._assert_checkpoint_tensors_restored(original, restored)
            buffer.copy_(expected)
        restored._non_persistent_buffers_set.remove("ga_steps")
        with pytest.raises(AssertionError, match="ga_steps"):
            module._assert_checkpoint_tensors_restored(original, restored)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_native_router_counters_are_not_checkpoint_state(tmp_path):
    import torch.distributed as dist
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.transformer_config import TransformerConfig

    source = Path(__file__).resolve().parents[3] / "functional_tests/models/speech_cascades/test_sequence_parallel.py"
    spec = importlib.util.spec_from_file_location("speech_sp_gpu_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=8,
            moe_router_topk=8,
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=0.001,
            use_cpu_initialization=True,
            moe_router_fusion=False,
        )
        groups = SimpleNamespace(
            tp=dist.group.WORLD, cp=dist.group.WORLD, tp_cp=dist.group.WORLD, tp_dp_cp=dist.group.WORLD
        )
        with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
            router = TopKRouter(config, pg_collection=groups)
            restored = TopKRouter(config, pg_collection=groups)
        router.set_layer_number(1)
        for length in (32, 33):
            scores = torch.full((length, 8), 1 / 8, requires_grad=True)
            router._apply_global_aux_loss(scores, scores, torch.ones(length, 8, dtype=torch.bool))
        torch.testing.assert_close(router.global_tokens_per_expert, torch.full((8,), 65.0))
        assert router.ga_steps.item() == 2
        assert {"global_tokens_per_expert", "ga_steps"} <= router._non_persistent_buffers_set
        assert {"global_tokens_per_expert", "ga_steps"}.isdisjoint(router.state_dict())
        restored.load_state_dict(router.state_dict(), strict=True)
        assert not restored.global_tokens_per_expert.any()
        with pytest.raises(AssertionError):
            torch.testing.assert_close(restored.global_tokens_per_expert, router.global_tokens_per_expert)
        module._assert_checkpoint_tensors_restored(router, restored)
    finally:
        dist.destroy_process_group()


class _DistributedLanguageModel(nn.Module):
    def forward(self, *, decoder_input, labels, **kwargs):
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        full = gather_from_sequence_parallel_region(
            decoder_input, tensor_parallel_output_grad=False, group=self.pg_collection.tp
        )
        return full.square().sum(-1).transpose(0, 1)


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(42)
        reference = _build_model(False)
        actual = _build_model(True)
        actual.load_state_dict(reference.state_dict())
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
        )
        embedding = LanguageModelEmbedding(
            config,
            vocab_size=32,
            max_sequence_length=16,
            position_embedding_type="rope",
            scatter_to_sequence_parallel=False,
            tp_group=dist.group.WORLD,
        )
        with torch.no_grad():
            embedding.word_embeddings.weight.copy_(reference.language_model.embedding.weight.chunk(2)[rank])
        actual.language_model = _DistributedLanguageModel()
        actual.language_model.embedding = embedding
        actual.language_model.pg_collection = SimpleNamespace(tp=dist.group.WORLD)
        for length in (6, 7):
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
            inputs = torch.arange(length).unsqueeze(0)
            inputs[:, 1:3] = -1
            kwargs = {"audio_input_features": torch.randn(1, 2, 4), "audio_embed_sizes": torch.tensor([2])}
            expected = reference(inputs, labels=inputs.clamp_min(0), **kwargs)
            with (
                patch("torch.cuda.current_device", return_value=torch.device("cpu")),
                patch(
                    "megatron.core.tensor_parallel.get_cuda_rng_tracker",
                    return_value=SimpleNamespace(fork=nullcontext),
                ),
            ):
                output = actual(inputs, labels=inputs.clamp_min(0), **kwargs)
                output.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(output, expected)
            for name, parameter in actual.audio_model.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(reference.audio_model.named_parameters())[name].grad)
            torch.testing.assert_close(
                embedding.word_embeddings.weight.grad, reference.language_model.embedding.weight.grad.chunk(2)[rank]
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_two_rank_native_embedding_and_sp_audio_gradients(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2, join=True)


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(32, 8))

    def forward(self, input_ids, position_ids):
        return torch.nn.functional.embedding(input_ids, self.weight).transpose(0, 1)


class _Audio(nn.Linear):
    def forward(self, features, mask):
        return super().forward(features)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.pg_collection = SimpleNamespace(tp=None)
        self.share_embeddings_and_output_weights = False
        self.last_input = None

    def forward(self, *, decoder_input, labels, **kwargs):
        self.last_input = decoder_input
        self.last_padding_mask = kwargs.get("padding_mask")
        loss = decoder_input.square().sum(-1).transpose(0, 1)
        return loss if labels is not None else decoder_input.transpose(0, 1).contiguous()


def _build_model(sequence_parallel):
    config = SimpleNamespace(
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=2 if sequence_parallel else 1,
        scatter_embedding_sequence_parallel=True,
        hidden_size=8,
    )
    language_model = _LanguageModel()
    with (
        patch("megatron.bridge.models.speech_cascades.model.GPTModelProvider.provide", return_value=language_model),
        patch("megatron.bridge.models.speech_cascades.model.CascadesAudioEmbedding", return_value=_Audio(4, 8)),
    ):
        model = SpeechCascadesModel(config, audio_token_id=31, audio_encoder_config={})
    assert config.scatter_embedding_sequence_parallel is (not sequence_parallel)
    return model


@pytest.mark.unit
@pytest.mark.parametrize("length", [6, 7])
@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("with_labels", [False, True])
def test_audio_is_merged_before_sp_and_padding_is_removed(length, with_audio, with_labels):
    torch.manual_seed(17)
    reference = _build_model(False)
    actual = _build_model(True)
    actual.load_state_dict(reference.state_dict())
    inputs = torch.arange(length).unsqueeze(0)
    kwargs = {}
    if with_audio:
        inputs[:, 1:3] = -1
        kwargs = {
            "audio_input_features": torch.randn(1, 2, 4),
            "audio_embed_sizes": torch.tensor([2]),
        }
    labels = inputs.clamp_min(0) if with_labels else None
    expected = reference(inputs, labels=labels, **kwargs)
    with patch(
        "megatron.core.tensor_parallel.scatter_to_sequence_parallel_region", side_effect=lambda tensor, **kw: tensor
    ) as scatter:
        output = actual(inputs, labels=labels, **kwargs)
    assert output.shape[:2] == (1, length)
    assert scatter.call_count == (2 if length % 2 else 1)
    assert scatter.call_args_list[0].args[0].shape[0] % 2 == 0
    if length % 2:
        mask = actual.language_model.last_padding_mask
        assert mask.shape == (1, length + 1)
        assert mask[:, -1].all() and not mask[:, :-1].any()
    torch.testing.assert_close(output, expected)
    expected.sum().backward()
    output.sum().backward()
    for name, parameter in reference.named_parameters():
        actual_grad = dict(actual.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, parameter.grad)
