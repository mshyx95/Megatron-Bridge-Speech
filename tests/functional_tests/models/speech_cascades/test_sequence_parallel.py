import logging
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _barrier(device_id=None):
    if not dist.is_initialized():
        return
    backend = dist.get_backend()
    if backend == "nccl" and device_id is not None:
        dist.barrier(device_ids=[int(device_id)])
        return
    dist.barrier()


def _assert_checkpoint_tensors_restored(original, restored):
    reference_parameters = dict(original.named_parameters())
    restored_parameters = dict(restored.named_parameters())
    assert restored_parameters.keys() == reference_parameters.keys(), "Restored parameter names differ"
    for name, parameter in restored_parameters.items():
        torch.testing.assert_close(
            parameter, reference_parameters[name], rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
        )
    reference_buffers = dict(original.named_buffers())
    restored_buffers = dict(restored.named_buffers())
    assert restored_buffers.keys() == reference_buffers.keys(), "Restored buffer names differ"
    for name, buffer in restored_buffers.items():
        parent, _, leaf = name.rpartition(".")
        reference_owner = original.get_submodule(parent)
        restored_owner = restored.get_submodule(parent)
        persistent = leaf not in reference_owner._non_persistent_buffers_set
        assert persistent == (leaf not in restored_owner._non_persistent_buffers_set), name
        if persistent:
            torch.testing.assert_close(
                buffer, reference_buffers[name], rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
            )
        else:
            logging.getLogger(__name__).info("Runtime-only buffer not checkpointed: %s", name)


def _restore_model(config, peft, checkpoint):
    from examples.models.speech_cascades.lora_eval import load_lora_weights

    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint

    model = config.provide().cuda().to(torch.bfloat16)
    _load_model_weights_from_checkpoint(str(checkpoint), [model], fully_parallel_load=True, strict=True)
    model = peft(model)
    load_lora_weights([model], peft, str(checkpoint))
    if dist.is_initialized():
        _barrier(torch.cuda.current_device() if torch.cuda.is_available() else None)
    return model


def _gpu_worker(rank, output):
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401
    from megatron.bridge.models.speech_cascades.peft import SpeechCascadesLoRA
    from megatron.bridge.models.speech_cascades.provider import SpeechCascadesModelProvider

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{output}/init", rank=rank, world_size=2, timeout=timedelta(minutes=5)
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    try:
        assert parallel_state.get_tensor_model_parallel_world_size() == 2
        assert parallel_state.get_expert_model_parallel_world_size() == 2
        model_parallel_cuda_manual_seed(1234)
        # Keep CPU-side parameter initialization deterministic across TP ranks.
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        config = SpeechCascadesModelProvider(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
            ffn_hidden_size=512,
            vocab_size=512,
            seq_length=128,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            num_moe_experts=8,
            moe_router_topk=8,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            moe_shared_expert_gate=True,
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=0.001,
            moe_grouped_gemm=True,
            moe_permute_fusion=True,
            moe_token_dispatcher_type="alltoall",
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            sequence_parallel=True,
            attention_backend="flash",
            attention_dropout=0.0,
            gradient_accumulation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            audio_encoder_config={
                "input_size": 80,
                "attention_dim": 64,
                "attention_heads": 4,
                "linear_units": 128,
                "num_blocks": 1,
                "dropout_rate": 0.0,
                "input_layer": "conv2d",
                "time_reduction": 8,
                "cnn_out": 32,
                "chunk_size": 10,
                "left_chunk": 2,
                "activation_checkpointing": "",
                "encoder_embedding_config": {"input_size": 80},
                "activation": "swish",
                "conv_activation": "swish",
                "conv_glu_type": "swish",
            },
            audio_compression_rate=8,
        )
        config.finalize()
        model = config.provide().cuda().to(torch.bfloat16)
        peft = SpeechCascadesLoRA(dim=320, alpha=640, dropout=0.01, normalize_moe_lora=True)
        model = peft(model)
        torch.manual_seed(31)
        for length in (32, 33):
            model.zero_grad(set_to_none=True)
            inputs = torch.randint(3, 512, (1, length), device="cuda")
            inputs[:, 1:5] = -1
            features = torch.randn(1, 80, 80, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(1, 80, device="cuda", dtype=torch.long)
            sizes = torch.tensor([4], device="cuda")
            labels = inputs.clamp_min(0)
            losses = model(
                inputs,
                audio_input_features=features,
                audio_attention_mask=mask,
                audio_embed_sizes=sizes,
                labels=labels,
            )
            assert losses.shape == labels.shape and torch.isfinite(losses).all()
            losses.float().mean().backward()
            audio_grads = [parameter.grad for parameter in model.audio_model.audio_projection.parameters()]
            assert audio_grads and all(grad is not None and torch.isfinite(grad).all() for grad in audio_grads)
            assert sum(grad.float().abs().sum().item() for grad in audio_grads) > 0
            adapter_grads = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if "adapter" in name and parameter.grad is not None
            ]
            assert adapter_grads and all(torch.isfinite(grad).all() for grad in adapter_grads)
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None
        model.eval()
        with torch.no_grad():
            expected = model(
                inputs,
                audio_input_features=features,
                audio_attention_mask=mask,
                audio_embed_sizes=sizes,
                labels=labels,
            )
        checkpoint = Path(output) / "checkpoint"
        checkpoint.mkdir(exist_ok=True)
        _barrier(rank)
        dist_checkpointing.save({"model": model.sharded_state_dict()}, str(checkpoint))
        _barrier(rank)
        logging.getLogger(__name__).warning("Rank %d entering restore validation stage", rank)
        restored = _restore_model(config, peft, checkpoint)
        restored.eval()
        restore_assertion_failed = torch.zeros(1, device="cuda", dtype=torch.int32)
        try:
            _assert_checkpoint_tensors_restored(model, restored)
        except Exception:
            logging.getLogger(__name__).exception("Rank %d restore tensor assertion failed", rank)
            restore_assertion_failed.fill_(1)
        dist.all_reduce(restore_assertion_failed, op=dist.ReduceOp.MAX)
        _barrier(rank)
        if int(restore_assertion_failed.item()) != 0:
            raise AssertionError("Checkpoint tensor-restore validation failed on at least one rank")
        with torch.no_grad():
            actual = restored(
                inputs,
                audio_input_features=features,
                audio_attention_mask=mask,
                audio_embed_sizes=sizes,
                labels=labels,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        logging.getLogger(__name__).warning(
            "TP2/SP native speech MoE LoRA forward/backward/checkpoint PASS rank=%d", rank
        )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_two_gpu_native_speech_moe_lora_sequence_parallel(tmp_path):
    assert torch.cuda.device_count() >= 2, "Two distinct CUDA devices required; do not skip this submission gate"
    mp.spawn(_gpu_worker, args=(str(tmp_path),), nprocs=2, join=True)
