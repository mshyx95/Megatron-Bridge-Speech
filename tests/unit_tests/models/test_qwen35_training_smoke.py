# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from transformers import Qwen3_5TextConfig

from megatron.bridge import AutoBridge


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3.5 GDN requires CUDA")
def test_qwen35_mixed_attention_two_training_steps() -> None:
    """Exercise the native Qwen3.5 GDN and full-attention training path."""
    text_config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=32,
        hidden_size=256,
        intermediate_size=512,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_value_head_dim=64,
        max_position_embeddings=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
        torch_dtype="bfloat16",
        vocab_size=512,
    )
    provider = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    distributed_was_initialized = dist.is_initialized()

    try:
        model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )[0]
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 2
        sequence_length = 32
        input_ids = torch.randint(
            0,
            text_config.vocab_size,
            (batch_size, sequence_length),
            device="cuda",
        )
        labels = torch.randint(
            0,
            text_config.vocab_size,
            (batch_size, sequence_length),
            device="cuda",
        )
        position_ids = torch.arange(sequence_length, device="cuda").expand(batch_size, -1)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            token_losses = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
            )
            loss = token_losses.float().mean()
            assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            assert gradients, "Expected trainable parameters to receive gradients"
            assert all(torch.isfinite(gradient).all() for gradient in gradients), (
                "Expected all gradients to be finite"
            )
            optimizer.step()
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if not distributed_was_initialized and dist.is_initialized():
            dist.destroy_process_group()