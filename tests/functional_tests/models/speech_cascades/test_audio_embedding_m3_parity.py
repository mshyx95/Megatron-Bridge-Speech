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

"""End-to-end m3 parity: real 40k weights + real log-mel batch reproduce the captured audio embedding.

Gated on real artifacts (skipped in CI). Set these environment variables to the production fixtures:

- ``SPEECH_CASCADES_FULL_CHECKPOINT`` — consolidated DeepSpeed ``mp_rank_00_model_states.pt``.
- ``SPEECH_CASCADES_REFERENCE_BATCH`` — captured batch with ``inputs_embeds`` and ``audio_attention_mask``.
- ``SPEECH_CASCADES_REFERENCE_AUDIO`` — captured ``{'audio_features': ...}`` reference.

Acceptance (``acceptance_baseline.yaml`` encoder gate): ``bf16_max_absolute_difference_max = 0.01``.
"""

import os

import pytest


torch = pytest.importorskip("torch")
CascadesAudioEmbedding = pytest.importorskip(
    "megatron.bridge.models.speech_cascades.audio_embedding"
).CascadesAudioEmbedding


_FULL_CHECKPOINT = os.environ.get("SPEECH_CASCADES_FULL_CHECKPOINT")
_REFERENCE_BATCH = os.environ.get("SPEECH_CASCADES_REFERENCE_BATCH")
_REFERENCE_AUDIO = os.environ.get("SPEECH_CASCADES_REFERENCE_AUDIO")

_ARTIFACTS_PRESENT = all(
    path and os.path.isfile(path) for path in (_FULL_CHECKPOINT, _REFERENCE_BATCH, _REFERENCE_AUDIO)
)

# Production encoder config (qwen3-5-4b-s2 run); activation checkpointing is off (no forward-numeric effect).
_PROD_ENCODER_CONFIG = dict(
    input_size=80,
    attention_dim=1280,
    attention_heads=20,
    linear_units=4480,
    dropout_rate=0.0,
    num_blocks=12,
    kernel_size=3,
    causal=True,
    batch_norm=False,
    activation="swish",
    cnn_layer_norm=True,
    ext_pw_out_channel=0,
    depthwise_seperable_out_channel=0,
    depthwise_multiplier=1,
    chunk_size=10,
    left_chunk=7,
    bias_in_glu=True,
    input_layer="conv2d",
    time_reduction=8,
    cnn_out=512,
    relative_k=True,
    conv_threshold=4,
    activation_checkpointing="",
    encoder_embedding_config={"input_size": 80},
)
_HIDDEN_SIZE = 2560
_MAX_ABS_DIFF = 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="real Cascades artifacts not provided")
def test_audio_embedding_reproduces_m3_reference():
    device = "cuda"
    dtype = torch.bfloat16

    checkpoint = torch.load(_FULL_CHECKPOINT, map_location="cpu", mmap=True, weights_only=False)
    state_dict = checkpoint["module"] if "module" in checkpoint else checkpoint

    module = CascadesAudioEmbedding(
        _PROD_ENCODER_CONFIG,
        hidden_size=_HIDDEN_SIZE,
        projection_cls="mlp",
        compression_rate=8,
        downsample_rate=1,
    )
    report = module.load_full_model_weights({k: v.float() for k, v in state_dict.items()})
    assert report.projection_tensor_count == 4
    module = module.to(device, dtype).eval()

    batch = torch.load(_REFERENCE_BATCH, map_location="cpu", weights_only=False)
    xs_pad = batch["inputs_embeds"].to(device, dtype)
    masks = batch["audio_attention_mask"].to(device)
    reference = torch.load(_REFERENCE_AUDIO, map_location="cpu", weights_only=False)["audio_features"].to(device)

    with torch.no_grad():
        output = module(xs_pad, masks)

    assert tuple(output.shape) == tuple(reference.shape)
    max_abs_diff = float((output.float() - reference.float()).abs().max())
    assert max_abs_diff <= _MAX_ABS_DIFF, f"max_abs_diff={max_abs_diff} exceeds {_MAX_ABS_DIFF}"
