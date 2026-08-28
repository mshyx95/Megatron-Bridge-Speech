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

"""Regression parity test for the vendored Cascades ConformerEncoder.

The golden signature was captured from the read-only Megatron-LM baseline encoder; a
byte-for-byte port must reproduce it bit-identically. See
docs/speech_cascades/execution_plan.md (gate P3).
"""

import pytest


torch = pytest.importorskip("torch")
ConformerEncoder = pytest.importorskip(
    "megatron.bridge.models.speech_cascades.encoder"
).ConformerEncoder


# Small stand-in for the production encoder that exercises the same code path as the
# qwen3-5-4b-s2 run config: conv2d streaming subsampling, relative_k positional bias, swish.
# The production standalone-init contract discovered from the baseline: build, then post_init({}).
_ENCODER_KWARGS = dict(
    input_size=80,
    attention_dim=256,
    attention_heads=4,
    linear_units=1024,
    dropout_rate=0.0,
    num_blocks=2,
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

# Golden signature captured from the baseline encoder in the pinned container.
_GOLDEN_SHAPE = (2, 31, 256)
_GOLDEN_ABS_SUM = 11621.912059334023
_GOLDEN_NORM = 125.98369993974671
_GOLDEN_SLICE = [
    -0.9923161268234253,
    0.3564288914203644,
    0.06754285097122192,
    0.44348692893981934,
    -0.6628372073173523,
]


def _run_encoder() -> "torch.Tensor":
    torch.manual_seed(0)
    encoder = ConformerEncoder(**_ENCODER_KWARGS)
    encoder.post_init({})
    encoder.eval()
    generator = torch.Generator().manual_seed(0)
    xs_pad = torch.randn(2, 256, 80, generator=generator)
    masks = torch.ones(2, 256, dtype=torch.long)
    with torch.no_grad():
        output = encoder(xs_pad, masks)
    return output[0] if isinstance(output, (tuple, list)) else output


def test_encoder_forward_is_deterministic():
    assert torch.equal(_run_encoder(), _run_encoder())


def test_encoder_matches_baseline_signature():
    output = _run_encoder()

    assert tuple(output.shape) == _GOLDEN_SHAPE
    assert output.dtype == torch.float32
    assert output.double().abs().sum().item() == pytest.approx(_GOLDEN_ABS_SUM, rel=1e-6)
    assert output.double().norm().item() == pytest.approx(_GOLDEN_NORM, rel=1e-6)
    for actual, expected in zip(output[0, 0, :5].double().tolist(), _GOLDEN_SLICE):
        assert actual == pytest.approx(expected, abs=1e-6)
