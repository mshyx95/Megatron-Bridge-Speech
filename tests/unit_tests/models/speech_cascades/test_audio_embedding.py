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

"""Structure and determinism test for the Cascades audio embedding (encoder + projection)."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")
CascadesAudioEmbedding = pytest.importorskip(
    "megatron.bridge.models.speech_cascades.audio_embedding"
).CascadesAudioEmbedding


_ENCODER_CONFIG = dict(
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

_HIDDEN_SIZE = 512
_GOLDEN_SHAPE = (2, 31, _HIDDEN_SIZE)
_GOLDEN_ABS_SUM = 4846.649317195261
_GOLDEN_NORM = 33.90680707151569
_GOLDEN_SLICE = [
    0.08218643814325333,
    0.036331817507743835,
    0.0632021576166153,
    0.042237527668476105,
    0.25838470458984375,
]


def _run():
    torch.manual_seed(0)
    module = CascadesAudioEmbedding(
        _ENCODER_CONFIG, hidden_size=_HIDDEN_SIZE, projection_cls="mlp", compression_rate=8, downsample_rate=1
    )
    module.eval()
    generator = torch.Generator().manual_seed(0)
    xs_pad = torch.randn(2, 256, 80, generator=generator)
    masks = torch.ones(2, 256, dtype=torch.long)
    with torch.no_grad():
        return module(xs_pad, masks)


def test_audio_embedding_projection_wiring():
    torch.manual_seed(0)
    module = CascadesAudioEmbedding(_ENCODER_CONFIG, hidden_size=_HIDDEN_SIZE, projection_cls="mlp")
    linears = [m for m in module.audio_projection if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2
    assert linears[0].in_features == _ENCODER_CONFIG["attention_dim"]
    assert linears[-1].out_features == _HIDDEN_SIZE


def test_audio_embedding_is_deterministic():
    assert torch.equal(_run(), _run())


def test_audio_embedding_matches_signature():
    output = _run()
    assert tuple(output.shape) == _GOLDEN_SHAPE
    assert output.dtype == torch.float32
    assert output.double().abs().sum().item() == pytest.approx(_GOLDEN_ABS_SUM, rel=1e-6)
    assert output.double().norm().item() == pytest.approx(_GOLDEN_NORM, rel=1e-6)
    for actual, expected in zip(output[0, 0, :5].double().tolist(), _GOLDEN_SLICE):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_unsupported_projection_raises():
    with pytest.raises(NotImplementedError):
        CascadesAudioEmbedding(_ENCODER_CONFIG, hidden_size=_HIDDEN_SIZE, projection_cls="qformer")


@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_releasing_attention_maps_preserves_outputs_and_gradients(freeze_encoder):
    from megatron.bridge.models.speech_cascades.encoder.audio_blocks.cascade_encoder.utils import MultiHeadedAttention

    torch.manual_seed(3)
    retained = CascadesAudioEmbedding(_ENCODER_CONFIG, hidden_size=32).train()
    released = CascadesAudioEmbedding(_ENCODER_CONFIG, hidden_size=32, retain_attention_maps=False).train()
    released.load_state_dict(retained.state_dict())
    inputs = torch.randn(2, 128, 80)
    masks = torch.ones(2, 128, dtype=torch.long)
    masks[1, 96:] = 0
    for module in (retained, released):
        module.encoder.requires_grad_(not freeze_encoder)
    for length in (128, 96, 112):
        outputs = []
        for module in (retained, released):
            module.zero_grad(set_to_none=True)
            torch.manual_seed(7)
            np.random.seed(7)
            output = module(inputs[:, :length].clone(), masks[:, :length].clone())
            output.square().mean().backward()
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for name, parameter in retained.named_parameters():
            other = dict(released.named_parameters())[name]
            assert (parameter.grad is None) == (other.grad is None), name
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
        retained_maps = [
            module.attn for module in retained.encoder.modules() if isinstance(module, MultiHeadedAttention)
        ]
        released_maps = [
            module.attn for module in released.encoder.modules() if isinstance(module, MultiHeadedAttention)
        ]
        assert len(retained_maps) == len(released_maps) == _ENCODER_CONFIG["num_blocks"]
        assert all(isinstance(value, torch.Tensor) for value in retained_maps)
        assert all(value is None for value in released_maps)
