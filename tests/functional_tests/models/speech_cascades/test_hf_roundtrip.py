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

"""Megatron -> HF export round trip for the native speech model.

Gated on the converted HF directory (skipped in CI). Set:

- ``SPEECH_CASCADES_HF_DIR`` — converted HF directory (dispatches to ``SpeechCascadesBridge``).

Builds the native :class:`SpeechCascadesModel`, exports every weight back to HF format, and checks
that each exported tensor matches the source checkpoint. This exercises the bidirectional bridge for
the Qwen3.5 GDN language model **and** the replicated audio encoder / projection / CMVN buffers.
"""

import os

import pytest


torch = pytest.importorskip("torch")

_HF_DIR = os.environ.get("SPEECH_CASCADES_HF_DIR")
_ARTIFACTS_PRESENT = bool(_HF_DIR) and os.path.isdir(_HF_DIR or "")


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="SPEECH_CASCADES_HF_DIR not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_hf_export_round_trip_matches_source():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge import AutoBridge

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29615")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    bridge = AutoBridge.from_hf_pretrained(_HF_DIR)
    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)

    checked = 0
    mismatched: list[str] = []
    missing: list[str] = []
    max_abs_diff = 0.0
    saw_audio_encoder = False
    saw_audio_projection = False
    saw_cmvn = False

    for name, weight in bridge.export_hf_weights(model, cpu=True, show_progress=False):
        if name not in bridge.hf_pretrained.state:
            missing.append(name)
            continue
        original = bridge.hf_pretrained.state[name]
        exported = weight.float()
        source = original.float()
        if exported.shape != source.shape:
            mismatched.append(f"{name} shape {tuple(exported.shape)} != {tuple(source.shape)}")
            continue
        diff = (exported - source).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        if not torch.allclose(exported, source, atol=1e-2):
            mismatched.append(f"{name} max_abs_diff {diff:.4g}")
        checked += 1
        if "embed_tokens_extend.encoder" in name:
            saw_audio_encoder = True
        if "embed_tokens_extend.audio_projection" in name:
            saw_audio_projection = True
        if "global_mean" in name or "global_invstd" in name:
            saw_cmvn = True

    assert not missing, f"exported names absent from source: {missing[:5]} ({len(missing)} total)"
    assert not mismatched, f"{len(mismatched)} mismatched: {mismatched[:5]}"
    assert checked > 700, f"expected the full checkpoint, only checked {checked}"
    assert saw_audio_encoder and saw_audio_projection and saw_cmvn, (
        f"audio export coverage gap: encoder={saw_audio_encoder} projection={saw_audio_projection} cmvn={saw_cmvn}"
    )
    assert max_abs_diff < 1e-2, f"max_abs_diff {max_abs_diff}"
