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

"""Real training-path step-zero loss: the causal-shift forward step reproduces the reference loss.

Gated on real artifacts (skipped in CI). Set:

- ``SPEECH_CASCADES_HF_DIR`` — converted HF directory (dispatches to ``SpeechCascadesBridge``).
- ``SPEECH_CASCADES_REFERENCE_BATCH`` — captured ALM batch (``m1_capture/batch.pt``).

Acceptance: masked-mean cross-entropy within 1% of the reference ``0.20644``. This exercises the
training-path causal label shift (``get_alm_batch``) on the real 4.3B weights. The loss is computed
memory-safely by projecting only the supervised positions (avoiding a full-vocab logits tensor when
GPU memory is tight).
"""

import os

import pytest


torch = pytest.importorskip("torch")

_HF_DIR = os.environ.get("SPEECH_CASCADES_HF_DIR")
_REFERENCE_BATCH = os.environ.get("SPEECH_CASCADES_REFERENCE_BATCH")
_ARTIFACTS_PRESENT = all(p and os.path.exists(p) for p in (_HF_DIR, _REFERENCE_BATCH))

_REFERENCE_LOSS = 0.20644
_REL_TOL = 0.01


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="SPEECH_CASCADES_HF_DIR / REFERENCE_BATCH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_forward_step_reproduces_reference_loss():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge import AutoBridge
    from megatron.bridge.data.speech_chunk import ALMDataCollator
    from megatron.bridge.models.speech_cascades.step import get_alm_batch

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    bridge = AutoBridge.from_hf_pretrained(_HF_DIR)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0]
    model.eval()
    inner = model.module if hasattr(model, "module") else model

    ref = torch.load(_REFERENCE_BATCH, map_location="cpu", weights_only=False)
    frames = ref["audio_frames"]
    features = [ref["inputs_embeds"][i, : int(frames[i])].clone() for i in range(ref["inputs_embeds"].shape[0])]
    example = {
        "input_ids": ref["input_ids"],
        "labels": ref["labels"],
        "input_embeds": features,
        "audio_embed_sizes": ref["audio_embed_sizes"],
        "audio_frames": frames,
        "is_text_sample": False,
    }
    collator = ALMDataCollator(pad_token_id=int(ref["input_ids"].max()) + 1, audio_key="inputs_embeds")
    batch = collator([example])

    prepared = get_alm_batch(iter([batch]))
    loss_mask = prepared.pop("loss_mask")

    gpt = inner.language_model
    weight = gpt.shared_embedding_or_output_weight()

    # Memory-safe loss: capture the output-layer input (post-final-norm hidden) and project only the
    # supervised positions, avoiding a full-vocab logits tensor when GPU memory is tight.
    captured: dict[str, torch.Tensor] = {}

    class _Stop(Exception):
        pass

    def _capture(_module, args):
        captured["hidden"] = args[0]
        raise _Stop()

    handle = gpt.output_layer.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            inner(
                input_ids=prepared["input_ids"],
                position_ids=None,
                attention_mask=None,
                labels=prepared["labels"],
                audio_input_features=prepared["audio_input_features"].to(torch.bfloat16),
                audio_attention_mask=prepared["audio_attention_mask"],
                audio_embed_sizes=prepared["audio_embed_sizes"],
            )
    except _Stop:
        pass
    finally:
        handle.remove()

    hidden = captured["hidden"]  # [seq, batch, hidden]
    valid = loss_mask[0].bool()  # [seq]
    hidden_valid = hidden[:, 0, :][valid].unsqueeze(1)  # [n_valid, 1, hidden]
    targets = prepared["labels"][0][valid]  # [n_valid]

    with torch.no_grad():
        logits, _ = gpt.output_layer(hidden_valid, weight=weight)  # [n_valid, 1, vocab]
        loss = torch.nn.functional.cross_entropy(logits.squeeze(1).float(), targets)

    rel = abs(loss.item() - _REFERENCE_LOSS) / _REFERENCE_LOSS
    assert rel < _REL_TOL, f"loss {loss.item():.5f} vs ref {_REFERENCE_LOSS} (rel {rel:.4f})"
