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

"""Real-data inference for the native speech model.

Gated on real artifacts (skipped in CI). Set:

- ``SPEECH_CASCADES_HF_DIR`` — converted HF directory.
- ``SPEECH_CASCADES_REFERENCE_BATCH`` — captured ALM batch (``m1_capture/batch.pt``).

Two checks on the real 4.3B weights and the real captured audio batch:

1. Teacher-forced next-token top-1 accuracy over the supervised positions is high (``>= 0.9``),
   confirming the model transcribes the real audio correctly.
2. ``greedy_generate`` decodes continuation tokens from the audio prompt without error.
"""

import os

import pytest


torch = pytest.importorskip("torch")

_HF_DIR = os.environ.get("SPEECH_CASCADES_HF_DIR")
_REFERENCE_BATCH = os.environ.get("SPEECH_CASCADES_REFERENCE_BATCH")
_ARTIFACTS_PRESENT = all(p and os.path.exists(p) for p in (_HF_DIR, _REFERENCE_BATCH))

_ACC_BAR = 0.9


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="SPEECH_CASCADES_HF_DIR / REFERENCE_BATCH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_real_audio_inference():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401  (registers the bridge)
    from megatron.bridge import AutoBridge
    from megatron.bridge.data.speech_chunk import ALMDataCollator
    from megatron.bridge.models.speech_cascades.generate import greedy_generate
    from megatron.bridge.models.speech_cascades.step import get_alm_batch

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29616")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    bridge = AutoBridge.from_hf_pretrained(_HF_DIR)
    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)[0]
    model.eval()
    inner = model.module if hasattr(model, "module") else model
    gpt = inner.language_model
    weight = gpt.shared_embedding_or_output_weight()

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
    audio_features = batch["inputs_embeds"].to("cuda", torch.bfloat16)
    audio_mask = batch["audio_attention_mask"].cuda()
    audio_sizes = batch["audio_embed_sizes"].cuda()

    # --- 1. Teacher-forced top-1 accuracy over supervised (shifted) positions. ---
    prepared = get_alm_batch(iter([batch]))
    loss_mask = prepared.pop("loss_mask")
    captured: dict[str, torch.Tensor] = {}

    class _Stop(Exception):
        pass

    def _capture(_m, args):
        captured["hidden"] = args[0]
        raise _Stop()

    handle = gpt.output_layer.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            inner(
                input_ids=prepared["input_ids"],
                position_ids=None,
                attention_mask=None,
                audio_input_features=audio_features,
                audio_attention_mask=audio_mask,
                audio_embed_sizes=audio_sizes,
            )
    except _Stop:
        pass
    finally:
        handle.remove()

    valid = loss_mask[0].bool()
    hidden_valid = captured["hidden"][:, 0, :][valid].unsqueeze(1)
    targets = prepared["labels"][0][valid]
    with torch.no_grad():
        logits, _ = gpt.output_layer(hidden_valid, weight=weight)
        preds = logits.squeeze(1).argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    assert accuracy >= _ACC_BAR, f"teacher-forced top-1 accuracy {accuracy:.4f} < {_ACC_BAR}"

    # --- 2. Greedy generation from the audio prompt runs and extends the sequence. ---
    prompt = ref["input_ids"].cuda()  # full prompt keeps all audio placeholders consistent with sizes
    out = greedy_generate(
        inner,
        prompt,
        audio_input_features=audio_features,
        audio_attention_mask=audio_mask,
        audio_embed_sizes=audio_sizes,
        max_new_tokens=4,
    )
    assert out.shape[1] == prompt.shape[1] + 4
