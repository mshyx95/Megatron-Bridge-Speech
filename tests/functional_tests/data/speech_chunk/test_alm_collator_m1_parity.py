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

"""m1 collator parity: rebuild the captured reference batch bit-identically from its examples.

Gated on the captured batch (skipped in CI). Set:

- ``SPEECH_CASCADES_REFERENCE_BATCH`` — captured ALM batch with ``input_ids``, ``labels``,
  ``inputs_embeds``, ``audio_attention_mask``, ``audio_embed_sizes``, ``audio_frames``.

The batch is decomposed back into per-audio (un-padded) features and a single ALM example, then
re-collated. Every produced key must match the captured tensors exactly (``torch.equal``).
"""

import os

import pytest


torch = pytest.importorskip("torch")

from megatron.bridge.data.speech_chunk import ALMDataCollator  # noqa: E402


_REFERENCE_BATCH = os.environ.get("SPEECH_CASCADES_REFERENCE_BATCH")
_ARTIFACTS_PRESENT = bool(_REFERENCE_BATCH) and os.path.isfile(_REFERENCE_BATCH or "")


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="SPEECH_CASCADES_REFERENCE_BATCH not set")
def test_alm_collator_reproduces_reference_batch_bit_identical():
    ref = torch.load(_REFERENCE_BATCH, map_location="cpu", weights_only=False)

    input_ids = ref["input_ids"]
    labels = ref["labels"]
    inputs_embeds = ref["inputs_embeds"]
    audio_frames = ref["audio_frames"]
    audio_embed_sizes = ref["audio_embed_sizes"]

    # Decompose the padded (N, T_max, F) audio tensor back into per-audio un-padded features.
    features = [inputs_embeds[i, : int(audio_frames[i])].clone() for i in range(inputs_embeds.shape[0])]

    example = {
        "input_ids": input_ids,
        "labels": labels,
        "input_embeds": features,
        "audio_embed_sizes": audio_embed_sizes,
        "audio_frames": audio_frames,
        "is_text_sample": False,
    }

    # Choose a pad id absent from the sequence so attention_mask is all-ones like the reference.
    pad_token_id = int(input_ids.max()) + 1
    collator = ALMDataCollator(pad_token_id=pad_token_id, audio_key="inputs_embeds", padding_mask=True)
    batch = collator([example])

    for key in ("input_ids", "labels", "inputs_embeds", "audio_attention_mask", "audio_embed_sizes", "audio_frames"):
        assert key in batch, f"missing key {key}"
        assert batch[key].shape == ref[key].shape, f"{key} shape {batch[key].shape} != {ref[key].shape}"
        assert torch.equal(batch[key], ref[key]), f"{key} mismatch"

    if "attention_mask" in ref:
        assert torch.equal(batch["attention_mask"], ref["attention_mask"].bool())
