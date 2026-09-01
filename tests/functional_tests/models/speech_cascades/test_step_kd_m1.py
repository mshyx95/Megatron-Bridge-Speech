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

"""Gated real-weights step-zero check for the WP-A2 dual-forward KD loss (STUB).

Skipped in CI: requires the real 4.3B artifacts and a GPU. Set:

- ``SPEECH_CASCADES_HF_DIR`` — converted HF directory (dispatches to ``SpeechCascadesBridge``).
- ``SPEECH_CASCADES_KD_BATCH`` — a captured KD ALM batch that already carries the teacher tensors
  (``teacher_input_ids`` / ``teacher_labels`` / ``teacher_attention_mask``) produced by the WP-A1
  collator with ``knowledge_distillation=True``.

Acceptance (to fill in when artifacts are wired by WP-A3):
- KD ``forward_step`` returns a finite scalar loss at step zero.
- With ``distillation_weight -> 0`` the loss matches the ASR masked CE; with ``-> 1`` it matches the
  pure chunked KL.
- The teacher forward is detached: no grad reaches the frozen language-model weights.
"""

import os

import pytest


torch = pytest.importorskip("torch")

_HF_DIR = os.environ.get("SPEECH_CASCADES_HF_DIR")
_KD_BATCH = os.environ.get("SPEECH_CASCADES_KD_BATCH")
_ARTIFACTS_PRESENT = all(p and os.path.exists(p) for p in (_HF_DIR, _KD_BATCH))


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="SPEECH_CASCADES_HF_DIR / SPEECH_CASCADES_KD_BATCH not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_kd_forward_step_zero_is_finite():
    pytest.skip("WP-A2 gated KD step-zero check stub: enable once WP-A3 wires the real KD artifacts")
