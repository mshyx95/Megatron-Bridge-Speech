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

"""Unit tests for the speech->text knowledge-distillation loss (CPU, no model parallel)."""

import pytest
import torch

from megatron.bridge.models.speech_cascades.distillation import chunked_kl_divergence, distillation_loss


def _head(vocab: int, hidden: int) -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(hidden, vocab, bias=False)


def test_identical_hidden_states_give_zero_kl():
    head = _head(64, 16)
    h = torch.randn(10, 16)
    kl = chunked_kl_divergence(h, h.clone(), head, temperature=2.0, chunk_size=4)
    assert torch.isclose(kl, torch.zeros(()), atol=1e-5)


def test_chunking_matches_single_chunk():
    head = _head(64, 16)
    s = torch.randn(20, 16)
    t = torch.randn(20, 16)
    one = chunked_kl_divergence(s, t, head, temperature=2.0, chunk_size=64)
    many = chunked_kl_divergence(s, t, head, temperature=2.0, chunk_size=4)
    assert torch.isclose(one, many, atol=1e-5)


def test_averaged_over_response_tokens():
    head = _head(32, 8)
    s = torch.randn(7, 8)
    t = torch.randn(7, 8)
    kl = chunked_kl_divergence(s, t, head, temperature=1.5, chunk_size=3)
    assert kl.ndim == 0 and kl.item() >= 0.0


def test_gradient_flows_to_student_not_teacher():
    head = _head(32, 8)
    s = torch.randn(6, 8, requires_grad=True)
    t = torch.randn(6, 8, requires_grad=True)
    chunked_kl_divergence(s, t, head, temperature=2.0, chunk_size=2).backward()
    assert s.grad is not None and torch.any(s.grad != 0)
    assert t.grad is None  # teacher distribution is detached under no_grad


def test_shape_mismatch_and_empty_raise():
    head = _head(16, 4)
    with pytest.raises(ValueError):
        chunked_kl_divergence(torch.randn(3, 4), torch.randn(2, 4), head, 2.0, 2)
    with pytest.raises(ValueError):
        chunked_kl_divergence(torch.empty(0, 4), torch.empty(0, 4), head, 2.0, 2)


def test_distillation_loss_mixing():
    kl = torch.tensor(2.0)
    ce = torch.tensor(1.0)
    assert torch.isclose(distillation_loss(kl, ce, 1.0), kl)  # pure KL
    assert torch.isclose(distillation_loss(kl, ce, 0.75), torch.tensor(0.75 * 2.0 + 0.25 * 1.0))
