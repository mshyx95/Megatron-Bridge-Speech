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

import io
from types import SimpleNamespace

import pytest
import torch
from rich.console import Console

from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size, weights_verification_table


class _FakeBridge:
    def __init__(self, converted: torch.Tensor, original: torch.Tensor) -> None:
        self.converted = converted
        self.hf_pretrained = SimpleNamespace(state={"weight": original})

    def export_hf_weights(self, megatron_model, show_progress=False):
        yield "weight", self.converted


@pytest.mark.unit
@pytest.mark.parametrize(("original", "expected"), [([1.0, 2.0], "✅"), ([1.0, 3.0], "❌")])
def test_weights_verification_table_supports_mixed_dtypes(original, expected) -> None:
    bridge = _FakeBridge(
        converted=torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        original=torch.tensor(original, dtype=torch.float32),
    )

    stream = io.StringIO()
    Console(file=stream, color_system=None).print(weights_verification_table(bridge, None))
    assert expected in stream.getvalue()


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (None, None),
        (2048, 2048),
        ((2047, 0), 2048),
        ([2047, 0], 2048),
    ],
)
def test_mcore_to_hf_window_size(window_size, expected):
    assert mcore_to_hf_window_size(window_size) == expected


def test_mcore_to_hf_window_size_rejects_malformed_pair():
    with pytest.raises(ValueError, match="two-element MCore window"):
        mcore_to_hf_window_size([2047])
