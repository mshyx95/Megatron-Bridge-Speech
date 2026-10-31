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

from pathlib import Path

import pytest
import torch
from torch import nn

from megatron.bridge.models.speech_cascades.checkpoint import (
    load_legacy_deepspeed_state_dict,
    load_legacy_deepspeed_weights,
)


class _LoRALinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(3, 2, bias=False)
        self.lora_A = nn.Parameter(torch.zeros(1, 3))


def _save_consolidated(checkpoint_dir: Path, module: dict[str, torch.Tensor] | None, *, zero: bool = False) -> None:
    filename = "zero_pp_rank_0_mp_rank_00_model_states.pt" if zero else "mp_rank_00_model_states.pt"
    torch.save({"module": module}, checkpoint_dir / filename)


@pytest.mark.unit
def test_loads_consolidated_checkpoint_and_remaps_lora_base_layer(tmp_path: Path) -> None:
    source_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    _save_consolidated(tmp_path, {"weight": source_weight}, zero=True)
    model = _LoRALinear()

    report = load_legacy_deepspeed_weights(
        model,
        tmp_path,
        ignored_source_keys=(),
        ignored_target_keys=("lora_A",),
    )

    torch.testing.assert_close(model.base_layer.weight, source_weight)
    assert report.source_format == "consolidated"
    assert report.loaded_tensor_count == 1
    assert report.remapped_keys == (("weight", "base_layer.weight"),)


@pytest.mark.unit
def test_loads_layer_sharded_checkpoint_in_numeric_order(tmp_path: Path) -> None:
    _save_consolidated(tmp_path, None)
    torch.save({"weight": torch.tensor([10.0])}, tmp_path / "layer_10-model_0-model_states.pt")
    torch.save({"weight": torch.tensor([2.0])}, tmp_path / "layer_2-model_0-model_states.pt")

    state_dict, source_format = load_legacy_deepspeed_state_dict(tmp_path)

    assert source_format == "layer_sharded"
    assert list(state_dict) == ["layers.2.weight", "layers.10.weight"]
    torch.testing.assert_close(state_dict["layers.2.weight"], torch.tensor([2.0]))


@pytest.mark.unit
def test_rejects_non_bijective_checkpoint_mapping(tmp_path: Path) -> None:
    _save_consolidated(
        tmp_path,
        {
            "weight": torch.ones(2, 3),
            "unexpected": torch.ones(1),
        },
    )
    model = _LoRALinear()

    with pytest.raises(ValueError, match=r"unexpected source=\['unexpected'\]"):
        load_legacy_deepspeed_weights(
            model,
            tmp_path,
            ignored_source_keys=(),
            ignored_target_keys=("lora_A",),
        )