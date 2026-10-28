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

"""Tests for the torch.distributed.checkpoint no_dist compatibility shim."""

import inspect

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch.distributed.checkpoint")

import torch.distributed.checkpoint as dcp  # noqa: E402

from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: E402


def test_apply_is_noop_when_load_supports_no_dist(monkeypatch):
    def fake_load(state_dict, storage_reader=None, *, no_dist=False, planner=None):
        return "native"

    monkeypatch.setattr(dcp_compat, "_original_load", fake_load)
    monkeypatch.setattr(dcp, "load", fake_load)
    # load already accepts no_dist -> patch must be a no-op.
    assert dcp_compat.apply() is False
    assert dcp.load is fake_load


def test_apply_patches_and_routes_no_dist(monkeypatch):
    calls = {}

    def fake_load(state_dict, *, checkpoint_id=None, storage_reader=None, planner=None, process_group=None):
        calls["load"] = True
        return "native-load"

    def fake_load_state_dict(state_dict, storage_reader, process_group=None, no_dist=False, planner=None):
        calls["load_state_dict"] = dict(no_dist=no_dist, storage_reader=storage_reader, planner=planner)
        return "legacy-load"

    monkeypatch.setattr(dcp_compat, "_original_load", fake_load)
    monkeypatch.setattr(dcp, "load", fake_load)
    monkeypatch.setattr(dcp, "load_state_dict", fake_load_state_dict)

    # fake_load has no no_dist param -> apply() installs the shim.
    assert "no_dist" not in inspect.signature(fake_load).parameters
    assert dcp_compat.apply() is True

    # MCore-shaped call (positional storage_reader + planner + no_dist) routes to load_state_dict.
    result = dcp.load({"w": 1}, "READER", planner="PLANNER", no_dist=True)
    assert result == "legacy-load"
    assert calls["load_state_dict"] == {"no_dist": True, "storage_reader": "READER", "planner": "PLANNER"}

    # Calls without no_dist fall through to the original load.
    assert dcp.load({"w": 1}, checkpoint_id="DIR") == "native-load"
