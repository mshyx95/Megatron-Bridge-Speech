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

import re
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


_CONSOLIDATED_FILENAMES = (
    "mp_rank_00_model_states.pt",
    "zero_pp_rank_0_mp_rank_00_model_states.pt",
)
_LAYER_FILE_PATTERN = re.compile(r"layer_(\d+)-model_(\d+)-model_states\.pt")


@dataclass(frozen=True)
class LegacyCheckpointLoadReport:
    """Audit result for a legacy DeepSpeed checkpoint load."""

    source_format: str
    source_tensor_count: int
    loaded_tensor_count: int
    remapped_keys: tuple[tuple[str, str], ...]
    ignored_source_keys: tuple[str, ...]
    ignored_target_keys: tuple[str, ...]


def _load_checkpoint_file(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_tensor_mapping(value: object, *, source: Path) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping in {source}, got {type(value).__name__}")

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected string-to-tensor entries in {source}, got {key!r}: {type(tensor).__name__}")
        state_dict[key] = tensor
    return state_dict


def load_legacy_deepspeed_state_dict(checkpoint_dir: str | Path) -> tuple[OrderedDict[str, torch.Tensor], str]:
    """Load a legacy speech DeepSpeed checkpoint without changing tensor names or values.

    Supports the consolidated ``module`` payload and the layer-sharded fallback emitted by the
    production speech training stack.

    Args:
        checkpoint_dir: Directory containing legacy DeepSpeed model-state files.

    Returns:
        The CPU state dictionary and detected source format.

    Raises:
        FileNotFoundError: If no supported checkpoint file exists.
        TypeError: If a checkpoint payload has an invalid structure.
        ValueError: If layer shards contain duplicate tensor names.
    """
    checkpoint_path = Path(checkpoint_dir)
    consolidated_path = next(
        (checkpoint_path / filename for filename in _CONSOLIDATED_FILENAMES if (checkpoint_path / filename).is_file()),
        None,
    )
    if consolidated_path is None:
        raise FileNotFoundError(
            f"No legacy DeepSpeed checkpoint found in {checkpoint_path}; expected one of {_CONSOLIDATED_FILENAMES}"
        )

    payload = _load_checkpoint_file(consolidated_path)
    if not isinstance(payload, Mapping) or "module" not in payload:
        raise TypeError(f"Expected a DeepSpeed payload with a 'module' entry in {consolidated_path}")
    if payload["module"] is not None:
        return _as_tensor_mapping(payload["module"], source=consolidated_path), "consolidated"

    layer_files = sorted(
        (path for path in checkpoint_path.iterdir() if _LAYER_FILE_PATTERN.fullmatch(path.name)),
        key=lambda path: tuple(int(index) for index in _LAYER_FILE_PATTERN.fullmatch(path.name).groups()),
    )
    if not layer_files:
        raise FileNotFoundError(f"Checkpoint {consolidated_path} has no module payload or layer state files")

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer_file in layer_files:
        match = _LAYER_FILE_PATTERN.fullmatch(layer_file.name)
        layer_index = int(match.group(1))
        layer_state = _as_tensor_mapping(_load_checkpoint_file(layer_file), source=layer_file)
        for key, tensor in layer_state.items():
            target_key = f"layers.{layer_index}.{key}"
            if target_key in state_dict:
                raise ValueError(f"Duplicate legacy checkpoint tensor: {target_key}")
            state_dict[target_key] = tensor
    return state_dict, "layer_sharded"


def _source_key_for_target(target_key: str, source_keys: Collection[str]) -> str | None:
    if target_key in source_keys:
        return target_key
    if "base_layer." in target_key:
        legacy_key = target_key.replace("base_layer.", "")
        if legacy_key in source_keys:
            return legacy_key
    return None


def load_legacy_deepspeed_weights(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    ignored_source_keys: Collection[str],
    ignored_target_keys: Collection[str],
) -> LegacyCheckpointLoadReport:
    """Load a legacy speech checkpoint after proving a one-to-one tensor mapping.

    Args:
        model: Target speech model.
        checkpoint_dir: Directory containing legacy DeepSpeed model-state files.
        ignored_source_keys: Explicitly accepted checkpoint tensors not owned by ``model``.
        ignored_target_keys: Explicitly accepted model tensors initialized outside the checkpoint.

    Returns:
        An immutable load report suitable for P2 acceptance evidence.

    Raises:
        ValueError: If any non-ignored tensor is missing, unexpected, duplicated, or shape-incompatible.
    """
    source_state, source_format = load_legacy_deepspeed_state_dict(checkpoint_dir)
    target_state = model.state_dict()
    ignored_source = set(ignored_source_keys)
    ignored_target = set(ignored_target_keys)
    unknown_ignored_source = ignored_source - source_state.keys()
    unknown_ignored_target = ignored_target - target_state.keys()
    if unknown_ignored_source or unknown_ignored_target:
        raise ValueError(
            "Ignored checkpoint keys must exist; "
            f"unknown source={sorted(unknown_ignored_source)}, unknown target={sorted(unknown_ignored_target)}"
        )

    mapped_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    consumed_source_keys: set[str] = set()
    remapped_keys: list[tuple[str, str]] = []
    missing_target_keys: list[str] = []
    shape_mismatches: list[str] = []
    for target_key, target_tensor in target_state.items():
        if target_key in ignored_target:
            continue
        source_key = _source_key_for_target(target_key, source_state.keys())
        if source_key is None:
            missing_target_keys.append(target_key)
            continue
        if source_key in consumed_source_keys:
            raise ValueError(f"Legacy checkpoint tensor maps more than once: {source_key}")
        source_tensor = source_state[source_key]
        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(f"{source_key} -> {target_key}: {tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}")
            continue
        mapped_state[target_key] = source_tensor
        consumed_source_keys.add(source_key)
        if source_key != target_key:
            remapped_keys.append((source_key, target_key))

    unexpected_source_keys = source_state.keys() - consumed_source_keys - ignored_source
    if missing_target_keys or unexpected_source_keys or shape_mismatches:
        raise ValueError(
            "Legacy checkpoint mapping is not one-to-one; "
            f"missing target={sorted(missing_target_keys)}, "
            f"unexpected source={sorted(unexpected_source_keys)}, "
            f"shape mismatches={sorted(shape_mismatches)}"
        )

    incompatible = model.load_state_dict(mapped_state, strict=False)
    residual_missing = set(incompatible.missing_keys) - ignored_target
    if residual_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"Model rejected audited state dict; missing={sorted(residual_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )

    return LegacyCheckpointLoadReport(
        source_format=source_format,
        source_tensor_count=len(source_state),
        loaded_tensor_count=len(mapped_state),
        remapped_keys=tuple(sorted(remapped_keys)),
        ignored_source_keys=tuple(sorted(ignored_source)),
        ignored_target_keys=tuple(sorted(ignored_target)),
    )