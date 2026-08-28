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

"""Compatibility shim for ``torch.distributed.checkpoint.load``.

The pinned Megatron-Core distributed-checkpoint loader calls
``torch.distributed.checkpoint.load(state_dict, storage_reader, planner=..., no_dist=True)``. PyTorch
2.6 removed the ``no_dist`` argument from ``load`` (it survives only on the legacy
``load_state_dict``), so resuming a distributed checkpoint raises
``TypeError: load() got an unexpected keyword argument 'no_dist'``. Importing this module patches
``load`` to route ``no_dist`` calls to ``load_state_dict``. The patch is applied only when the
running PyTorch's ``load`` lacks ``no_dist``, so it is a no-op on versions that still support it.
"""

import inspect

import torch.distributed.checkpoint as _dcp


_SENTINEL = object()
_original_load = _dcp.load


def _load_with_no_dist(
    state_dict,
    storage_reader=None,
    *,
    planner=None,
    no_dist=_SENTINEL,
    process_group=None,
    checkpoint_id=None,
    **kwargs,
):
    if no_dist is not _SENTINEL:
        return _dcp.load_state_dict(
            state_dict,
            storage_reader,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )
    return _original_load(
        state_dict,
        storage_reader=storage_reader,
        planner=planner,
        process_group=process_group,
        checkpoint_id=checkpoint_id,
        **kwargs,
    )


def apply() -> bool:
    """Patch ``torch.distributed.checkpoint.load`` if it does not accept ``no_dist``.

    Returns ``True`` if the patch was applied, ``False`` if the running PyTorch already supports it.
    """
    if "no_dist" in inspect.signature(_original_load).parameters:
        return False
    _dcp.load = _load_with_no_dist
    return True


apply()
