from unittest.mock import MagicMock

import pytest
import torch
from megatron.core import parallel_state

from megatron.bridge.models.speech_cascades import obs_logging


def test_audio_hours_use_cpu_gloo_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    group = object()
    all_reduce = MagicMock(side_effect=lambda tensor, group: tensor.add_(5.0))

    monkeypatch.setattr(obs_logging, "_AUDIO_SECONDS_INTERVAL", 7.0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group_gloo", lambda: group)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    interval_hours = obs_logging._pop_interval_audio_hours()

    tensor = all_reduce.call_args.args[0]
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert all_reduce.call_args.kwargs["group"] is group
    assert interval_hours == pytest.approx(12.0 / 3600.0)
    assert obs_logging._AUDIO_SECONDS_INTERVAL == 0.0