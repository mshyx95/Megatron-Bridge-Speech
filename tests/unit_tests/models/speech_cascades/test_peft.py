# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.speech_cascades.peft import SpeechCascadesLoRA


class _GDNState(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(2, 2, 2)
        self.A_log = nn.Parameter(torch.ones(2))
        self.dt_bias = nn.Parameter(torch.ones(2))
        self.in_proj = nn.Linear(2, 2)


class _SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.ModuleDict({"gdn": _GDNState(), "dense": nn.Linear(2, 2)})
        self.audio_model = nn.Linear(2, 2)


@pytest.mark.unit
def test_speech_lora_freezes_lm_but_trains_audio_and_gdn_state():
    model = _SpeechModel()
    peft = SpeechCascadesLoRA()

    with patch("megatron.bridge.models.speech_cascades.peft.unwrap_model", return_value=model):
        peft.freeze_model(model)

    trainable_lm = {name for name, param in model.language_model.named_parameters() if param.requires_grad}
    assert trainable_lm == {"gdn.A_log", "gdn.dt_bias", "gdn.conv1d.weight", "gdn.conv1d.bias"}
    assert all(param.requires_grad for param in model.audio_model.parameters())