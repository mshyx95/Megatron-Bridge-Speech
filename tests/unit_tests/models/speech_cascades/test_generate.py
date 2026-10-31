import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn


@pytest.mark.unit
@pytest.mark.parametrize("tied,expected_token", [(False, 2), (True, 1)])
def test_greedy_generate_uses_correct_output_weight(monkeypatch, tied, expected_token):
    transformer = ModuleType("megatron.core.transformer")
    transformer.MegatronModule = nn.Module
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer)
    path = (
        Path(__file__).resolve().parents[4]
        / "src/megatron/bridge/models/speech_cascades/generate.py"
    )
    spec = importlib.util.spec_from_file_location("speech_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class OutputLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 1.0], [4.0, 0.0]]))
            self.override = None

        def forward(self, hidden, weight=None):
            assert not torch.is_grad_enabled()
            self.override = weight
            return hidden @ (self.weight if weight is None else weight).T, None

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.share_embeddings_and_output_weights = tied
            self.embedding_weight = nn.Parameter(torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0]]))
            self.output_layer = OutputLayer()
            self.shared_weight_calls = 0

        def shared_embedding_or_output_weight(self):
            self.shared_weight_calls += 1
            return self.embedding_weight

    class SpeechModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

        def forward(self, input_ids, **kwargs):
            hidden = torch.tensor([1.0, 0.0]).expand(input_ids.shape[1], 1, 2)
            return self.language_model.output_layer(hidden)

    model = SpeechModel()
    generated = module.greedy_generate(model, torch.tensor([[0]]), max_new_tokens=3, eos_token_ids=[expected_token])
    assert generated.tolist() == [[0, expected_token]]
    assert model.language_model.shared_weight_calls == int(tied)
    expected_weight = model.language_model.embedding_weight if tied else None
    assert model.language_model.output_layer.override is expected_weight
    assert not model.language_model.output_layer._forward_pre_hooks