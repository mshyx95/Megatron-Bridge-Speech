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

"""Tests for finetune.py --train-config-yaml merge (yaml fills defaults, CLI overrides win)."""

import argparse
import importlib.util
from pathlib import Path

import pytest


_FINETUNE = Path(__file__).resolve().parents[4] / "examples" / "models" / "speech_cascades" / "finetune.py"


def _load_finetune():
    spec = importlib.util.spec_from_file_location("_ft_under_test", _FINETUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = dict(train_config_yaml=None, s2_kd=False, global_batch_size=8, lr=None, mtp_num_hidden_layers=0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_yaml_fills_unpassed_args(tmp_path, monkeypatch):
    ft = _load_finetune()
    cfg = tmp_path / "train.yaml"
    cfg.write_text("s2_kd: true\nglobal_batch_size: 64\nmtp_num_hidden_layers: 4\n")
    monkeypatch.setattr("sys.argv", ["finetune.py", "--train-config-yaml", str(cfg)])
    args = _args(train_config_yaml=str(cfg))
    ft._apply_train_config_yaml(args, argparse.ArgumentParser())
    assert args.s2_kd is True
    assert args.global_batch_size == 64
    assert args.mtp_num_hidden_layers == 4


def test_cli_overrides_yaml(tmp_path, monkeypatch):
    ft = _load_finetune()
    cfg = tmp_path / "train.yaml"
    cfg.write_text("global_batch_size: 64\nlr: 4.0e-5\n")
    # --global-batch-size passed on the CLI must win over the yaml; lr (not passed) comes from yaml.
    monkeypatch.setattr("sys.argv", ["finetune.py", "--train-config-yaml", str(cfg), "--global-batch-size", "7"])
    args = _args(train_config_yaml=str(cfg), global_batch_size=7)
    ft._apply_train_config_yaml(args, argparse.ArgumentParser())
    assert args.global_batch_size == 7
    assert args.lr == 4.0e-5


def test_unknown_key_rejected(tmp_path, monkeypatch):
    ft = _load_finetune()
    cfg = tmp_path / "train.yaml"
    cfg.write_text("not_a_real_flag: 1\n")
    monkeypatch.setattr("sys.argv", ["finetune.py", "--train-config-yaml", str(cfg)])
    with pytest.raises(SystemExit):
        ft._apply_train_config_yaml(_args(train_config_yaml=str(cfg)), argparse.ArgumentParser())


def test_no_yaml_is_noop(monkeypatch):
    ft = _load_finetune()
    monkeypatch.setattr("sys.argv", ["finetune.py"])
    args = _args(train_config_yaml=None, global_batch_size=8)
    ft._apply_train_config_yaml(args, argparse.ArgumentParser())
    assert args.global_batch_size == 8
