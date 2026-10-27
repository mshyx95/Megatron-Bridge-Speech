import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
@pytest.mark.parametrize("exit_interval", [160, 0, -1])
@pytest.mark.parametrize("overlap", [None, True, False])
@pytest.mark.parametrize("storage", [None, True, False])
def test_finetune_transport_overlap_and_exit_preserve_schedule(monkeypatch, exit_interval, overlap, storage):
    spec = importlib.util.spec_from_file_location(
        "speech_finetune_cli", ROOT / "examples/models/speech_cascades/finetune.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    received = []
    cfg = SimpleNamespace(
        train=SimpleNamespace(train_iters=12000, exit_interval=None),
        ddp=SimpleNamespace(overlap_grad_reduce=False, overlap_param_gather=False),
    )

    def recipe(**kwargs):
        assert kwargs["train_iters"] == 12000
        assert kwargs["streaming_use_storage_api"] is storage
        return cfg

    monkeypatch.setattr(
        cli,
        "_load_recipe",
        lambda: SimpleNamespace(
            speech_cascades_finetune_config=recipe,
            speech_cascades_s2_kd_config=recipe,
        ),
    )
    monkeypatch.setattr("megatron.bridge.models.speech_cascades.obs_logging.install_speech_obs_logging", lambda: None)
    monkeypatch.setattr("megatron.bridge.training.finetune.finetune", lambda config, **kwargs: received.append(config))
    args = [
        "finetune.py",
        "--hf-path",
        "/hf",
        "--train-iters",
        "12000",
        "--exit-interval",
        str(exit_interval),
    ]
    if overlap is not None:
        args.append("--overlap-grad-reduce" if overlap else "--no-overlap-grad-reduce")
    if storage is not None:
        args.append("--streaming-use-storage-api" if storage else "--no-streaming-use-storage-api")
    monkeypatch.setattr(sys, "argv", args)
    if exit_interval <= 0:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        assert not received
    else:
        cli.main()
        assert received == [cfg]
        assert cfg.train.exit_interval == 160
        assert cfg.train.train_iters == 12000
        assert cfg.ddp.overlap_grad_reduce is (overlap is True)
        assert cfg.ddp.overlap_param_gather is False
