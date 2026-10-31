#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=$(pwd)/src:$(pwd)/3rdparty/Megatron-LM
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --no-project --python "$(command -v python)" python - <<'PY'
import logging
from importlib.metadata import version
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
requirements = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
for dependency in requirements:
    requirement = Requirement(dependency)
    if requirement.name in {"transformers", "nvidia-resiliency-ext", "mlflow", "tensorboard"}:
        installed = version(requirement.name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"B1 runtime mismatch: {requirement}, installed={installed}; use the rebuilt image")
        logging.info("B1 dependency %s=%s", requirement.name, installed)
if Version(version("nvidia-resiliency-ext")) < Version("0.6.0"):
    raise RuntimeError("Megatron-Core requires nvidia-resiliency-ext>=0.6.0")
import megatron.bridge
import megatron.core

root = Path.cwd().resolve()
for module, expected in ((megatron.bridge, root / "src"), (megatron.core, root / "3rdparty/Megatron-LM")):
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(expected):
        raise RuntimeError(f"Unexpected source loaded: {location}, expected under {expected}")
    logging.warning("B1 source: %s", location)
PY

if [[ "${B1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  uv run --no-project --python "$(command -v python)" python examples/models/speech_cascades/finetune.py --help
  exit 0
fi

: "${BASE_HF:?Set clean Qwen3.5-4B path}"
: "${ENCODER_DIR:?Set Phi-4 MM 24L encoder directory}"
: "${OUTPUT:?Set a fresh B1 output directory}"
export MODEL_HF=${OUTPUT}/initial_hf
export SPEECH_TOKENIZER_PATH=${BASE_HF}
export TRAIN_CONFIG=projects/cluster/train_s2tt_b1_4b.yaml
export DATA_CONFIG=projects/cluster/data_s2tt_b1.yaml

uv run --no-project --python "$(command -v python)" python - <<'PY'
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import timedelta

import torch.distributed as dist
import yaml

logging.basicConfig(level=logging.INFO)
dist.init_process_group("gloo", timeout=timedelta(hours=2))
status = [None]
if dist.get_rank() == 0:
    try:
        hf = Path(os.environ["MODEL_HF"])
        encoder = Path(os.environ["ENCODER_DIR"])
        done = hf / "b1_prepared.json"
        source = {
            "base_hf": os.environ["BASE_HF"], "encoder": str(encoder),
            "blocks": 24, "projector_seed": 1234,
        }
        if done.exists():
            if json.loads(done.read_text()) != source:
                raise ValueError("Existing prepared model has different B1 provenance")
        else:
            if hf.exists() and any(hf.iterdir()):
                raise ValueError("Incomplete prepared model exists; use a fresh output directory")
            subprocess.run([
                sys.executable, "examples/conversion/convert_speech_cascades_stage1_to_hf.py",
                "--base-hf", source["base_hf"], "--encoder-checkpoint", str(encoder / "model.checkpoint"),
                "--encoder-num-blocks", "24", "--mean-file", str(encoder / "mean.vec"),
                "--invstd-file", str(encoder / "Invstd.vec"), "--output-dir", str(hf), "--seed", "1234",
            ], check=True)
            done.write_text(json.dumps(source, indent=2))
        config = json.loads((hf / "config.json").read_text())
        assert config["hidden_size"] == 2560
        assert config["audio_processor"]["config"]["num_blocks"] == 24
        assert config["embd_layer"]["projection_cls"] == "mlp"
        training = yaml.safe_load(Path(os.environ["TRAIN_CONFIG"]).read_text())
        assert training["lora_rank"] is None and training["mtp_num_hidden_layers"] == 0
        assert training["distillation_weight"] == 1.0
        logging.info("B1 initialization verified: Phi24L, clean 4B, random MLP 9.18016M, no LoRA/MTP")
    except Exception as error:
        logging.exception("B1 initialization failed")
        status[0] = str(error)
dist.broadcast_object_list(status, src=0)
dist.destroy_process_group()
if status[0] is not None:
    raise RuntimeError(status[0])
PY

uv run --no-project --python "$(command -v python)" python examples/models/speech_cascades/finetune.py \
  --train-config-yaml "${TRAIN_CONFIG}" \
  --hf-path "${MODEL_HF}" \
  --streaming-config-yaml "${DATA_CONFIG}" \
  --output-dir "${OUTPUT}" --name run