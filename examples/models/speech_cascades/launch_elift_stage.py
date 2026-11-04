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

"""Launch one ELIFT stage with the native Megatron-Bridge trainer (replaces the DeepSpeed ``ds_train_advanced.py``).

The stage's training profile, data snapshot, clean LLM and frozen encoder come from ``catalog.yaml`` and
``launch.yaml``. Rank 0 builds the prepared speech HF (clean LLM + frozen encoder + random MLP projector); every
rank then runs ``finetune.py``. Continuation stages (P2, P4, P5, P6, P4 Hindi) additionally restore weights from a
complete native Megatron checkpoint configured in ``launch.yaml``. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "projects/cluster/elift/v0.1"
PREPARED_MARKER = "elift_prepared.json"


@dataclass(frozen=True)
class Plan:
    """Commands and provenance for one stage."""

    stage: str
    output: str
    model_hf: str
    provenance: dict[str, object]
    prepare: list[str]
    finetune: list[str]


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a mapping")
    return value


def _absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {value!r}")
    return value


def build_plan(stage: str, *, configs: Path = CONFIGS, output_root: str | None = None) -> Plan:
    """Build the preparation and training commands for a catalog stage.

    Args:
        stage: Stage name from ``catalog.yaml``.
        configs: Directory containing ``catalog.yaml``, ``launch.yaml``, ``train/`` and ``data/``.
        output_root: Optional experiment root overriding ``launch.yaml``.

    Raises:
        ValueError: The stage is unknown, reference-only, or lacks a required complete native checkpoint.
    """
    catalog = _load(configs / "catalog.yaml")
    launch = _load(configs / "launch.yaml")
    stages = catalog["stages"]
    if not isinstance(stages, dict) or stage not in stages:
        raise ValueError(f"Unknown ELIFT stage {stage!r}; choose from {', '.join(sorted(map(str, stages)))}")
    profile = stages[stage]
    if profile["train_config"] is None:
        raise ValueError(f"Stage {stage} is reference-only ({profile['runtime_status']}) and cannot be launched")

    base = _absolute(launch["base_models"][profile["language_model"]], "base model")
    encoder = launch["encoders"][profile["encoder"]]
    for key in ("checkpoint", "mean_file", "invstd_file"):
        _absolute(encoder[key], f"{profile['encoder']} {key}")
    output = str(Path(_absolute(output_root or launch["output_root"], "output root")) / stage)
    model_hf = f"{output}/initial_hf"

    provenance: dict[str, object] = {
        "base_hf": base,
        "encoder": profile["encoder"],
        "encoder_checkpoint": encoder["checkpoint"],
        "num_blocks": encoder["num_blocks"],
        "projector_seed": 1234,
    }
    prepare = [
        sys.executable,
        "examples/conversion/convert_speech_cascades_stage1_to_hf.py",
        "--base-hf",
        base,
        "--encoder-checkpoint",
        encoder["checkpoint"],
        "--encoder-num-blocks",
        str(encoder["num_blocks"]),
        "--mean-file",
        encoder["mean_file"],
        "--invstd-file",
        encoder["invstd_file"],
        "--output-dir",
        model_hf,
        "--seed",
        "1234",
    ]
    finetune = [
        sys.executable,
        "examples/models/speech_cascades/finetune.py",
        "--train-config-yaml",
        f"projects/cluster/elift/v0.1/{profile['train_config']}",
        "--hf-path",
        model_hf,
        "--streaming-config-yaml",
        f"projects/cluster/elift/v0.1/{profile['data_config']}",
        "--output-dir",
        output,
        "--name",
        stage,
    ]
    initialization = str(profile["initialization"])
    if "native-mtp" in initialization or initialization.startswith("complete-"):
        checkpoint = launch["init_checkpoints"].get(stage)
        if checkpoint is None:
            raise ValueError(
                f"Stage {stage} ({initialization}) needs a complete native Megatron checkpoint: set "
                f"init_checkpoints.{stage} in projects/cluster/elift/v0.1/launch.yaml"
            )
        _absolute(checkpoint, f"init_checkpoints.{stage}")
        provenance["init_checkpoint"] = checkpoint
        finetune += ["--init-checkpoint", checkpoint]
    return Plan(stage, output, model_hf, provenance, prepare, finetune)


def prepare_model(plan: Plan) -> None:
    """Create the prepared speech HF once, refusing a directory with different provenance."""
    hf = Path(plan.model_hf)
    marker = hf / PREPARED_MARKER
    if marker.exists():
        if json.loads(marker.read_text()) != plan.provenance:
            raise ValueError(f"Existing prepared model {hf} has different provenance; use a fresh output")
    else:
        if hf.exists() and any(hf.iterdir()):
            raise ValueError(f"Incomplete prepared model exists at {hf}; use a fresh output")
        subprocess.run(plan.prepare, check=True)
        marker.write_text(json.dumps(plan.provenance, indent=2))
    config = json.loads((hf / "config.json").read_text())
    if config["audio_processor"]["config"]["num_blocks"] != plan.provenance["num_blocks"]:
        raise ValueError("Prepared encoder block count does not match the stage")
    if config["embd_layer"]["projection_cls"] != "mlp":
        raise ValueError("Prepared projector is not an MLP")
    LOGGER.info("Prepared speech HF verified: %s", hf)


def main() -> None:
    """Prepare the model on rank 0, then run native training on every rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Catalog stage, for example b2, p4 or p5")
    parser.add_argument("--output-root", help="Fresh experiment root; defaults to launch.yaml output_root")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without preparing or training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plan = build_plan(args.stage, output_root=args.output_root)
    if args.dry_run:
        print(json.dumps(asdict(plan), indent=2))
        return

    distributed = "WORLD_SIZE" in os.environ
    status: list[str | None] = [None]
    if distributed:
        from datetime import timedelta

        import torch.distributed as dist

        dist.init_process_group("gloo", timeout=timedelta(hours=2))
    if not distributed or dist.get_rank() == 0:
        try:
            prepare_model(plan)
        except Exception as error:
            LOGGER.exception("Model preparation failed")
            status[0] = str(error)
    if distributed:
        dist.broadcast_object_list(status, src=0)
        dist.destroy_process_group()
    if status[0] is not None:
        raise RuntimeError(status[0])
    sys.exit(subprocess.run(plan.finetune, check=False).returncode)


if __name__ == "__main__":
    main()
