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

"""Prepare ELIFT data and a speech HF model, then run native Megatron-Bridge training.

The public Amulet entry passes the training/data files, model, encoder and output paths explicitly.
The historical ``--stage`` interface remains available for catalog reproducibility. Rank 0 prepares
the data and speech HF; every rank then runs ``finetune.py``. Run from the repository root.
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
CONFIGS = ROOT / "projects/ELIFT"
PREPARED_MARKER = "elift_prepared.json"


@dataclass(frozen=True)
class Plan:
    """Commands and provenance for one stage."""

    stage: str
    output: str
    model_hf: str
    provenance: dict[str, object]
    prepare: list[str]
    prepare_data: list[str]
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
    checkpoint = None
    initialization = str(profile["initialization"])
    if "native-mtp" in initialization or initialization.startswith("complete-"):
        checkpoint = launch["init_checkpoints"].get(stage)
        if checkpoint is None:
            raise ValueError(
                f"Stage {stage} ({initialization}) needs a complete native Megatron checkpoint: set "
                f"init_checkpoints.{stage} in projects/ELIFT/launch.yaml"
            )
        _absolute(checkpoint, f"init_checkpoints.{stage}")
    return build_explicit_plan(
        name=stage,
        train_config=Path(f"projects/ELIFT/{profile['train_config']}"),
        data_config=Path("projects/ELIFT/data/mixture.yaml"),
        data_recipe=profile["data_recipe"],
        data_hindi=profile.get("data_hindi"),
        base_hf=base,
        encoder_name=profile["encoder"],
        encoder_checkpoint=encoder["checkpoint"],
        encoder_num_blocks=encoder["num_blocks"],
        mean_file=encoder["mean_file"],
        invstd_file=encoder["invstd_file"],
        output_dir=output,
        init_checkpoint=checkpoint,
        streaming_compression=profile.get("data_streaming_compression"),
    )


def build_explicit_plan(
    *,
    name: str,
    train_config: Path,
    data_config: Path,
    data_recipe: str | None,
    data_hindi: str | None,
    base_hf: str,
    encoder_name: str,
    encoder_checkpoint: str,
    encoder_num_blocks: int,
    mean_file: str,
    invstd_file: str,
    output_dir: str,
    init_checkpoint: str | None = None,
    pretrained_checkpoint: str | None = None,
    streaming_compression: bool | None = None,
) -> Plan:
    """Build commands from explicit inputs without consulting the experiment catalog.

    ``data_recipe=None`` passes a standalone streaming YAML directly to training.
    Otherwise ``data_config`` must be the shared mixture, rendered with its adjacent source mappings.
    MTP profiles that disable HF loading require a complete native initialization checkpoint.
    """
    for path in (train_config, data_config):
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(f"Configuration does not exist: {path}")
    settings = _load(train_config if train_config.is_absolute() else ROOT / train_config)
    if settings.get("load_hf_weights") is False and settings.get("mtp_num_hidden_layers", 0) > 0:
        if init_checkpoint is None:
            raise ValueError("This MTP profile needs a complete native Megatron checkpoint: set INIT_CHECKPOINT")
    if init_checkpoint is not None and pretrained_checkpoint is not None:
        raise ValueError("Choose INIT_CHECKPOINT or PRETRAINED_CHECKPOINT, not both")
    if init_checkpoint is not None and not settings.get("s2_kd", False):
        raise ValueError("INIT_CHECKPOINT is a KD input; use PRETRAINED_CHECKPOINT for S1/LoRA")
    if settings.get("lora_rank") and pretrained_checkpoint is None:
        raise ValueError("LoRA requires a native PRETRAINED_CHECKPOINT")
    if data_recipe is None and (data_hindi is not None or streaming_compression is not None):
        raise ValueError("Hindi/compression overlays require a composed data recipe, not a standalone data YAML")
    if data_recipe is not None and data_config.name != "mixture.yaml":
        raise ValueError("A composed recipe requires data/mixture.yaml; use DATA_RECIPE=none for standalone data YAML")
    if not name or not encoder_name or encoder_num_blocks <= 0:
        raise ValueError("Run/encoder names must be nonempty and encoder block count must be positive")
    base = _absolute(base_hf, "base model")
    checkpoint = _absolute(encoder_checkpoint, "encoder checkpoint")
    mean = _absolute(mean_file, "encoder mean")
    invstd = _absolute(invstd_file, "encoder inverse standard deviation")
    output = _absolute(output_dir, "output directory")
    model_hf = f"{output}/initial_hf"
    provenance: dict[str, object] = {
        "base_hf": base,
        "encoder": encoder_name,
        "encoder_checkpoint": checkpoint,
        "num_blocks": encoder_num_blocks,
        "projector_seed": 1234,
    }
    prepare = [
        sys.executable,
        "examples/conversion/convert_speech_cascades_stage1_to_hf.py",
        "--base-hf",
        base,
        "--encoder-checkpoint",
        checkpoint,
        "--encoder-num-blocks",
        str(encoder_num_blocks),
        "--mean-file",
        mean,
        "--invstd-file",
        invstd,
        "--output-dir",
        model_hf,
        "--seed",
        "1234",
    ]
    streaming_config = str(data_config)
    prepare_data = []
    if data_recipe is not None:
        streaming_config = f"{output}/data.yaml"
        prepare_data = [
            sys.executable,
            "examples/models/speech_cascades/prepare_elift_data.py",
            "config",
            "--config-root",
            str(data_config.parent),
            "--recipe",
            data_recipe,
            "--tokenizer",
            base,
            "--output",
            streaming_config,
        ]
        if data_hindi is not None:
            prepare_data += ["--hindi", data_hindi]
        if streaming_compression is not None:
            prepare_data += ["--streaming-compression" if streaming_compression else "--no-streaming-compression"]
    finetune = [
        sys.executable,
        "examples/models/speech_cascades/finetune.py",
        "--train-config-yaml",
        str(train_config),
        "--hf-path",
        model_hf,
        "--streaming-config-yaml",
        streaming_config,
        "--output-dir",
        output,
        "--name",
        name,
    ]
    if init_checkpoint is not None:
        provenance["init_checkpoint"] = _absolute(init_checkpoint, "initialization checkpoint")
        finetune += ["--init-checkpoint", init_checkpoint]
    if pretrained_checkpoint is not None:
        provenance["pretrained_checkpoint"] = _absolute(pretrained_checkpoint, "pretrained checkpoint")
        finetune += ["--pretrained-checkpoint", pretrained_checkpoint]
    return Plan(name, output, model_hf, provenance, prepare, prepare_data, finetune)


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", help="Historical catalog stage; not used by the public Amulet entry")
    mode.add_argument("--train-config-yaml", type=Path, help="Explicit training profile")
    parser.add_argument("--output-root", help="Fresh experiment root; defaults to launch.yaml output_root")
    parser.add_argument("--data-config-yaml", type=Path)
    parser.add_argument("--data-recipe", help="base, new-2k, new-5k; none passes a standalone data YAML unchanged")
    parser.add_argument("--data-hindi", choices=("none", "2k", "5k"))
    parser.add_argument("--data-streaming-compression", choices=("recipe", "on", "off"))
    parser.add_argument("--base-hf")
    parser.add_argument("--encoder-name")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--encoder-num-blocks", type=int)
    parser.add_argument("--mean-file")
    parser.add_argument("--invstd-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--name")
    parser.add_argument("--init-checkpoint", help="Complete native MTP checkpoint, or none")
    parser.add_argument("--pretrained-checkpoint", help="Native pretrained checkpoint for S1/LoRA, or none")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without preparing or training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    explicit = {
        "train_config": args.train_config_yaml,
        "data_config": args.data_config_yaml,
        "data_recipe": args.data_recipe,
        "data_hindi": args.data_hindi,
        "streaming_compression": args.data_streaming_compression,
        "base_hf": args.base_hf,
        "encoder_name": args.encoder_name,
        "encoder_checkpoint": args.encoder_checkpoint,
        "encoder_num_blocks": args.encoder_num_blocks,
        "mean_file": args.mean_file,
        "invstd_file": args.invstd_file,
        "output_dir": args.output_dir,
        "name": args.name,
        "init_checkpoint": args.init_checkpoint,
        "pretrained_checkpoint": args.pretrained_checkpoint,
    }
    if args.stage is not None:
        if any(value is not None for value in explicit.values()):
            parser.error("--stage cannot be combined with explicit training inputs")
        plan = build_plan(args.stage, output_root=args.output_root)
    else:
        if args.output_root is not None:
            parser.error("Use --output-dir for explicit training, not --output-root")
        optional = {"data_hindi", "streaming_compression", "init_checkpoint", "pretrained_checkpoint"}
        missing = [key for key, value in explicit.items() if value is None and key not in optional]
        if missing:
            parser.error(f"Explicit training is missing inputs: {', '.join(missing)}")
        for key in ("data_recipe", *optional):
            if explicit[key] == "none":
                explicit[key] = None
        explicit["streaming_compression"] = {None: None, "recipe": None, "on": True, "off": False}[
            args.data_streaming_compression
        ]
        plan = build_explicit_plan(**explicit)
    if args.dry_run:
        sys.stdout.write(json.dumps(asdict(plan), indent=2) + "\n")
        return

    distributed = "WORLD_SIZE" in os.environ
    status: list[str | None] = [None]
    if distributed:
        from datetime import timedelta

        import torch.distributed as dist

        dist.init_process_group("gloo", timeout=timedelta(hours=2))
    if not distributed or dist.get_rank() == 0:
        try:
            if plan.prepare_data:
                subprocess.run(plan.prepare_data, check=True)
            prepare_model(plan)
        except Exception as error:
            LOGGER.exception("Stage preparation failed")
            status[0] = str(error)
    if distributed:
        dist.broadcast_object_list(status, src=0)
        dist.destroy_process_group()
    if status[0] is not None:
        raise RuntimeError(status[0])
    sys.exit(subprocess.run(plan.finetune, check=False).returncode)


if __name__ == "__main__":
    main()
