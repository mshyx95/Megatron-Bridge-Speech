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

"""Export a native SpeechCascadesModel back to a HuggingFace-format directory.

Usage:
    uv run python examples/models/speech_cascades/export_hf.py \
        --hf-dir /path/to/initial_hf \
        --checkpoint /path/to/checkpoints/iter_0004000 \
        --vllm \
        --output /path/to/exported_hf_dir

For an HF round-trip without trained weights:
    uv run python -m torch.distributed.run --nproc_per_node=1 \
        examples/models/speech_cascades/export_hf.py \
        --hf-dir /path/to/converted_hf_dir \
        --output /path/to/exported_hf_dir
"""

import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def prepare_vllm_config(output: Path) -> None:
    """Adapt exported HF metadata to the Qwen3.5 speech-vLLM runtime."""
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("The speech-vLLM runtime requires a qwen3_5_text or qwen3_5_moe_text backbone")
    if config.get("audio_processor", {}).get("name") != "cascades":
        raise ValueError("The speech-vLLM runtime requires the Cascades audio encoder")
    embedding = config["embd_layer"]
    if embedding.get("embedding_cls", "audio") != "audio":
        raise ValueError("Expected an audio embedding layer")
    if config.get("n_embd", config["hidden_size"]) != config["hidden_size"]:
        raise ValueError("n_embd disagrees with hidden_size")
    index = json.loads((output / "model.safetensors.index.json").read_text())
    if not index.get("weight_map"):
        raise ValueError("Expected a nonempty safetensors weight map")
    if not any(name.startswith("mtp.") for name in index["weight_map"]):
        for field in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"):
            config[field] = 0
    config["architectures"] = ["Qwen3_5AudioForCausalLM"]
    config["n_embd"] = config["hidden_size"]
    embedding["embedding_cls"] = "audio"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    """Export trained Megatron weights, or explicitly round-trip an initial HF model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, help="Converted HF directory to build the model from.")
    parser.add_argument(
        "--checkpoint", help="Trained Megatron iteration directory containing .metadata and run_config.yaml."
    )
    parser.add_argument("--vllm", action="store_true", help="Write metadata for the speech-vLLM inference image.")
    parser.add_argument("--output", required=True, help="Destination directory for the exported HF checkpoint.")
    args = parser.parse_args()

    if args.vllm and not args.checkpoint:
        parser.error("--vllm requires --checkpoint; initial HF weights are not an evaluation checkpoint")
    if args.checkpoint:
        for filename in (".metadata", "run_config.yaml"):
            if not (Path(args.checkpoint) / filename).is_file():
                parser.error(f"--checkpoint must be a complete iteration directory: missing {filename}")
    if Path(args.output).resolve() == Path(args.hf_dir).resolve():
        parser.error("--output must not overwrite --hf-dir")
    if Path(args.output).exists() and (not Path(args.output).is_dir() or any(Path(args.output).iterdir())):
        parser.error("--output must be a new or empty directory")

    import megatron.bridge.models.speech_cascades.bridge  # noqa: F401
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401

    bridge = AutoBridge.from_hf_pretrained(args.hf_dir)
    if args.checkpoint:
        bridge.export_ckpt(args.checkpoint, args.output, strict=True, source_path=args.hf_dir)
        if args.vllm:
            prepare_vllm_config(Path(args.output))
        logger.info("Exported trained checkpoint %s to %s", args.checkpoint, args.output)
        return

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    ps.initialize_model_parallel(1, 1)
    torch.cuda.set_device(0)
    model_parallel_cuda_manual_seed(0)

    model = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)
    bridge.save_hf_pretrained(model, args.output)
    if ps.get_data_parallel_rank() == 0:
        logger.info("Exported initial HF round-trip to %s (no trained checkpoint supplied)", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
