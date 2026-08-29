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

"""CLI entry to fine-tune the Cascades speech ALM with the native Megatron-Bridge path.

Loads a previously trained model either from a converted HF directory (``--load-hf-weights``) or a
Megatron checkpoint (``--pretrained-checkpoint``), and resumes automatically from ``checkpoint.load``
(defaults to the save directory). Launch under ``torch.distributed`` (or the platform's MPI env).

Example (single node):
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        examples/models/speech_cascades/finetune.py \
        --hf-path /path/to/converted_hf --train-shards /path/to/train_000.pt \
        --load-hf-weights --tensor-model-parallel-size 2 --train-iters 8
"""

import argparse
import os
import sys
import types


def _load_recipe():
    """Import the speech-cascades recipe without triggering the recipes aggregator.

    ``megatron.bridge.recipes.__init__`` eagerly imports every model family, several of which pull
    optional dependencies (``diffusers``, ``megatron.energon``, ...) that a minimal training image
    need not carry. Registering a lightweight stub package lets the submodule import resolve against
    the real source tree while skipping the aggregator's heavy imports.
    """
    import megatron.bridge

    if "megatron.bridge.recipes" not in sys.modules:
        stub = types.ModuleType("megatron.bridge.recipes")
        stub.__path__ = [os.path.join(os.path.dirname(megatron.bridge.__file__), "recipes")]
        sys.modules["megatron.bridge.recipes"] = stub

    from megatron.bridge.recipes.speech_cascades.speech_cascades import speech_cascades_finetune_config

    return speech_cascades_finetune_config


def main() -> None:
    """Build the speech-cascades fine-tune config and run the native training loop."""
    # Patch torch.distributed.checkpoint.load for PyTorch>=2.6 (drops the no_dist kwarg MCore passes).
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401
    from megatron.bridge.models.speech_cascades.obs_logging import install_speech_obs_logging
    from megatron.bridge.models.speech_cascades.step import forward_step
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.pretrain import pretrain

    # Add audio-hours / ppl to MLflow and a rank-0 loss line (built-in log prints on the last rank).
    install_speech_obs_logging()

    speech_cascades_finetune_config = _load_recipe()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Converted HF directory (model config + optional weights).")
    parser.add_argument("--train-shards", nargs="+", default=None, help="Local .pt ALM example-record shards.")
    parser.add_argument("--streaming-config-yaml", default=None, help="OmegaConf data config for real streaming data.")
    parser.add_argument("--streaming-force-local", action="store_true", help="Read streaming data from mounts.")
    parser.add_argument("--streaming-token-budget", type=int, default=None, help="Override per-batch frame budget.")
    parser.add_argument("--val-shards", nargs="+", default=None, help="Optional validation shards.")
    parser.add_argument("--output-dir", default=None, help="Experiment root (checkpoints + logs).")
    parser.add_argument("--name", default="speech_cascades_finetune")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Megatron checkpoint dir to cold-start from.")
    parser.add_argument("--load-hf-weights", action="store_true", help="Cold-start weights from --hf-path directly.")
    parser.add_argument("--freeze-language-model", action="store_true", help="Freeze the LLM (stage-1).")
    parser.add_argument("--freeze-audio-model", action="store_true", help="Freeze the audio encoder.")
    parser.add_argument("--activation-recompute", action="store_true", help="Full activation recompute.")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr-warmup-iters", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10, help="Console/MLflow logging cadence.")
    args = parser.parse_args()

    cfg = speech_cascades_finetune_config(
        hf_path=args.hf_path,
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        streaming_config_yaml=args.streaming_config_yaml,
        streaming_force_local=args.streaming_force_local,
        streaming_token_budget=args.streaming_token_budget,
        output_dir=args.output_dir,
        name=args.name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_hf_weights=args.load_hf_weights,
        freeze_language_model=args.freeze_language_model,
        freeze_audio_model=args.freeze_audio_model,
        activation_recompute=args.activation_recompute,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        save_interval=args.save_interval,
        lr=args.lr,
        lr_warmup_iters=args.lr_warmup_iters,
        log_interval=args.log_interval,
    )
    # With --load-hf-weights the provider already bakes weights into the model at build time,
    # so no checkpoint load is required; pretrain() still auto-resumes from checkpoint.load if
    # a prior iteration exists. finetune() (which hard-requires a checkpoint source) is used only
    # when cold-starting from a Megatron --pretrained-checkpoint.
    if args.load_hf_weights and not args.pretrained_checkpoint:
        pretrain(cfg, forward_step_func=forward_step)
    else:
        finetune(cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
