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
    """Import the speech-cascades recipe module without triggering the recipes aggregator.

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

    from megatron.bridge.recipes.speech_cascades import speech_cascades as recipe_module

    return recipe_module


def _apply_train_config_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Fill args from a training-config yaml for every option not passed explicitly on the CLI.

    Keeps data (``--streaming-config-yaml``) and run-specific paths on the CLI while the reusable
    training/model hyperparameters live in the yaml. CLI flags win over yaml; yaml wins over argparse
    defaults. Unknown yaml keys raise, so a typo can never silently create a phantom setting.
    """
    if not args.train_config_yaml:
        return
    import yaml

    with open(args.train_config_yaml) as handle:
        train_cfg = yaml.safe_load(handle) or {}
    if not isinstance(train_cfg, dict):
        parser.error(f"--train-config-yaml must be a mapping, got {type(train_cfg).__name__}")

    passed = {token[2:].split("=", 1)[0].replace("-", "_") for token in sys.argv[1:] if token.startswith("--")}
    for key, value in train_cfg.items():
        dest = str(key).replace("-", "_")
        if not hasattr(args, dest):
            parser.error(f"--train-config-yaml has unknown key '{key}'")
        if dest not in passed:
            setattr(args, dest, value)


def main() -> None:
    """Build the speech-cascades fine-tune config and run the native training loop."""
    # Patch torch.distributed.checkpoint.load for PyTorch>=2.6 (drops the no_dist kwarg MCore passes).
    from megatron.bridge.models.speech_cascades import dcp_compat  # noqa: F401
    from megatron.bridge.models.speech_cascades.obs_logging import install_speech_obs_logging
    from megatron.bridge.models.speech_cascades.peft import SpeechCascadesLoRA
    from megatron.bridge.models.speech_cascades.step import forward_step
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.pretrain import pretrain

    # Add audio-hours / ppl to MLflow and a rank-0 loss line (built-in log prints on the last rank).
    install_speech_obs_logging()

    recipe_module = _load_recipe()
    speech_cascades_finetune_config = recipe_module.speech_cascades_finetune_config
    speech_cascades_s2_kd_config = recipe_module.speech_cascades_s2_kd_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Converted HF directory (model config + optional weights).")
    parser.add_argument("--train-shards", nargs="+", default=None, help="Local .pt ALM example-record shards.")
    parser.add_argument("--streaming-config-yaml", default=None, help="OmegaConf data config for real streaming data.")
    parser.add_argument(
        "--train-config-yaml",
        default=None,
        help="Training/model config yaml (recipe hyperparameters). CLI flags override yaml values.",
    )
    parser.add_argument("--streaming-force-local", action="store_true", help="Read streaming data from mounts.")
    parser.add_argument(
        "--streaming-use-storage-api",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override runtime storage transport without rewriting the source data config.",
    )
    parser.add_argument("--streaming-token-budget", type=int, default=None, help="Override per-batch frame budget.")
    parser.add_argument("--streaming-num-workers", type=int, default=1)
    parser.add_argument("--streaming-prefetch-factor", type=int, default=16)
    parser.add_argument("--val-shards", nargs="+", default=None, help="Optional validation shards.")
    parser.add_argument("--output-dir", default=None, help="Experiment root (checkpoints + logs).")
    parser.add_argument("--name", default="speech_cascades_finetune")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Megatron checkpoint dir to cold-start from.")
    parser.add_argument("--load-hf-weights", action="store_true", help="Cold-start weights from --hf-path directly.")
    parser.add_argument("--freeze-language-model", action="store_true", help="Freeze the LLM (stage-1).")
    parser.add_argument("--freeze-audio-model", action="store_true", help="Freeze the audio encoder.")
    parser.add_argument("--activation-recompute", action="store_true", help="Full activation recompute.")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "flash", "fused", "unfused", "local"],
        default=None,
        help="Attention backend override for the language model.",
    )
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--expert-model-parallel-size", type=int, default=1)
    parser.add_argument("--expert-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=None, help="Iterations (defaults per recipe/phase).")
    parser.add_argument(
        "--exit-interval", type=int, default=None, help="Save and exit early without shortening the LR schedule."
    )
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--save-interval", type=int, default=None, help="Checkpoint cadence (defaults per recipe).")
    parser.add_argument("--lr", type=float, default=None, help="Peak LR (defaults per recipe).")
    parser.add_argument("--lr-warmup-iters", type=int, default=None, help="LR warmup (defaults per recipe).")
    parser.add_argument("--log-interval", type=int, default=10, help="Console/MLflow logging cadence.")
    parser.add_argument("--timing-log-level", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--overlap-grad-reduce", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--normalize-moe-lora", action="store_true")
    parser.add_argument("--per-expert-lora", action="store_true")
    # --- Stage-2 knowledge distillation + MTP (WP-A3) ---
    parser.add_argument("--s2-kd", action="store_true", help="Use the stage-2 KD+MTP recipe (freeze encoder + LLM).")
    parser.add_argument("--kd-phase", choices=["warm", "full"], default="full", help="Two-phase KD schedule.")
    parser.add_argument("--warm-iters", type=int, default=500, help="Warm-phase iterations.")
    parser.add_argument("--full-iters", type=int, default=4000, help="Full-phase iterations.")
    parser.add_argument("--init-checkpoint", default=None, help="Warm checkpoint to init the full phase from.")
    parser.add_argument("--distillation-weight", type=float, default=0.75, help="Main-path KL vs CE mix weight.")
    parser.add_argument("--distillation-temperature", type=float, default=2.0, help="KD softmax temperature.")
    parser.add_argument("--distillation-kl-chunk-size", type=int, default=128, help="KL vocab chunk size.")
    parser.add_argument("--mtp-num-hidden-layers", type=int, default=4, help="Number of MTP depth heads.")
    parser.add_argument("--mtp-loss-weight", type=float, default=1.0, help="Overall MTP loss scale.")
    parser.add_argument(
        "--mtp-loss-depth-weights",
        type=float,
        nargs="+",
        default=[0.50, 0.25, 0.15, 0.10],
        help="Per-depth MTP loss weights.",
    )
    parser.add_argument("--mtp-distillation-weight", type=float, default=0.75, help="MTP per-depth KL vs CE mix.")
    args = parser.parse_args()

    _apply_train_config_yaml(args, parser)
    if args.exit_interval is not None and args.exit_interval <= 0:
        parser.error("--exit-interval must be positive")

    peft = None
    if args.lora_rank is not None:
        if args.lora_rank <= 0:
            parser.error("--lora-rank must be positive")
        if args.pretrained_checkpoint is None:
            parser.error("--lora-rank requires --pretrained-checkpoint")
        if args.s2_kd:
            parser.error("--lora-rank is not supported together with --s2-kd")
        peft = SpeechCascadesLoRA(
            dim=args.lora_rank,
            alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank,
            dropout=args.lora_dropout,
            normalize_moe_lora=args.normalize_moe_lora,
            share_expert_adapters=not args.per_expert_lora,
        )

    if args.s2_kd:
        if args.streaming_config_yaml is None:
            parser.error("--s2-kd requires --streaming-config-yaml")
        kd_kwargs = {}
        if args.train_iters is not None:
            kd_kwargs["train_iters"] = args.train_iters
        if args.save_interval is not None:
            kd_kwargs["save_interval"] = args.save_interval
        if args.lr is not None:
            kd_kwargs["lr"] = args.lr
        if args.lr_warmup_iters is not None:
            kd_kwargs["lr_warmup_iters"] = args.lr_warmup_iters
        cfg = speech_cascades_s2_kd_config(
            hf_path=args.hf_path,
            streaming_config_yaml=args.streaming_config_yaml,
            streaming_force_local=args.streaming_force_local,
            streaming_use_storage_api=args.streaming_use_storage_api,
            streaming_token_budget=args.streaming_token_budget,
            streaming_num_workers=args.streaming_num_workers,
            streaming_prefetch_factor=args.streaming_prefetch_factor,
            output_dir=args.output_dir,
            name=args.name,
            load_hf_weights=args.load_hf_weights,
            pretrained_checkpoint=args.pretrained_checkpoint,
            phase=args.kd_phase,
            init_checkpoint=args.init_checkpoint,
            warm_iters=args.warm_iters,
            full_iters=args.full_iters,
            distillation_weight=args.distillation_weight,
            distillation_temperature=args.distillation_temperature,
            distillation_kl_chunk_size=args.distillation_kl_chunk_size,
            mtp_num_hidden_layers=args.mtp_num_hidden_layers,
            mtp_loss_weight=args.mtp_loss_weight,
            mtp_loss_depth_weights=tuple(args.mtp_loss_depth_weights),
            mtp_distillation_weight=args.mtp_distillation_weight,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            sequence_parallel=args.sequence_parallel,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            activation_recompute=args.activation_recompute,
            log_interval=args.log_interval,
            timing_log_level=args.timing_log_level,
            **kd_kwargs,
        )
    else:
        cfg = speech_cascades_finetune_config(
            hf_path=args.hf_path,
            train_shards=args.train_shards,
            val_shards=args.val_shards,
            streaming_config_yaml=args.streaming_config_yaml,
            streaming_force_local=args.streaming_force_local,
            streaming_use_storage_api=args.streaming_use_storage_api,
            streaming_token_budget=args.streaming_token_budget,
            streaming_num_workers=args.streaming_num_workers,
            streaming_prefetch_factor=args.streaming_prefetch_factor,
            output_dir=args.output_dir,
            name=args.name,
            pretrained_checkpoint=args.pretrained_checkpoint,
            load_hf_weights=args.load_hf_weights,
            freeze_language_model=args.freeze_language_model,
            freeze_audio_model=args.freeze_audio_model,
            activation_recompute=args.activation_recompute,
            peft=peft,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            sequence_parallel=args.sequence_parallel,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            train_iters=args.train_iters if args.train_iters is not None else 8,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            save_interval=args.save_interval if args.save_interval is not None else 4,
            lr=args.lr if args.lr is not None else 5e-6,
            lr_warmup_iters=args.lr_warmup_iters if args.lr_warmup_iters is not None else 2,
            log_interval=args.log_interval,
            timing_log_level=args.timing_log_level,
        )
    if args.attention_backend is not None:
        cfg.model.attention_backend = args.attention_backend
    if args.overlap_grad_reduce is not None:
        cfg.ddp.overlap_grad_reduce = args.overlap_grad_reduce
    if args.exit_interval is not None:
        cfg.train.exit_interval = args.exit_interval
    # With --load-hf-weights the provider already bakes weights into the model at build time,
    # so no checkpoint load is required; pretrain() still auto-resumes from checkpoint.load if
    # a prior iteration exists. finetune() (which hard-requires a checkpoint source) is used only
    # when cold-starting from a Megatron --pretrained-checkpoint.
    if args.load_hf_weights and not args.pretrained_checkpoint and not args.init_checkpoint:
        pretrain(cfg, forward_step_func=forward_step)
    else:
        finetune(cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
