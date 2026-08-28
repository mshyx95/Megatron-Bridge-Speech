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

"""Fine-tuning recipe for the Cascades speech audio-language model (Qwen3.5 LM + Cascades encoder).

Loading a previously trained model:

- **Cold start** from the imported 40k weights: convert the checkpoint once with
  ``examples/conversion/convert_checkpoints.py import`` and pass the resulting Megatron directory as
  ``pretrained_checkpoint``. Weights are loaded; optimizer / RNG / step are reset.
- **Resume** a run: ``checkpoint.load`` defaults to ``checkpoint.save`` so an interrupted run
  auto-resumes from its latest ``iter_*`` (weights + optimizer + RNG + step).
"""

import os
from typing import Any, Optional, Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.speech_chunk import ALMLocalDatasetProvider

# Import the bridge module so its @register_bridge decorator registers the
# Qwen3_5SpeechCascadesForConditionalGeneration architecture with AutoBridge.
from megatron.bridge.models.speech_cascades import bridge as _speech_cascades_bridge  # noqa: F401
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def speech_cascades_finetune_config(
    hf_path: str,
    train_shards: Optional[list[str]] = None,
    *,
    output_dir: str | None = None,
    name: str = "speech_cascades_finetune",
    pretrained_checkpoint: Optional[str] = None,
    load_hf_weights: bool = False,
    val_shards: Optional[list[str]] = None,
    # Real streaming speech data (mutually exclusive with train_shards)
    streaming_config_yaml: Optional[str] = None,
    streaming_force_local: bool = False,
    streaming_token_budget: Optional[int] = None,
    # Parallelism
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 8,
    global_batch_size: int = 1,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 5e-6,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 2,
    lr_decay_iters: Optional[int] = None,
    weight_decay: float = 0.01,
    adam_eps: float = 1e-7,
    lr_decay_style: str = "linear",
    eval_interval: int = 1000,
    save_interval: int = 4,
    pad_token_id: int = 0,
    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    # Freeze options
    freeze_language_model: bool = False,
    freeze_audio_model: bool = False,
    activation_recompute: bool = False,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
    # Console / MLflow logging cadence (MLflow attaches to the ambient AzureML run in obs_logging).
    log_interval: int = 10,
) -> ConfigContainer:
    """Create a fine-tuning configuration for the Cascades speech ALM.

    Args:
        hf_path: Converted HF directory (``architectures`` dispatches to ``SpeechCascadesBridge``);
            used to build the model provider (config only, weights loaded separately).
        train_shards: Local ``.pt`` ALM example-record shards for the training split.
        pretrained_checkpoint: Optional Megatron checkpoint directory to cold-start weights from.
        load_hf_weights: When ``True`` load the ``hf_path`` weights into the model at build time
            (cold-start directly from the converted HF directory, no Megatron conversion needed).
        val_shards: Optional validation shards.

    Returns:
        A fully populated :class:`ConfigContainer` for ``finetune`` / ``pretrain``.
    """
    base_output_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Build the SpeechCascades provider from the converted HF config (no weights here).
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=load_hf_weights)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.freeze_language_model = freeze_language_model
    if activation_recompute:
        # Full recompute (every layer): the frozen LLM's activations are still needed to backprop
        # into the audio encoder, so recomputing them keeps peak memory bounded.
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1
    model_cfg.freeze_audio_model = freeze_audio_model
    model_cfg.seq_length = seq_length

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        start_weight_decay=weight_decay,
        end_weight_decay=weight_decay,
        adam_eps=adam_eps,
        lr_decay_style=lr_decay_style,
    )

    dataset_cfg: Any
    if streaming_config_yaml is not None:
        from megatron.bridge.data.speech_chunk import SpeechChunkStreamingDatasetProvider

        dataset_cfg = SpeechChunkStreamingDatasetProvider(
            seq_length=seq_length,
            speech_config_yaml=streaming_config_yaml,
            force_local=streaming_force_local,
            token_budget_override=streaming_token_budget,
            num_workers=1,
            prefetch_factor=16,
            data_sharding=True,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        if not train_shards:
            raise ValueError("provide either train_shards or streaming_config_yaml")
        dataset_cfg = ALMLocalDatasetProvider(
            seq_length=seq_length,
            shard_paths=train_shards,
            val_shard_paths=val_shards,
            pad_token_id=pad_token_id,
            num_workers=0,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
        )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=0,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=log_interval,
            tensorboard_dir=tensorboard_dir,
            tensorboard_log_interval=1,
            timing_log_level=1,
            log_throughput=True,
            log_throughput_to_tensorboard=True,
            log_memory_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_exp_name=wandb_exp_name,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        mixed_precision=precision_config,
    )

    return cfg
