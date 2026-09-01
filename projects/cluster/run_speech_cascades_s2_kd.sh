#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native Megatron-Bridge stage-2 speech->text knowledge-distillation + MTP training on Singularity
# (aiservices-vc02-eu2 / eus2). Freezes the stage-1 Conformer encoder and the Qwen3.5 language model
# and trains only the audio projector plus the MTP depth heads, with the loss = main-path KL+CE
# distillation (teacher = text-forward of the frozen LM with audio spans replaced by transcripts) plus
# the per-depth MTP self-distillation. Two phases mirror the reference
# qwen3-5-4b-s2-proj20m-p1-mtp4-kd-{warm500,full4000}: a short warm phase then the full run initialized
# from the warm checkpoint. Streams the S2 KD data mixture (projects/cluster/data_en_s2_kd.yaml, with
# knowledge_distillation: True). Under mpi: True + process_count_per_node the platform sets the
# torch.distributed env vars, so finetune.py is launched directly.
set -e -o xtrace

cd "$(dirname "$0")/../.."

if [[ -d /opt/bridge-overlay ]]; then
  OVERLAY=/opt/bridge-overlay
else
  OVERLAY=${OVERLAY_DIR:-$HOME/bridge-overlay}
  if [[ "${LOCAL_RANK:-0}" == "0" && ! -f "${OVERLAY}/.done" ]]; then
    python -m pip install --target "${OVERLAY}" --upgrade "numpy==1.26.4" "transformers>=5.8.1,<5.9.0"
    python -m pip install --target "${OVERLAY}" --upgrade --no-deps \
      --extra-index-url https://pypi.nvidia.com "nvidia-resiliency-ext==0.6.0" "nvidia-modelopt==0.44.0rc5"
    touch "${OVERLAY}/.done"
  fi
  for _ in $(seq 1 180); do [[ -f "${OVERLAY}/.done" ]] && break; sleep 5; done
fi

export PYTHONPATH=${OVERLAY}:$(pwd)/src:$(pwd)/3rdparty/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage-1 artifacts used to initialize stage-2: the HF directory supplies the model architecture
# (LM + encoder + projector); the trained Megatron checkpoint supplies the stage-1 weights. The MTP
# depth heads are added by the provider and initialized from the LM's first decoder layer.
S1_HF=${S1_HF:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-torch210-full-v3/stage1_hf}
S1_CKPT=${S1_CKPT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-torch210-full-v3/run/checkpoints/iter_0012000}
OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/s2-kd-mtp4-torch210}
DATA_YAML=${DATA_YAML:-projects/cluster/data_en_s2_kd.yaml}
TP=${TP:-1}
PP=${PP:-1}
SEQ_LEN=${SEQ_LEN:-8192}
GLOBAL_BSZ=${GLOBAL_BSZ:-64}
MICRO_BSZ=${MICRO_BSZ:-1}
WARM_ITERS=${WARM_ITERS:-500}
FULL_ITERS=${FULL_ITERS:-4000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LR=${LR:-4.0e-5}
LR_WARMUP=${LR_WARMUP:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}

WARM_CKPT="${OUTPUT}/warm/checkpoints/iter_$(printf '%07d' "${WARM_ITERS}")"

COMMON_ARGS=(
  --s2-kd
  --hf-path "${S1_HF}"
  --streaming-config-yaml "${DATA_YAML}"
  --activation-recompute
  --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --global-batch-size "${GLOBAL_BSZ}" --micro-batch-size "${MICRO_BSZ}" --seq-length "${SEQ_LEN}"
  --save-interval "${SAVE_INTERVAL}" --lr "${LR}" --lr-warmup-iters "${LR_WARMUP}"
  --mtp-num-hidden-layers 4 --mtp-loss-weight 1.0 --mtp-distillation-weight 0.75
  --mtp-loss-depth-weights 0.50 0.25 0.15 0.10
  --distillation-weight 0.75 --distillation-temperature 2.0 --distillation-kl-chunk-size 128
  --log-interval "${LOG_INTERVAL}"
)

# Phase 1 (warm): init the projector + MTP heads from the stage-1 checkpoint.
python examples/models/speech_cascades/finetune.py \
  "${COMMON_ARGS[@]}" \
  --kd-phase warm --warm-iters "${WARM_ITERS}" \
  --pretrained-checkpoint "${S1_CKPT}" \
  --output-dir "${OUTPUT}" --name warm

# Phase 2 (full): initialize from the warm checkpoint (weights only; optimizer / step reset).
python examples/models/speech_cascades/finetune.py \
  "${COMMON_ARGS[@]}" \
  --kd-phase full --full-iters "${FULL_ITERS}" \
  --init-checkpoint "${WARM_CKPT}" \
  --output-dir "${OUTPUT}" --name full
