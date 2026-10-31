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

# Model source for stage-2. Two modes:
#   (a) from scratch  — build the HF dir from base Qwen3.5 + the cascades encoder (random projector),
#       leave S1_CKPT empty, and cold-start the warm phase via --load-hf-weights.
#   (b) from stage-1  — point MODEL_HF at a trained stage1_hf and S1_CKPT at its Megatron checkpoint.
BASE_HF=${BASE_HF:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
ENC_CKPT=${ENC_CKPT:-/datablob1/users/ruchaofan/cascades_speech_encoder/checkpoint_22L/model.checkpoint}
OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/s2-kd-mtp4-torch210}
MODEL_HF=${MODEL_HF:-${OUTPUT}/stage1_hf}
S1_CKPT=${S1_CKPT:-}
# Reusable configs: TRAIN_CONFIG holds the model/training hyperparameters, DATA_CONFIG the data mixture.
TRAIN_CONFIG=${TRAIN_CONFIG:-projects/cluster/train_s2_kd_mtp4.yaml}
DATA_CONFIG=${DATA_CONFIG:-projects/cluster/data_en_s2_kd.yaml}
# MTP depth heads: set MTP_LAYERS=0 to train the projector only (the MTP+HF-load bridge path is WIP).
MTP_LAYERS=${MTP_LAYERS:-4}

# Prep (rank 0): build MODEL_HF from base Qwen3.5 + the cascades encoder when it does not exist yet
# (random projector). Skipped when MODEL_HF already points at a trained stage1_hf.
RANK=${RANK:-0}
PREP_DONE=${OUTPUT}/prep.done
if [[ "${RANK}" == "0" && ! -f "${PREP_DONE}" && ! -d "${MODEL_HF}" ]]; then
  mkdir -p "${OUTPUT}"
  python examples/conversion/convert_speech_cascades_stage1_to_hf.py \
    --base-hf "${BASE_HF}" --encoder-checkpoint "${ENC_CKPT}" --output-dir "${MODEL_HF}"
  touch "${PREP_DONE}"
fi
if [[ ! -d "${MODEL_HF}" ]]; then
  for _ in $(seq 1 360); do [[ -f "${PREP_DONE}" || -d "${MODEL_HF}" ]] && break; sleep 5; done
fi

# The warm checkpoint path depends on warm_iters, which lives in TRAIN_CONFIG (single source of truth).
WARM_ITERS=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('warm_iters',500))" "${TRAIN_CONFIG}")
WARM_CKPT="${OUTPUT}/warm/checkpoints/iter_$(printf '%07d' "${WARM_ITERS}")"

# Hyperparameters (KD + MTP + optimizer + parallelism) come from TRAIN_CONFIG; only the run-specific
# data, model, phase, MTP override and output are passed on the CLI (CLI wins over the yaml).
COMMON_ARGS=(
  --train-config-yaml "${TRAIN_CONFIG}"
  --hf-path "${MODEL_HF}"
  --streaming-config-yaml "${DATA_CONFIG}"
  --mtp-num-hidden-layers "${MTP_LAYERS}"
)

# Phase 1 (warm): cold-start from MODEL_HF (random projector) via --load-hf-weights, or init from a
# stage-1 Megatron checkpoint when S1_CKPT is set.
WARM_INIT=(--load-hf-weights)
if [[ -n "${S1_CKPT}" ]]; then
  WARM_INIT=(--load-hf-weights --pretrained-checkpoint "${S1_CKPT}")
fi
python examples/models/speech_cascades/finetune.py \
  "${COMMON_ARGS[@]}" \
  --kd-phase warm \
  "${WARM_INIT[@]}" \
  --output-dir "${OUTPUT}" --name warm

# Phase 2 (full): initialize from the warm checkpoint (weights only; optimizer / step reset).
python examples/models/speech_cascades/finetune.py \
  "${COMMON_ARGS[@]}" \
  --kd-phase full \
  --init-checkpoint "${WARM_CKPT}" \
  --output-dir "${OUTPUT}" --name full
