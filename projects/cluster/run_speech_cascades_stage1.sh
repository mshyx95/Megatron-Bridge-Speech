#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native Megatron-Bridge EN-ASR stage-1 training (encoder + MLP projector; LLM frozen) on real
# streaming speech data, launched on Singularity/amlt (aiservices-vc02-eu2). Under mpi: True +
# process_count_per_node, the platform sets the torch.distributed env vars, so finetune.py is launched
# directly. The primary rank converts the base Qwen3.5-4B + 22L encoder into the combined stage-1 HF
# checkpoint once; all ranks then stream real English ASR data through the vendored speech-chunk
# pipeline.
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

BASE_HF=${BASE_HF:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
ENC_CKPT=${ENC_CKPT:-/datablob1/users/ruchaofan/cascades_speech_encoder/checkpoint_22L/model.checkpoint}
OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-smoke}
# Reusable configs: TRAIN_CONFIG holds the model/training hyperparameters, DATA_CONFIG the data mixture.
TRAIN_CONFIG=${TRAIN_CONFIG:-projects/cluster/train_en_asr_s1.yaml}
DATA_CONFIG=${DATA_CONFIG:-projects/cluster/data_en_asr_s1.yaml}

export STAGE1_HF=${OUTPUT}/stage1_hf
PREP_DONE=${OUTPUT}/prep.done

RANK=${RANK:-0}
if [[ "${RANK}" == "0" && ! -f "${PREP_DONE}" ]]; then
  mkdir -p "${OUTPUT}"
  python examples/conversion/convert_speech_cascades_stage1_to_hf.py \
    --base-hf "${BASE_HF}" --encoder-checkpoint "${ENC_CKPT}" --output-dir "${STAGE1_HF}"
  touch "${PREP_DONE}"
fi
for _ in $(seq 1 360); do [[ -f "${PREP_DONE}" ]] && break; sleep 5; done

# Hyperparameters (freeze/optimizer/iters/parallelism) come from TRAIN_CONFIG; only the run-specific
# model dir, data and output are passed on the CLI.
python examples/models/speech_cascades/finetune.py \
  --train-config-yaml "${TRAIN_CONFIG}" \
  --hf-path "${STAGE1_HF}" \
  --load-hf-weights \
  --streaming-config-yaml "${DATA_CONFIG}" \
  --output-dir "${OUTPUT}" --name run
