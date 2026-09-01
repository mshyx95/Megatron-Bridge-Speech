#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BigBench-Audio (BBH) speech-to-text eval of a native speech-cascades checkpoint on Singularity.
# Mirrors run_speech_cascades_openasr.sh: one process loads the 4.3B model once and greedily answers
# each spoken-question utterance in the BBH manifest, sharded across the data-parallel ranks. Rank 0
# writes predictions.json + judge_input.json; the GPT-4.1 CORRECT/INCORRECT accuracy is computed by
# bbh_gpt_judge.py (optionally in the same job when AZURE_OPENAI_ENDPOINT/credentials are exported).
set -e -o xtrace

cd "$(dirname "$0")/../.."

if [[ -d /opt/bridge-overlay ]]; then
  OVERLAY=/opt/bridge-overlay
else
  OVERLAY=${OVERLAY_DIR:-$HOME/bridge-overlay}
fi

export PYTHONPATH=${OVERLAY}:$(pwd)/src:$(pwd)/3rdparty/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
# Eager decode: torch.compile/dynamo recompiles per audio length during variable-length generation,
# which dominated runtime and desynced ranks past the NCCL init timeout on the large subsets.
export TORCHDYNAMO_DISABLE=1

OUTPUT=${OUTPUT:-/modelblob/projects/yuxuanhu_s2s_qwen/bridge-results/en-asr-s1-v2}
CHECKPOINT_ITER=${CHECKPOINT_ITER:-12000}
MANIFEST=${MANIFEST:-/datablob1/users/ruchaofan/Evaluation/BigBenchAudio/manifest.jsonl}
AUDIO_ROOT=${AUDIO_ROOT:-}
TOKENIZER=${TOKENIZER:-/datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-4B}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

# Model to evaluate: set MODEL_HF + MODEL_CKPT to eval any model; they default to the OUTPUT+ITER layout.
MODEL_HF=${MODEL_HF:-${OUTPUT}/stage1_hf}
MODEL_CKPT=${MODEL_CKPT:-$(printf "%s/run/checkpoints/iter_%07d" "${OUTPUT}" "${CHECKPOINT_ITER}")}
RESULT_DIR=${RESULT_DIR:-${OUTPUT}/eval_output/bbh_iter${CHECKPOINT_ITER}}
mkdir -p "${RESULT_DIR}"

AUDIO_ROOT_ARG=()
if [[ -n "${AUDIO_ROOT}" ]]; then
  AUDIO_ROOT_ARG=(--audio-root "${AUDIO_ROOT}")
fi

# One process loads the 4.3B model once and answers every BBH utterance (avoids the per-run model
# reload + NCCL re-init that tripped the init timeout). Writes predictions.json + judge_input.json.
python examples/models/speech_cascades/eval_bbh.py \
  --hf-path "${MODEL_HF}" \
  --checkpoint "${MODEL_CKPT}" \
  --manifest "${MANIFEST}" \
  "${AUDIO_ROOT_ARG[@]}" \
  --tokenizer "${TOKENIZER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${RESULT_DIR}"

# GPT-4.1 judge (rank 0 only). Requires AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (or
# AZURE_OPENAI_BEARER_TOKEN) in the env; never hardcode secrets. Skipped when no endpoint is set.
if [[ "${RANK:-0}" == "0" && -n "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
  python examples/models/speech_cascades/bbh_gpt_judge.py \
    --input "${RESULT_DIR}/judge_input.json" \
    --cache "${RESULT_DIR}/judge_cache.jsonl" \
    --output "${RESULT_DIR}/bbh_accuracy.json"
else
  echo "[bbh] skipping GPT-4.1 judge (set AZURE_OPENAI_ENDPOINT + credentials to enable)."
fi
