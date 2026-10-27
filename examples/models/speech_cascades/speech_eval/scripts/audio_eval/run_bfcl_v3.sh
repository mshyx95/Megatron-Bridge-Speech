#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
MODEL_FAMILY=${MODEL_FAMILY:?MODEL_FAMILY must be lfg3, qwen35-audio, or qwen35-text}
MODEL=${MODEL:?MODEL must point to the model or checkpoint}
DATA_DIR=${DATA_DIR:?DATA_DIR must contain BFCL_v3_<category>.json files}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set}
CATEGORIES=${CATEGORIES:-"simple multiple parallel parallel_multiple irrelevance"}
NUM_WORKERS=${NUM_WORKERS:-1}
BATCH_SIZE=${BATCH_SIZE:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

read -r -a CATEGORY_ARGS <<< "$CATEGORIES"
if ((NUM_WORKERS < 1 || NUM_WORKERS > ${#CATEGORY_ARGS[@]})); then
    echo "NUM_WORKERS must be between 1 and ${#CATEGORY_ARGS[@]}" >&2
    exit 2
fi
if [[ "$MODEL_FAMILY" == "qwen35-audio" && -z "${TOKENIZER:-}" ]]; then
    echo "TOKENIZER must be set for MODEL_FAMILY=qwen35-audio" >&2
    exit 2
fi

if [[ "${PREPARE_BFCL:-0}" == "1" ]]; then
    SOURCE_DIR=${SOURCE_DIR:?SOURCE_DIR must be set when PREPARE_BFCL=1}
    "$PYTHON" "$SCRIPT_DIR/prepare_bfcl_speech.py" \
        --source-dir "$SOURCE_DIR" \
        --output-dir "$DATA_DIR" \
        --categories "${CATEGORY_ARGS[@]}" \
        --limit-per-category "${LIMIT_PER_CATEGORY:-0}" \
        --seed "${SEED:-42}" \
        --audio-mode "${AUDIO_MODE:-reference}"
fi

mkdir -p "$OUTPUT_DIR/logs"
common_args=(
    --model-family "$MODEL_FAMILY"
    --model "$MODEL"
    --data-dir "$DATA_DIR"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --device-map "${DEVICE_MAP:-auto}"
    --dtype "${DTYPE:-bfloat16}"
    --attn "${ATTN:-flash_attention_2}"
)
if [[ -n "${TOKENIZER:-}" ]]; then
    common_args+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "${ASR_TRANSCRIPTS:-}" ]]; then
    if [[ "$MODEL_FAMILY" != "qwen35-text" ]]; then
        echo "ASR_TRANSCRIPTS requires MODEL_FAMILY=qwen35-text" >&2
        exit 2
    fi
    common_args+=(--asr-transcripts "$ASR_TRANSCRIPTS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
    common_args+=(--batch-size "$BATCH_SIZE")
fi
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then
    common_args+=(--enable-thinking)
fi
if [[ "${NO_RESUME:-0}" == "1" ]]; then
    common_args+=(--no-resume)
fi

pids=()
worker_outputs=()
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
    worker_categories=()
    for index in "${!CATEGORY_ARGS[@]}"; do
        if ((index % NUM_WORKERS == worker)); then
            worker_categories+=("${CATEGORY_ARGS[$index]}")
        fi
    done
    worker_output="$OUTPUT_DIR/responses.worker${worker}.json"
    worker_outputs+=("$worker_output")
    echo "Launching worker $worker on GPU $worker: ${worker_categories[*]}"
    CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" "$SCRIPT_DIR/eval_bfcl.py" \
        "${common_args[@]}" \
        --output "$worker_output" \
        --categories "${worker_categories[@]}" \
        > >(tee "$OUTPUT_DIR/logs/worker${worker}.log") 2>&1 &
    pids+=("$!")
done

status=0
for worker in "${!pids[@]}"; do
    if ! wait "${pids[$worker]}"; then
        echo "Worker $worker failed" >&2
        status=1
    fi
done
if ((status != 0)); then
    exit "$status"
fi

if [[ "${SPEECH_EVAL_WORLD_SIZE:-1}" != 1 ]]; then
    exit 0
fi

"$PYTHON" - "$OUTPUT_DIR/responses.json" "${worker_outputs[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
records = []
for source in map(Path, sys.argv[2:]):
    records.extend(json.loads(source.read_text(encoding="utf-8")))
records.sort(key=lambda row: row["audio_ids"][0])
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
temporary.replace(output)
print(f"Merged {len(records)} records into {output}")
PY

"$PYTHON" "$SCRIPT_DIR/report_bfcl.py" \
    "$OUTPUT_DIR/responses.json" \
    "$DATA_DIR" \
    "$OUTPUT_DIR/report"