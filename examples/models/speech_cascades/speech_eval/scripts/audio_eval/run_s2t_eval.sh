#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
TASK=${TASK:-${1:-}}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH must be set}
TOKENIZER=${TOKENIZER:?TOKENIZER must be set}
DATASET_PATH=${DATASET_PATH:?DATASET_PATH must be set}
OUTPUT_DIR=${OUTPUT_DIR:-${AMLT_OUTPUT_DIR:-$REPO_ROOT/${TASK}_output}}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen3_5_audio_nothink_instruct}
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-0}}}

case "$TASK" in
    voicebench)
        task_name=voice_bench_s2t
        sub_task=s2t
        output_file="$OUTPUT_DIR/generate_voice_bench_s2t.json"
        instruction_first=true
        RUN_EVAL=${RUN_EVAL:-0}
        ;;
    mmsu)
        task_name=mmsu_s2t
        sub_task=s2t
        output_file="$OUTPUT_DIR/generate_mmsu_s2t.json"
        instruction_first=true
        RUN_EVAL=${RUN_EVAL:-1}
        ;;
    mmlu)
        task_name=mmlu_speech
        sub_task=dummy
        output_file="$OUTPUT_DIR/mmlu_speech_en.json"
        instruction_first=false
        RUN_EVAL=${RUN_EVAL:-1}
        ;;
    bigbench)
        task_name=bigbench_audio_s2t
        sub_task=s2t
        output_file="$OUTPUT_DIR/generate_bigbench_audio_s2t.json"
        instruction_first=true
        RUN_EVAL=${RUN_EVAL:-1}
        ;;
    *)
        echo "Usage: TASK={voicebench|mmsu|mmlu|bigbench} $0" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"
eval_args=(
    "$SCRIPT_DIR/eval.py" "$MODEL_PATH"
    -t "$TOKENIZER"
    -tn "$task_name"
    -st "$sub_task"
    -ct "$CHAT_TEMPLATE"
    -dp "$DATASET_PATH"
    -op "$output_file"
    -bs "${BATCH_SIZE:-8}"
    -mt "${MAX_NEW_TOKENS:-2048}"
)
if [[ "${MTP_SPECULATIVE:-false}" == "True" || "${MTP_SPECULATIVE:-false}" == "true" || "${MTP_SPECULATIVE:-false}" == "1" ]]; then
    eval_args+=(-mtp_spec true)
fi
if [[ "$instruction_first" == "true" ]]; then
    eval_args+=(-if true)
fi
"$PYTHON" "${eval_args[@]}"

if [[ "$rank" == "0" && ( "${MTP_SPECULATIVE:-false}" == "True" || "${MTP_SPECULATIVE:-false}" == "true" || "${MTP_SPECULATIVE:-false}" == "1" ) ]]; then
    "$PYTHON" "$SCRIPT_DIR/summarize_mtp_spec_stats.py" "$OUTPUT_DIR"
fi

if [[ "$rank" != "0" || "${POSTPROCESS:-1}" != "1" ]]; then
    exit 0
fi

case "$TASK" in
    voicebench)
        "$PYTHON" "$SCRIPT_DIR/speech_out/convert_voice_bench.py" \
            --root_dir "$(dirname "$DATASET_PATH")" \
            --results_dir "$OUTPUT_DIR" \
            --s2t_results_file "$output_file"
        if [[ "$RUN_EVAL" == "1" ]]; then
            evaluation_args=(
                --results-dir "$OUTPUT_DIR"
                --output-dir "$OUTPUT_DIR/evaluation"
            )
            if [[ -n "${VOICEBENCH_OFFICIAL_REPO:-}" ]]; then
                evaluation_args+=(--official-repo "$VOICEBENCH_OFFICIAL_REPO")
            fi
            if [[ "${RUN_JUDGE:-0}" == "1" ]]; then
                evaluation_args+=(--run-judge)
            fi
            "$PYTHON" "$SCRIPT_DIR/evaluate_voicebench_s2t.py" \
                "${evaluation_args[@]}"
        fi
        ;;
    mmsu)
        if [[ "$RUN_EVAL" == "1" ]]; then
            "$PYTHON" "$SCRIPT_DIR/evaluate_mmsu_s2t.py" \
                --prediction-file "$output_file" \
                --manifest-file "$DATASET_PATH" \
                --output-dir "$OUTPUT_DIR/evaluation"
        fi
        ;;
    mmlu)
        if [[ "$RUN_EVAL" == "1" ]]; then
            "$PYTHON" "$SCRIPT_DIR/report_mmlu.py" \
                "$output_file" \
                "$(dirname "$DATASET_PATH")" \
                "$OUTPUT_DIR"
        fi
        ;;
    bigbench)
        judge_input="$OUTPUT_DIR/eval_results.json"
        "$PYTHON" "$SCRIPT_DIR/format_bigbench_audio_judge_input.py" \
            --manifest "$DATASET_PATH" \
            --predictions "$output_file" \
            --output "$judge_input"
        if [[ "${RUN_JUDGE:-1}" == "1" ]]; then
            GPT_JUDGE_SCRIPT=${GPT_JUDGE_SCRIPT:?GPT_JUDGE_SCRIPT must be set when RUN_JUDGE=1}
            "$PYTHON" "$GPT_JUDGE_SCRIPT" \
                --input_path "$judge_input" \
                --output_path "$OUTPUT_DIR/output_judged.json" \
                --model_name "${GPT_JUDGE_MODEL:-gpt-4.1}" \
                --num_workers "${GPT_JUDGE_WORKERS:-2}"
        fi
        ;;
esac