#!/bin/bash

set -e -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-0}}}

# export PYTHONPATH=/scratch/amlt_code
# cd /scratch/amlt_code/scripts/audio_eval/

exp_id=$1
exp_name=$2
checkpoints=$3
tokenizer=$4
eval_set=$5
n_beams=${6:-1}
nrns=${7:-0}
use_lid=${8:-false}
chat_template=${9:-"phi_3"}
instruction_first=${10:-"false"}
tag=${11:-$(date +%Y%m%d%H%M)}
suffix=${12:-""}
batch_size=${13:-1}
chunk_type=${14:-"svad"}
ter_type=${15:-"ter"}  # not effective for this script, just for downstream ter evaluation script
data_path_prefix=${16:-"/datablob1"}
asr_prompt=${ASR_PROMPT:-${17:-"default"}}
mtp_speculative=${MTP_SPECULATIVE:-false}
mtp_args=()
if [[ "${mtp_speculative}" == "True" || "${mtp_speculative}" == "true" || "${mtp_speculative}" == "1" ]]; then
    mtp_args=(-mtp_spec true)
fi

summarize_mtp_stats() {
    local result_dir=$1
    if [[ ${#mtp_args[@]} -gt 0 && "$rank" == "0" ]]; then
        python "$script_dir/summarize_mtp_spec_stats.py" "$result_dir"
    fi
}

tok_dir=$tokenizer
temperature=1.0

[[ "$eval_set" == openasr || "$eval_set" == openasr_ml || "$eval_set" == openasr_long ]] || exit 2
if [[ "${SPEECH_SMOKE_SAMPLES:-0}" != 0 ]]; then
    [[ "$eval_set" == openasr && "$SPEECH_SMOKE_SAMPLES" =~ ^[0-9]+$ ]] || exit 2
    ((SPEECH_SMOKE_SAMPLES >= 1 && SPEECH_SMOKE_SAMPLES <= 128)) || exit 2
fi
eval_script="eval.py"
for checkpoint in $checkpoints; do
    model_rel_path="projects/${exp_name}/amlt-results/${exp_id}/${checkpoint}"
    chkp_dir=${MODEL:-/modelblob/$model_rel_path}
    if [ $eval_set == "openasr" ]; then
        max_new_tokens=${OPENASR_MAX_NEW_TOKENS:-128}
        # huggingface open ASR Leaderboard (updated June 2026)
        # Removed: tedlium (license), voxpopuli (replaced by AA cleaned)
        # VoxPopuli-Cleaned-AA merged from sr_open_aa
        # AMI and Gigaspeech use cleaned splits (>1s) via openasr_clean task
        datadir="${OPENASR_ESB_ROOT:?Set OPENASR_ESB_ROOT}"
        aa_datadir="${OPENASR_AA_ROOT:?Set OPENASR_AA_ROOT}"
        subsets="librispeech-clean librispeech-other earnings22 spgispeech"
        subsets_clean="ami gigaspeech"
        subsets_aa="VoxPopuli-Cleaned-AA"
        if [[ "${SPEECH_SMOKE_SAMPLES:-0}" != 0 ]]; then
            subsets="librispeech-clean"
            subsets_clean=""
            subsets_aa=""
        fi
        result_dir=${OUTPUT_DIR:-/modelblob/${model_rel_path}/eval_output/sr_openasr${suffix}/${tag}}
        mkdir -p $result_dir

        # Standard subsets (no duration filter)
        for set in $subsets; do
            dataset_path="$datadir $set"
            result_path=$result_dir/generate_${set}.txt

            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr" \
                -st "hfwer_en" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done

        # Cleaned subsets (min 1s duration filter)
        for set in $subsets_clean; do
            dataset_path="$datadir $set"
            result_path=$result_dir/generate_${set}_clean.txt

            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr_clean" \
                -st "hfwer_en" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done

        # AA cleaned VoxPopuli (replaces original voxpopuli)
        for set in $subsets_aa; do
            dataset_path="$aa_datadir $set"
            result_path=$result_dir/generate_${set}.txt

            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr_aa" \
                -st "hfwer_en" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done
        summarize_mtp_stats "$result_dir"
    fi

    if [ $eval_set == "openasr_ml" ]; then
        max_new_tokens=${OPENASR_MAX_NEW_TOKENS:-128}
        # huggingface open ASR Leaderboard
        datadir="${OPENASR_ML_ROOT:?Set OPENASR_ML_ROOT}"
        subsets="fleurs_de fleurs_es fleurs_fr fleurs_it fleurs_pt mcv_de mcv_es mcv_fr mcv_it mcv_pt mls_es mls_fr mls_it mls_pt"
        result_dir=${OUTPUT_DIR:-/modelblob/${model_rel_path}/eval_output/sr_openasr_ml${suffix}/${tag}}
        mkdir -p $result_dir

        for set in $subsets; do
            dataset_path="$datadir $set"
            result_path=$result_dir/generate_${set}.txt
            lang=${set##*_}
            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr_ml" \
                -st "hfwer_${lang}" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done
        summarize_mtp_stats "$result_dir"
    fi

    if [ $eval_set == "openasr_long" ]; then
        max_new_tokens=${OPENASR_LONG_MAX_NEW_TOKENS:-512}
        # huggingface open ASR Leaderboard (long-form, updated June 2026)
        # Removed: tedlium (license)
        # Added: Earnings22-Cleaned-AA (merged from sr_open_aa_long)
        datadir="${OPENASR_LONG_ROOT:?Set OPENASR_LONG_ROOT}"
        aa_datadir="${OPENASR_AA_ROOT:?Set OPENASR_AA_ROOT}"
        subsets="earnings21 earnings22 coraal"
        subsets_aa="Earnings22-Cleaned-AA"
        result_dir=${OUTPUT_DIR:-/modelblob/${model_rel_path}/eval_output/sr_openasr_long${suffix}/${tag}}
        mkdir -p $result_dir

        for set in $subsets; do
            dataset_path="$datadir $set"
            result_path=$result_dir/generate_${set}.txt
            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr_long" \
                -st "hfwer_en" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done

        # AA cleaned Earnings22 (long-form)
        for set in $subsets_aa; do
            dataset_path="$aa_datadir $set"
            result_path=$result_dir/generate_aa_earnings22_clean.txt
            python ${eval_script} $chkp_dir \
                -t $tok_dir \
                -tn "openasr_aa_long" \
                -st "hfwer_en" \
                -dp "$dataset_path" \
                -op $result_path \
                -nb $n_beams \
                -tp "$temperature" \
                -nrns $nrns \
                -ct ${chat_template} \
                -if ${instruction_first} \
                -lid "$use_lid" \
                -ap ${asr_prompt} \
                -bs ${batch_size} \
                -mt ${max_new_tokens} \
                "${mtp_args[@]}"
        done
        summarize_mtp_stats "$result_dir"
    fi

done
