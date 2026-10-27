#!/bin/bash

set -e -o pipefail

# export PYTHONPATH=/scratch/amlt_code
# cd /scratch/amlt_code/scripts/audio_eval/

exp_id=$1
exp_name=$2
checkpoints=$3
tokenizer=$4
eval_set=${5:-"teams_segments"}
use_lid=${6:-false}
chat_template=${7:-"phi_3"}
instruction_first=${8:-"false"}
tag=${9:-$(date +%Y%m%d%H%M)}
batch_size=${10:-1}
data_path_prefix=${11:-"/datablob1"}

if [ -z "$exp_id" ]; then
    echo "exp_id is empty"
    exit 1
fi

tok_dir=$tokenizer

[[ "$eval_set" == fleurs ]] || exit 2
eval_script="eval.py"
for checkpoint in $checkpoints; do
    model_rel_path="projects/${exp_name}/amlt-results/${exp_id}/${checkpoint}"
    chkp_dir=${MODEL:-/modelblob/$model_rel_path}
    if [ $eval_set == "fleurs" ]; then

        # FLEURS X->Y
        language="EN DE ES FR IT JA PT ZH"
        result_dir=${OUTPUT_DIR:-/modelblob/${model_rel_path}/eval_output/st_fleurs/${tag}}
        mkdir -p $result_dir

        for src_lang in $language; do
            for tgt_lang in $language; do
                if [ $src_lang == $tgt_lang ]; then
                    continue
                fi
                dataset_path=${FLEURS_ROOT:?Set FLEURS_ROOT}/$src_lang/$tgt_lang/test.tsv
                result_path=$result_dir/generate_${src_lang}_${tgt_lang}.txt
                python ${eval_script} $chkp_dir \
                    -t $tok_dir \
                    -tn "fleurs" \
                    -st "bleu_${src_lang}_${tgt_lang}" \
                    -ct ${chat_template} \
                    -if ${instruction_first} \
                    -dp $dataset_path \
                    -op $result_path \
                    -lid "$use_lid" \
                    -bs ${batch_size}
            done
        done
    fi


done
