#!/bin/bash
set -e
# bash local_split/run.sh --stage 2 --stop_stage 5 --config_f local_split/only_text.yaml --skip_generate true --k 100
# bash local_split/run.sh --stage 2 --stop_stage 5 --config_f local_split/only_text-v2.yaml --skip_generate true --k 100
# bash local_split/run.sh --stage 2 --stop_stage 5 --config_f local_split/default.yaml --skip_generate true --k 100

# Config
# 仅保留：config + input/output + nj
config_f="local_split/default.yaml"
input_jsonl="data/part2_4/metadata.dur_20_30.debug.jsonl"
work_dir="data/part2_4/dur_20_30.debug"
nj=256

stage=1
stop_stage=10000
skip_generate=false
k=-1

. utils/parse_options.sh


mkdir -p "${work_dir}"

config_basename=$(basename "${config_f}" .yaml)

step1_metadata_jsonl="${work_dir}/metadata.step1_vad.jsonl"
step2_metadata_jsonl="${work_dir}/metadata.step2_caption.jsonl"
step3_metadata_jsonl="${work_dir}/metadata.step3_refine.jsonl"

export PYTHONPATH=".:${PYTHONPATH}"

if [ "${skip_generate}" = true ]; then
    echo "Mode: skip metadata generation, assemble top-k=${k}"
else
    echo "Mode: run metadata generation"
fi

# Step 1: VAD Split
# This script uses joblib for parallel processing
if [ ${stage} -le 1 ] && [ 1 -le ${stop_stage} ]; then
    echo "Starting Step 1: VAD Split"
    python3 local_split/step1_vad.py \
        --input_jsonl "${input_jsonl}" \
        --output_dir "${work_dir}" \
        --output_jsonl "${step1_metadata_jsonl}" \
        --nj 256 \
        --config_path "${config_f}"
    echo "Step 1 finished. Output: ${step1_metadata_jsonl}"
fi

# Step 2: Captioning
# Requires vLLM server running. Adjust URL and model if needed.
# Reference: /mnt/home/xungong-andr-1766e0/prep/audio_edit/step2_5/captioner.py
if [ ${stage} -le 2 ] && [ 2 -le ${stop_stage} ]; then
    echo "Running Step 2: Captioning..."
 if [ "${skip_generate}" = true ]; then
     echo "Skipping Step 2 metadata generation, using existing: ${step2_metadata_jsonl}"
    else
    python3 local_split/step2_caption.py \
        --input_jsonl "${step1_metadata_jsonl}" \
        --output_jsonl "${step2_metadata_jsonl}" \
        --nj "${nj}" \
        --config_path "${config_f}"

    echo "Step 2 finished. Output: ${step2_metadata_jsonl}"
fi
    for caption_mode in "main" "concat_split"; do
        step2_dataset_jsonl="${work_dir}/dialogue.step2_${caption_mode}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl "${step2_metadata_jsonl}" \
            --output_jsonl "${step2_dataset_jsonl}" \
            --caption_mode "${caption_mode}" \
            -k "${k}"
        echo "Dataset at: ${step2_dataset_jsonl}"
    done
fi

# Step 3: Refine Captions
if [ ${stage} -le 3 ] && [ 3 -le ${stop_stage} ]; then
 if [ "${skip_generate}" = true ]; then
     echo "Skipping Step 3 metadata generation, using existing: ${step3_metadata_jsonl}"
    else
    echo "Starting Step 3: Refine Captions"
    python3 local_split/step3_refine_caption.py \
        --input_jsonl "${step2_metadata_jsonl}" \
        --output_jsonl "${step3_metadata_jsonl}" \
        --nj "${nj}" \
        --config_path "${config_f}"

    echo "Step 3 finished. Output: ${step3_metadata_jsonl}"
fi

    for caption_mode in "main" "concat_split"; do
        step3_dataset_jsonl="${work_dir}/dialogue.step3_${caption_mode}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl "${step3_metadata_jsonl}" \
            --output_jsonl "${step3_dataset_jsonl}" \
            --caption_mode "${caption_mode}" \
            -k "${k}"
        echo "Dataset at: ${step3_dataset_jsonl}"
    done
fi


# Step 4: merge old Step4+Step5
#   1) main-audio <- repeat split1, split2 <- split1
#   2) rewrite main-caption on top of Step4 metadata
step4_audio_metadata_jsonl="${work_dir}/metadata.step4_repeat_gen.${config_basename}.jsonl"
step4_metadata_jsonl="${work_dir}/metadata.step4_repeat_rewrite_main.${config_basename}.jsonl"
step4_caption_metadata_jsonl="${work_dir}/metadata.step4_repeat_caption.${config_basename}.jsonl"
if [ ${stage} -le 4 ] && [ 4 -le ${stop_stage} ]; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 4 metadata generation, using existing: ${step4_metadata_jsonl}"
    else
        echo "Starting Step 4.1: Build repeated main audio by repeat split1"
        python3 local_split/step4_repeat_gen.py \
            --input_jsonl "${step3_metadata_jsonl}" \
            --output_jsonl "${step4_audio_metadata_jsonl}" \
            --nj 256 \
            --config_path "${config_f}"

        echo "Starting Step 4.2: LLM rewrite main caption based on Step4.1 metadata"
        python3 local_split/step4_repeat_rewrite_main.py \
            --input_jsonl "${step4_audio_metadata_jsonl}" \
            --output_jsonl "${step4_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"

        echo "Starting Step 4.3: Captioner generates main caption from repeated audio"
        python3 local_split/step4_repeat_caption.py \
            --input_jsonl "${step4_audio_metadata_jsonl}" \
            --output_jsonl "${step4_caption_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"

        echo "Step 4 finished. Metadata: ${step4_metadata_jsonl}"
    fi

    for caption_mode in "main" "concat_split"; do
        step4_dataset_jsonl="${work_dir}/dialogue.step4_${caption_mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl "${step4_metadata_jsonl}" \
            --output_jsonl "${step4_dataset_jsonl}" \
            --caption_mode "${caption_mode}" \
            -k "${k}"
        echo "Dataset at: ${step4_dataset_jsonl}"
    done

    caption_mode=main
    step4_caption_dataset_jsonl="${work_dir}/dialogue.step4_caption_${caption_mode}.${config_basename}.jsonl"
    python3 local_split/assemble_dialogue.py \
        --input_jsonl "${step4_caption_metadata_jsonl}" \
        --output_jsonl "${step4_caption_dataset_jsonl}" \
        --caption_mode "${caption_mode}" \
        -k "${k}"
    echo "Dataset at: ${step4_caption_dataset_jsonl}"
fi

# Step 5: edit split2 + generate edited main-caption (metadata only)
step5_metadata_jsonl="${work_dir}/metadata.step5_edit_merge.${config_basename}.jsonl"
if [ ${stage} -le 5 ] && [ 5 -le ${stop_stage} ]; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 5 metadata generation, using existing: ${step5_metadata_jsonl}"
    else
        echo "Starting Step 5: Edit split2 and generate edited main caption"
        python3 local_split/step5_edit_split2_merge_main.py \
            --input_jsonl "${step3_metadata_jsonl}" \
            --output_jsonl "${step5_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"

        echo "Step 5 finished. Metadata: ${step5_metadata_jsonl}"
    fi

    for caption_mode in "main" "concat_split"; do
        step5_dataset_jsonl="${work_dir}/dialogue.step5_${caption_mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl "${step5_metadata_jsonl}" \
            --output_jsonl "${step5_dataset_jsonl}" \
            --caption_mode "${caption_mode}" \
            -k "${k}"
        echo "Dataset at: ${step5_dataset_jsonl}"
    done

fi
