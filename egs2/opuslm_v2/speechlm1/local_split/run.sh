#!/bin/bash
set -e

# Config
# Adjust input path as needed, using the one from context as example or argument
input_jsonl="data/part2_4/metadata.dur_20_30.debug.jsonl"
work_dir="data/part2_4/dur_20_30.debug"
stage=1

. utils/parse_options.sh

mkdir -p "${work_dir}"

step1_jsonl="${work_dir}/metadata.step1_vad.jsonl"
step2_jsonl="${work_dir}/metadata.step2_caption.jsonl"
step3_jsonl="${work_dir}/metadata.step3_refine.jsonl"

export PYTHONPATH=".:${PYTHONPATH}"

# Step 1: VAD Split
# This script uses joblib for parallel processing
if [ $stage -le 1 ]; then
     echo "Starting Step 1: VAD Split"
     python3 local_split/step1_vad.py \
         --input_jsonl "${input_jsonl}" \
         --output_dir "${work_dir}" \
         --output_jsonl "${step1_jsonl}" \
         --nj 256
     echo "Step 1 finished. Output: ${step1_jsonl}"
fi

# Step 2: Captioning
# Requires vLLM server running. Adjust URL and model if needed.
# Reference: /mnt/home/xungong-andr-1766e0/prep/audio_edit/step2_5/captioner.py
if [ $stage -le 2 ]; then
    vllm_url="http://localhost:8000/v1"
    model_name="Qwen/Qwen3-Omni-30B-A3B-Captioner"

    echo "Running Step 2: Captioning..."
    python3 local_split/step2_caption.py \
        --input_jsonl "${step1_jsonl}" \
        --output_jsonl "${step2_jsonl}" \
        --vllm_url "${vllm_url}" \
        --model "${model_name}" \
        --nj 512

    echo "Step 2 finished. Output: ${step2_jsonl}"
fi

# Step 3: Refine Captions
if [ ${stage} -le 3 ]; then
    vllm_url_instruct="http://localhost:8000/v1"
    model_name_instruct="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    
    echo "Starting Step 3: Refine Captions"
    python3 local_split/step3_refine_caption.py \
        --input_jsonl "${step2_jsonl}" \
        --output_jsonl "${step3_jsonl}" \
        --vllm_url "${vllm_url_instruct}" \
        --model "${model_name_instruct}" \
        --nj 512

    echo "Step 3 finished. Output: ${step3_jsonl}"
fi

