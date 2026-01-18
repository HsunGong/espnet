#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# SFT Data Simulation for Text-to-Audio Generation
# This script generates user requests and reasoning traces from rich captions.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Default parameters
stage=1
stop_stage=100
version=
# vLLM URLs: use ":" to separate multiple URLs serving the same model
# Example: "http://host1:8000/v1:http://host2:8000/v1:http://host3:8000/v1"
vllm_url_stage1="http://cnode1-002:9000/v1:http://cnode1-003:9000/v1:http://cnode1-005:9000/v1"                                   
vllm_url_stage2="http://cnode1-002:9000/v1:http://cnode1-003:9000/v1:http://cnode1-005:9000/v1"
vllm_url_stage4="http://cnode1-002:9000/v1:http://cnode1-003:9000/v1:http://cnode1-005:9000/v1"
model_stage1="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage2="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage4="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
input_file="/mnt/home/haoranw4-andr-49167f/data/sft_data/part2_pretrain_curation/metadata.jsonl"
output_base="data/sft_part2"
num_workers=2400
timeout=1200
resume=true
num_samples=-1  # -1 means process all samples

log "$0 $*"
. utils/parse_options.sh

# Check required parameter
if [ -z "${version}" ]; then
    log "Error: --version is required"
    log "Usage: $0 --version <version_tag> [options]"
    exit 1
fi

# Set output directory with version
output_dir="${output_base}/${version}"
log "Output directory: ${output_dir}"

# Create directory structure
mkdir -p "${output_dir}/stage1_user_requests"
mkdir -p "${output_dir}/stage2_reasoning_traces"
mkdir -p "${output_dir}/stage3_dialogues"
mkdir -p "${output_dir}/stage4_quality_judge"

# Resume flag
resume_flag=""
if ${resume}; then
    resume_flag="--resume"
fi

# Stage 1: Generate user requests
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Generating user requests from rich captions"

    python3 local/sft_data_simulation/sft_generate_user_requests.py \
        --input_file "${input_file}" \
        --output_dir "${output_dir}/stage1_user_requests" \
        --vllm_url "${vllm_url_stage1}" \
        --model "${model_stage1}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        ${resume_flag}

    # Count results
    stage1_output="${output_dir}/stage1_user_requests/user_requests.jsonl"
    if [ -f "${stage1_output}" ]; then
        count=$(wc -l < "${stage1_output}")
        log "Stage 1 completed: ${count} user requests generated"
    fi
fi

# Stage 2: Generate reasoning traces for each detail level
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating reasoning traces"

    stage1_output="${output_dir}/stage1_user_requests/user_requests.jsonl"
    if [ ! -f "${stage1_output}" ]; then
        log "Error: Stage 1 output not found: ${stage1_output}"
        exit 1
    fi

    for level in realistic imaginary; do
        log "Stage 2: Processing detail level: ${level}"

        python3 local/sft_data_simulation/sft_generate_reasoning_traces.py \
            --input_file "${stage1_output}" \
            --output_dir "${output_dir}/stage2_reasoning_traces" \
            --detail_level "${level}" \
            --vllm_url "${vllm_url_stage2}" \
            --model "${model_stage2}" \
            --num_workers ${num_workers} \
            --timeout ${timeout} \
            --num_samples ${num_samples} \
            --version "${version}" \
            ${resume_flag}

        # Count results
        stage2_output="${output_dir}/stage2_reasoning_traces/reasoning_${level}.jsonl"
        if [ -f "${stage2_output}" ]; then
            count=$(wc -l < "${stage2_output}")
            log "Stage 2 (${level}) completed: ${count} reasoning traces generated"
        fi
    done
fi

# Stage 3: Assemble dialogues
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Assembling dialogues"

    for level in realistic imaginary; do
        stage2_output="${output_dir}/stage2_reasoning_traces/reasoning_${level}.jsonl"

        if [ ! -f "${stage2_output}" ]; then
            log "Warning: Stage 2 output not found for ${level}: ${stage2_output}"
            continue
        fi

        log "Stage 3: Assembling dialogues for detail level: ${level}"

        python3 local/sft_data_simulation/sft_assemble_dialogue.py \
            --input_file "${stage2_output}" \
            --output_dir "${output_dir}/stage3_dialogues" \
            --detail_level "${level}" \
            --version "${version}"

        # Count results
        stage3_output="${output_dir}/stage3_dialogues/dialogues_${level}.jsonl"
        if [ -f "${stage3_output}" ]; then
            count=$(wc -l < "${stage3_output}")
            log "Stage 3 (${level}) completed: ${count} dialogues assembled"
        fi
    done
fi

# Stage 4: LLM-as-judge quality validation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: LLM-as-judge quality validation"

    for level in realistic imaginary; do
        stage3_output="${output_dir}/stage3_dialogues/dialogues_${level}.jsonl"

        if [ ! -f "${stage3_output}" ]; then
            log "Warning: Stage 3 output not found for ${level}: ${stage3_output}"
            continue
        fi

        log "Stage 4: Judging quality for detail level: ${level}"

        python3 local/sft_data_simulation/sft_judge_quality.py \
            --input_file "${stage3_output}" \
            --output_dir "${output_dir}/stage4_quality_judge" \
            --detail_level "${level}" \
            --vllm_url "${vllm_url_stage4}" \
            --model "${model_stage4}" \
            --num_workers ${num_workers} \
            --timeout ${timeout} \
            --num_samples ${num_samples} \
            ${resume_flag}

        # Count results
        stage4_output="${output_dir}/stage4_quality_judge/judge_${level}.jsonl"
        if [ -f "${stage4_output}" ]; then
            count=$(wc -l < "${stage4_output}")
            log "Stage 4 (${level}) completed: ${count} samples judged"
        fi
    done

    # Merge summaries
    log "Stage 4: Generating combined summary"
    python3 local/sft_data_simulation/sft_merge_summaries.py \
        --output_dir "${output_dir}/stage4_quality_judge"
fi

# Stage 5: Merge all dialogues
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Merging all dialogues"

    merged_output="${output_dir}/stage3_dialogues/dialogues_all.jsonl"
    > "${merged_output}"  # Clear file

    for level in realistic imaginary; do
        stage3_output="${output_dir}/stage3_dialogues/dialogues_${level}.jsonl"
        if [ -f "${stage3_output}" ]; then
            cat "${stage3_output}" >> "${merged_output}"
        fi
    done

    if [ -f "${merged_output}" ]; then
        count=$(wc -l < "${merged_output}")
        log "Stage 5 completed: ${count} total dialogues merged"
    fi
fi

# Stage 6: Generate statistics
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Generating statistics"

    stats_file="${output_dir}/stats.txt"
    {
        echo "SFT Data Simulation Statistics"
        echo "==============================="
        echo "Version: ${version}"
        echo "Generated at: $(date)"
        echo ""

        echo "Stage 1 - User Requests:"
        if [ -f "${output_dir}/stage1_user_requests/user_requests.jsonl" ]; then
            echo "  $(wc -l < "${output_dir}/stage1_user_requests/user_requests.jsonl") samples"
        fi
        echo ""

        echo "Stage 2 - Reasoning Traces:"
        for level in realistic imaginary; do
            f="${output_dir}/stage2_reasoning_traces/reasoning_${level}.jsonl"
            if [ -f "${f}" ]; then
                echo "  ${level}: $(wc -l < "${f}") samples"
            fi
        done
        echo ""

        echo "Stage 3 - Dialogues:"
        for level in realistic imaginary; do
            f="${output_dir}/stage3_dialogues/dialogues_${level}.jsonl"
            if [ -f "${f}" ]; then
                echo "  ${level}: $(wc -l < "${f}") samples"
            fi
        done
        echo ""

        if [ -f "${output_dir}/stage3_dialogues/dialogues_all.jsonl" ]; then
            echo "Total merged dialogues: $(wc -l < "${output_dir}/stage3_dialogues/dialogues_all.jsonl")"
        fi
    } > "${stats_file}"

    cat "${stats_file}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
