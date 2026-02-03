#!/bin/bash
# Wrapper script for evaluating LibriMix enhancement results
#
# Two-stage evaluation:
#   Stage 1: Prepare - Parse results.json and dialogues_all.jsonl to create scp files
#   Stage 2: Eval - Compute metrics (stoi, estoi, si_snr, sdr, pesq, mos, wer)
#
# Usage:
#   # Run both stages
#   ./local_librimix/run_eval.sh \
#       --results_json exp/.../results.json \
#       --dialogue_jsonl /path/to/dialogues_all.jsonl \
#       --output_dir exp/.../eval_results
#
#   # Run only stage 1 (prepare)
#   ./local_librimix/run_eval.sh --stage 1 --stop_stage 1 ...
#
#   # Run only stage 2 (eval) - assumes scp files already exist
#   ./local_librimix/run_eval.sh --stage 2 --stop_stage 2 ...
#
# Example:
#   ./local_librimix/run_eval.sh \
#       --results_json exp/opuslm_v2_stage3_sft_librimix_enh-v3/inference/inference_step_351881/dialogue_librimix_spk1_enh-v3_test100/results.json \
#       --dialogue_jsonl /mnt/home/xungong-andr-1766e0/prep/data/librimix_sft/data_mix_single/test100/spk1_enh/a2a_enh-v3/stage3_dialogues/dialogues_all.jsonl \
#       --output_dir exp/opuslm_v2_stage3_sft_librimix_enh-v3/eval/test100 \
#       --metrics "stoi estoi si_snr sdr pesq wer"

set -e
set -o pipefail

# Stage control
stage=1
stop_stage=2

# Input paths
results_json=""
dialogue_jsonl=""
output_dir=""

# Metric options (space-separated list: stoi estoi si_snr sdr pesq mos wer)
metrics="stoi estoi si_snr sdr pesq mos wer"
target_sr=16000

# WER options
whisper_model="base"
device="cuda"

# Parse options
. utils/parse_options.sh || exit 1

# Validate required arguments
if [ -z "${output_dir}" ]; then
    echo "Error: --output_dir is required"
    exit 1
fi

# Create output directory
mkdir -p "${output_dir}"

scp_dir="${output_dir}/scp"

# =============================================================================
# Stage 1: Prepare - Create scp files from results.json and dialogue_jsonl
# =============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo ""
    echo "=============================================="
    echo "Stage 1: Preparing evaluation directory"
    echo "=============================================="
    
    if [ -z "${results_json}" ] || [ -z "${dialogue_jsonl}" ]; then
        echo "Error: --results_json and --dialogue_jsonl are required for stage 1"
        exit 1
    fi
    
    python3 local_librimix/prepare_eval_dir.py \
        --results_json "${results_json}" \
        --dialogue_jsonl "${dialogue_jsonl}" \
        --output_dir "${scp_dir}"
    
    echo "Stage 1 completed. SCP files created in: ${scp_dir}"
fi

# =============================================================================
# Stage 2: Eval - Compute metrics
# =============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo ""
    echo "=============================================="
    echo "Stage 2: Computing evaluation metrics"
    echo "=============================================="
    
    # Check if scp files exist
    if [ ! -f "${scp_dir}/enh.scp" ] || [ ! -f "${scp_dir}/ref.scp" ]; then
        echo "Error: SCP files not found in ${scp_dir}. Run stage 1 first."
        exit 1
    fi
    
    # Build eval command
    eval_cmd="python3 local_librimix/eval_enh.py"
    eval_cmd="${eval_cmd} --enh_scp ${scp_dir}/enh.scp"
    eval_cmd="${eval_cmd} --ref_scp ${scp_dir}/ref.scp"
    eval_cmd="${eval_cmd} --output_dir ${output_dir}"
    eval_cmd="${eval_cmd} --target_sr ${target_sr}"
    eval_cmd="${eval_cmd} --metrics ${metrics}"
    
    # Add mix_scp if exists
    if [ -f "${scp_dir}/mix.scp" ]; then
        eval_cmd="${eval_cmd} --mix_scp ${scp_dir}/mix.scp"
    fi
    
    # Add WER options if wer is in metrics
    if echo "${metrics}" | grep -q "wer"; then
        eval_cmd="${eval_cmd} --whisper_model ${whisper_model}"
        eval_cmd="${eval_cmd} --device ${device}"
        
        # Use dialogue_jsonl for transcript extraction if provided
        if [ -n "${dialogue_jsonl}" ]; then
            eval_cmd="${eval_cmd} --dialogue_jsonl ${dialogue_jsonl}"
        fi
    fi
    
    echo "Running: ${eval_cmd}"
    eval ${eval_cmd}
    
    echo "Stage 2 completed. Results saved to: ${output_dir}"
fi
