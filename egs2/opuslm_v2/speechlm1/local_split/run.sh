#!/bin/bash
set -e
# Run all steps (default):
#   bash local_split/run.sh
#
# Run specific steps only via run_stages (comma-separated, overrides stage/stop_stage):
#   bash local_split/run.sh --run_stages 1,2,3
#   bash local_split/run.sh --run_stages 4,6   # repeat-only, supply --step4_input_jsonl
#
# run_repeat shortcut (no VAD/caption, input is already split1 metadata):
#   bash local_split/run_repeat.sh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config_f="local_split/conf/default.yaml"
input_jsonl_raw="data/part2_4/metadata.dur_20_30.debug.jsonl"
expdir="data/part2_4/dur_20_30.debug"
yaml_path="data/debug.yaml"
name_prefix="debug"
skip_repeat_gen=false # for stage 4
nj=1024

# Stage range (used when run_stages is empty)
stage=1
stop_stage=10000

# Comma-separated explicit stage list, e.g. "1,2,4,6".
# When set, overrides stage/stop_stage entirely.
run_stages=""

skip_generate=false
k=-1

. utils/parse_options.sh

mkdir -p "${expdir}"

config_basename=$(basename "${config_f}" .yaml)

# Helper: returns 0 (true) if stage N should run
should_run() {
    local n=$1
    if [ -n "${run_stages}" ]; then
        # check membership in comma-separated list
        echo ",${run_stages}," | grep -q ",${n},"
    else
        [ "${stage}" -le "${n}" ] && [ "${n}" -le "${stop_stage}" ]
    fi
}

# Downstream steps read from the raw input directly
input_jsonl="${input_jsonl_raw}"

step0_metadata_jsonl="${expdir}/metadata.step0_prepare.jsonl"
step1_metadata_jsonl="${expdir}/metadata.step1_vad.jsonl"
step2_metadata_jsonl="${expdir}/metadata.step2_caption.jsonl"
step3_metadata_jsonl="${expdir}/metadata.step3_refine.jsonl"

step4_audio_metadata_jsonl="${expdir}/metadata.step4_repeat_gen.${config_basename}.jsonl"
step5_metadata_jsonl="${expdir}/metadata.step5_repeat_rewrite_main.${config_basename}.jsonl"
step6_caption_metadata_jsonl="${expdir}/metadata.step6_repeat_caption.${config_basename}.jsonl"

step7_metadata_jsonl="${expdir}/metadata.step7_edit_split2_merge_main.${config_basename}.jsonl"
step8_metadata_jsonl="${expdir}/metadata.step8_edit_split1_merge_main.${config_basename}.jsonl"

export PYTHONPATH=".:${PYTHONPATH}"

if [ "${skip_generate}" = true ]; then
    echo "Mode: skip metadata generation, assemble top-k=${k}"
else
    echo "Mode: run metadata generation"
fi
if [ -n "${run_stages}" ]; then
    echo "Active stages: ${run_stages}"
else
    echo "Stage range: ${stage}..${stop_stage}"
fi

# shuf set seed =7 for reproducibility when selecting top-k
if [ "${k}" -gt 0 ]; then
    if [ -f "${expdir}/metadata.top-${k}.jsonl" ]; then
        echo "Top-k=${k} metadata already exists: ${expdir}/metadata.top-${k}.jsonl"
    else
        echo "Will select top-k=${k} per type for output JSONL"
        SEED=7
        shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED" -nosalt </dev/zero 2>/dev/null) \
            "${input_jsonl}" | head -n "${k}" \
            > "${expdir}/metadata.top-${k}.jsonl"
        # shuf --random-source=/dev/urandom "${input_jsonl}" | head -n "${k}" > "${expdir}/metadata.top-${k}.jsonl"
        input_jsonl="${expdir}/metadata.top-${k}.jsonl"
    fi
    k=-1
fi

# ---------------------------------------------------------------------------
# Step 0: Prepare — set main=split1, no VAD (for pre-segmented input)
# ---------------------------------------------------------------------------
if should_run 0; then
    echo "Starting Step 0: Prepare (set main=split1, skip VAD)"
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 0 metadata generation, using existing: ${step0_metadata_jsonl}"
    else
        python3 local_split/step0_prepare.py \
            --input_jsonl  "${input_jsonl}" \
            --output_jsonl "${step0_metadata_jsonl}" \
            --config_path  "${config_f}"
        echo "Step 0 finished. Output: ${step0_metadata_jsonl}"
    fi

    mode="main2main"
    step0_dataset_jsonl="${expdir}/dialogue.step0.${mode}.jsonl"
    python3 local_split/assemble_dialogue.py \
        --input_jsonl  "${step0_metadata_jsonl}" \
        --output_jsonl "${step0_dataset_jsonl}" \
        --mode "${mode}" \
        --yaml-path "${yaml_path}" \
        --name-prefix "${name_prefix}" \
        --k "${k}"

fi

# ---------------------------------------------------------------------------
# Step 1: VAD Split
# ---------------------------------------------------------------------------
if should_run 1; then
    echo "Starting Step 1: VAD Split"
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 1 metadata generation, using existing: ${step1_metadata_jsonl}"
    else
    python3 local_split/step1_vad.py \
        --input_jsonl "${input_jsonl}" \
        --output_dir "${expdir}" \
        --output_jsonl "${step1_metadata_jsonl}" \
        --nj 256 \
        --config_path "${config_f}"
    echo "Step 1 finished. Output: ${step1_metadata_jsonl}"
    fi

    for mode in "main2split1" "main2main"; do
        step1_dataset_jsonl="${expdir}/dialogue.step1.${mode}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step1_metadata_jsonl}" \
            --output_jsonl "${step1_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 2: Caption Audio Segments
# Requires vLLM server running. Adjust URL and model if needed.
# ---------------------------------------------------------------------------
if should_run 2; then
    echo "Running Step 2: Audio Caption Generate for all Segment..."
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

    for mode in "cat2split1" "cat2main" "t2a_t2a" "a2t_t2a"; do
        step2_dataset_jsonl="${expdir}/dialogue.step2.${mode}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step2_metadata_jsonl}" \
            --output_jsonl "${step2_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 3: Refine Captions
# WARNING: Step3 is not necessary, and only for debug
# ---------------------------------------------------------------------------
if should_run 3; then
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

    for mode in "cat2split1" "cat2main" "t2a_t2a" "a2t_t2a"; do
        step3_dataset_jsonl="${expdir}/dialogue.step3.${mode}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step3_metadata_jsonl}" \
            --output_jsonl "${step3_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# Auto-detect best available upstream metadata: step3 → step2 → step1 → step0 → raw
if [ -f "${step3_metadata_jsonl}" ]; then
    _step4_in="${step3_metadata_jsonl}"
elif [ -f "${step2_metadata_jsonl}" ]; then
    _step4_in="${step2_metadata_jsonl}"
elif [ -f "${step1_metadata_jsonl}" ]; then
    _step4_in="${step1_metadata_jsonl}"
elif [ -f "${step0_metadata_jsonl}" ]; then
    _step4_in="${step0_metadata_jsonl}"
else
    echo "WARNING: No prepared metadata found for step4. Using raw input (may fail)."
    _step4_in="${input_jsonl}"
fi

echo ">>>> We use ${_step4_in} as input for Step 4 and downstream steps. If this is not expected, please check previous steps' outputs."

# ---------------------------------------------------------------------------
# Step 4: Repeat Gen — build repeated main audio from split1
# WARNING: main audio_caption has not been changed
# ---------------------------------------------------------------------------
if should_run 4; then
    echo "Step 4 input: ${_step4_in}"
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 4 metadata generation, using existing: ${step4_audio_metadata_jsonl}"
    else
        echo "Starting Step 4: Repeat Gen (build repeated main audio from split1)"
        python3 local_split/step4_repeat_gen.py \
            --input_jsonl "${_step4_in}" \
            --output_jsonl "${step4_audio_metadata_jsonl}" \
            --nj $nj \
            --skip_gen ${skip_repeat_gen} \
            --config_path "${config_f}"
        echo "Step 4 finished. Output: ${step4_audio_metadata_jsonl}"
    fi

    for mode in "cat2split1" "cat2main" "t2a_t2a" "a2t_t2a"; do
        step4_dataset_jsonl="${expdir}/dialogue.step4.${mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step4_audio_metadata_jsonl}" \
            --output_jsonl "${step4_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 5: Repeat Rewrite Main Caption — LLM rewrites main caption
# ---------------------------------------------------------------------------
if should_run 5; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 5 metadata generation, using existing: ${step5_metadata_jsonl}"
    else
        echo "Starting Step 5: Repeat Rewrite Main Caption"
        python3 local_split/step5_repeat_rewrite_main.py \
            --input_jsonl "${step4_audio_metadata_jsonl}" \
            --output_jsonl "${step5_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"
        echo "Step 5 finished. Output: ${step5_metadata_jsonl}"
    fi

    for mode in "main2split1" "main2main"; do
        step5_dataset_jsonl="${expdir}/dialogue.step5.${mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step5_metadata_jsonl}" \
            --output_jsonl "${step5_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 6: Repeat Caption — captioner generates main caption from repeated audio
# ---------------------------------------------------------------------------
if should_run 6; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 6 metadata generation, using existing: ${step6_caption_metadata_jsonl}"
    else
        echo "Starting Step 6: Repeat Caption (captioner on repeated audio)"
        python3 local_split/step6_repeat_caption.py \
            --input_jsonl "${step4_audio_metadata_jsonl}" \
            --output_jsonl "${step6_caption_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"
        echo "Step 6 finished. Output: ${step6_caption_metadata_jsonl}"
    fi

    for mode in "main2split1" "main2main"; do
        step6_dataset_jsonl="${expdir}/dialogue.step6.${mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step6_caption_metadata_jsonl}" \
            --output_jsonl "${step6_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 7: Edit Split2 + Merge Main Caption
# WARNING: there will be no groud-truth split2/main audio-path
# ---------------------------------------------------------------------------
if should_run 7; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 7 metadata generation, using existing: ${step7_metadata_jsonl}"
    else
        echo "Starting Step 7: Edit Split2 and Merge Main Caption"
        python3 local_split/step7_edit_split2_merge_main.py \
            --input_jsonl "${_step4_in}" \
            --output_jsonl "${step7_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"
        echo "Step 7 finished. Output: ${step7_metadata_jsonl}"
    fi

    for mode in "cat2split1" "main2split1" "t2a_t2a" "a2t_t2a"; do
        step7_dataset_jsonl="${expdir}/dialogue.step7.${mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step7_metadata_jsonl}" \
            --output_jsonl "${step7_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi

# ---------------------------------------------------------------------------
# Step 8: Edit Split1 + Regenerate Main Caption
# ---------------------------------------------------------------------------
if should_run 8; then
    if [ "${skip_generate}" = true ]; then
        echo "Skipping Step 8 metadata generation, using existing: ${step8_metadata_jsonl}"
    else
        echo "Starting Step 8: Edit Split1 and Regenerate Main Caption"
        python3 local_split/step8_edit_split1_merge_main.py \
            --input_jsonl "${_step4_in}" \
            --output_jsonl "${step8_metadata_jsonl}" \
            --nj "${nj}" \
            --config_path "${config_f}"
        echo "Step 8 finished. Output: ${step8_metadata_jsonl}"
    fi

    for mode in "cat2split1" "main2split1" "t2a_t2a" "a2t_t2a"; do
        step8_dataset_jsonl="${expdir}/dialogue.step8.${mode}.${config_basename}.jsonl"
        python3 local_split/assemble_dialogue.py \
            --input_jsonl  "${step8_metadata_jsonl}" \
            --output_jsonl "${step8_dataset_jsonl}" \
            --mode "${mode}" \
            --yaml-path "${yaml_path}" \
            --name-prefix "${name_prefix}" \
            --k "${k}"
    done
fi
