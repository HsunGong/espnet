#!/usr/bin/env bash
set -euo pipefail

# --------- user configs ----------
REPO_ID="insight/model-model"     # dataset repo
REPO_TYPE="dataset"

exp_dir="exp/opuslm_v2_stage3_sft_vctk_vc-5epoch"
step=350545

. utils/parse_options.sh || exit 1

base="$(basename ${exp_dir})"
base="${base#opuslm_v2_stage3_sft_}" 
REMOTE_NAME="$base-step_${step}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERR] 'hf' command not found. Install/upgrade huggingface_hub and ensure hf is in PATH."
  exit 1
fi

[ -d ${exp_dir}/checkpoints/step_${step} ] || {
  echo "[ERR] Step dir not found: ${exp_dir}/checkpoints/step_${step}"
  exit 1
}

cp $exp_dir/train.yaml ${exp_dir}/checkpoints/step_${step}

echo "[INFO] Uploading to Hugging Face..."
# This uploads the staged folder into a remote subfolder named REMOTE_NAME/
hf upload "${REPO_ID}" "${exp_dir}/checkpoints/step_${step}" \
  "models/${REMOTE_NAME}" \
  --repo-type "${REPO_TYPE}" \
  --commit-message "Add ${REMOTE_NAME}"

echo "[OK] Done. Uploaded as: ${REPO_ID}/${REMOTE_NAME}/"
