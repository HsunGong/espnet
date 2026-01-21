#!/usr/bin/env bash
set -euo pipefail

# --------- user configs ----------
REPO_ID="insight/model-model"     # dataset repo
REPO_TYPE="dataset"

exp_dir="exp/opuslm_v2_stage3_sft_vctk_vc-5epoch"
step=350545

# restore to this exp dir on the new machine (can be different)
dst_exp_dir="${exp_dir}"

# optional: where to place downloaded cache/temp
# HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# --------------------------------

. utils/parse_options.sh || exit 1

base="$(basename "${exp_dir}")"
base="${base#opuslm_v2_stage3_sft_}"
REMOTE_NAME="${base}-step_${step}"

SRC_PATH_IN_REPO="models/${REMOTE_NAME}"   # remote folder in HF dataset repo
DST_STEP_DIR="${dst_exp_dir}/checkpoints/step_${step}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERR] 'hf' command not found. Install/upgrade huggingface_hub and ensure hf is in PATH."
  exit 1
fi

mkdir -p "${DST_STEP_DIR}"

echo "[INFO] Downloading from Hugging Face..."
echo "       repo: ${REPO_ID} (type=${REPO_TYPE})"
echo "       path-in-repo: ${SRC_PATH_IN_REPO}"
echo "       to: ${DST_STEP_DIR}"

# Download all files under that folder. hf will place them under local-dir / local-dir-use-symlinks
hf download "${REPO_ID}" \
  --repo-type "${REPO_TYPE}" \
  --local-dir "${DST_STEP_DIR}" \
  --local-dir-use-symlinks False \
  --include "${SRC_PATH_IN_REPO}/**"

# After download, files will be located at:
#   ${DST_STEP_DIR}/${SRC_PATH_IN_REPO}/...
# We want to "restore" to ${DST_STEP_DIR}/ directly (flatten one level).
if [[ -d "${DST_STEP_DIR}/${SRC_PATH_IN_REPO}" ]]; then
  echo "[INFO] Restoring directory layout..."
  rsync -a "${DST_STEP_DIR}/${SRC_PATH_IN_REPO}/" "${DST_STEP_DIR}/"
  rm -rf "${DST_STEP_DIR:?}/${SRC_PATH_IN_REPO}"
fi

echo "[OK] Done. Restored to: ${DST_STEP_DIR}"
