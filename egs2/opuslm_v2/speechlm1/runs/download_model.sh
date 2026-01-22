#!/usr/bin/env bash
set -euo pipefail

# --------- user configs ----------
REPO_ID="insight/model-model"
REPO_TYPE="dataset"

exp_dir="exp/opuslm_v2_stage3_sft_vctk_vc-5epoch"
step=350545

. utils/parse_options.sh || exit 1

base="$(basename "${exp_dir}")"
base="${base#opuslm_v2_stage3_sft_}"
REMOTE_NAME="${base}-step_${step}"

SRC_PATH="models/${REMOTE_NAME}"  # path in repo
DST_DIR="${exp_dir}/checkpoints/step_${step}"

echo "[INFO] Downloading ${SRC_PATH} from ${REPO_ID}..."
# Download to ./exp/models/...
hf download "${REPO_ID}" --repo-type "${REPO_TYPE}" --local-dir "./exp" --include "${SRC_PATH}/**"

DOWNLOADED_DIR="exp/${SRC_PATH}"
mkdir -p "$(dirname "${DST_DIR}")"

if [ -d "${DOWNLOADED_DIR}" ]; then
    # Symlink: exp/models/remote_name -> exp/exp_dir/checkpoints/step_X
    ln -sfn "$(realpath "${DOWNLOADED_DIR}")" "${DST_DIR}"
    echo "[INFO] Linked ${DOWNLOADED_DIR} -> ${DST_DIR}"
    
    # Restore train.yaml
    if [ -f "${DOWNLOADED_DIR}/train.yaml" ]; then
        cp "${DOWNLOADED_DIR}/train.yaml" "${exp_dir}/"
        echo "[INFO] Restored train.yaml"
    fi
else
    echo "[ERR] Download failed or path incorrect: ${DOWNLOADED_DIR}"
    exit 1
fi

echo "[OK] Done. Restored to: ${DST_DIR}"
