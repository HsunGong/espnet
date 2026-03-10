#!/usr/bin/env bash
# rename_eval_dirs.sh
#
# Moves eval-test_clean* directories under exp/*/inference/*/ to a new layout:
#
#   eval-test_clean_audioset-v3-{suffix}  ->  {suffix}/test_clean/freeform-edit
#   eval-test_clean_audioset-v2-{suffix}  ->  {suffix}/test_clean/audio_edit-v2
#   eval-test_clean-v2-{suffix}           ->  {suffix}/test_clean/speech_edit-v2
#   eval-test_clean-v1-{suffix}           ->  {suffix}/test_clean/speech_edit-v1
#
# Only the last path component (after the inference/YYY/ prefix) changes.
#
# Usage:
#   bash rename_eval_dirs.sh           # dry-run (prints moves, no action)
#   bash rename_eval_dirs.sh --apply   # actually move

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # speechlm1/
DRY_RUN=true
[[ "${1:-}" == "--apply" ]] && DRY_RUN=false

moved=0; skipped=0

while IFS= read -r src; do
    parent="$(dirname "$src")"
    base="$(basename "$src")"

    suffix=""
    subdir=""

    if [[ "$base" =~ ^eval-test_clean_audioset-v3-(.+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        subdir="test_clean/freeform-edit"
    elif [[ "$base" =~ ^eval-test_clean_audioset-v2-(.+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        subdir="test_clean/audio_edit-v2"
    elif [[ "$base" =~ ^eval-test_clean-v2-(.+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        subdir="test_clean/speech_edit-v2"
    elif [[ "$base" =~ ^eval-test_clean-v1-(.+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        subdir="test_clean/speech_edit-v1"
    else
        echo "  SKIP (no rule): $src"
        (( skipped++ )) || true
        continue
    fi

    dst="${parent}/${suffix}/${subdir}"

    if [[ -e "$dst" ]]; then
        echo "  SKIP (dst exists): $src -> $dst"
        (( skipped++ )) || true
        continue
    fi

    echo "  MOVE: $src"
    echo "     -> $dst"

    if ! $DRY_RUN; then
        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst"
    fi
    (( moved++ )) || true

done < <(find "$ROOT/exp" -maxdepth 5 -type d -name 'eval-test_clean*' | sort)

echo ""
if $DRY_RUN; then
    echo "[DRY-RUN] Would move $moved dir(s), skip $skipped. Re-run with --apply to execute."
else
    echo "[DONE] Moved $moved dir(s), skipped $skipped."
fi
