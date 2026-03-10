#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioSet → metadata.jsonl Converter
====================================

Loads AudioSet (balanced train + test), saves each audio to
``data/audioset/wavs/<video_id>.wav`` and writes a single
``data/audioset/metadata.jsonl`` with all original keys preserved,
``audio_path`` as an absolute path, and ``audio_caption`` generated
by a round-robin captioner (vLLM).

Usage:
    python local_eval/non_speech/audioset.py

All configuration lives in this file (no external config needed).
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import concatenate_datasets, load_dataset
from joblib import Parallel, delayed
from tqdm import tqdm

# ── repo root so we can import local_split ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # speechlm1/
sys.path.insert(0, str(REPO_ROOT))
from local_split.sft_vllm_client import VLLMClient

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# =====================================================================
# Configuration  (edit in-place — no external config file)
# =====================================================================

# Captioner vLLM URLs (comma-separated for round-robin)
CAPTIONER_URLS = ",".join([
    "http://localhost:8000/v1",
])
CAPTIONER_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Captioner"

# AudioSet subsets to load
AUDIOSET_SPLITS = [
    ("agkphysics/AudioSet", "balanced", "train"),
    ("agkphysics/AudioSet", "balanced", "test"),
]

# Output paths (relative to REPO_ROOT)
OUT_DIR = REPO_ROOT / "data" / "audioset"
WAV_DIR = OUT_DIR / "wavs"
METADATA_JSONL = OUT_DIR / "metadata.jsonl"

# Audio processing
TARGET_SR = 16000
MIN_DUR = 2.0   # seconds – minimum duration to keep
MAX_DUR = 30.0  # seconds – maximum duration to keep

# Parallelism
N_JOBS = 64              # number of parallel workers (joblib)
PARALLEL_BACKEND = "threading"  # "threading" for I/O-bound captioning
CAPTIONER_TIMEOUT = 600  # per-request timeout (seconds)
CAPTIONER_RETRIES = 5

# Resume support: skip video_ids already in the output JSONL
RESUME = True

# =====================================================================
# AudioSet Loading
# =====================================================================


def load_audioset():
    """Load and concatenate requested AudioSet splits."""
    log.info("Loading AudioSet splits …")
    parts = []
    for repo, subset, split in AUDIOSET_SPLITS:
        log.info(f"  {repo}  subset={subset}  split={split}")
        ds = load_dataset(repo, subset, split=split, trust_remote_code=True)
        parts.append(ds)
    ds = concatenate_datasets(parts)
    log.info(f"Total rows before filtering: {len(ds)}")
    return ds


def _dur_filter_batch(batch):
    """Vectorised duration filter applied via ``Dataset.filter``."""
    keep = []
    for a in batch["audio"]:
        dur = len(a["array"]) / a["sampling_rate"]
        keep.append(MIN_DUR <= dur <= MAX_DUR)
    return keep


# =====================================================================
# Audio helpers
# =====================================================================


def normalise_audio(y: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(y)
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def save_wav(audio_dict: dict, out_path: Path) -> float:
    """Resample to TARGET_SR, normalise, and write a mono WAV.

    Returns duration in seconds.
    """
    y = np.asarray(audio_dict["array"], dtype=np.float32)
    sr = int(audio_dict["sampling_rate"])
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y = normalise_audio(y)
    sf.write(str(out_path), y, TARGET_SR)
    return len(y) / TARGET_SR


# =====================================================================
# Captioner
# =====================================================================


def get_caption(client: VLLMClient, wav_path: str) -> str | None:
    """Call the multimodal captioner on a saved WAV file."""
    try:
        resp = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{wav_path}"},
                        }
                    ],
                }
            ],
            temperature=0.65,
            max_tokens=1024,
        )
        return resp
    except Exception as e:
        log.warning("Captioner error for %s: %s", wav_path, e)
        return None


# =====================================================================
# Per-sample processing
# =====================================================================


def process_one(
    idx: int,
    row: dict,
    client: VLLMClient,
    wav_dir: Path,
) -> dict | None:
    """Process a single AudioSet row → save WAV, caption, emit record."""
    try:
        video_id = row.get("video_id") or f"audioset_{idx:07d}"
        wav_path = wav_dir / f"{video_id}.wav"

        # ── save audio ──────────────────────────────────────────────
        dur = save_wav(row["audio"], wav_path)

        # ── caption ─────────────────────────────────────────────────
        abs_path = str(wav_path.resolve())
        caption = get_caption(client, abs_path)
        if caption is None:
            log.warning("No caption for %s – skipping", video_id)
            return None

        # ── build record (keep ALL original keys) ───────────────────
        record = {k: v for k, v in row.items() if k != "audio"}
        record["audio_path"] = abs_path
        record["audio_caption"] = caption
        record["duration"] = round(dur, 4)
        return record

    except Exception as e:
        log.warning("Error processing idx=%d: %s", idx, e)
        return None


# =====================================================================
# Main
# =====================================================================


def main():
    # ── directories ─────────────────────────────────────────────────
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load dataset ────────────────────────────────────────────────
    ds = load_audioset()
    log.info("Applying duration filter [%.1f, %.1f] s …", MIN_DUR, MAX_DUR)
    ds = ds.filter(
        _dur_filter_batch,
        batched=True,
        batch_size=64,
        num_proc=min(64, os.cpu_count() or 1),
    )
    log.info("Rows after duration filter: %d", len(ds))

    # ── resume: collect already-done video_ids ──────────────────────
    done_ids: set[str] = set()
    if RESUME and METADATA_JSONL.exists():
        with open(METADATA_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    vid = rec.get("video_id")
                    if vid:
                        done_ids.add(vid)
                except Exception:
                    pass
        log.info("Resume: %d samples already done – will skip them", len(done_ids))

    # ── build work list (indices into the HF dataset) ───────────────
    work: list[int] = []
    for i in range(len(ds)):
        vid = ds[i].get("video_id")
        if vid and vid in done_ids:
            continue
        work.append(i)
    log.info("Work items: %d (skipped %d already done)", len(work), len(ds) - len(work))

    if not work:
        log.info("Nothing to do – exiting.")
        return

    # ── captioner client (round-robin across URLs) ──────────────────
    client = VLLMClient(
        base_url=CAPTIONER_URLS,
        model=CAPTIONER_MODEL,
        timeout=CAPTIONER_TIMEOUT,
        max_retries=CAPTIONER_RETRIES,
    )

    # ── parallel processing ─────────────────────────────────────────
    write_mode = "a" if done_ids else "w"
    pbar = tqdm(total=len(work), desc="AudioSet→JSONL")
    success = fail = 0

    def _worker(i: int):
        row = ds[i]
        return process_one(i, row, client, WAV_DIR)

    with open(METADATA_JSONL, write_mode, encoding="utf-8") as fout:
        for ret in Parallel(
            n_jobs=N_JOBS,
            backend=PARALLEL_BACKEND,
            return_as="generator",
            pre_dispatch="n_jobs",
        )(delayed(_worker)(i) for i in work):
            if ret is not None:
                fout.write(json.dumps(ret, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
            else:
                fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=success, fail=fail, refresh=False)

    pbar.close()
    log.info(
        "Done.  success=%d  fail=%d  output=%s",
        success, fail, METADATA_JSONL,
    )


if __name__ == "__main__":
    main()
