#!/usr/bin/env python3
"""
mixup_data.py

Merge XXX.scp (target audio paths) into XXX.jsonl's target_audio_path field,
generate target_audio_caption via a multimodal captioner, and write to
output_dir/XXX.jsonl.

Usage:
    python local_eval/speech/mixup_data.py \
        --scp_dir  data/part4/speech_edit-v2/split6/cv3 \
        --jsonl_dir data/part4/speech_edit-v2 \
        --output_dir data/part4/speech_edit-v2/with_audio \
        --captioner_url http://localhost:9000/v1 \
        --captioner_model Qwen/Qwen3-Omni-30B-A3B-Captioner \
        --nj 16
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import joblib
from tqdm import tqdm

from local_split.sft_vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scp(scp_path: str) -> dict:
    """Load a Kaldi-style SCP file into {utt_id: audio_path}."""
    mapping = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def get_id(data: dict) -> Optional[str]:
    """Return the utterance ID from a JSONL record."""
    return data.get("id") or data.get("utt_id") or data.get("example_id")


def get_captioner_ref(captioner_client: VLLMClient, audio_path: str) -> str:
    """Query the multimodal captioner and return the caption string."""
    return captioner_client.chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"},
                }
            ],
        }],
    )


# ---------------------------------------------------------------------------
# Per-record processing (called by joblib threads)
# ---------------------------------------------------------------------------

def process_record(
    line: str,
    scp: dict,
    captioner_client: VLLMClient,
) -> Optional[dict]:
    """
    Fill target_audio_path from SCP and generate target_audio_caption.

    Returns updated record dict, or None if the utt_id is not in the SCP.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    utt_id = get_id(data)
    if utt_id is None or utt_id not in scp:
        return None

    target_audio_path = os.path.abspath(scp[utt_id])
    data["target_audio_path"] = target_audio_path

    try:
        caption = get_captioner_ref(captioner_client, target_audio_path)
        data["target_audio_caption"] = caption
    except Exception as e:
        print(f"[captioner] Error on {utt_id}: {e}")
        data["target_audio_caption"] = None

    return data


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_jsonl(
    jsonl_path: str,
    scp_path: str,
    output_jsonl: str,
    captioner_client: VLLMClient,
    n_jobs: int,
) -> None:
    """Process one JSONL+SCP pair and write merged output."""
    scp = load_scp(scp_path)
    print(f"  Loaded {len(scp)} SCP entries from {scp_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    print(f"  Processing {len(lines)} JSONL records with {n_jobs} threads …")

    # joblib threading generator — yields results as they finish
    results = joblib.Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
        joblib.delayed(process_record)(line, scp, captioner_client)
        for line in lines
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    written = skipped = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for result in tqdm(results, total=len(lines), desc=Path(jsonl_path).stem):
            if result is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
            else:
                skipped += 1

    print(f"  → {written} written, {skipped} skipped  →  {output_jsonl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SCP target-audio paths into JSONL and caption them."
    )
    parser.add_argument(
        "--scp_dir",
        default="data/part4/speech_edit-v2/split6/cv3",
        help="Directory containing *.scp files  (default: %(default)s)",
    )
    parser.add_argument(
        "--jsonl_dir",
        default="data/part4/speech_edit-v2",
        help="Directory containing *.jsonl files  (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for merged JSONL files",
    )
    parser.add_argument(
        "--captioner_url",
        default="http://localhost:9000/v1",
        help="vLLM base URL for the captioner  (default: %(default)s)",
    )
    parser.add_argument(
        "--captioner_model",
        default="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        help="Captioner model name  (default: %(default)s)",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=16,
        help="Number of parallel threads  (default: %(default)s)",
    )
    args = parser.parse_args()

    captioner_client = VLLMClient(
        base_url=args.captioner_url,
        model=args.captioner_model,
        max_concurrent=args.nj * 2,
        timeout=1200,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Index SCP and JSONL files by stem name
    scp_files  = {Path(p).stem: p for p in glob.glob(os.path.join(args.scp_dir,  "*.scp"))}
    jsonl_files = {Path(p).stem: p for p in glob.glob(os.path.join(args.jsonl_dir, "*.jsonl"))}

    common = sorted(set(scp_files) & set(jsonl_files))
    if not common:
        print("No matching SCP/JSONL pairs found — check --scp_dir and --jsonl_dir.")
        return

    print(f"Found {len(common)} matching pair(s): {common}")

    for name in common:
        print(f"\n=== {name} ===")
        output_jsonl = os.path.join(args.output_dir, f"{name}.jsonl")
        process_jsonl(
            jsonl_path=jsonl_files[name],
            scp_path=scp_files[name],
            output_jsonl=output_jsonl,
            captioner_client=captioner_client,
            n_jobs=args.nj,
        )


if __name__ == "__main__":
    main()
