#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption wav files from StepAudioX speech-edit-v2 inference results,
then patch the metadata JSONL with the new target_audio_caption.

Usage:
    python local_eval/speech/get_step_cap.py \
        --exp_dir exp/stepaudiox/test_clean/speech_edit-v2 \
        --data_dir data/test_clean/speech_edit-v2 \
        --output_dir data/test_clean/speech_edit-v2-fixed \
        --host cnode1-004 \
        --port_start 8000 \
        --port_end 8007 \
        --nj 16
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# -- resolve project root so the import works regardless of cwd --
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent          # speechlm1/
sys.path.insert(0, str(_PROJECT_ROOT))

from local_split.sft_vllm_client import VLLMClient


# =========================================================
# SCP & JSONL helpers
# =========================================================

def read_scp(scp_path: str) -> List[Tuple[str, str]]:
    """Read a Kaldi-style scp: <utt_id>  <wav_path>"""
    entries = []
    with open(scp_path, "r") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
    return entries


def load_jsonl_as_dict(jsonl_path: str, key: str = "utt_id") -> Dict[str, dict]:
    """Load a JSONL file into {key_value: record} dict."""
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            records[d[key]] = d
    return records


def write_jsonl(records: List[dict], out_path: str):
    """Write records list to JSONL (preserves original order)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# Captioning
# =========================================================

def caption_one_wav(captioner: VLLMClient, wav_path: str) -> str:
    """Send a wav file to the captioner and return the caption text."""
    resp = captioner.chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"file://{wav_path}"}},
            ],
        }],
    )
    return resp


def caption_batch(
    captioner: VLLMClient,
    scp_entries: List[Tuple[str, str]],
    nj: int,
    desc: str = "",
) -> Dict[str, str]:
    """Caption all wavs in scp_entries in parallel.

    Returns:
        {utt_id: caption_text}
    """
    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=nj) as pool:
        future_map = {
            pool.submit(caption_one_wav, captioner, wav_path): utt_id
            for utt_id, wav_path in scp_entries
        }
        pbar = tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc=desc or "Captioning",
        )
        for fut in pbar:
            utt_id = future_map[fut]
            try:
                cap = fut.result()
                if cap and isinstance(cap, str) and len(cap.strip()) > 10:
                    results[utt_id] = cap.strip()
                else:
                    tqdm.write(f"[WARN] Empty/short caption for {utt_id}")
            except Exception as e:
                tqdm.write(f"[ERROR] {utt_id}: {e}")
        pbar.close()

    return results


# =========================================================
# Patch metadata JSONL
# =========================================================

def patch_jsonl(
    src_jsonl: str,
    dst_jsonl: str,
    new_captions: Dict[str, str],
    key: str = "utt_id",
    caption_field: str = "target_audio_caption",
):
    """Copy src_jsonl → dst_jsonl, replacing caption_field where available."""
    patched, total = 0, 0
    records = []
    with open(src_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            total += 1
            uid = d.get(key)
            if uid and uid in new_captions:
                d[caption_field] = new_captions[uid]
                patched += 1
            records.append(d)

    os.makedirs(os.path.dirname(dst_jsonl), exist_ok=True)
    write_jsonl(records, dst_jsonl)
    print(f"  Patched {patched}/{total} records in {dst_jsonl}")
    return patched, total


# =========================================================
# Main
# =========================================================

def build_captioner_urls(host: str, port_start: int, port_end: int) -> str:
    """Build comma-separated vLLM endpoint URLs for round-robin."""
    urls = [f"http://{host}:{p}/v1" for p in range(port_start, port_end + 1)]
    return ",".join(urls)


def main():
    parser = argparse.ArgumentParser(
        description="Caption StepAudioX wav outputs and patch metadata JSONL."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp/stepaudiox/test_clean/speech_edit-v2",
        help="Directory containing *.scp files and wav subdirs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/test_clean/speech_edit-v2",
        help="Source metadata directory with *.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/test_clean/speech_edit-v2-fixed",
        help="Destination directory (backup + patched metadata).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="cnode1-004",
        help="Captioner vLLM host.",
    )
    parser.add_argument(
        "--port_start",
        type=int,
        default=8000,
        help="First captioner port (inclusive).",
    )
    parser.add_argument(
        "--port_end",
        type=int,
        default=8007,
        help="Last captioner port (inclusive).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        help="Captioner model name.",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=16,
        help="Number of parallel captioning workers.",
    )
    parser.add_argument(
        "--caption_field",
        type=str,
        default="target_audio_caption",
        help="JSONL field name to overwrite with the new caption.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache captions (optional). "
             "If set, captions are saved/loaded as <cache_dir>/<subset>.json "
             "to avoid re-captioning on re-runs.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # ---- 1. Discover scp files ----
    scp_files = sorted(exp_dir.glob("*.scp"))
    if not scp_files:
        print(f"[ERROR] No .scp files found in {exp_dir}")
        sys.exit(1)
    print(f"Found {len(scp_files)} scp file(s): {[s.name for s in scp_files]}")

    # ---- 2. Build captioner client (round-robin across ports) ----
    cap_urls = build_captioner_urls(args.host, args.port_start, args.port_end)
    captioner = VLLMClient(
        base_url=cap_urls,
        model=args.model,
        max_concurrent=args.nj * 2,
    )

    # ---- 3. Copy data_dir → output_dir (full backup) ----
    if output_dir.exists():
        print(f"[INFO] Output dir already exists: {output_dir}")
    else:
        print(f"Copying {data_dir} → {output_dir} ...")
        shutil.copytree(str(data_dir), str(output_dir))
        print("  Done.")

    # ---- 4. Caption each subset and patch ----
    for scp_file in scp_files:
        subset = scp_file.stem  # e.g. "style_emotion", "style_whisper"
        print(f"\n{'='*60}")
        print(f"Processing subset: {subset}")
        print(f"{'='*60}")

        # Read scp
        scp_entries = read_scp(str(scp_file))
        print(f"  {len(scp_entries)} wav entries in {scp_file.name}")

        # Check for cached captions
        cache_path = None
        cached_captions: Dict[str, str] = {}
        if args.cache_dir:
            cache_path = Path(args.cache_dir) / f"{subset}.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    cached_captions = json.load(f)
                print(f"  Loaded {len(cached_captions)} cached captions from {cache_path}")

        # Filter out already-cached entries
        todo_entries = [
            (uid, wp) for uid, wp in scp_entries
            if uid not in cached_captions
        ]
        print(f"  {len(todo_entries)} entries need captioning "
              f"({len(scp_entries) - len(todo_entries)} cached)")

        # Caption remaining
        if todo_entries:
            new_captions = caption_batch(
                captioner, todo_entries, args.nj,
                desc=f"Caption [{subset}]",
            )
            # Merge with cache
            cached_captions.update(new_captions)

            # Save cache
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cached_captions, f, ensure_ascii=False, indent=2)
                print(f"  Saved {len(cached_captions)} captions to {cache_path}")

        print(f"  Total captions: {len(cached_captions)}/{len(scp_entries)}")

        # Patch the JSONL in output_dir
        src_jsonl = data_dir / f"{subset}.jsonl"
        dst_jsonl = output_dir / f"{subset}.jsonl"
        if src_jsonl.exists():
            patch_jsonl(
                str(src_jsonl),
                str(dst_jsonl),
                cached_captions,
                caption_field=args.caption_field,
            )
        else:
            print(f"  [WARN] Source JSONL not found: {src_jsonl}")

    print(f"\nAll done. Patched metadata saved to: {output_dir}")


if __name__ == "__main__":
    main()
