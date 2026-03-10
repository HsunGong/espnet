#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creative Audio-Edit Caption Generator (v3)
===========================================

Pipeline (caption-level only — no target audio synthesis):

  1. Sample source audio from AudioSet / local JSONL pools.
  2. Obtain source caption via Captioner (or reuse existing).
  3. LLM creatively proposes an edit → edit_prompt + target_caption.
  4. Judge validates quality, coherence, and style consistency.
  5. Emit { audio_path, audio_caption, target_caption, metadata }.

Key differences from step2_gen.py:
  - No mixing / target-audio generation.
  - The LLM itself invents the edit (rich, diverse edit_type_pool).
  - Optional "style bank" for writing-style consistency.
"""

import argparse
import base64
import io
import json
import multiprocessing
import os
import random
import re
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import jinja2
import librosa
import numpy as np
import soundfile as sf
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from local_split.sft_vllm_client import VLLMClient
from datasets import load_dataset, concatenate_datasets

random.seed(7)
warnings.filterwarnings("ignore")

SR = 16000

# =========================================================
# AudioSet Loading
# =========================================================

_ASET_DS = concatenate_datasets([
    load_dataset("agkphysics/AudioSet", "balanced", split="train", trust_remote_code=True),
    load_dataset("agkphysics/AudioSet", "balanced", split="test", trust_remote_code=True),
])


def _dur_filter_batch(batch):
    keep = []
    for a in batch["audio"]:
        dur = len(a["array"]) / a["sampling_rate"]
        keep.append(5.0 <= dur <= 15.0)
    return keep


_ASET_DS = _ASET_DS.filter(_dur_filter_batch, batched=True, batch_size=64, num_proc=256)


# =========================================================
# Audio Helpers
# =========================================================

def formulate_audio(y):
    y = np.nan_to_num(y)
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


# =========================================================
# Data Helpers
# =========================================================

def filter_data(jsonl_path):
    """Load local JSONL and filter by duration / caption availability."""
    out = []
    if not jsonl_path or not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["audio_caption"] = d.get("audio_caption", d.pop("qwen_caption", None))
            if not d["audio_caption"]:
                continue
            if d["duration"] < 5.0 or d["duration"] > 15.0:
                continue
            d["id"] = d.get("utt_id") or d.get("id") or uuid.uuid4().hex
            d["text"] = "The speaker says: \u201c" + d["text"] + "\u201d"
            out.append(d)
    return out


def classify_audioset(labels: Iterable[str]) -> str:
    text = " | ".join(str(x).strip().lower() for x in labels if x and str(x).strip())
    if not text:
        return "sound"

    SING = [
        r"\bsinging\b", r"\bsong\b", r"\brapping\b", r"\bchoir\b",
        r"\ba\s*capella\b", r"\bvocal music\b", r"\bsynthetic singing\b",
        r"\bmale singing\b", r"\bfemale singing\b", r"\bchild singing\b",
        r"\byodeling\b",
    ]
    SPEECH = [
        r"\bspeech\b", r"\bspeaking\b", r"\bconversation\b",
        r"\bnarration\b", r"\bmonologue\b", r"\bwhispering\b",
        r"\bchild speech\b", r"\bkid speaking\b",
        r"\bmale speech\b", r"\bfemale speech\b",
        r"\bspeech synthesizer\b", r"\bspeech babble\b",
        r"\bspeech noise\b", r"\bhubbub\b",
        r"\bshout\b", r"\byell\b",
    ]
    MUSIC = [r"\bmusic\b"]

    def hit(pats):
        return any(re.search(p, text) for p in pats)

    if hit(SING):
        return "sing"
    if hit(SPEECH):
        return "speech"
    if hit(MUSIC):
        return "music"
    return "sound"


def _map_audioset_labels(batch, indices):
    main_types, uids = [], []
    labels_list = batch.get("human_labels", [[]] * len(indices))
    vid_list = batch.get("video_id", [None] * len(indices))
    for labels, vid in zip(labels_list, vid_list):
        main_types.append(classify_audioset(labels))
        uids.append(vid if vid else uuid.uuid4().hex)
    return {"main_type": main_types, "uid": uids, "audioset_idx": indices}


POOLS = defaultdict(list)


def update_audioset_pools(**kwargs):
    print("Building AudioSet POOLS from labels …")
    num_proc = min(16, max(1, multiprocessing.cpu_count() - 2))
    for row in _ASET_DS.map(
        _map_audioset_labels,
        with_indices=True,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["audio", "labels"],
        desc="Classifying AudioSet labels",
    ):
        row["text"] = "The audio contains audio events: " + ",".join(row["human_labels"]) + "."
        POOLS[row["main_type"]].append(row)


# =========================================================
# Audio Loading & Captioning
# =========================================================

def _load_audio(d, min_dur: float = 5.0, max_dur: float = 15.0):
    if "audioset_idx" in d:
        row = _ASET_DS[d["audioset_idx"]]
        obj = row["audio"]
    else:
        obj = d["audio_path"]

    if isinstance(obj, dict) and "array" in obj and "sampling_rate" in obj:
        y = np.asarray(obj["array"], dtype=np.float32)
        dur = y.shape[-1] / obj["sampling_rate"]
        assert min_dur <= dur <= max_dur, f"Duration {dur:.2f}s out of range ({min_dur}-{max_dur})"
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        if int(obj["sampling_rate"]) != SR:
            y = librosa.resample(y, orig_sr=int(obj["sampling_rate"]), target_sr=SR)
    elif isinstance(obj, str) and os.path.exists(obj):
        y, _ = librosa.load(obj, sr=SR, mono=True)
        dur = y.shape[-1] / SR
        assert min_dur <= dur <= max_dur, f"Duration {dur:.2f}s out of range ({min_dur}-{max_dur})"
    else:
        raise FileNotFoundError(f"Cannot load audio: {obj}")
    return formulate_audio(y)


def get_caption_text(captioner, path_or_audio):
    """Send audio (file-path or numpy array) to the captioner and return text."""
    if isinstance(path_or_audio, (str, Path)):
        resp = captioner.chat_completion(
            messages=[{
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": f"file://{path_or_audio}"}}],
            }]
        )
    else:
        buf = io.BytesIO()
        sf.write(buf, path_or_audio, SR, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        resp = captioner.chat_completion(
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": "wav"},
                }],
            }]
        )
    return resp


def resolve_caption(d, y_audio, captioner):
    """Return the existing caption if available, otherwise caption the audio."""
    if d and d.get("audio_caption"):
        return d["audio_caption"]
    return get_caption_text(captioner, y_audio)


# =========================================================
# Dynamic Style Bank
# =========================================================

def build_style_bank(captioner, n: int = 10) -> List[str]:
    """Sample N entries uniformly across ALL POOLS and caption them for style-reference."""
    # Flatten all pool entries into one list so every type is represented
    n = n // len(POOLS)
    samples = []
    for pool_type, entries in POOLS.items():
        samples.extend(random.sample(entries, n))

    bank: List[str] = []

    def _caption_one(entry):
        # If the entry already has a good caption, use it directly
        existing = entry.get("audio_caption")
        if existing and isinstance(existing, str) and len(existing.strip()) > 20:
            return existing.strip()
        # Otherwise load audio and caption it
        y = _load_audio(entry)
        return get_caption_text(captioner, y)

    with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
        futures = {pool.submit(_caption_one, entry): entry for entry in samples}
        for f in as_completed(futures):
            try:
                cap = f.result()
                if cap and isinstance(cap, str) and len(cap.strip()) > 20:
                    bank.append(cap.strip())
            except Exception:
                pass

    print(f"Built dynamic style bank with {len(bank)} exemplars "
          f"(sampled from {len(samples)} total pool entries)")
    return bank


# =========================================================
# LLM Calls
# =========================================================

def format_edit_types(pool: list, n_sample: int = 8) -> str:
    """Return a random subset of edit types to present to the LLM.

    Showing a SUBSET (rather than the full pool) per call forces the model
    into different creative corners each time, dramatically improving diversity.
    """
    k = min(n_sample, len(pool))
    subset = random.sample(pool, k)
    return "\n".join(f"  - {t}" for t in subset)


def creative_edit_and_caption(llm, sys_p: str, usr_p: str, **kwargs):
    """
    Single LLM call that proposes a creative edit AND generates the target caption.
    Returns (edit_type, edit_prompt, target_caption).
    """
    rendered = jinja2.Template(usr_p).render(**kwargs)
    resp = llm.chat_completion(
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": rendered},
        ],
        json_mode=True,
    )
    if resp is None:
        raise RuntimeError("LLM returned None")
    return resp


def judge_edit(judge, sys_p: str, usr_p: str, **kwargs):
    """Call the judge to validate the creative edit."""
    rendered = jinja2.Template(usr_p).render(**kwargs)
    resp = judge.chat_completion(
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": rendered},
        ],
        json_mode=True,
    )
    if resp is None:
        return False, "Judge returned None"
    return resp.get("valid", False), resp.get("reason", "No reason provided")


# =========================================================
# Per-sample Processing
# =========================================================

def process_single_sample(
    idx: int,
    main_type: str,
    cfg: dict,
    llm: VLLMClient,
    judge_client: VLLMClient,
    cap: VLLMClient,
    out_dir: Path,
    style_bank: List[str],
    max_retries: int = 3,
):
    """
    Sample one audio, caption it, propose a creative edit, judge it.
    Retries internally up to max_retries times.
    """
    base = f"{main_type}_{idx:06d}"
    wav_path = (out_dir / f"{base}.wav").absolute()

    retries_left = max_retries
    while retries_left > 0:
        retries_left -= 1
        try:
            # 1. Sample random entry from the pool
            data = random.choice(POOLS[main_type])
            y = _load_audio(data)

            # 2. Obtain source caption
            source_caption = resolve_caption(data, y, cap)
            if not source_caption or not isinstance(source_caption, str) or len(source_caption.strip()) < 10:
                raise ValueError("Source caption too short or missing")

            # 3. Select style exemplars (random 2-3 from bank)
            n_pick = min(3, len(style_bank))
            exemplars = random.sample(style_bank, n_pick) if n_pick > 0 else []

            # 4. LLM: creative edit + target caption (single call)
            #    Random subset of edit types per call → forces diversity
            resp = creative_edit_and_caption(
                llm,
                cfg["creative_edit_params"]["system_prompt"],
                cfg["creative_edit_params"]["user_prompt"],
                source_caption=source_caption,
                style_exemplars=exemplars,
                edit_types=format_edit_types(cfg.get("edit_type_pool", []), n_sample=8),
            )

            # 5. Judge
            is_valid, reason = judge_edit(
                judge_client,
                cfg["judge_params"]["system_prompt"],
                cfg["judge_params"]["judge_user_prompt"],
                source_caption=source_caption,
                edit_prompt=resp["edit_prompt"],
                target_caption=resp["target_caption"],
            )

            if not is_valid:
                raise ValueError(f"Judge rejected: {reason}")

            # 6. Persist source audio to disk (AudioSet has no local path)
            if "audioset_idx" in data:
                sf.write(wav_path, y, SR)
                audio_path = str(wav_path)
            else:
                audio_path = data.get("audio_path", str(wav_path))
                if not os.path.exists(audio_path):
                    sf.write(wav_path, y, SR)
                    audio_path = str(wav_path)

            return {
                "id": uuid.uuid4().hex,
                "audio_type": main_type,
                "audio_path": audio_path,
                "audio_caption": source_caption,
                "target_audio_caption": resp.pop("target_caption", ""),
                "edit_prompt": resp.pop("edit_prompt", ""),
                "judge_reason": reason,
                **resp,
            }

        except Exception as e:
            tqdm.write(f"[{main_type}][retry_left={retries_left}] {e}")

    return None  # all retries exhausted


# =========================================================
# Orchestration
# =========================================================

def process(mode: str, cfg: dict, args, style_bank: List[str]):
    """Generate creative-edit samples for one audio type (mode)."""
    wav_dir = args.output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    llm = VLLMClient(
        base_url=cfg["vllm"]["url"],
        model=cfg["vllm"]["model"],
        max_concurrent=args.nj * 2,
    )
    judge_client = VLLMClient(
        base_url=cfg["judge"]["url"],
        model=cfg["judge"]["model"],
        max_concurrent=args.nj * 2,
    )
    cap = VLLMClient(
        base_url=cfg["captioner"]["url"],
        model=cfg["captioner"]["model"],
        max_concurrent=args.nj * 2,
    )

    out_file = args.output_dir / f"{mode}_creative_edit.jsonl"
    valid_count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=args.k, desc=f"Collecting {mode} creative edits")
        for res in Parallel(n_jobs=args.nj, backend="threading", return_as="generator")(
            delayed(process_single_sample)(
                i,
                mode,
                cfg,
                llm,
                judge_client,
                cap,
                out_dir=wav_dir,
                style_bank=style_bank,
                max_retries=args.max_retries,
            )
            for i in range(args.k)
        ):
            pbar.update(1)
            if res is None:
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            valid_count += 1
            pbar.set_postfix({"valid": valid_count})
        pbar.close()

    print(f"[{mode}] Saved {valid_count}/{args.k} valid samples → {out_file}")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Creative audio-edit caption generator (v3). "
                    "Proposes imaginative edits for source captions and generates target captions."
    )
    parser.add_argument("--speech", default="audioset", help="Path to JSONL or 'audioset'")
    parser.add_argument("--music",  default="audioset", help="Path to JSONL or 'audioset'")
    parser.add_argument("--sing",   default="audioset", help="Path to JSONL or 'audioset'")
    parser.add_argument("--sound",  default="audioset", help="Path to JSONL or 'audioset'")
    parser.add_argument("-o", "--output_dir", type=Path, required=True,
                        help="Directory for output JSONL and wavs/")
    parser.add_argument("-c", "--config", required=True, help="Path to gen_v3.yaml")
    parser.add_argument("-k", "--k", type=int, default=100,
                        help="Number of samples to generate per audio type")
    parser.add_argument("--nj", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per sample before giving up")
    parser.add_argument("--mode", type=str, default="speech,music,sing,sound",
                        help="Comma-separated audio types to process")
    parser.add_argument("--style_bank_size", type=int, default=0,
                        help="Pre-caption N random AudioSet samples as dynamic style exemplars "
                             "(0 = only use exemplars from YAML config)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Build audio pools
    print("Loading data …")
    update_audioset_pools(
        speech=args.speech == "audioset",
        music=args.music == "audioset",
        sing=args.sing == "audioset",
        sound=args.sound == "audioset",
    )
    if args.speech != "audioset":
        POOLS["speech"] = filter_data(args.speech)
    if args.music != "audioset":
        POOLS["music"] = filter_data(args.music)
    if args.sing != "audioset":
        POOLS["sing"] = filter_data(args.sing)
    if args.sound != "audioset":
        POOLS["sound"] = filter_data(args.sound)

    for k, v in POOLS.items():
        print(f"  {k}: {len(v)} samples")

    # Build style bank: static (yaml) + optional dynamic (captioner)
    style_bank = list(cfg.get("style_exemplars", []))
    n_static = len(style_bank)
    if args.style_bank_size > 0:
        cap_client = VLLMClient(
            base_url=cfg["captioner"]["url"],
            model=cfg["captioner"]["model"],
            max_concurrent=16,
        )
        dynamic = build_style_bank(cap_client, n=args.style_bank_size)
        style_bank.extend(dynamic)
    print(f"Style bank: {len(style_bank)} exemplars "
          f"({n_static} static + {len(style_bank) - n_static} dynamic)")

    # Process each requested audio type
    for mode in args.mode.split(","):
        mode = mode.strip()
        if mode not in POOLS or len(POOLS[mode]) == 0:
            print(f"Skipping '{mode}': no samples available")
            continue
        process(mode, cfg, args, style_bank)

    print("Done.")


if __name__ == "__main__":
    main()
