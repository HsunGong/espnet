#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Mix-edit dataset generator:
  - Generates Source/Target wavs and runs Captioner on BOTH.
  - LLM strictly generates the edit_prompt.
  - Judger validates the captioner's textual outputs against the delta text.
  - Worker threads loop internally (resample -> filter -> judge) until 1 valid sample is produced.
  - Supports mixing remote AudioSet and local JSONL sources via duck-typing.
"""

import argparse
import json
import os
import random
import uuid
import warnings
import multiprocessing
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

# =========================
# Audio Helpers
# =========================

_ASET_DS = concatenate_datasets([load_dataset("agkphysics/AudioSet", "balanced", split="train", trust_remote_code=True), load_dataset("agkphysics/AudioSet", "balanced", split="test", trust_remote_code=True)])

def dur_gt_5_batch(batch):
    keep = []
    for a in batch["audio"]:
        arr = a["array"]
        sr = a["sampling_rate"]
        keep.append(15 >= (len(arr) / sr) >= 5.0)
    return keep

_ASET_DS = _ASET_DS.filter(dur_gt_5_batch, batched=True, batch_size=64, num_proc=256)

def formulate_audio(y):
    y = np.nan_to_num(y)
    m = np.max(np.abs(y))
    return y / m if m > 0 else y

def mix_audio(y1, y2):
    out2 = np.pad(y2, (0, len(y1) - len(y2))) if len(y1) > len(y2) else y2[: len(y1)]
    y1_rms = float(np.mean(np.abs(y1)) + 1e-8)
    y2_rms = float(np.mean(np.abs(out2)) + 1e-8)
    scale = y2_rms / y1_rms
    mixed = y1 + (0.8 * out2 / scale if scale > 0.8 else out2)
    return formulate_audio(mixed)

# =========================
# Data Helpers
# =========================

def filter_data(jsonl_path):
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
            d["text"] = "The speaker says: “" + d["text"] + "”"
            out.append(d)
    return out
import re
from typing import Iterable

def classify_audioset(labels: Iterable[str]) -> str:
    """
    Classify AudioSet-style labels into: speech / sing / music / sound

    Strategy:
    - Join all tags into one normalized string
    - Match by shared keywords (regex / substring)
    - Fallback to "sound"
    """
    # 1) join -> one string
    text = " | ".join(str(x).strip().lower() for x in labels if x and str(x).strip())
    if not text:
        return "sound"

    # 2) common keys (keep small & robust)
    # NOTE: use word boundaries where possible to avoid accidental matches
    SING_PATTERNS = [
        r"\bsinging\b", r"\bsong\b", r"\brapping\b", r"\bchoir\b",
        r"\ba\s*capella\b", r"\bvocal music\b", r"\bsynthetic singing\b",
        r"\bmale singing\b", r"\bfemale singing\b", r"\bchild singing\b",
        r"\byodeling\b",  # 你的标签里有
    ]

    SPEECH_PATTERNS = [
        r"\bspeech\b", r"\bspeaking\b", r"\bconversation\b",
        r"\bnarration\b", r"\bmonologue\b",
        r"\bwhispering\b", r"\bchild speech\b", r"\bkid speaking\b",
        r"\bmale speech\b", r"\bfemale speech\b",
        r"\bspeech synthesizer\b",
        r"\bspeech babble\b", r"\bspeech noise\b", r"\bhubbub\b",
        r"\bshout\b", r"\byell\b",
    ]

    # music: 你的标签大量是 "xxx music"，用一个公共 key 覆盖
    MUSIC_PATTERNS = [
        r"\bmusic\b",  # 覆盖 "rock music", "theme music", "background music", 以及 "Music" soundtrack
    ]

    def hit(patterns):
        return any(re.search(p, text) for p in patterns)

    if hit(SING_PATTERNS):
        return "sing"
    if hit(SPEECH_PATTERNS):
        return "speech"
    if hit(MUSIC_PATTERNS):
        return "music"
    return "sound"

def _map_audioset_labels(batch, indices):
    main_types = []
    uids = []
    labels_list = batch.get("human_labels", [[]] * len(indices))
    vid_list = batch.get("video_id", [None] * len(indices))

    for labels, vid in zip(labels_list, vid_list):
        main_types.append(classify_audioset(labels))
        uids.append(vid if vid else uuid.uuid4().hex)

    return {"main_type": main_types, "uid": uids, "audioset_idx": indices}

POOLS = defaultdict(list)  # type: ignore[assignment]

def update_audioset_pools(speech: bool = False, music: bool = False, sing: bool = False, sound: bool = False):
    print("Building AudioSet POOLS from labels using HF concurrency...")
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

def sample_single_task(main_type, op):
    def pick_delta(dtype):
        cands = POOLS[dtype]
        return random.choice(cands)

    main_data = random.choice(POOLS[main_type])
    if main_type == "speech" or main_type == "sing":
        dtype = random.choice(["music", "sound"])
    elif main_type == "music":
        dtype = random.choice(["speech", "sing", "sound"])
    elif main_type == "sound":
        dtype = random.choice(["speech", "sing", "music", "sound"])

    if op == "ADD":
        y_data = pick_delta(dtype)
        return {"op": "ADD", "main_type": main_type, "main": main_data, "y": y_data}
    elif op == "REMOVE":
        x_data = pick_delta(dtype)
        return {"op": "REMOVE", "main_type": main_type, "main": main_data, "x": x_data}
    elif op == "REPLACE":
        x_data = pick_delta(dtype)
        y_data = pick_delta(dtype)
        return {
            "op": "REPLACE",
            "main_type": main_type,
            "main": main_data,
            "x": x_data,
            "y": y_data,
        }
    return None

# =========================
# Execution Logic
# =========================
import base64
import io

def get_caption_text(captioner, path_or_audio):
    if isinstance(path_or_audio, str) or isinstance(path_or_audio, Path):
        resp = captioner.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": f"file://{path_or_audio}"}}],
                }
            ]
        )
    else:
        _audio_io = io.BytesIO()
        sf.write(_audio_io, path_or_audio, SR, format="WAV")
        audio_base64 = base64.b64encode(_audio_io.getvalue()).decode("utf-8")
        resp = captioner.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav",
                        },
                    }],
                }
            ]
        )

    return resp

def resolve_caption(d, y_audio, captioner):
    if d and d.get("audio_caption"):
        return d["audio_caption"]
    return get_caption_text(captioner, y_audio)


def _load_audio(d, min_dur: float = 5.0, max_dur: float = 15.0):
    if "audioset_idx" in d:
        row = _ASET_DS[d["audioset_idx"]]
        obj = row["audio"]
    else:
        obj = d["audio_path"]

    # Otherwise, it's from a local JSONL
    if isinstance(obj, dict) and "array" in obj and "sampling_rate" in obj:
        y = np.asarray(obj["array"], dtype=np.float32)
        duration = y.shape[-1] / obj["sampling_rate"]
        assert min_dur <= duration <= max_dur, f"Audio duration {duration:.2f}s out of bounds ({min_dur}-{max_dur}s)"

        if y.ndim > 1:
            y = np.mean(y, axis=0)
        if int(obj["sampling_rate"]) != SR:
            y = librosa.resample(y, orig_sr=int(obj["sampling_rate"]), target_sr=SR)

    elif isinstance(obj, str) and os.path.exists(obj):
        y, _ = librosa.load(obj, sr=SR, mono=True)
        duration = y.shape[-1] / SR
        assert min_dur <= duration <= max_dur, f"Audio duration {duration:.2f}s out of bounds ({min_dur}-{max_dur}s)"
    return y

def judge_edit(judge, sys_p, usr_p, **kwargs):
    prompt = jinja2.Template(usr_p).render(**kwargs)
    resp = judge.chat_completion(
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}],
        json_mode=True,
    )
    return resp.get("valid", False), resp.get("reason", "No reason provided")

def get_edit_prompt(llm, sys_p, usr_p, **kwargs):
    prompt = jinja2.Template(usr_p).render(**kwargs)
    resp = llm.chat_completion(
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}],
        json_mode=True,
    )
    return resp["edit_prompt"]

def construct_target_caption(llm, sys_p, usr_p, **kwargs):
    prompt = jinja2.Template(usr_p).render(**kwargs)
    resp = llm.chat_completion(
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}],
        json_mode=True,
    )
    return resp["target_caption"], resp["edit_prompt"]


def process_single_sample_with_retry(
    idx: int, main_type: str, op: str, cfg, llm, judge, cap, out_dir: Path, max_retries: int
):
    base = f"{main_type}_{op.lower()}_{idx:05d}"
    in_wav = (out_dir / f"{base}_input.wav").absolute()
    tgt_wav = (out_dir / f"{base}_target.wav").absolute()

    while max_retries > 0:
        max_retries -= 1
        try:
            task = sample_single_task(main_type, op)
            y_main = _load_audio(task["main"])

            if op == "ADD":
                # main + y = target
                y_Y = _load_audio(task["y"])
                sf.write(in_wav, y_main, SR)
                sf.write(tgt_wav, mix_audio(y_main, y_Y), SR)

                # qwen-caption
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_main = executor.submit(resolve_caption, task["main"], in_wav, captioner=cap)
                    future_y    = executor.submit(resolve_caption, task["y"], y_Y, captioner=cap)
                    future_tgt  = executor.submit(get_caption_text, cap, tgt_wav)

                    task["main"]["audio_caption"] = future_main.result()
                    task["y"]["audio_caption"]    = future_y.result()
                    _tgt_caption                  = future_tgt.result()

                # llm-caption
                tgt_cap_llm, edit_prompt = construct_target_caption(
                    llm,
                    cfg["system_prompt"],
                    cfg["user_prompt"],
                    source_caption=task["main"]["audio_caption"],
                    add_caption=task["y"]["audio_caption"],
                    add_event=task["y"]["text"], # plain text to be added
                )

                task["target"] = {
                    "audio_path": str(tgt_wav),
                    "audio_caption": _tgt_caption,
                    "audio_caption_llm": tgt_cap_llm,
                }

                src_cap = task["main"]["audio_caption"]
                tgt_cap = task["target"]["audio_caption_llm"]
                # edit_prompt

            elif op == "REMOVE":
                # source - x = main
                y_X = _load_audio(task["x"])
                sf.write(in_wav, mix_audio(y_main, y_X), SR)
                sf.write(tgt_wav, y_main, SR)

                with ThreadPoolExecutor(max_workers=3) as executor:
                    f_main = executor.submit(resolve_caption, task["main"], tgt_wav, captioner=cap)
                    f_x    = executor.submit(resolve_caption, task["x"], y_X, captioner=cap)
                    f_src  = executor.submit(get_caption_text, cap, in_wav)

                    task["main"]["audio_caption"] = f_main.result()
                    task["x"]["audio_caption"]    = f_x.result()
                    _src_caption                  = f_src.result()

                src_cap_llm, _ = construct_target_caption(
                    llm,
                    cfg["system_prompt"],
                    cfg["user_prompt"],
                    source_caption=task["main"]["audio_caption"],
                    add_caption=task["x"]["audio_caption"],
                    add_event=task["x"]["text"], # plain text to be removed
                ) # src = main + x

                task["source"] = {
                    "audio_path": str(in_wav),
                    "audio_caption": _src_caption,
                    "audio_caption_llm": src_cap_llm,
                }

                edit_prompt = get_edit_prompt(
                    llm,
                    cfg["edit_system_prompt"],
                    cfg["edit_user_prompt"],
                    source_caption=task["source"]["audio_caption_llm"],
                    target_caption=task["main"]["audio_caption"],
                    remove_caption=task["x"]["audio_caption"],
                    remove_event=task["x"]["text"], # plain text to be removed
                )

                src_cap = task["source"]["audio_caption_llm"]
                tgt_cap = task["main"]["audio_caption"]

            elif op == "REPLACE":
                y_X = _load_audio(task["x"])
                y_Y = _load_audio(task["y"])
                sf.write(in_wav, mix_audio(y_main, y_X), SR)
                sf.write(tgt_wav, mix_audio(y_main, y_Y), SR)

                with ThreadPoolExecutor(max_workers=5) as executor:
                    f_main = executor.submit(resolve_caption, task["main"], tgt_wav, captioner=cap)
                    f_x    = executor.submit(resolve_caption, task["x"], y_X, captioner=cap)
                    f_y    = executor.submit(resolve_caption, task["y"], y_Y, captioner=cap)
                    f_src  = executor.submit(get_caption_text, cap, in_wav)
                    f_tgt  = executor.submit(get_caption_text, cap, tgt_wav)

                    task["main"]["audio_caption"] = f_main.result()
                    task["x"]["audio_caption"]    = f_x.result()
                    task["y"]["audio_caption"]    = f_y.result()
                    _src_caption                  = f_src.result()
                    _tgt_caption                  = f_tgt.result()

                src_cap_llm, _ = construct_target_caption(
                    llm,
                    cfg["system_prompt"],
                    cfg["user_prompt"],
                    source_caption=task["main"]["audio_caption"],
                    add_caption=task["y"]["audio_caption"],
                    add_event=task["y"]["text"], # plain text to be added
                ) # src = main + x
                tgt_cap_llm, _ = construct_target_caption(
                    llm,
                    cfg["system_prompt"],
                    cfg["user_prompt"],
                    source_caption=task["main"]["audio_caption"],
                    add_caption=task["y"]["audio_caption"],
                    add_event=task["y"]["text"], # plain text to be added
                ) # tgt = main + y
                task["source"] = {
                    "audio_path": str(in_wav),
                    "audio_caption": _src_caption,
                    "audio_caption_llm": src_cap_llm,
                }
                task["target"] = {
                    "audio_path": str(tgt_wav),
                    "audio_caption": _tgt_caption,
                    "audio_caption_llm": tgt_cap_llm,
                }

                edit_prompt = get_edit_prompt(
                    llm,
                    cfg["edit_system_prompt"],
                    cfg["edit_user_prompt"],
                    source_caption=task["source"]["audio_caption"],
                    target_caption=task["target"]["audio_caption_llm"],
                    remove_caption=task["x"]["audio_caption"],
                    remove_event=task["x"]["text"], # plain text to be removed
                    add_caption=task["y"]["audio_caption"],
                    add_event=task["y"]["text"], # plain text to be added
                )

                src_cap = task["source"]["audio_caption_llm"]
                tgt_cap = task["target"]["audio_caption_llm"]
            else:
                raise

            is_valid, reason = judge_edit(
                judge,
                cfg["judge_system_prompt"],
                cfg["judge_user_prompt"],
                source_caption=src_cap,
                target_caption=tgt_cap,
                edit_prompt=edit_prompt,
            )
            # print(">>", task)
            if not is_valid:
                raise ValueError("Judgement failed: " + reason)

            return {
                "id": uuid.uuid4().hex,
                "operation": op,
                "audio_path": str(in_wav),
                "audio_caption": src_cap, # should possible come from llm
                "target_audio_path": str(tgt_wav),
                "target_audio_caption": tgt_cap, # should possible come from llm
                "edit_prompt": edit_prompt,
                "metadata": task,
                "judge_reason": reason,
            }
        except Exception as e:
            tqdm.write(f"[max_retry={max_retries}] Failed due to {e}")


# =========================
# Main
# =========================

def process(mode: str, op: str, cfg:dict, args):
    wav_dir = args.output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    llm = VLLMClient(
        base_url=cfg["vllm"]["url"], model=cfg["vllm"]["model"], max_concurrent=args.nj * 2
    )
    judge = VLLMClient(
        base_url=cfg["judge"]["url"], model=cfg["judge"]["model"], max_concurrent=args.nj * 2
    )
    cap = VLLMClient(
        base_url=cfg["captioner"]["url"],
        model=cfg["captioner"]["model"],
        max_concurrent=args.nj * 2,
    )

    valid_count = 0
    with open(args.output_dir / f"{mode}_{op.lower()}_mix.jsonl", "w", encoding="utf-8") as f:
        pbar = tqdm(total=args.k, desc=f"Collecting {mode}_{op.lower()} samples")
        for res in Parallel(n_jobs=args.nj, backend="threading", return_as="generator")(
            delayed(process_single_sample_with_retry)(
                i,
                mode,
                op,
                cfg[f"{op.lower()}_params"],
                llm=llm,
                judge=judge,
                cap=cap,
                out_dir=wav_dir,
                max_retries=3,
            )
            for i in range(args.k)
        ):
            pbar.update(1)
            if res is None:
                continue
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            valid_count += 1
            pbar.set_postfix({"valid": valid_count})
        pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech", default="audioset", help="Path to jsonl or 'audioset'")
    parser.add_argument("--music", default="audioset", help="Path to jsonl or 'audioset'")
    parser.add_argument("--sing", default="audioset", help="Path to jsonl, 'audioset', or None")
    parser.add_argument("--sound", default="audioset", help="Path to jsonl or 'audioset'")
    parser.add_argument("-o", "--output_dir", type=Path)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-k", "--k", type=int, default=100)
    parser.add_argument("--nj", type=int, default=8)
    parser.add_argument("--mode", type=str, default="speech,music,sing,sound")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("Loading data...")
    update_audioset_pools(speech=args.speech=="audioset", music=args.music=="audioset", sing=args.sing=="audioset", sound=args.sound=="audioset")

    if args.speech != "audioset":
        POOLS["speech"] = filter_data(args.speech)
    if args.music != "audioset":
        POOLS["music"] = filter_data(args.music)
    if args.sing != "audioset":
        POOLS["sing"] = filter_data(args.sing)
    if args.sound != "audioset":
        POOLS["sound"] = filter_data(args.sound)

    for k, v in POOLS.items():
        print(f"Category {k} has {len(v)} samples after filtering.")

    for mode in args.mode.split(","):
        for op in ["ADD", "REMOVE", "REPLACE"]:
            process(mode, op, cfg=cfg, args=args)

if __name__ == "__main__":
    main()