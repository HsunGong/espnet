#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference client script for Qwen-tts (vLLM-Omni OpenAI-compatible Speech API).
Model selection is done by URL (different endpoints), NOT by 'model' field in payload.

Rules:
- Content/text modification => Base (voice-clone): task_type="Base", with ref_audio/ref_text if available
- Style change => VoiceDesign (voice-design): task_type="VoiceDesign", with instructions
"""
import random
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from joblib import Parallel, delayed
from tqdm import tqdm


def encode_audio_to_base64(audio_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/wav"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def is_style_change(edit_type: str) -> bool:
    return (edit_type or "") in {"style_whisper", "style_style", "style_emotion", "audio_effect_speed"}


PARA_TAG_MAP = {
    "[sigh]":                "Sigh...",
    "[laugh]":               "ha ha!",
    "[exhale]":              "Haaa...",
    "[snort]":               "Humph!",
    "[cough]":               "Hack...",
    "[uhm]":                 "uhmm...",
    "[Surprise-oh]":         "oh!",
    "[Surprise-wa]":         "wa!",
    "[Dissatisfaction-hnn]": "hmm",
    "[Question-ah]":         "ah?",
    "[Question-yi]":         "yi?",
}

def map_paralinguistic_tokens(text: str) -> str:
    for src, dst in PARA_TAG_MAP.items():
        text = text.replace(src, dst)
    return text

def map_jsonl_to_payload_and_url(
    data: Dict[str, Any],
    *,
    base_url: str,
    design_url: str,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (request_url, payload, ref_audio_path_for_debug)
    """
    design_url = random.choice(design_url.split(","))
    base_url = random.choice(base_url.split(","))

    target_text= map_paralinguistic_tokens(data["target_text"])

    payload: Dict[str, Any] = {
        "input": target_text,
        "task_type": "Base",
        "ref_audio": encode_audio_to_base64(data["audio_path"]),
        "ref_text": data["text"],
        "max_new_tokens": 4096,
        "x_vector_only_mode": False # use ICL mode
    }
    payload["x_vector_only_mode"] = False

    return base_url, payload


def post_speech(
    url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> bytes:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    if resp.status_code != 200:
        try:
            err_txt = resp.content.decode("utf-8", errors="replace")
        except Exception:
            err_txt = "<non-utf8 error>"
        raise RuntimeError(f"HTTP {resp.status_code}: {err_txt}")

    # Guard: sometimes server returns JSON error with 200
    try:
        txt = resp.content.decode("utf-8")
        if txt.startswith('{"error"'):
            raise RuntimeError(f"Server error (200): {txt}")
    except UnicodeDecodeError:
        pass

    return resp.content


def process_one(
    data: Dict[str, Any],
    *,
    base_url: str,
    design_url: str,
    api_key: str,
    out_dir: str,
    voice: str,
    language: str,
    response_format: str,
    max_new_tokens: int,
    skip_existing: bool,
) -> Tuple[str, bool, str, str]:
    utt_id = str(data.get("id", data.get("utt_id", "")))
    if not utt_id:
        return "<missing-id>", False, "", "Missing id/utt_id"

    if is_style_change(data["edit_type"]):
        return utt_id, False, "", f"Can not do style change task while preserving the speaker identity ({data['edit_type']})"

    out_path = os.path.abspath(os.path.join(out_dir, f"{utt_id}.{response_format}"))
    if skip_existing and os.path.exists(out_path):
        return utt_id, True, out_path, "Already exists"

    url, payload = map_jsonl_to_payload_and_url(
        data,
        base_url=base_url,
        design_url=design_url,
        voice=voice,
        language=language,
        response_format=response_format,
        max_new_tokens=max_new_tokens,
    )

    try:
        audio_bytes = post_speech(url, api_key, payload)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return utt_id, True, out_path, f"Success ({payload.get('task_type')} -> {url})"
    except Exception as e:
        return utt_id, False, out_path, str(e)


def get_args():
    p = argparse.ArgumentParser("Concurrent Qwen-TTS client (model selected by URL)")
    # Two endpoints (each uniquely maps to one model)
    p.add_argument("--base-url", type=str, default="http://localhost:8091/v1/audio/speech", # "http://localhost:8000/v1/audio/speech,http://localhost:8001/v1/audio/speech,http://localhost:8002/v1/audio/speech,http://localhost:8003/v1/audio/speech",
                   help="Base/voice-clone endpoint")
    p.add_argument("--design-url", type=str,default="http://localhost:8004/v1/audio/speech,http://localhost:8005/v1/audio/speech,http://localhost:8006/v1/audio/speech,http://localhost:8007/v1/audio/speech",
                   help="VoiceDesign/voice-design endpoint")
    p.add_argument("--api-key", type=str, default="", help="Bearer token; empty for no auth")

    p.add_argument("--jsonl-files", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-workers", type=int, default=64)
    p.add_argument("--skip-existing", action="store_true")

    # Request params
    p.add_argument("--voice", type=str, default="vivian")
    p.add_argument("--language", type=str, default="Auto")
    p.add_argument("--response-format", type=str, default="wav",
                   choices=["wav", "mp3", "flac", "pcm", "aac", "opus"])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    return p.parse_args()

def process_jsonl(jsonl_file, args):
    if not os.path.exists(jsonl_file):
        print(f"[Warning] File not found: {jsonl_file}")
        return

    file_stem = Path(jsonl_file).stem
    save_dir = os.path.join(args.output_dir, file_stem)
    scp_path = os.path.join(args.output_dir, f"{file_stem}.scp")
    os.makedirs(save_dir, exist_ok=True)

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[Error] JSON decode failed: {jsonl_file}:{idx}")

    parallel_executor = Parallel(n_jobs=args.num_workers, return_as="generator", prefer="threads")

    ok_cnt = 0
    with open(scp_path, "w", encoding="utf-8") as fscp:
        pbar = tqdm(total=len(tasks), desc=f"Processing {file_stem}", unit="utt")
        for res in parallel_executor(
            delayed(process_one)(
                task,
                base_url=args.base_url,
                design_url=args.design_url,
                api_key=args.api_key,
                out_dir=save_dir,
                voice=args.voice,
                language=args.language,
                response_format=args.response_format,
                max_new_tokens=args.max_new_tokens,
                skip_existing=args.skip_existing,
            )
            for task in tasks
        ):
            utt_id, ok, out_path, msg = res
            pbar.update(1)
            if ok:
                fscp.write(f"{utt_id}\t{out_path}\n")
                ok_cnt += 1
                pbar.set_postfix_str(f"OK: {ok_cnt}/{pbar.n}")
            else:
                print(f"[Failed] {utt_id}: {msg}")

    print(f"--- Finished {file_stem} ({ok_cnt}/{len(tasks)} ok) ---")
    print(f"Audio saved → {save_dir}")
    print(f"SCP saved   → {scp_path}\n")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Info] Base URL   : {args.base_url}")
    print(f"[Info] Design URL : {args.design_url}")
    print(f"[Info] Workers    : {args.num_workers}")
    print(f"[Info] Voice      : {args.voice} | Language: {args.language} | Format: {args.response_format}")

    for _ in Parallel(n_jobs=args.num_workers)(delayed(process_jsonl)(jsonl_file, args) for jsonl_file in args.jsonl_files):
        pass

if __name__ == "__main__":
    main()
