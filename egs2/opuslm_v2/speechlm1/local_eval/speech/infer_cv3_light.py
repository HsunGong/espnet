"""
Batch inference client script for CosyVoice3.
Sends concurrent requests to a remote/local CosyVoice server using joblib.
"""

import argparse
import json
import os
import requests
from pathlib import Path

import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CosyVoice3 supported paralinguistic tokens
# ---------------------------------------------------------------------------
# Single Tags (Sound Insertion)
# Tag	Effect	Example
# <breath>	Breathing sound	I'm tired <breath> let's rest
# <quick_breath>	Quick breath	Running <quick_breath> almost there
# <laughter>	Laughter	That's hilarious <laughter>!
# <cough>	Cough	Excuse me <cough> sorry
# <sigh>	Sigh	Fine <sigh> I'll do it
# <gasp>	Gasp	Oh no <gasp> what happened?
# <noise>	Background noise	Walking <noise> through forest
# <hissing>	Hissing sound	The snake <hissing> away
# <vocalized-noise>	Vocalized noise	Hmm <vocalized-noise> interesting
# <lipsmack>	Lip smack	Delicious <lipsmack> food
# <mn>	"mn" sound	I think <mn> maybe
# <clucking>	Clucking sound	Disapproving <clucking>
# <accent>	Accent emphasis	Very <accent> important

CV3_SUPPORTED_TOKENS = {
    "[breath]", "[noise]", "[laughter]", "[cough]", "[clucking]",
    "[accent]", "[quick_breath]", "[hissing]", "[sigh]",
    "[vocalized-noise]", "[lipsmack]", "[mn]",
    "<laughter>", "</laughter>",
}

# We evaluate the effectiveness of instructed generation capabilities using the Expresso [59] datasetalongside an internal expressive dataset. The Expresso dataset is a multi-speaker expressive speechcollection featuring eight distinct speaking styles, evaluated on a subset of 3,000 samples. Ourinternal dataset includes 3,600 samples, matching the domains of the instruction-following trainingdataset and encompassing over 50 different emotions, speeds, dialects, accents, and role-playingspeaking styles.
# Map input paralinguistic tags → closest CV3-supported token
PARA_TAG_MAP = {
    "[sigh]":                "[sigh]",
    "[laugh]":               "[laughter]",
    "[exhale]":              "[breath]",
    "[snort]":               "[hissing]",
    "[cough]":               "[cough]",
    "[uhm]":                 "[mn]",
    "[Surprise-oh]":         "[vocalized-noise]",
    "[Surprise-wa]":         "[vocalized-noise]",
    "[Dissatisfaction-hnn]": "[mn]",
    "[Question-ah]":         "[vocalized-noise]",
    "[Question-yi]":         "[vocalized-noise]",
}

STYLE_INSTRUCTION = {
    "serious": "Speak in a serious and formal tone.", "arrogant": "Speak in an arrogant and proud tone.",
    "child": "Speak in a childlike, innocent voice.", "older": "Speak in an elderly, calm voice.",
    "girl": "Speak in a lively, young female voice.", "pure": "Speak in a pure and clear voice.",
    "sister": "Speak in a warm, elder-sister tone.", "sweet": "Speak in a sweet and gentle voice.",
    "ethereal": "Speak in an ethereal and dreamy voice.", "whisper": "Speak in a soft whisper.",
    "gentle": "Speak in a gentle and soft tone.", "recite": "Speak in a recitation style.",
    "generous": "Speak in a generous and magnanimous tone.", "act_coy": "Speak in a coy and playful manner.",
    "warm": "Speak in a warm and caring tone.", "shy": "Speak in a shy and timid voice.",
    "comfort": "Speak in a comforting and soothing tone.", "authority": "Speak in an authoritative tone.",
    "chat": "Speak in a casual, chatty style.", "radio": "Speak in a radio broadcast style.",
    "soulful": "Speak in a soulful, heartfelt voice.", "story": "Speak in a storytelling style.",
    "vivid": "Speak in a vivid and expressive style.", "program": "Speak in a program-host style.",
    "news": "Speak in a news anchor style.", "advertising": "Speak in an advertising and promotional style.",
    "roar": "Speak in a loud, roaring voice.", "murmur": "Speak in a low murmur.",
    "shout": "Shout the sentence loudly.", "deeply": "Speak in a deep, resonant voice.",
    "loudly": "Speak loudly and clearly.", "remove": "Speak in a plain, neutral voice.",
    "exaggerated": "Speak in an exaggerated, theatrical manner."
}

EMOTION_INSTRUCTION = {
    "happy": "Speak with happiness and joy.", "angry": "Speak with anger.",
    "sad": "Speak with sadness.", "humour": "Speak in a humorous and funny tone.",
    "confusion": "Speak with confusion and uncertainty.", "disgusted": "Speak with disgust.",
    "empathy": "Speak with empathy and understanding.", "embarrass": "Speak with embarrassment.",
    "fear": "Speak with fear.", "surprised": "Speak with surprise.",
    "excited": "Speak with excitement.", "depressed": "Speak with depression and low energy.",
    "coldness": "Speak in a cold and distant tone.", "admiration": "Speak with admiration."
}

def map_paralinguistic_tokens(text: str) -> str:
    for src, dst in PARA_TAG_MAP.items():
        if src not in CV3_SUPPORTED_TOKENS:
            text = text.replace(src, dst)
    return text

# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------
def map_jsonl_to_zero_shot_payload(data: dict):
    """
    将所有逻辑统一映射到 zero_shot 接口所需的 payload。
    """
    edit_type   = data.get("edit_type", "")
    kwargs      = data.get("edit_kwargs", {})
    target_text = data.get("target_text", data.get("text", ""))
    prompt_text = data.get("text", "") # 原始参考文本
    prompt_wav  = data.get("audio_path", "")

    # 1. 处理指令/风格/情绪 (原本 instruct2 的逻辑)
    instruction = ""
    if edit_type in {"style_whisper", "style_style"}:
        style = kwargs.get("style", "whisper")
        instruction = STYLE_INSTRUCTION.get(style, f"Speak in a {style} style.")
    elif edit_type == "style_emotion":
        emotion = kwargs.get("style", "happy")
        instruction = EMOTION_INSTRUCTION.get(emotion, f"Speak with {emotion}.")
    elif edit_type == "audio_effect_speed":
        rate = float(kwargs.get("speed_rate", 1.0))
        instruction = f"Speak at speed rate {rate}." # 简化描述

    # 2. 构造符合 Zero-Shot 格式的 Prompt
    # 格式通常为: 指令(可选) + 参考文本 + <|endofprompt|> + 目标文本
    full_prompt_text = f"You are a helpful assistant. {instruction}".strip()
    full_prompt_text += f"<|endofprompt|>{prompt_text}"

    # 3. 处理停顿/语气词映射
    final_target_text = map_paralinguistic_tokens(target_text)

    payload = {
        "tts_text": final_target_text,
        "prompt_text": full_prompt_text,
        "stream": False,
        "tts_model_name": "default"
    }

    return payload, prompt_wav

# ---------------------------------------------------------------------------
# Worker Function for Joblib
# ---------------------------------------------------------------------------

def process_single_task(data: dict, host_url: str, save_dir: str, sample_rate: int):
    utt_id = data["id"]
    save_audio_path = os.path.join(save_dir, f"{utt_id}.wav")

    payload, prompt_wav = map_jsonl_to_zero_shot_payload(data)

    if not os.path.exists(prompt_wav):
        return utt_id, False, save_audio_path, f"Prompt audio not found: {prompt_wav}"

    try:
        with open(prompt_wav, "rb") as f:
            files = {"prompt_wav": f}
            # 统一调用 inference_zero_shot
            response = requests.post(
                f"{host_url.rstrip('/')}/inference_zero_shot", 
                files=files, 
                data=payload, 
                timeout=60
            )

        if response.status_code != 200:
            return utt_id, False, save_audio_path, f"API Error {response.status_code}"

        audio_np = np.frombuffer(response.content, dtype=np.int16)
        if audio_np.size == 0:
            return utt_id, False, save_audio_path, "Empty audio returned"
            
        sf.write(save_audio_path, audio_np, samplerate=sample_rate, subtype="PCM_16")
        return utt_id, True, save_audio_path, "Success"

    except Exception as exc:
        return utt_id, False, save_audio_path, str(exc)

# ---------------------------------------------------------------------------
# CLI & Main Loop
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Concurrent CosyVoice3 client using Joblib")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="CosyVoice API host URL")
    parser.add_argument("--jsonl-files", type=str, nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=128, help="Number of concurrent requests")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate (CV3 defaults to 22050)")
    return parser.parse_args()


def process_jsonl():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Info] Target Host: {args.host}")
    print(f"[Info] Concurrency: {args.num_workers} workers")

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"[Warning] File not found: {jsonl_file}")
            continue

        file_stem = Path(jsonl_file).stem
        save_dir = os.path.join(args.output_dir, file_stem)
        scp_path = os.path.join(args.output_dir, f"{file_stem}.scp")
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Loading {jsonl_file} ---")
        
        # 1. Load all tasks into memory
        tasks = []
        with open(jsonl_file, "r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    tasks.append(data)
                except json.JSONDecodeError:
                    print(f"[Error] JSONDecodeError at line {line_idx}. Skipped.")

        # 2. Execute concurrently using Joblib with TQDM progress bar
        # return_as="generator" allows tqdm to update as soon as a worker finishes
        parallel_executor = Parallel(n_jobs=args.num_workers, return_as="generator", prefer="threads")

        # 3. Write results to SCP file
        success_count = 0
        with open(scp_path, "w", encoding="utf-8") as fscp:
            pbar = tqdm(total=len(tasks), desc=f"Processing {file_stem}", unit="task")
            for res in parallel_executor(
                delayed(process_single_task)(task, args.host, save_dir, args.sample_rate)
                for task in tasks
            ):
                utt_id, success, audio_path, msg = res
                pbar.update(1)
                if success:
                    fscp.write(f"{utt_id}\t{audio_path}\n")
                    success_count += 1
                    pbar.set_postfix_str(f"Success: {success_count}/{pbar.n}")
                else:
                    print(f"[Failed] {utt_id}: {msg}")

        print(f"--- Finished {file_stem} ({success_count}/{len(tasks)} succeeded) ---")
        print(f"Wavs saved at → {save_dir}")
        print(f"SCP saved at  → {scp_path}\n")


if __name__ == "__main__":
    process_jsonl()

# CUDA_VISIBLE_DEVICES=0 python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512 --data_type bfloat16 --load_trt true --port 8000
