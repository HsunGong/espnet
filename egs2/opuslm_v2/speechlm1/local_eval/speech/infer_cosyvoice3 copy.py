"""
Batch inference script for CosyVoice3 (with vLLM backend hardcoded).
"""
import os

# # <- force cuda126 toolkit
# os.environ["CUDA_HOME"] = "/mnt/home/xungong-andr-1766e0/tools/cuda126"
# os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ["PATH"]
# os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CosyVoice3 supported paralinguistic tokens & Mappings
# ---------------------------------------------------------------------------
CV3_SUPPORTED_TOKENS = {
    "[breath]", "[noise]", "[laughter]", "[cough]", "[clucking]",
    "[accent]", "[quick_breath]", "[hissing]", "[sigh]",
    "[vocalized-noise]", "[lipsmack]", "[mn]",
    "<laughter>", "</laughter>",
}

# eval -> cv3
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
        if dst in CV3_SUPPORTED_TOKENS:
            text = text.replace(src, dst)
    return text

# ---------------------------------------------------------------------------
# Core Inference Dispatcher
# ---------------------------------------------------------------------------

def synthesize_audio(model, data: dict):
    """
    根据任务类型 (edit_type)，分发到 CosyVoice3 不同的 inference 接口。
    """
    edit_type   = data.get("edit_type", "")
    kwargs      = data.get("edit_kwargs", {})
    target_text = data.get("target_text", data.get("text", ""))
    prompt_text = data.get("text", "")
    prompt_wav  = data.get("audio_path", "")

    if not target_text.strip():
        return None

    # 1. 零样本声音克隆 (文本增/删/改/替换) -> inference_zero_shot
    if edit_type in {"transcription_ins", "transcription_del", "transcription_sub", "transcription_replace_sentence"}:
        prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
        generator = model.inference_zero_shot(
            tts_text=target_text, 
            prompt_text=prompt, 
            prompt_speech_16k=prompt_wav, 
            stream=False
        )

    # 2. 跨语种/副语言插入 -> inference_cross_lingual
    elif edit_type == "transcription_add_paralinguistic":
        target_text = map_paralinguistic_tokens(target_text)
        prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"

        generator = model.inference_zero_shot(
            tts_text=target_text, 
            prompt_text=prompt, 
            prompt_speech_16k=prompt_wav, 
            stream=False
        )

    # 3. 指令控制 (风格/情感/语速/去混响) -> inference_instruct2
    elif edit_type in {"style_whisper", "style_style", "style_emotion", "audio_effect_speed", "audio_effect_dereverb"}:
        instruction = ""
        
        if edit_type in {"style_whisper", "style_style"}:
            style = kwargs.get("style", "whisper")
            instruction = STYLE_INSTRUCTION.get(style, f"Speak in a {style} style.")
        
        elif edit_type == "style_emotion":
            emotion = kwargs.get("style", "happy")
            instruction = EMOTION_INSTRUCTION.get(emotion, f"Speak with {emotion}.")
            
        elif edit_type == "audio_effect_speed":
            rate = float(kwargs.get("speed_rate", 1.0))
            if rate >= 1.5:   instruction = "Speak as fast as possible."
            elif rate > 1.0:  instruction = "Speak faster than normal."
            elif rate <= 0.6: instruction = "Speak as slowly as possible."
            elif rate < 1.0:  instruction = "Speak slower than normal."
            else:             instruction = "Speak at a normal pace."
            
        elif edit_type == "audio_effect_dereverb":
            instruction = "Speak in a clean, noise-free and reverb-free environment."

        instruct_text = f"You are a helpful assistant. {instruction}<|endofprompt|>"
        generator = model.inference_instruct2(
            tts_text=target_text, 
            instruct_text=instruct_text, 
            prompt_speech_16k=prompt_wav, 
            stream=False
        )

    else:
        # 未知任务类型
        return None

    # 从 generator 提取并拼接所有的 chunk 音频
    chunks = [chunk["tts_speech"] for chunk in generator]
    if not chunks:
        raise RuntimeError("Model returned no audio chunks.")
    
    return torch.cat(chunks, dim=-1)

# ---------------------------------------------------------------------------
# CLI & Main Loop
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 batch inference (vLLM enforced)")
    parser.add_argument("--model-dir", type=str, default="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--jsonl-files", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cosyvoice-dir", type=str, default='./CosyVoice', help="Path to CosyVoice root dir")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Python Path for CosyVoice
    if args.cosyvoice_dir:
        sys.path.insert(0, args.cosyvoice_dir)
        sys.path.insert(0, os.path.join(args.cosyvoice_dir, "third_party", "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel

    # 强制启用 vLLM 和 TRT 以保证性能
    print("[Info] Loading CosyVoice3 with vLLM backend...")
    model = AutoModel(model_dir=args.model_dir, load_vllm=True, load_trt=True, fp16=False)

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"[Warning] File not found: {jsonl_file}")
            continue

        file_stem = Path(jsonl_file).stem
        save_dir = os.path.join(args.output_dir, file_stem)
        scp_path = os.path.join(args.output_dir, f"{file_stem}.scp")
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--- Processing {jsonl_file} ---")

        with open(jsonl_file, "r", encoding="utf-8") as fin, \
             open(scp_path, "w", encoding="utf-8") as fscp:
            
            lines = [line for line in fin if line.strip()]
            for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Error] JSONDecodeError at line {line_idx}. Skipped.")
                    continue

                utt_id = data.get("id", f"utt_{line_idx}")
                prompt_wav = data.get("audio_path", "")

                if not os.path.exists(prompt_wav):
                    print(f"[Warning] Audio not found: {prompt_wav} (utt={utt_id}). Skipped.")
                    continue

                save_audio_path = os.path.abspath(os.path.join(save_dir, f"{utt_id}.wav"))

                try:
                    # 核心调用
                    audio_tensor = synthesize_audio(model, data)
                    
                    if audio_tensor is None:
                        print(f"[{utt_id}] Skipped unsupported edit_type: '{data.get('edit_type')}'.")
                        continue

                    torchaudio.save(save_audio_path, audio_tensor.cpu(), model.sample_rate)
                    fscp.write(f"{utt_id}\t{save_audio_path}\n")
                except Exception as exc:
                    print(f"\n[Error] Failed to process {utt_id}: {exc}")

        print(f"--- Finished {file_stem}. Wavs → {save_dir} | SCP → {scp_path} ---")

if __name__ == "__main__":
    main()
