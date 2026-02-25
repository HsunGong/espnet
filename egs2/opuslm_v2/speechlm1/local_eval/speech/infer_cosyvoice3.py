"""
Batch inference script for CosyVoice3 (with vLLM backend).

Input format mirrors infer_stepaudiox.py: one or more JSONL files where each line has:
  {
    "id":          <utterance-id>,
    "audio_path":  <path to prompt .wav>,
    "text":        <transcription of prompt audio>,
    "target_text": <text to synthesize>,        # or "text" field as fallback
    "edit_type":   <see SUPPORTED_EDIT_TYPES>,
    "edit_kwargs": { ... }                       # optional per-type params
  }

Supported edit_types
--------------------
  transcription_ins / transcription_del / transcription_sub /
  transcription_replace_sentence
      → zero-shot voice cloning (inference_zero_shot)

  transcription_add_paralinguistic
      → cross-lingual synthesis with paralinguistic tokens
        (maps unsupported tokens → closest CV3 equivalents)

  style_whisper / style_emotion / style_style
      → instruction-based synthesis (inference_instruct2)
        edit_kwargs["style"] carries the style/emotion label

  audio_effect_speed
      → speed control via inference_instruct2
        edit_kwargs["speed_rate"] (float, default 1.0)

  audio_effect_dereverb
      → denoise / dereverb via inference_instruct2

CosyVoice3 supported paralinguistic tokens
-------------------------------------------
  [breath] [noise] [laughter] <laughter></laughter> [cough] [clucking]
  [accent] [quick_breath] [hissing] [sigh] [vocalized-noise] [lipsmack] [mn]
"""

import argparse
import json
import os
import re
import sys
import torchaudio
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# CosyVoice3 supported paralinguistic tokens
# ---------------------------------------------------------------------------
CV3_SUPPORTED_TOKENS = {
    "[breath]", "[noise]", "[laughter]", "[cough]", "[clucking]",
    "[accent]", "[quick_breath]", "[hissing]", "[sigh]",
    "[vocalized-noise]", "[lipsmack]", "[mn]",
    "<laughter>", "</laughter>",
}

# Map input paralinguistic tags → closest CV3-supported token
PARA_TAG_MAP = {
    "[sigh]":                "[sigh]",           # direct
    "[laugh]":               "[laughter]",        # remap
    "[exhale]":              "[breath]",          # closest
    "[snort]":               "[hissing]",         # closest
    "[cough]":               "[cough]",           # direct
    "[uhm]":                 "[mn]",              # filler → mn
    "[Surprise-oh]":         "[vocalized-noise]",
    "[Surprise-wa]":         "[vocalized-noise]",
    "[Dissatisfaction-hnn]": "[mn]",
    "[Question-ah]":         "[vocalized-noise]",
    "[Question-yi]":         "[vocalized-noise]",
}

# ---------------------------------------------------------------------------
# Style → English instruction
# ---------------------------------------------------------------------------
STYLE_INSTRUCTION = {
    "serious":     "Speak in a serious and formal tone.",
    "arrogant":    "Speak in an arrogant and proud tone.",
    "child":       "Speak in a childlike, innocent voice.",
    "older":       "Speak in an elderly, calm voice.",
    "girl":        "Speak in a lively, young female voice.",
    "pure":        "Speak in a pure and clear voice.",
    "sister":      "Speak in a warm, elder-sister tone.",
    "sweet":       "Speak in a sweet and gentle voice.",
    "ethereal":    "Speak in an ethereal and dreamy voice.",
    "whisper":     "Speak in a soft whisper.",
    "gentle":      "Speak in a gentle and soft tone.",
    "recite":      "Speak in a recitation style.",
    "generous":    "Speak in a generous and magnanimous tone.",
    "act_coy":     "Speak in a coy and playful manner.",
    "warm":        "Speak in a warm and caring tone.",
    "shy":         "Speak in a shy and timid voice.",
    "comfort":     "Speak in a comforting and soothing tone.",
    "authority":   "Speak in an authoritative tone.",
    "chat":        "Speak in a casual, chatty style.",
    "radio":       "Speak in a radio broadcast style.",
    "soulful":     "Speak in a soulful, heartfelt voice.",
    "story":       "Speak in a storytelling style.",
    "vivid":       "Speak in a vivid and expressive style.",
    "program":     "Speak in a program-host style.",
    "news":        "Speak in a news anchor style.",
    "advertising": "Speak in an advertising and promotional style.",
    "roar":        "Speak in a loud, roaring voice.",
    "murmur":      "Speak in a low murmur.",
    "shout":       "Shout the sentence loudly.",
    "deeply":      "Speak in a deep, resonant voice.",
    "loudly":      "Speak loudly and clearly.",
    "remove":      "Speak in a plain, neutral voice.",
    "exaggerated": "Speak in an exaggerated, theatrical manner.",
}

# ---------------------------------------------------------------------------
# Emotion → English instruction
# ---------------------------------------------------------------------------
EMOTION_INSTRUCTION = {
    "happy":      "Speak with happiness and joy.",
    "angry":      "Speak with anger.",
    "sad":        "Speak with sadness.",
    "humour":     "Speak in a humorous and funny tone.",
    "confusion":  "Speak with confusion and uncertainty.",
    "disgusted":  "Speak with disgust.",
    "empathy":    "Speak with empathy and understanding.",
    "embarrass":  "Speak with embarrassment.",
    "fear":       "Speak with fear.",
    "surprised":  "Speak with surprise.",
    "excited":    "Speak with excitement.",
    "depressed":  "Speak with depression and low energy.",
    "coldness":   "Speak in a cold and distant tone.",
    "admiration": "Speak with admiration.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_paralinguistic_tokens(text: str) -> str:
    """
    Replace any unsupported paralinguistic tags in *text* with the nearest
    CV3-supported token.  Tags already supported are left unchanged.
    """
    for src, dst in PARA_TAG_MAP.items():
        if src not in CV3_SUPPORTED_TOKENS:
            text = text.replace(src, dst)
    return text


def build_cv3_prompt_text(raw_prompt_text: str) -> str:
    """
    Prepend the CosyVoice3 system-prompt header to the raw prompt transcription.
    """
    return f"You are a helpful assistant.<|endofprompt|>{raw_prompt_text}"


def build_cv3_instruct(instruction: str) -> str:
    """
    Wrap *instruction* with the CV3 instruct2 format.
    """
    return f"You are a helpful assistant. {instruction}<|endofprompt|>"


def map_jsonl_to_cv3_params(data: dict):
    """
    Map a JSONL record to CosyVoice3 inference parameters.

    Returns
    -------
    mode : str | None
        "zero_shot", "cross_lingual", "instruct2", or None (skip).
    cv3_params : dict
        Keyword arguments consumed by the corresponding CV3 method.
    """
    edit_type  = data.get("edit_type", "")
    kwargs     = data.get("edit_kwargs", {})
    target_text = data.get("target_text", data.get("text", ""))
    prompt_text = data.get("text", "")
    prompt_wav  = data.get("audio_path", "")

    mode      = None
    cv3_params = {}

    # ------------------------------------------------------------------
    # 1. Text insertion / deletion / substitution → zero-shot cloning
    # ------------------------------------------------------------------
    if edit_type in {
        "transcription_ins",
        "transcription_del",
        "transcription_sub",
        "transcription_replace_sentence",
    }:
        mode = "zero_shot"
        cv3_params = dict(
            tts_text=target_text,
            prompt_text=build_cv3_prompt_text(prompt_text),
            prompt_speech_16k=prompt_wav,
        )

    # ------------------------------------------------------------------
    # 2. Paralinguistic tag insertion → cross-lingual with token mapping
    # ------------------------------------------------------------------
    elif edit_type == "transcription_add_paralinguistic":
        converted = map_paralinguistic_tokens(target_text)
        mode = "cross_lingual"
        cv3_params = dict(
            tts_text=f"You are a helpful assistant.<|endofprompt|>{converted}",
            prompt_speech_16k=prompt_wav,
        )

    # ------------------------------------------------------------------
    # 3. Whisper / style → instruct2
    # ------------------------------------------------------------------
    elif edit_type in {"style_whisper", "style_style"}:
        style_label = kwargs.get("style", "whisper")
        instruction = STYLE_INSTRUCTION.get(
            style_label,
            f"Speak in a {style_label} style.",
        )
        mode = "instruct2"
        cv3_params = dict(
            tts_text=target_text,
            instruct_text=build_cv3_instruct(instruction),
            prompt_speech_16k=prompt_wav,
        )

    # ------------------------------------------------------------------
    # 4. Emotion → instruct2
    # ------------------------------------------------------------------
    elif edit_type == "style_emotion":
        emotion_label = kwargs.get("style", "happy")
        instruction = EMOTION_INSTRUCTION.get(
            emotion_label,
            f"Speak with {emotion_label}.",
        )
        mode = "instruct2"
        cv3_params = dict(
            tts_text=target_text,
            instruct_text=build_cv3_instruct(instruction),
            prompt_speech_16k=prompt_wav,
        )

    # ------------------------------------------------------------------
    # 5. Speed control → instruct2
    # ------------------------------------------------------------------
    elif edit_type == "audio_effect_speed":
        rate = float(kwargs.get("speed_rate", 1.0))
        if rate >= 1.5:
            instruction = "Speak as fast as possible."
        elif rate > 1.0:
            instruction = "Speak faster than normal."
        elif rate <= 0.6:
            instruction = "Speak as slowly as possible."
        elif rate < 1.0:
            instruction = "Speak slower than normal."
        else:
            instruction = "Speak at a normal pace."
        mode = "instruct2"
        cv3_params = dict(
            tts_text=target_text,
            instruct_text=build_cv3_instruct(instruction),
            prompt_speech_16k=prompt_wav,
        )

    # ------------------------------------------------------------------
    # 6. Dereverb / denoise → instruct2
    # ------------------------------------------------------------------
    elif edit_type == "audio_effect_dereverb":
        mode = "instruct2"
        cv3_params = dict(
            tts_text=target_text,
            instruct_text=build_cv3_instruct(
                "Speak in a clean, noise-free and reverb-free environment."
            ),
            prompt_speech_16k=prompt_wav,
        )

    print(f"  [map] edit_type={edit_type!r}  mode={mode}")
    return mode, cv3_params


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="CosyVoice3 batch inference from JSONL files (vLLM backend)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to the CosyVoice3 model directory.",
    )
    parser.add_argument(
        "--jsonl-files",
        type=str,
        nargs="+",
        required=True,
        help="One or more input JSONL files (e.g. metadata.jsonl transcription_del.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory to save wav files and scp files.",
    )
    # vLLM / model knobs
    parser.add_argument(
        "--load-vllm",
        action="store_true",
        default=True,
        help="Use vLLM backend for the LLM part (default: True).",
    )
    parser.add_argument(
        "--no-vllm",
        dest="load_vllm",
        action="store_false",
        help="Disable vLLM backend.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 precision (CV3 default is fp32/bf16; fp16=False recommended).",
    )
    parser.add_argument(
        "--load-trt",
        action="store_true",
        default=True,
        help="Use TensorRT for the flow-matching decoder (default: True).",
    )
    parser.add_argument(
        "--no-trt",
        dest="load_trt",
        action="store_false",
        help="Disable TensorRT.",
    )
    parser.add_argument(
        "--cosyvoice-dir",
        type=str,
        default='./CosyVoice',
        help="If CosyVoice is not on PYTHONPATH, provide its root dir here.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    if args.cosyvoice_dir:
        sys.path.insert(0, args.cosyvoice_dir)
        sys.path.insert(0, os.path.join(args.cosyvoice_dir, "third_party", "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

    print("LOAD trt", args.load_trt, "vllm", args.load_vllm, "fp16", args.fp16)
    model = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=False,
    )
    return model


# ---------------------------------------------------------------------------
# Inference dispatch
# ---------------------------------------------------------------------------

def run_inference(model, mode: str, cv3_params: dict):
    """
    Dispatch to the appropriate CosyVoice3 method and collect all chunks.
    Returns (waveform_tensor, sample_rate).
    """
    import torch  # noqa: PLC0415

    chunks = []
    sr = model.sample_rate

    if mode == "zero_shot":
        gen = model.inference_zero_shot(
            cv3_params["tts_text"],
            cv3_params["prompt_text"],
            cv3_params["prompt_speech_16k"],
            stream=False,
        )
    elif mode == "cross_lingual":
        gen = model.inference_cross_lingual(
            cv3_params["tts_text"],
            cv3_params["prompt_speech_16k"],
            stream=False,
        )
    elif mode == "instruct2":
        gen = model.inference_instruct2(
            cv3_params["tts_text"],
            cv3_params["instruct_text"],
            cv3_params["prompt_speech_16k"],
            stream=False,
        )
    else:
        raise ValueError(f"Unknown inference mode: {mode!r}")

    for _, chunk in enumerate(gen):
        chunks.append(chunk["tts_speech"])

    if not chunks:
        raise RuntimeError("Model returned no audio chunks.")

    audio = torch.cat(chunks, dim=-1)
    return audio, sr


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_jsonl():
    args = get_args()
    model = load_model(args)
    os.makedirs(args.output_dir, exist_ok=True)

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"[Warning] File not found: {jsonl_file}")
            continue

        file_stem = Path(jsonl_file).stem

        # Output sub-directory and scp file
        save_dir = os.path.join(args.output_dir, file_stem)
        os.makedirs(save_dir, exist_ok=True)
        scp_path = os.path.join(args.output_dir, f"{file_stem}.scp")

        print(f"\n--- Processing {jsonl_file} ---")

        with (
            open(jsonl_file, "r", encoding="utf-8") as fin,
            open(scp_path, "w", encoding="utf-8") as fscp,
        ):
            lines = fin.readlines()
            for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Error] JSONDecodeError at line {line_idx}. Skipped.")
                    continue

                utt_id      = data["id"]
                prompt_wav  = data.get("audio_path", "")
                prompt_text = data.get("text", "")

                if not os.path.exists(prompt_wav):
                    print(f"[Warning] Audio not found: {prompt_wav} (utt={utt_id}). Skipped.")
                    continue

                mode, cv3_params = map_jsonl_to_cv3_params(data)

                if mode is None:
                    print(
                        f"[{utt_id}] Skipped unsupported edit_type "
                        f"'{data.get('edit_type')}'."
                    )
                    continue

                save_audio_path = os.path.abspath(
                    os.path.join(save_dir, f"{utt_id}.wav")
                )

                try:
                    audio, sr = run_inference(model, mode, cv3_params)
                    torchaudio.save(save_audio_path, audio.cpu(), sr)
                    print(
                        f"[{utt_id}] mode={mode} | target={data['target_text'][:60]!r}"
                    )
                    fscp.write(f"{utt_id}\t{save_audio_path}\n")
                    fscp.flush()
                except Exception as exc:
                    print(f"[Error] Failed to process {utt_id}: {exc}")

        print(
            f"--- Finished {file_stem}. "
            f"Wavs → {save_dir}  |  SCP → {scp_path} ---"
        )


if __name__ == "__main__":
    process_jsonl()
