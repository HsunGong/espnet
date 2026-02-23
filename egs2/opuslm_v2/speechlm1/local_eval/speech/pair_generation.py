#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
from matplotlib.pyplot import step
import soundfile as sf
import librosa
import numpy as np
import scipy.signal
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import re
import random
try:
    from .rir import apply_rir
except ImportError:
    from rir import apply_rir

random.seed(42)

# Suppress librosa warnings
warnings.filterwarnings("ignore")

SAFE_OP = set(["REVERB", "VOLUME"])

MAIN_SINGLE_OPS = ["VOLUME", "SPEED", "PITCH", "REVERB"]
MOD_SINGLE_OPS = ["VOLUME", "SPEED", "PITCH", "REVERB", "REPEAT"]
MIX_OPS = ["REMIX", "LEFT", "RIGHT"]


def random_effect_for_single(op_name: str, *, main_dur: float = 0.0, mod_dur: float = 0.0):
    op = op_name.upper()

    if op == "VOLUME":
        sign = random.choice([-1, 1])
        gain = random.randint(1, 10) * sign
        return f"{gain:+d}dB"

    if op == "SPEED":
        return f"{random.choice([0.5, 1.5, 2.0]):.1f}x"

    if op == "PITCH":
        sign = random.choice([-1, 1])
        semitones = random.randint(1, 5) * sign
        return f"{semitones:+d}"

    if op == "REVERB":
        return random.choice(["small", "medium", "large"])

    if op == "REPEAT":
        max_repeat_by_len = 4
        if main_dur > 0 and mod_dur > 0:
            max_repeat_by_len = int(max(1, min(4, main_dur // mod_dur)))

        if max_repeat_by_len < 2:
            return 2
        return random.randint(2, max_repeat_by_len)

    return None


def random_mix_effect(op_name: str, main_dur: float, mod_dur: float):
    op = op_name.upper()
    if op == "REMIX":
        return None

    max_overlap = int(max(0, min(10, main_dur, mod_dur)))
    if op == "LEFT":
        return random.randint(0, max_overlap) if max_overlap > 0 else 0
    if op == "RIGHT":
        if max_overlap <= 0:
            return 0
        return random.randint(1, max_overlap)
    return None


def generate_random_ops(main_dur: float, mod_dur: float):
    main_op_name = random.choice(MAIN_SINGLE_OPS)
    mod_op_pool = list(MOD_SINGLE_OPS)
    if main_dur > 0 and mod_dur > 0 and (main_dur // mod_dur) < 2 and "REPEAT" in mod_op_pool:
        mod_op_pool.remove("REPEAT")
    mod_op_name = random.choice(mod_op_pool)
    mix_op_name = random.choice(MIX_OPS)

    main_op = {
        "scope": "main",
        "operation": main_op_name,
        "effect": random_effect_for_single(main_op_name, main_dur=main_dur, mod_dur=mod_dur),
    }
    mod_op = {
        "scope": "modifier",
        "operation": mod_op_name,
        "effect": random_effect_for_single(mod_op_name, main_dur=main_dur, mod_dur=mod_dur),
    }
    mix_op = {
        "scope": "mix",
        "operation": mix_op_name,
        "effect": random_mix_effect(mix_op_name, main_dur=main_dur, mod_dur=mod_dur),
    }
    return main_op, mod_op, mix_op

def get_caption(item):
    # User requested: "audio_caption (not qwen_caption)"
    return item.get("caption") or item.get("audio_caption") or item.get("qwen_caption") or ""

def safe_load(path, sr=None):
    if not path or not os.path.exists(path):
        return None, None
    try:
        y, s = librosa.load(path, sr=sr)
        return y, s
    except Exception as e:
        return None, None

def save_audio(y, sr, path):
    assert y is not None, f"can not save None at {path}"
    sf.write(path, y, sr)

def apply_reverb(y, sr, size_str):
    if y is None: return None, size_str
    # Map descriptive strings to room sizes (meters)
    actual_size = "medium"
    if "small" in size_str: 
        actual_size = "small"
    elif "medium" in size_str: 
        actual_size = "medium"
    elif "large" in size_str: 
        actual_size = "large"
    
    y_rev = apply_rir(y, sr, actual_size)
    
    # Match original length for simulation consistency
    if y_rev.ndim == 1:
        res = y_rev[:len(y)]
    else:
        res = y_rev[:, :y.shape[1]]
    return res, actual_size

def parse_gain(effect_str):
    # e.g. "+3dB", "-10dB"
    match = re.search(r'([+-]?\d+)', effect_str)
    if match:
        db = float(match.group(1))
    else:
        db = 0.0
    
    if abs(db) < 1.0:
        db = 5.0 if random.random() > 0.5 else -5.0
    return db

def parse_speed(effect_str):
    # e.g. "1.5x", "0.5x"
    speed = 1.0
    match = re.search(r'([\d\.]+)', effect_str)
    if match:
        speed = float(match.group(1))
    if "slow" in effect_str: speed = 0.8
    if "fast" in effect_str: speed = 1.2
    
    if speed <= 0: speed = 1.0
    if speed == 1.0: # random re-sample
        speed = 1.5 if random.random() > 0.5 else 0.75
    return speed

def parse_pitch(effect_str):
    # e.g. "+2 semitones"
    match = re.search(r'([+-]?\d+)', effect_str)
    pitch = 0
    if match:
        pitch = int(match.group(1))
    if "low" in effect_str: pitch = -2
    if "high" in effect_str: pitch = 2
    if abs(pitch) < 2:
        pitch = int(random.random() * 10 - 5) # Random between -5 and +5 semitones
    if pitch == 0: pitch = 3
    return pitch

def parse_repeat(effect_str):
    match = re.search(r'(\d+)', effect_str)
    count = 0
    if match:
        count = int(match.group(1))
    
    if count <= 1:
        count = 2 if random.random() > 0.5 else 3
    return count

def apply_low_quality(y, sr):
    if y is None: return None, None

    seed = random.random()

    # 1. Downsample to 8k and back to simulate bandwidth loss
    y_low = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_degraded = librosa.resample(y_low, orig_sr=8000, target_sr=sr)
    
    # Ensure length match
    if len(y_degraded) > len(y): y_degraded = y_degraded[:len(y)]
    elif len(y_degraded) < len(y): y_degraded = np.pad(y_degraded, (0, len(y) - len(y_degraded)))

    # 2. Add subtle white noise
    noise = np.random.randn(len(y_degraded))
    y_degraded = y_degraded + 0.005 * noise
    
    # 3. Quantize to simulate bit depth reduction (e.g., 8-bit)
    y_degraded = np.round(y_degraded * 127) / 127.0
    
    return y_degraded, "low"

def apply_volume(y, sr, delta):
    import pyloudnorm as pyln
    if y.ndim == 1:
        y = y[np.newaxis, :]
    
    # 5db -> 10db as maxmisze -> scale by 1/4
    delta = 10 * max(delta / 5, 1) ** (1 / 4)

    duration = y.shape[1] / sr

    # measure the loudness first
    meter = pyln.Meter(
        sr, block_size=min(0.4, duration - 1e-10)
    )  # create BS.1770 meter
    loudness = meter.integrated_loudness(y.T)

    # loudness normalize audio to target LUFS. We will ignore the warnings related to
    # clipping the audio.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loudness_normalized_audio = pyln.normalize.loudness(y.T, loudness, loudness + delta)

    return loudness_normalized_audio.T.squeeze()

def apply_op(y, sr, op_name, effect_val):
    if y is None: return None, effect_val
    op = op_name.upper()
    eff = str(effect_val).lower() if effect_val else ""

    if "REVERB" in op:
        y, effect_val = apply_reverb(y, sr, eff)
        if y is None:
            return None, effect_val

    if "VOLUME" in op:
        gain = parse_gain(eff)
        y = apply_volume(y, sr, gain)
        effect_val = f"{gain:+.1f}dB"
        if y is None:
            return None, effect_val
    
    if "SPEED" in op:
        if y is None:
            return None, effect_val
        rate = parse_speed(eff)
        y = librosa.effects.time_stretch(y, rate=rate)
        effect_val = f"{rate:.2f}x"

    if "PITCH" in op:
        if y is None:
            return None, effect_val
        steps = parse_pitch(eff) # n_steps = 12 ~ 1 octave range from -5~+5 semitones
        if steps != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        effect_val = f"{steps:+}"

    if "REPEAT" in op:
        if y is None:
            return None, effect_val
        repeat = parse_repeat(str(effect_val))
        y = np.tile(y, repeat)
        effect_val = repeat

    return formulate_audio(y), effect_val

def mix_tracks(base, mod, op_name, effect_val, sr: int = 16000):
    """
    Handle Two-Audio Ops: ADD, REMOVE, SHIFT, SEPARATE
    Returns: The RESULT of the operation and the updated effect value.
    """
    if base is None: return None, effect_val
    if mod is None: return base, effect_val

    op = op_name.upper()
    merged = None
    actual_eff = effect_val

    factor = len(base) / max(len(mod), 1)
    if factor > 10:
        repeat = random.randint(2, 5)
        mod = np.tile(mod, repeat)
    
    eps = 1e-10
    base_rms = np.sqrt(np.mean(base**2) + eps)
    mod_rms  = np.sqrt(np.mean(mod**2) + eps)

    snr_db = 20 * np.log10(base_rms / mod_rms)

    min_snr_db = 5.0

    if snr_db < min_snr_db:
        # target_snr_db = random.uniform(6.0, 14.0)  # 让分布更自然
        # need_gain_db = snr_db - target_snr_db   # 负数：需要衰减
        # scale = 10 ** (need_gain_db / 20.0)     # < 1

        # jitter = random.uniform(0.9, 1.1)
        # mod *= scale * jitter
        mod *= 0.8

    if "ADD" in op or "REMIX" in op or "MID INSERT" in op:
        # Align lengths for max-len
        if len(base) > len(mod):
            merged = np.copy(base)
            random_start = random.randint(0, len(base) - len(mod))
            merged[random_start:random_start+len(mod)] += mod
        else:
            random_start = 0
            merged = np.copy(base)
            merged += mod[:len(base)]

        actual_eff = round(random_start / 16000, 2)

    try: overlap_s = float(effect_val)
    except: overlap_s = random.random() * 4
    overlap_s = min(abs(overlap_s), len(base) / sr, len(mod) / sr)

    if "RIGHT" == op or "POST OVERLAP" == op:
        tot_len = len(base) + len(mod) - int(overlap_s * sr)
        merged = np.zeros(tot_len)
        merged[:len(base)] += base
        merged[-len(mod):] += mod
        actual_eff = overlap_s
    
    if "LEFT" == op or "PRE OVERLAP" == op:
        tot_len = len(base) + len(mod) - int(overlap_s * sr)
        merged = np.zeros(tot_len)
        merged[-len(base):] += base
        merged[:len(mod)] += mod
        actual_eff = overlap_s

    assert merged is not None, f"Unsupported mix operation: {op_name}"
    
    return formulate_audio(merged), actual_eff

def formulate_audio(y):
    # if y has nan -> set to zero
    y = np.nan_to_num(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y

def process_session(line: str, output_dir, idx) -> dict | None:
    if isinstance(line, str):
        session = json.loads(line)
    else:
        session = line

    # --- NEW MODE (Raw Step 1 Item) ---
    # Process once, generate 4 variations
    try:
        # 0. Initial validation
        audios = session.get("audios")
        if isinstance(audios, dict):
            main_info = audios.get("main", {})
            mod_info = audios.get("modifier", {})
        else:
            # New Step1 format from pair_selection_rule.py
            main_info = session.get("main", {})
            mod_info = session.get("modifier", {})

        main_cap = get_caption(main_info)
        mod_cap = get_caption(mod_info)

        main_dur = main_info.get("duration", 0)
        mod_dur = mod_info.get("duration", 0)    
        if main_dur < 5.0 and random.random() > 0.1:
            #  f"Main audio too short: {main_dur}s in session {idx}"
            return None

        main_path = main_info.get("audio_path")
        mod_path = mod_info.get("audio_path")

        # 1. Setup Base UID/Folder
        uid = f"{idx:05d}_{Path(main_path).stem[:10]}"
        base_path = os.path.join(output_dir, uid) # Shared folder for all sub-tasks
        SR = 16000
        # 2. Load Sources
        y_main_raw, _ = safe_load(main_path, SR)
        y_mod_raw, _ = safe_load(mod_path, SR)

        # Random ops (aligned with step1 generation OP space)
        main_op, mod_op, mix_op = generate_random_ops(main_dur=main_dur, mod_dur=mod_dur)

        # Apply Single Op on main: A -> A'
        try:
            main_out, main_op["effect"] = apply_op(y_main_raw, SR, main_op["operation"], main_op["effect"])

            # argument with low-quality, REVERB is a low-quality effect
            if main_info.get("quality") == "high" and main_op["operation"] != "REVERB" and random.random() > 0.8:
                main_out, main_op["quality"] = apply_low_quality(main_out, SR)
        except Exception as e:
            logging.warning(f"Single-Op Main failed for session {idx}: {e}")
            main_out = main_op = None
        save_audio(main_out, SR, f"{base_path}-main_op.wav")

        session["main_edited"] = {
            "audio_path": os.path.abspath(f"{base_path}-main_op.wav"),
            "audio_caption": None,
            "operations": [main_op],
        }

        # Apply Single Op on modifier: B -> B'
        try:
            mod_out, mod_op["effect"] = apply_op(y_mod_raw, SR, mod_op["operation"], mod_op["effect"])
        except Exception as e:
            logging.warning(f"Single-Op Modifier failed for session {idx}: {e}")
            mod_out = y_mod_raw
        save_audio(mod_out, SR, f"{base_path}-modifier_op.wav")

        session["modifier_edited"] = {
            "audio_path": os.path.abspath(f"{base_path}-modifier_op.wav"),
            "audio_caption": None,
            "operations": [mod_op],
        }

        # p = f"{base_path}-step1_{scope}_{op.split()[0]}.wav"
        # A'+B' <- A', B'
        # Step2: Apply mixup after random single-ops
        try:
            mix_out, mix_op["effect"] = mix_tracks(main_out, mod_out, mix_op["operation"], mix_op["effect"], sr=SR)

        except Exception as e:
            logging.warning(f"Mix operation failed for session {idx}: {e}")
            mix_out = None
            mix_op = None
        save_audio(mix_out, SR, f"{base_path}-mix_op.wav")

        mix_ops = [main_op, mod_op, mix_op]

        session["mixup"] = {
            "audio_path": os.path.abspath(f"{base_path}-mix_op.wav"),
            "audio_caption": None,
            "augment": "both",
            "operations": mix_ops,
        }

        return session
    except Exception as e:
        logging.exception(e)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Step 1 output JSONL")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_split", action="store_true", help="Disable automatic splitting of Step 1 output")
    args = parser.parse_args()

    wav_dir = os.path.join(args.output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    # Read all lines
    print(f"Reading tasks from {args.input_file}...")
    lines = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    line_count = len(lines)
    print(f"Total input lines: {line_count}")

    with open(os.path.join(args.output_dir, "audio_operation.jsonl"), "w") as fout:
        for ret in tqdm(Parallel(n_jobs=args.num_workers, backend="loky", return_as="generator")(
            delayed(process_session)(line, wav_dir, i)
            for i, line in enumerate(lines)), total=line_count, desc="Processing"
        ):
            if ret:
                fout.write(json.dumps(ret, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
