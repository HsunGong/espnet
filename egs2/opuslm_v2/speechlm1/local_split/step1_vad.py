#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import librosa
import soundfile as sf
import warnings

from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

# Suppress librosa warnings if any
warnings.filterwarnings("ignore")
# os.environ["NUMBA_DISABLE_JIT"] = "1"  # Disable numba JIT to avoid potential issues in some environments

def get_audio_type(path: str) -> str:
    """Infer audio type from path keywords (speech / music / sound)."""
    p = path.lower()
    if any(k in p for k in ["speech", "owsm", "emilia", "commonvoice", "voxforge"]):
        return "speech"
    elif any(k in p for k in ["music", "fma", "jamendo", "disco"]):
        return "music"
    else:
        return "sound"


def _evaluate_and_split(y, sr, intervals, min_ratio, max_ratio):
    """
    评估器：按顺序拼接 segments，验证是否能落在 [min_ratio, max_ratio] 区间内。
    如果切分块太大导致“一步跨过界”，则返回 False。
    """
    if len(intervals) <= 1:
        return False, None, None

    min_target = int(len(y) * min_ratio)
    max_target = int(len(y) * max_ratio)
    
    split1_end = 0

    for i, seg in enumerate(intervals):
        split1_end = seg[1]
        if split1_end >= min_target and split1_end <= max_target:
            y1 = y[:split1_end]
            y2 = y[split1_end:]
            if len(y2) < sr * 1.0:
                # concat with another 1.0 seconds silence
                y2 = np.concatenate((y2, np.zeros(sr)))  # 1 second of silence
            return True, y1, y2
        elif split1_end > max_target:
            # 已经超过 max_target 了，说明这个切分块太大了，无法命中区间
            return False, None, None

    return False, None, None

# ================= 切分方法库 =================

def _energy_split(y, sr, vad_top_db = 30, **kwargs):
    """1. 全局能量 (适合干净的语音)"""
    return librosa.effects.split(y, top_db=vad_top_db)

def _rms_split(y, sr, rms_hop=10, rms_db = -25, **kwargs):
    """2. 局部相对能量 (适合带背景音的语音或部分音乐)"""
    hop = int(sr * rms_hop / 1000)  # 转换为样本数
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    mask = librosa.amplitude_to_db(rms, ref=np.max) > rms_db
    
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    if len(starts) == 0:
        return np.array([[0, len(y)]])
    return librosa.frames_to_samples(np.column_stack((starts, ends)), hop_length=hop)

def _beat_split(y, sr, n_beat: int = 4, **kwargs):
    """3. 节拍切分 (适合音乐)"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(beats) < 4:
        return np.array([[0, len(y)]])
    
    beat_samples = librosa.frames_to_samples(beats)
    # 每 4 个节拍切一刀，粒度较细，更容易命中 ratio 区间
    split_idx = list(range(0, len(beat_samples), n_beat))
    
    intervals = []
    current_start = 0 
    for i in range(1, len(split_idx)):
        end = beat_samples[split_idx[i]]
        intervals.append([current_start, end])
        current_start = end
        
    if current_start < len(y):
        intervals.append([current_start, len(y)])
    return np.array(intervals)

def _fallback_split(y, sr, min_ratio, max_ratio, fallback_ratio: float | None = None, **kwargs):
    fallback_ratio = (fallback_ratio or max_ratio)
    max_duration_len = int(len(y) * fallback_ratio)
    return np.array([[0, max_duration_len], [max_duration_len, len(y)]])

# ================= 主流程 =================

def process_one(line, output_dir, min_ratio=0.4, max_ratio=0.6, **kwargs):
    data = json.loads(line)
    audio_path = data.get("audio_path") or data.get("wav_path") or data.get("file_path")
    if not audio_path or not os.path.exists(audio_path):
        return None

    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0: 
        return None
    
    # 1. 定义流水线顺序
    pipeline = [
        ("Energy", _energy_split),
        # ("RMS", _rms_split), # it is simlar to Energy
        ("Beat", _beat_split),
        # ("Force", _force_split)
        # ("Fallback", _fallback_split)
    ]
    
    valid = False
    y1, y2 = None, None
    used_method = None
    
    # 2. 执行流水线
    for method_name, method_func in pipeline:
        intervals = method_func(y, sr, min_ratio=min_ratio, max_ratio=max_ratio, **kwargs)
        # print(intervals, len(y))
        valid, y1, y2 = _evaluate_and_split(y, sr, intervals, min_ratio, max_ratio)
        
        if valid:
            used_method = method_name
            break
    else:
        # 如果所有方法都失败了，使用 fallback 方法
        used_method = "Fallback"
        intervals = _fallback_split(y, sr, min_ratio=min_ratio, max_ratio=max_ratio, **kwargs)
        valid, y1, y2 = _evaluate_and_split(y, sr, intervals, min_ratio, max_ratio=kwargs.get("fallback_ratio", max_ratio))

    assert y1 is not None and y2 is not None, f"All methods failed for {audio_path} with {pipeline}"
    # print(f"{audio_path}: Method: {method_name} Split1 Dur: {len(y1) / sr:.2f}s  Split2 Dur: {len(y2)/sr:.2f}s, {min_ratio=}, {max_ratio=}, {intervals=}")

    # 4. 写入文件
    name, ext = os.path.splitext(os.path.basename(audio_path))

    path1 = output_dir / "split1" / f"{name}.flac"
    sf.write(path1, y1, sr)

    path2 = output_dir / "split2" / f"{name}.flac"
    sf.write(path2, y2, sr)

    # 5. 返回数据
    data["audio_caption"] = data.pop("qwen_caption", "")
    return {
        "main": data,
        "split1": {"audio_path": path1.absolute().as_posix(), "duration": len(y1)/sr},
        "split2": {"audio_path": path2.absolute().as_posix(), "duration": len(y2)/sr},
        "meta": {"split_method": used_method} # 记录用了什么方法，方便排查
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--nj", type=int, default=32)
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default="loky",
        choices=["threading", "loky"],
        help="joblib backend",
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already written in output_jsonl using main.audio_path as key",
    )
    parser.add_argument(
        "--max_ratio", type=float, default=1.0,
        help="Maximum fraction of total active audio that split1 may contain. "
             "1.0 = no cap (all active audio can go to split1, fallback halves it); "
             "0.5 = split1 ≤ 50%% of active audio. Default: 1.0."
    )
    parser.add_argument(
        "--min_ratio", type=float, default=0.5,
        help="Minimum fraction of total active audio that split1 must contain. "
             "0.0 = no floor; 0.2 = split1 ≥ 20%% of active audio. Default: 0.0."
    )
    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step1_vad")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir / "split1", exist_ok=True)
    os.makedirs(args.output_dir / "split2", exist_ok=True)

    # Capture args into closure so process_fn matches (idx, line) signature.
    _output_dir = args.output_dir
    _kwargs = {k: v for k, v in vars(args).items()
               if k not in ("input_jsonl", "output_jsonl", "output_dir",
                            "nj", "parallel_backend", "resume", "config_path")}

    def _process(idx: int, line: str) -> dict | None:
        return process_one(line, _output_dir, **_kwargs)

    runner = JsonlParallelRunner(
        input_jsonl=str(args.input_jsonl),
        output_jsonl=str(args.output_jsonl),
        process_fn=_process,
        n_jobs=args.nj,
        backend=args.parallel_backend,
        desc="Processing VAD Split",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["main"]["audio_path"],
    )
    runner.run()

    # ---------------------------------------------------------------- stats
    from collections import defaultdict
    import numpy as _np

    type_ratios: dict = defaultdict(list)
    method_ratios: dict = defaultdict(list)
    total_written = 0

    try:
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except Exception:
                    continue
                total_written += 1
                split1_dur = res.get("split1", {}).get("duration")
                main_dur = res.get("main", {}).get("duration")
                audio_path = res.get("main", {}).get("audio_path", "")
                split_method = res.get("meta", {}).get("split_method", "Unknown")
                if split1_dur and main_dur:
                    ratio = split1_dur / main_dur
                    type_ratios[get_audio_type(audio_path)].append(ratio)
                    method_ratios[split_method].append(ratio)
    except FileNotFoundError:
        pass

    vad_top_db = getattr(args, "vad_top_db", 30)
    print(f"\n=== Step1 VAD Split Statistics ===")
    print(f"  vad_top_db={vad_top_db}  min_ratio={args.min_ratio}  max_ratio={args.max_ratio}")
    print(f"  Valid records written: {total_written}")
    print()

    hdr_type = f"  {'Type':<8}  {'N':>6}  {'Mean(s1/act)':>13}  {'Min':>8}  {'Max':>8}"
    print(hdr_type)
    print("  " + "-" * (len(hdr_type) - 2))
    for atype in ["speech", "music", "sound"]:
        ratios = type_ratios.get(atype, [])
        if not ratios:
            print(f"  {atype:<8}  {'0':>6}  {'N/A':>13}  {'N/A':>8}  {'N/A':>8}")
        else:
            arr = _np.array(ratios)
            print(
                f"  {atype:<8}  {len(arr):>6}  "
                f"{arr.mean():>13.4f}  {arr.min():>8.4f}  {arr.max():>8.4f}"
            )
    print()

    print("=== Split Method Statistics ===")
    hdr_method = f"  {'Method':<10}  {'N':>6}  {'Mean(s1/act)':>13}  {'Min':>8}  {'Max':>8}"
    print(hdr_method)
    print("  " + "-" * (len(hdr_method) - 2))
    for method, ratios in sorted(method_ratios.items(), key=lambda item: len(item[1]), reverse=True):
        if not ratios:
            continue
        arr = _np.array(ratios)
        print(
            f"  {method:<10}  {len(arr):>6}  "
            f"{arr.mean():>13.4f}  {arr.min():>8.4f}  {arr.max():>8.4f}"
        )
    print()

if __name__ == "__main__":
    main()
