#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from local_split.local_config import apply_step_config

# Suppress librosa warnings if any
warnings.filterwarnings("ignore")


def get_audio_type(path: str) -> str:
    """Infer audio type from path keywords (speech / music / sound)."""
    p = path.lower()
    if any(k in p for k in ["speech", "owsm", "emilia", "commonvoice", "voxforge"]):
        return "speech"
    elif any(k in p for k in ["music", "fma", "jamendo", "disco"]):
        return "music"
    else:
        return "sound"


def process_one(line, output_dir, split_ratio=0.5, vad_top_db=30):
    try:
        data = json.loads(line)
        
        # Heuristic to find audio path
        audio_path = data.get("audio_path")
        if not audio_path and "audios" in data:
            # Fallback based on some schemas, but usually "audio_path" is top level
            # If not found, log and skip (or try to infer)
            pass

        if not audio_path or not os.path.exists(audio_path):
             # Try common variations
            if "wav_path" in data and os.path.exists(data["wav_path"]):
                audio_path = data["wav_path"]
            elif "file_path" in data and os.path.exists(data["file_path"]):
                audio_path = data["file_path"]
        
        if not audio_path:
             return None

        # Load audio
        # Using librosa to load. Sr=None to preserve native sampling rate.
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            # If loading fails, return None
            return None
        
        # specific logic for stereo: convert to mono for VAD? 
        # But split should apply to all channels.
        # Librosa load converts to mono by default unless mono=False. 
        # Standard VAD usually works on mono. Steps:
        # 1. Load mono for VAD.
        # 2. Apply split points to original multi-channel if needed.
        # But commonly these datasets are mono. Let's assume mono (y is 1D).
        
        # Energy VAD
        # Use librosa.effects.split; top_db is tunable via --vad_top_db
        try:
            intervals = librosa.effects.split(y, top_db=vad_top_db)
        except Exception:
            # Fallback if VAD fails
            intervals = np.array([])
        
        # If no speech detected, treat whole file as one segment
        if len(intervals) == 0:
            intervals = np.array([[0, len(y)]])
            
        # Collect segments
        segments = [y[start:end] for (start, end) in intervals]
        
        # Concatenate check logic
        # "If multiple segments, first part concatenated <= 1/2 total active duration"
        total_active_samples = sum(len(s) for s in segments)
        target_split_samples = total_active_samples * split_ratio
        
        part1_segments = []
        part2_segments = []
        current_samples = 0
        
        # Greedy accumulation for part 1
        for i, seg in enumerate(segments):
            if current_samples + len(seg) <= target_split_samples:
                part1_segments.append(seg)
                current_samples += len(seg)
            else:
                # If adding this segment exceeds target
                if len(part1_segments) == 0:
                     # If the very first segment is > target (i.e., > 0.5 total), 
                     # we must put it in part1? Or maybe part1 is just this segment?
                     # Let's assign it to part1 to ensure part1 is not empty.
                     part1_segments.append(seg)
                     part2_segments.extend(segments[i+1:])
                else:
                    # Current segments in part1 are good. The rest go to part2.
                    part2_segments.append(seg)
                    part2_segments.extend(segments[i+1:])
                break
        
        # If loop finished and didn't break (very rare with logic above unless total=0 or 1 seg matches exactly)
        if not part2_segments and len(part1_segments) < len(segments):
             # Should be covered by break, but just in case
             pass
        elif not part2_segments and len(segments) > 0 and len(part1_segments) == len(segments):
             # Everything went to part 1? Means target was met exactly at end?
             # Or single segment.
             pass

        # Concatenate
        if len(part1_segments) > 0:
            y1 = np.concatenate(part1_segments)
        else:
            y1 = np.array([], dtype=y.dtype)

        if len(part2_segments) > 0:
            y2 = np.concatenate(part2_segments)
        else:
            y2 = np.array([], dtype=y.dtype)
            
        # Handling the case where one part is empty logic:
        # If original file had only one segment > split_ratio? -> It went to part 1. Part 2 empty.
        # User said "If multiple segments...". If single segment or effectively single:
        if len(part2_segments) == 0 and len(y1) > 0:
             # Split y1 in half blindly
             mid = len(y1) // 2
             y2 = y1[mid:]
             y1 = y1[:mid]
        elif len(part1_segments) == 0 and len(y2) > 0:
             # Should not happen with logic above, but for safety
             mid = len(y2) // 2
             y1 = y2[:mid]
             y2 = y2[mid:]

        # Save files
        basename = os.path.basename(audio_path)
        name, ext = os.path.splitext(basename)
        if not ext: ext = ".wav" # Default
        
        # Ensure unique filenames in output dir to avoid collisions if basenames are not unique
        # Use a hash or just folder structure?
        # Assuming basenames are unique or we can prefix with index?
        # The input metadata usually has unique paths.
        # Let's use relative path structure if possible, but output_dir is flat here?
        # "output-dir 的两个 split" usually implies two subfolders? 
        # User said "audio 写入到 output-dir 的两个 split".
        # This might mean: `output_dir/split1/file.wav` and `output_dir/split2/file.wav`.
        
        out_dir1 = os.path.abspath(os.path.join(output_dir, "split1"))
        out_dir2 = os.path.abspath(os.path.join(output_dir, "split2"))
        os.makedirs(out_dir1, exist_ok=True)
        os.makedirs(out_dir2, exist_ok=True)
        
        out_path1 = os.path.abspath(os.path.join(out_dir1, f"{name}{ext}"))
        out_path2 = os.path.abspath(os.path.join(out_dir2, f"{name}{ext}"))
        
        sf.write(out_path1, y1, sr)
        sf.write(out_path2, y2, sr)
        
        # Prepare Metadata
        data["audio_caption"] = data.pop("qwen_caption")
        record = {
            "main": data,
            "split1": {
                "audio_path": out_path1,
                "duration": len(y1) / sr
            },
            "split2": {
                "audio_path": out_path2,
                "duration": len(y2) / sr
            }
        }
        return record

    except Exception as e:
        # print(f"Error processing {line}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--nj", type=int, default=32)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--vad_top_db", type=float, default=30,
        help="Energy threshold (dB) for librosa VAD silence detection. "
             "Lower = more sensitive (detects quieter speech). Default: 30."
    )
    parser.add_argument(
        "--max_split1_ratio", type=float, default=1.0,
        help="Maximum fraction of total active audio that split1 may contain. "
             "1.0 = no cap (all active audio can go to split1, fallback halves it); "
             "0.5 = split1 ≤ 50%% of active audio. Default: 1.0."
    )
    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step1_vad")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = Parallel(n_jobs=args.nj)(
        delayed(process_one)(
            line, args.output_dir,
            split_ratio=args.max_split1_ratio,
            vad_top_db=args.vad_top_db,
        )
        for line in tqdm(lines, desc="Processing VAD Split")
    )

    # ---------------------------------------------------------------- write
    valid_results = []
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for res in results:
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                valid_results.append(res)

    # ---------------------------------------------------------------- stats
    from collections import defaultdict
    # per type: list of split1_dur / main_dur ratios
    type_ratios: dict = defaultdict(list)
    for res in valid_results:
        main_dur = res.get("main", {}).get("duration")
        split1_dur = res.get("split1", {}).get("duration")
        audio_path = res.get("main", {}).get("audio_path", "")
        if main_dur and split1_dur and main_dur > 0:
            atype = get_audio_type(audio_path)
            type_ratios[atype].append(split1_dur / main_dur)

    print(f"\n=== Step1 VAD Split Statistics ===")
    print(f"  vad_top_db={args.vad_top_db}  max_split1_ratio={args.max_split1_ratio}")
    print(f"  Valid records written: {len(valid_results)} / {len(lines)}")
    print()
    hdr = f"  {'Type':<8}  {'N':>6}  {'Mean':>8}  {'Min':>8}  {'Max':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for atype in ["speech", "music", "sound"]:
        ratios = type_ratios.get(atype, [])
        if not ratios:
            print(f"  {atype:<8}  {'0':>6}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
        else:
            import numpy as _np
            arr = _np.array(ratios)
            print(
                f"  {atype:<8}  {len(arr):>6}  "
                f"{arr.mean():>8.4f}  {arr.min():>8.4f}  {arr.max():>8.4f}"
            )
    print()


if __name__ == "__main__":
    main()
