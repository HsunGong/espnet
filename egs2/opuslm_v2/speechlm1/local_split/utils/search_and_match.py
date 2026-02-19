#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import soundfile as sf

DATA_DIR = Path("/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/part2_4/dur_20_30.debug")
# EXP_BASE = Path("/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/inference/inference_audio_continue_step_350000")
# EXP_BASE = Path("/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/inference/inference_speech_step_350000")
EXP_BASE = Path("/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/exp/opuslm_v2_stage2_pretrain_base/inference/inference_speech_continue_step_350000")

def parse_args():
    p = argparse.ArgumentParser(
        description="Two independent scans: (1) jsonl by name_key/path_key; (2) exp wavs by name_key on parent path and path_key on basename."
    )
    p.add_argument("-n", "--name-key", action="append", default=[],
                   help="Substring filter. JSONL: filename; EXP: wav parent path. Repeatable.")
    p.add_argument("-p", "--path-key", action="append", default=[],
                   help="Substring filter. JSONL: line must contain; EXP: wav basename must contain. Repeatable.")
    p.add_argument("--mode", choices=["all", "any"], default="all",
                   help="Match mode for keys: all=all keys must match; any=any key may match.")
    return p.parse_args()

def match_keys(s: str, keys, mode: str) -> bool:
    if not keys:
        return True
    return all(k in s for k in keys) if mode == "all" else any(k in s for k in keys)

def scan_jsonl(name_keys, path_keys, mode):
    # jsonl scan: name_key filters filename; path_key filters raw line text (fast substring)
    for jf in DATA_DIR.glob("metadata.*.jsonl"):
        if not match_keys(jf.name, name_keys, mode):
            continue
        with jf.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not match_keys(line, path_keys, mode):
                    continue
                try:
                    obj = json.loads(line)
                    ap = obj["split1"]["audio_path"]
                except json.JSONDecodeError:
                    continue

                print(jf)
                for seg in ("split1", "split2", "split2_old", "main"):
                    ap = (obj.get(seg, {}) or {}).get("audio_path")
                    print(f"- {seg}")
                    if isinstance(ap, str) and ap.strip():
                        print(f"\t{ap}")
                    if seg in obj and "duration" in obj[seg]:
                        print("\tduration", obj[seg]["duration"])
                    if seg == "split1":
                        print("\tcaption1", repr(obj[seg]["audio_caption"]))
                    elif seg == "split2":
                        print("\tcaption2", repr(obj[seg]["audio_caption"]))
                    elif seg == "split2_old" and seg in obj:
                        print("\tcaption2-old", repr(obj[seg]["audio_caption"]))
                    elif seg == "main":
                        print("\tcaption-main", repr(obj[seg]["audio_caption"]))
                

def scan_exp(name_keys, path_keys, mode):
    # exp scan: name_key on wav.parent path string; path_key on wav basename (filename)
    for subdir in EXP_BASE.iterdir():
        if not subdir.is_dir():
            continue
        if not match_keys(str(subdir), name_keys, mode):
            continue
        # print(subdir)

        for wav in list(subdir.rglob("*.wav")) + list(subdir.rglob("*.flac")):
            # print(wav)
            if not match_keys(wav.name, path_keys, mode):
                continue
            print(f"> duration={sf.info(wav).duration}", wav)

def main():
    args = parse_args()

    # Independent scans (no dependency)
    scan_exp(args.name_key, args.path_key, args.mode)
    scan_jsonl(args.name_key, args.path_key, args.mode)

if __name__ == "__main__":
    main()
