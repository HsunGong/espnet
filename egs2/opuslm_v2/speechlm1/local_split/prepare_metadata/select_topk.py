#!/usr/bin/env python3
"""Select top-k samples per inferred audio type from a metadata JSONL.

Example:
    python3 local_split/select_topk.py \
        --input_jsonl data/part2_4/metadata.dur_20_30.jsonl \
        --output_jsonl data/part2_4/metadata.dur_20_30.top2k_each.jsonl \
        --k 2000
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def get_audio_type(path: str) -> str:
    p = path.lower()
    if any(k in p for k in ["speech", "owsm", "emilia", "commonvoice", "voxforge"]):
        return "speech"
    elif any(k in p for k in ["music", "fma", "jamendo", "disco"]):
        return "music"
    else:
        return "sound"


def extract_audio_path(record: dict) -> str:
    """Best-effort extraction of audio path from common metadata schemas."""
    main = record.get("main", {})
    if isinstance(main, dict) and main.get("audio_path"):
        return str(main["audio_path"])

    split1 = record.get("split1", {})
    if isinstance(split1, dict) and split1.get("audio_path"):
        return str(split1["audio_path"])

    if record.get("audio_path"):
        return str(record["audio_path"])

    if record.get("path"):
        return str(record["path"])

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-k samples for each inferred audio type (speech/music/sound)."
    )
    parser.add_argument("-i", "--input_jsonl", type=Path, required=True, help="Input metadata JSONL path")
    parser.add_argument("-o", "--output_jsonl", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("-k", "--k", type=int, required=True, help="Max number of samples per type")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for per-type shuffle")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError(f"--k must be > 0, got {args.k}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    total_lines = 0
    valid_lines = 0
    skipped_no_path = 0
    skipped_bad_json = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                skipped_bad_json += 1
                continue

            audio_path = extract_audio_path(record)
            if not audio_path:
                skipped_no_path += 1
                continue

            valid_lines += 1
            t = get_audio_type(audio_path)
            grouped[t].append(record)

    rng = random.Random(args.seed)
    selected: dict[str, list[dict]] = {}
    for t in ["speech", "music", "sound"]:
        bucket = grouped.get(t, [])
        rng.shuffle(bucket)
        selected[t] = bucket[: args.k]

    n_written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for t in ["speech", "music", "sound"]:
            for r in selected.get(t, []):
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_written += 1

    print("Done select_topk")
    print(f"Input:  {args.input_jsonl}")
    print(f"Output: {args.output_jsonl}")
    print(f"k/type: {args.k}")
    print(f"seed:   {args.seed}")
    print(f"Total lines:      {total_lines}")
    print(f"Valid lines:      {valid_lines}")
    print(f"Skipped bad json: {skipped_bad_json}")
    print(f"Skipped no path:  {skipped_no_path}")
    print(
        "Selected counts: "
        f"speech={len(selected.get('speech', []))}, "
        f"music={len(selected.get('music', []))}, "
        f"sound={len(selected.get('sound', []))}"
    )
    print(f"Written lines:    {n_written}")


if __name__ == "__main__":
    main()

