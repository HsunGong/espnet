#!/usr/bin/env python3
"""step0_prepare — no-VAD prepare: treat each flat metadata record as split1 (= main).

Use this when the input is already segment-level audio (e.g. pre-split split1 files)
and you want to run step4+ without running the VAD split (step1).

Output schema matches step1_vad (nested):
    {
        "main":   { audio_path, duration, audio_caption, ... },
        "split1": { audio_path, duration },
        "meta":   { "split_method": "identity" }
    }
    (no split2)

Usage
-----
python3 local_split/step0_prepare.py \\
    --input_jsonl  data/part2_4/metadata.split1.jsonl \\
    --output_jsonl data/part2_4/exp/metadata.step0_prepare.jsonl
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from local_split.local_config import apply_step_config


# ---------------------------------------------------------------------------
# Per-record conversion
# ---------------------------------------------------------------------------

def process_one(record: dict) -> dict | None:
    """Convert a flat metadata record to nested {main, split1} format."""
    audio_path = (
        record.get("audio_path")
        or record.get("wav_path")
        or record.get("file_path")
    )
    if not audio_path:
        return None

    # Shallow copy to avoid mutating the original
    flat = dict(record)

    # Normalise caption field name (qwen_caption → audio_caption)
    if "qwen_caption" in flat and "audio_caption" not in flat:
        flat["audio_caption"] = flat.pop("qwen_caption")

    return {
        "main": flat,
        "split1": flat,
        # no split2 — this record has only one segment
        "meta": {"split_method": "identity"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="step0: prepare flat metadata as nested split1=main records (no VAD)."
    )
    parser.add_argument("--input_jsonl",  type=Path, required=True,
                        help="Input flat metadata JSONL.")
    parser.add_argument("--output_jsonl", type=Path, required=True,
                        help="Output nested metadata JSONL.")
    parser.add_argument("--nj", type=int, default=1,
                        help="Unused; kept for CLI compatibility with other steps.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional YAML config path.")
    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step0_prepare")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        lines = [line.strip() for line in fin if line.strip()]

    n_ok = 0
    n_skip = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="step0_prepare"):
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"Warning: bad JSON — {e}")
                n_skip += 1
                continue

            res = process_one(record)
            if res is None:
                n_skip += 1
                continue

            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"\n[step0_prepare] Written: {n_ok}  |  Skipped: {n_skip}")
    print(f"[step0_prepare] Output:  {args.output_jsonl}")


if __name__ == "__main__":
    main()
