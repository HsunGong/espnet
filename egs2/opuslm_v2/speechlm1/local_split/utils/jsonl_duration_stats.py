#!/usr/bin/env python3
"""
Compute split1/main duration ratio statistics from one or more step4 JSONL files.

Usage:
    python local_split/utils/jsonl_duration_stats.py <jsonl> [<jsonl> ...]
    python local_split/utils/jsonl_duration_stats.py data/part2_4/full/*/metadata.step4_repeat_gen.default.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def stats(arr: np.ndarray) -> dict:
    return {
        "n":    len(arr),
        "mean": float(arr.mean()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "var":  float(arr.var()),
        "std":  float(arr.std()),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
    }


def fmt(d: dict) -> str:
    return (
        f"n={d['n']:>7}  "
        f"mean={d['mean']:>7.3f}  "
        f"min={d['min']:>7.3f}  "
        f"max={d['max']:>7.3f}  "
        f"var={d['var']:>7.4f}  "
        f"std={d['std']:>6.4f}  "
        f"p25={d['p25']:>7.3f}  "
        f"p50={d['p50']:>7.3f}  "
        f"p75={d['p75']:>7.3f}"
    )


def load_ratios(path: str) -> tuple[list[float], list[float], list[float]]:
    """Return (ratios, split1_durs, main_durs) from a JSONL file."""
    ratios, s1, main = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                d_main  = rec.get("main",   {}).get("duration")
                d_split1 = rec.get("split1", {}).get("duration")
                if d_main and d_split1 and d_main > 0:
                    ratios.append(d_split1 / d_main)
                    s1.append(d_split1)
                    main.append(d_main)
            except Exception:
                continue
    return ratios, s1, main


def print_stats(label: str, ratios: list[float], s1: list[float], main: list[float]) -> None:
    if not ratios:
        print(f"  {label}: no valid records")
        return
    print(f"\n  [{label}]")
    print(f"    ratio (split1/main):  {fmt(stats(np.array(ratios)))}")
    print(f"    split1 duration (s):  {fmt(stats(np.array(s1)))}")
    print(f"    main   duration (s):  {fmt(stats(np.array(main)))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Duration ratio stats from step4 JSONL files.")
    parser.add_argument("jsonl", nargs="+", help="Input JSONL file(s)")
    args = parser.parse_args()

    all_ratios, all_s1, all_main = [], [], []

    for path in args.jsonl:
        r, s, m = load_ratios(path)
        label = Path(path).parent.name + "/" + Path(path).name
        print_stats(label, r, s, m)
        all_ratios.extend(r)
        all_s1.extend(s)
        all_main.extend(m)

    if len(args.jsonl) > 1:
        print_stats("ALL (aggregate)", all_ratios, all_s1, all_main)

    print()


if __name__ == "__main__":
    main()
