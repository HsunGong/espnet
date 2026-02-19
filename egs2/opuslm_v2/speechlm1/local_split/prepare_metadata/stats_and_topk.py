#!/usr/bin/env python3
"""Compute per-type duration statistics and optionally select top-k samples.

The script works on *flat* metadata records that contain at least:
    audio_path  (str)   – used to infer audio type (speech / music / sound)
    duration    (float) – duration in seconds
    dataset     (str, optional) – source dataset name

Statistics printed
------------------
Per inferred type:
  • count and total / mean duration
  • duration-bucket histogram: 0-5 | 5-10 | 10-15 | 15-20 | 20+
  • per-dataset breakdown (count + mean dur)

Selection (optional, requires --output_jsonl and --k > 0)
----------------------------------------------------------
After optional duration filtering (--min-dur / --max-dur):
  • randomly shuffle each bucket then take first k samples
  • write to output JSONL (speech → music → sound order)

Example
-------
# Stats only
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl

# Filter + top-k
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl \\
    --output_jsonl data/part2_4/metadata.dur_20_30.top2k_each.jsonl \\
    --min_dur 20 --max_dur 30 --k 2000
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Audio-type inference (same logic as select_topk.py)
# ---------------------------------------------------------------------------

def get_audio_type(path: str) -> str:
    p = path.lower()
    if any(k in p for k in ["speech", "owsm", "emilia", "commonvoice", "voxforge"]):
        return "speech"
    elif any(k in p for k in ["music", "fma", "jamendo", "disco"]):
        return "music"
    else:
        return "sound"


def get_dur_bucket(dur: float) -> str:
    if dur < 5:
        return "0-5s"
    elif dur < 10:
        return "5-10s"
    elif dur < 15:
        return "10-15s"
    elif dur < 20:
        return "15-20s"
    else:
        return "20+s"


DUR_BUCKETS = ["0-5s", "5-10s", "10-15s", "15-20s", "20+s"]
AUDIO_TYPES = ["speech", "music", "sound"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class TypeStats:
    def __init__(self) -> None:
        self.count: int = 0
        self.total_dur: float = 0.0
        self.buckets: DefaultDict[str, int] = defaultdict(int)
        # dataset → [count, total_dur]
        self.datasets: DefaultDict[str, List] = defaultdict(lambda: [0, 0.0])

    def add(self, dur: float, dataset: str) -> None:
        self.count += 1
        self.total_dur += dur
        self.buckets[get_dur_bucket(dur)] += 1
        self.datasets[dataset][0] += 1
        self.datasets[dataset][1] += dur

    @property
    def mean_dur(self) -> float:
        return self.total_dur / self.count if self.count else 0.0


def print_stats(stats: Dict[str, TypeStats], title: str = "=== Statistics ===") -> None:
    print()
    print(title)
    total_all = sum(s.count for s in stats.values())
    print(f"Total records: {total_all}")
    print()

    # Header row for bucket table
    bucket_header = "  ".join(f"{b:>8}" for b in DUR_BUCKETS)
    print(f"{'Type':<8}  {'Count':>8}  {'MeanDur':>8}  {bucket_header}")
    print("-" * (8 + 2 + 8 + 2 + 8 + 2 + len(bucket_header)))

    for t in AUDIO_TYPES:
        s = stats[t]
        if s.count == 0:
            bucket_row = "  ".join(f"{'0':>8}" for _ in DUR_BUCKETS)
            print(f"{t:<8}  {'0':>8}  {'0.00':>8}  {bucket_row}")
            continue
        bucket_row = "  ".join(f"{s.buckets.get(b, 0):>8}" for b in DUR_BUCKETS)
        print(f"{t:<8}  {s.count:>8}  {s.mean_dur:>8.2f}  {bucket_row}")

    # Per-dataset breakdown
    print()
    print("--- Per-dataset breakdown ---")
    for t in AUDIO_TYPES:
        s = stats[t]
        if s.count == 0:
            continue
        print(f"\n  [{t}]")
        sorted_ds = sorted(s.datasets.items(), key=lambda x: -x[1][0])
        print(f"    {'Dataset':<40}  {'Count':>8}  {'MeanDur':>8}")
        print(f"    {'-'*40}  {'-'*8}  {'-'*8}")
        for ds, (cnt, tdur) in sorted_ds:
            mean = tdur / cnt if cnt else 0.0
            print(f"    {ds:<40}  {cnt:>8}  {mean:>8.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stats + optional top-k selection from flat metadata JSONL."
    )
    parser.add_argument("-i", "--input_jsonl", type=Path, required=True,
                        help="Input metadata JSONL (flat records with audio_path + duration)")
    parser.add_argument("-o", "--output_jsonl", type=Path, default=None,
                        help="Output JSONL path (required when --k > 0)")
    parser.add_argument("-k", "--k", type=int, default=0,
                        help="Max samples per audio type to write; 0 = stats only")
    parser.add_argument("--min_dur", type=float, default=None,
                        help="Minimum duration (seconds, inclusive) for selection")
    parser.add_argument("--max_dur", type=float, default=None,
                        help="Maximum duration (seconds, exclusive) for selection")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for per-type shuffle (default: 7)")
    args = parser.parse_args()

    if args.k > 0 and args.output_jsonl is None:
        parser.error("--output_jsonl is required when --k > 0")

    # ------------------------------------------------------------------ read
    # stats over *all* valid records (before duration filter)
    stats_all: Dict[str, TypeStats] = {t: TypeStats() for t in AUDIO_TYPES}
    # buckets used for selection (after duration filter)
    grouped: DefaultDict[str, List[dict]] = defaultdict(list)

    total_lines = 0
    skipped_bad_json = 0
    skipped_no_path = 0
    skipped_no_dur = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                skipped_bad_json += 1
                continue

            audio_path: Optional[str] = record.get("audio_path") or record.get("path")
            if not audio_path:
                skipped_no_path += 1
                continue

            raw_dur = record.get("duration")
            if raw_dur is None:
                skipped_no_dur += 1
                continue
            try:
                dur = float(raw_dur)
            except (TypeError, ValueError):
                skipped_no_dur += 1
                continue

            dataset: str = record.get("dataset", "<unknown>")
            atype = get_audio_type(str(audio_path))

            # always accumulate for global stats
            stats_all[atype].add(dur, dataset)

            # apply duration filter for selection
            if args.min_dur is not None and dur < args.min_dur:
                continue
            if args.max_dur is not None and dur >= args.max_dur:
                continue

            grouped[atype].append(record)

    # ----------------------------------------------------------------- stats
    print_stats(stats_all, title="=== Statistics (all valid records) ===")

    # stats after filter (only when a filter is active)
    filter_active = (args.min_dur is not None) or (args.max_dur is not None)
    if filter_active:
        stats_filtered: Dict[str, TypeStats] = {t: TypeStats() for t in AUDIO_TYPES}
        for atype in AUDIO_TYPES:
            for rec in grouped[atype]:
                audio_path_str = rec.get("audio_path") or rec.get("path", "")
                dur = float(rec.get("duration", 0))
                ds = rec.get("dataset", "<unknown>")
                stats_filtered[atype].add(dur, ds)
        dur_range = (
            f"{args.min_dur if args.min_dur is not None else '-∞'}"
            f" ≤ dur < "
            f"{args.max_dur if args.max_dur is not None else '+∞'}"
        )
        print_stats(stats_filtered, title=f"=== Statistics after duration filter [{dur_range}] ===")

    print(f"Input file:       {args.input_jsonl}")
    print(f"Total lines:      {total_lines}")
    print(f"Skipped bad json: {skipped_bad_json}")
    print(f"Skipped no path:  {skipped_no_path}")
    print(f"Skipped no dur:   {skipped_no_dur}")

    # --------------------------------------------------------------- top-k
    if args.k <= 0:
        return

    rng = random.Random(args.seed)
    selected: Dict[str, List[dict]] = {}
    for t in AUDIO_TYPES:
        bucket = grouped.get(t, [])
        rng.shuffle(bucket)
        selected[t] = bucket[: args.k]

    assert args.output_jsonl is not None
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for t in AUDIO_TYPES:
            for rec in selected.get(t, []):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print()
    print("=== Selection ===")
    print(f"k / type:  {args.k}")
    print(f"seed:      {args.seed}")
    for t in AUDIO_TYPES:
        print(f"  {t}: {len(selected.get(t, []))}")
    print(f"Written:   {n_written}")
    print(f"Output:    {args.output_jsonl}")


if __name__ == "__main__":
    main()
