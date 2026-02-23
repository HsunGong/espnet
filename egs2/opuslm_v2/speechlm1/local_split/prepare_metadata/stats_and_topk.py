#!/usr/bin/env python3
"""Compute per-type duration statistics and optionally select top-k samples.

The script works on *flat* metadata records that contain at least:
    audio_path  (str)   – used to infer audio type (speech / music / sound)
    duration    (float, optional) – duration in seconds; auto-filled via
                                    soundfile if missing
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
  • single type  → write to output_jsonl as-is
  • multiple types → write to output_jsonl stem per type:
      e.g. metadata.out.jsonl → metadata.out.speech.jsonl + .music.jsonl + .sound.jsonl

Example
-------
# Stats only
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl

# Filter + top-k, all types (→ separate per-type files)
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl \\
    --output_jsonl data/part2_4/metadata.dur_20_30.top2k_each.jsonl \\
    --min_dur 20 --max_dur 30 --k 2000

# Single type → single output file
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl \\
    --output_jsonl data/part2_4/metadata.sound.top2k.jsonl \\
    --type sound --k 2000

# Two types → two output files
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.dur_20_30.jsonl \\
    --output_jsonl data/part2_4/metadata.top2k.jsonl \\
    --type speech sound --k 2000

# Split by duration bins → separate files per bin
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.jsonl \\
    --output_jsonl data/part2_4/metadata.top2k.jsonl \\
    --dur_bins 5 10 --k 2000
# produces: metadata.top2k.min_0.max_5.jsonl
#           metadata.top2k.min_5.max_10.jsonl
#           metadata.top2k.min_10.jsonl

# Both type and dur_bins → one file per (type, bin)
python3 local_split/prepare_metadata/stats_and_topk.py \\
    --input_jsonl data/part2_4/metadata.jsonl \\
    --output_jsonl data/part2_4/metadata.top2k.jsonl \\
    --type speech sound --dur_bins 10 20 --k 2000
# produces: metadata.top2k.speech.min_0.max_10.jsonl  etc.
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Duration-bin helpers
# ---------------------------------------------------------------------------

def _float_label(v: float) -> str:
    """Format a float as a clean string for use in file names."""
    if v == int(v):
        return str(int(v))
    return str(v)


def make_dur_bins(breakpoints: List[float]):
    """Convert a list of breakpoints into (lo, hi, file_suffix) triples.

    Example: [5, 10] -> [(0, 5, 'min_0.max_5'),
                         (5, 10, 'min_5.max_10'),
                         (10, inf, 'min_10')]
    """
    bps = sorted(set(breakpoints))
    bins = []
    lo = 0.0
    for hi in bps:
        bins.append((lo, hi, f"min_{_float_label(lo)}.max_{_float_label(hi)}"))
        lo = hi
    bins.append((lo, float("inf"), f"min_{_float_label(lo)}"))
    return bins


def _in_bin(dur: float, lo: float, hi: float) -> bool:
    return lo <= dur < hi


def _fill_duration(record: dict) -> Optional[float]:
    """Return duration for the record, reading from file if not already present."""
    raw = record.get("duration")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    # Try to read from audio file
    audio_path: Optional[str] = record.get("audio_path") or record.get("path")
    if not audio_path:
        return None
    if not os.path.exists(audio_path):
        logging.warning("Audio file not found: %s", audio_path)
        return None
    try:
        info = sf.info(audio_path)
        dur = float(info.duration)
        record["duration"] = dur   # write back so output carries the field
        return dur
    except Exception as e:
        logging.warning("Cannot read duration from %s: %s", audio_path, e)
        return None

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
    parser.add_argument("-k", "--k", type=int, default=-1,
                        help="Max samples per audio type to write; 0 = stats only")
    parser.add_argument("--min_dur", type=float, default=None,
                        help="Minimum duration (seconds, inclusive) for selection")
    parser.add_argument("--max_dur", type=float, default=None,
                        help="Maximum duration (seconds, exclusive) for selection")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for per-type shuffle (default: 7)")
    parser.add_argument("--type", dest="audio_types", type=str, default=None,
                        nargs="+", choices=AUDIO_TYPES,
                        help="Only process these audio types (speech/music/sound); "
                             "multiple values allowed. Default: all types.")
    parser.add_argument("--dur_bins", type=float, nargs="+", default=None,
                        metavar="BP",
                        help="Duration breakpoints (seconds) to split output into "
                             "separate files. E.g. '5 10' creates bins (0,5), "
                             "(5,10), (10,+\u221e). Each bin is written to "
                             "stem.min_X[.max_Y].jsonl. Requires --output_jsonl.")
    args = parser.parse_args()

    # Resolve the active type set
    active_types: List[str] = args.audio_types if args.audio_types else AUDIO_TYPES
    multi_type: bool = len(active_types) > 1
    dur_bins = make_dur_bins(args.dur_bins) if args.dur_bins else None  # [(lo,hi,suffix), ...]

    if args.k != 0 and args.output_jsonl is None:
        parser.error("--output_jsonl is required when --k != 0")
    if args.dur_bins and args.output_jsonl is None:
        parser.error("--output_jsonl is required when --dur_bins is set")

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

            dur = _fill_duration(record)
            if dur is None:
                skipped_no_dur += 1
                continue

            dataset: str = record.get("dataset", "<unknown>")
            atype = get_audio_type(str(audio_path))

            # skip if --type filter is set and doesn't match
            if atype not in active_types:
                continue

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
    if args.k == 0:
        return  # stats only, no output
    elif args.k < 0:
        rng = random.Random(args.seed)
        selected: Dict[str, List[dict]] = {}
        for t in active_types:
            bucket = grouped.get(t, [])
            rng.shuffle(bucket)
            selected[t] = bucket[: args.k]
    else:
        selected = {t: grouped.get(t, [])[:args.k] for t in active_types}

    assert args.output_jsonl is not None
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Build stem (strip .jsonl suffix)
    base = args.output_jsonl
    stem = base.with_suffix("") if base.suffix == ".jsonl" else base

    print()
    print("=== Selection ===")
    print(f"k / type:    {args.k}")
    print(f"seed:        {args.seed}")
    print(f"multi_type:  {multi_type}")
    print(f"dur_bins:    {[(lo, hi, s) for lo, hi, s in dur_bins] if dur_bins else 'none'}")

    def _write_records(records: List[dict], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w", encoding="utf-8") as fout:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n

    n_written = 0

    # Dimensions: type (optional split) × dur_bin (optional split)
    for t in active_types:
        recs = selected.get(t, [])

        if dur_bins is None:
            # No bin splitting
            if not multi_type:
                out_path = args.output_jsonl
            else:
                out_path = stem.parent / f"{stem.name}.{t}.jsonl"
            cnt = _write_records(recs, out_path)
            n_written += cnt
            print(f"  {t}: {cnt}  →  {out_path}")
        else:
            # Split by duration bin
            for lo, hi, bin_suffix in dur_bins:
                bin_recs = [r for r in recs
                            if _in_bin(float(r.get("duration", 0)), lo, hi)]
                if multi_type:
                    out_path = stem.parent / f"{stem.name}.{t}.{bin_suffix}.jsonl"
                else:
                    out_path = stem.parent / f"{stem.name}.{bin_suffix}.jsonl"
                cnt = _write_records(bin_recs, out_path)
                n_written += cnt
                print(f"  {t} [{bin_suffix}]: {cnt}  →  {out_path}")

    print(f"Written total: {n_written}")


if __name__ == "__main__":
    main()
