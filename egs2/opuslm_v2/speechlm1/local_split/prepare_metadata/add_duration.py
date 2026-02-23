#!/usr/bin/env python3
"""Add 'duration' field to JSONL metadata records.

For each record:
  - If 'duration' already exists, the record is passed through unchanged.
  - Otherwise, soundfile is used to read the duration from 'audio_path'.

Usage
-----
python3 local_split/prepare_metadata/add_duration.py \\
    --input_jsonl  data/part2_4/metadata.jsonl \\
    --output_jsonl data/part2_4/metadata.with_dur.jsonl \\
    --nj 32
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Per-record worker (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _process_record(record: dict) -> dict:
    """Return record with 'duration' filled in if missing."""
    if "duration" in record:
        return record                       # already present – skip

    audio_path = record.get("audio_path") or record.get("path")
    if not audio_path:
        logging.warning("Record has no audio_path/path field: %s", record)
        return record

    if not os.path.exists(audio_path):
        logging.warning("Audio file not found: %s", audio_path)
        return record

    try:
        info = sf.info(audio_path)
        record["duration"] = info.duration
    except Exception as e:
        logging.warning("Failed to read %s: %s", audio_path, e)

    return record


def _process_line(line: str):
    """Parse a JSON line and call _process_record; returns None on bad JSON."""
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except Exception as e:
        logging.warning("Bad JSON line (%s): %s", e, line[:120])
        return None
    return _process_record(record)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add 'duration' field to JSONL metadata if missing."
    )
    parser.add_argument("-i", "--input_jsonl", type=Path, required=True,
                        help="Input metadata JSONL.")
    parser.add_argument("-o", "--output_jsonl", type=Path, required=True,
                        help="Output metadata JSONL (with duration filled).")
    parser.add_argument("--nj", type=int, default=32,
                        help="Number of parallel workers (default: 32).")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    lines = args.input_jsonl.read_text(encoding="utf-8").splitlines()
    print(f"[add_duration] Input:  {args.input_jsonl}  ({len(lines)} lines)")

    n_written = 0
    n_skipped = 0
    n_already = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        if args.nj <= 1:
            for line in tqdm(lines, desc="add_duration"):
                rec = _process_line(line)
                if rec is None:
                    n_skipped += 1
                    continue
                if "duration" in rec:
                    n_already += 1 if rec.get("_had_dur") else 0
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
        else:
            with ProcessPoolExecutor(max_workers=args.nj) as pool:
                futures = {pool.submit(_process_line, ln): i
                           for i, ln in enumerate(lines)}
                results: list = [None] * len(lines)
                for fut in tqdm(as_completed(futures),
                                total=len(futures), desc="add_duration"):
                    idx = futures[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        logging.warning("Worker error at line %d: %s", idx, e)
                        results[idx] = None

            for rec in results:
                if rec is None:
                    n_skipped += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[add_duration] Written: {n_written}  |  Skipped: {n_skipped}")
    print(f"[add_duration] Output:  {args.output_jsonl}")


if __name__ == "__main__":
    main()

