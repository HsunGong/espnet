#!/usr/bin/env python3
"""
Collect speech-edit evaluation results from .results JSONL files,
compute per-metric averages, and produce one TSV per dataset.

Usage:
    python3 local_eval/speech/collect_summary.py \\
        --input_globs \\
            "exp/ct-100k-default-mt/inference/*/eval-test_clean-v1-*" \\
            "exp/minguniaudioedit/test_clean/speech_edit" \\
        --output_dir results/summary

Output layout:
    results/summary/
        transcription_ins.tsv
        audio_effect_speed.tsv
        ...
    Each TSV:
        Row  = experiment / checkpoint / eval_mode
        Cols = metric scores (+ useful sub-metrics from extra)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================================
# Extra sub-metrics worth extracting as separate columns
# ============================================================================
# key = metric name, value = list of (extra_field, column_suffix)
EXTRA_FIELDS: dict[str, list[tuple[str, str]]] = {
    "asr_wer": [("edit_acc", "edit_acc")],
    "speaker_similarity_wespeaker": [("sim", "sim")],
    "speaker_similarity_wavlm": [("sim", "sim")],
    "pseudo_mos": [
        ("utmos", "utmos"),
        ("dns_overall", "dns_overall"),
        ("dns_p808", "dns_p808"),
    ],
}


# ============================================================================
# Path Parsing  (unified — works for any directory structure)
# ============================================================================

def parse_eval_dir(dirpath: str) -> dict[str, str]:
    """
    Extract a human-readable row identity from any eval directory.

    Handles:
        exp/<model>/inference/<ckpt>/eval-<testset>-<mode>
        exp/<model>/test_clean/<suite>
        exp/<model>/test_clean/<suite>-short
    """
    parts = Path(dirpath).parts
    try:
        exp_idx = list(parts).index("exp")
    except ValueError:
        exp_idx = 0

    experiment = parts[exp_idx + 1] if exp_idx + 1 < len(parts) else "unknown"

    if "inference" in parts:
        inf_idx = list(parts).index("inference")
        checkpoint = parts[inf_idx + 1] if inf_idx + 1 < len(parts) else ""
        eval_dir_name = parts[inf_idx + 2] if inf_idx + 2 < len(parts) else ""

        step_m = re.search(r"step_(\d+)", checkpoint)
        step = step_m.group(1) if step_m else "0"

        mode_m = re.search(r"eval-[^-]+-[^-]+-(.+)$", eval_dir_name)
        eval_mode = mode_m.group(1) if mode_m else eval_dir_name

        label = f"{experiment}/s{int(step) // 1000}k/{eval_mode}"
    else:
        checkpoint = ""
        step = "0"
        eval_mode = parts[-1] if len(parts) > exp_idx + 1 else ""
        suffix = f"/{eval_mode}" if eval_mode not in (experiment, "") else ""
        label = f"{experiment}{suffix}"

    return {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "eval_mode": eval_mode,
        "step": step,
        "label": label,
    }


# ============================================================================
# Read .results JSONL and compute per-metric averages
# ============================================================================

def read_results_file(filepath: str) -> list[dict]:
    """Read a .results JSONL file, return list of parsed dicts."""
    samples = []
    with open(filepath) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON error in {filepath}:{line_no}: {e}",
                      file=sys.stderr)
    return samples


def compute_metrics(
    samples: list[dict],
    wer_filter: float | None = None,
) -> dict[str, dict]:
    """
    Compute average for each metric + extra sub-fields.

    If wer_filter is set, samples with WER > threshold are excluded
    from ALL metrics (not just WER) to keep comparison fair.
    """
    if not samples:
        return {}

    # Build exclusion set from WER filter
    excluded_ids = set()
    if wer_filter is not None:
        for s in samples:
            wer_data = s.get("metrics", {}).get("asr_wer", {})
            if wer_data.get("valid") and wer_data.get("score") is not None:
                if wer_data["score"] > wer_filter:
                    excluded_ids.add(s.get("id"))

    all_metrics = set()
    for s in samples:
        all_metrics.update(s.get("metrics", {}).keys())

    result: dict[str, dict] = {}
    for metric_name in sorted(all_metrics):
        scores: list[float] = []
        total, valid, filtered = 0, 0, 0
        extra_accum: dict[str, list[float]] = {}
        extra_defs = EXTRA_FIELDS.get(metric_name, [])
        for ef_field, _ in extra_defs:
            extra_accum[ef_field] = []

        for s in samples:
            m_data = s.get("metrics", {}).get(metric_name)
            if m_data is None:
                continue
            total += 1

            if wer_filter is not None and s.get("id") in excluded_ids:
                filtered += 1
                continue

            if m_data.get("valid") and m_data.get("score") is not None:
                scores.append(m_data["score"])
                valid += 1

                extra = m_data.get("extra") or {}
                for ef_field, _ in extra_defs:
                    v = extra.get(ef_field)
                    if v is not None and isinstance(v, (int, float)):
                        extra_accum[ef_field].append(float(v))

        result[metric_name] = {
            "avg": sum(scores) / len(scores) if scores else None,
            "valid": valid,
            "total": total,
            "filtered": filtered,
        }

        for ef_field, ef_suffix in extra_defs:
            vals = extra_accum[ef_field]
            result[f"{metric_name}.{ef_suffix}"] = {
                "avg": sum(vals) / len(vals) if vals else None,
                "valid": len(vals),
                "total": total,
                "filtered": filtered,
            }

    return result


# ============================================================================
# Discover all eval directories matching input globs
# ============================================================================

def discover_eval_dirs(input_globs: list[str]) -> list[str]:
    """Expand globs and return deduplicated list of directories."""
    dirs: list[str] = []
    seen: set[str] = set()
    for pattern in input_globs:
        for path in sorted(glob.glob(pattern)):
            rp = os.path.realpath(path)
            if os.path.isdir(rp) and rp not in seen:
                seen.add(rp)
                dirs.append(path)
    return dirs


# ============================================================================
# Main collection logic
# ============================================================================

def collect_all(
    input_globs: list[str],
    wer_filter: float | None = None,
) -> dict[str, list[dict]]:
    """
    Collect from all directories.
    Returns: { dataset_name: [ {label, experiment, ..., <metric>: avg, ...} ] }
    """
    dirs = discover_eval_dirs(input_globs)
    tables: dict[str, list[dict]] = defaultdict(list)

    for d in dirs:
        info = parse_eval_dir(d)
        results_files = sorted(
            f for f in os.listdir(d) if f.endswith(".results")
        )
        for rf in results_files:
            dataset = rf[:-8]  # strip ".results"
            samples = read_results_file(os.path.join(d, rf))
            if not samples:
                continue

            metric_avgs = compute_metrics(samples, wer_filter=wer_filter)

            row: dict = {
                "label": info["label"],
                "experiment": info["experiment"],
                "checkpoint": info["checkpoint"],
                "eval_mode": info["eval_mode"],
                "step": info["step"],
                "n_samples": len(samples),
            }
            for mkey, mval in metric_avgs.items():
                row[mkey] = mval["avg"]
                row[f"{mkey}__valid"] = mval["valid"]
                row[f"{mkey}__total"] = mval["total"]
                row[f"{mkey}__filtered"] = mval["filtered"]

            tables[dataset].append(row)

    return dict(tables)


# ============================================================================
# Helpers
# ============================================================================

def _metric_columns(rows: list[dict]) -> list[str]:
    """Return sorted list of metric column names."""
    meta_keys = {"label", "experiment", "checkpoint", "eval_mode", "step", "n_samples"}
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    return sorted(
        k for k in all_keys
        if k not in meta_keys
        and not k.endswith("__valid")
        and not k.endswith("__total")
        and not k.endswith("__filtered")
    )


# ============================================================================
# Write per-dataset TSV files
# ============================================================================

def write_dataset_tsv(
    dataset: str,
    rows: list[dict],
    output_dir: str,
) -> str:
    """Write a single dataset TSV.  Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{dataset}.tsv")

    metric_cols = _metric_columns(rows)

    # Header: metadata cols + for each metric col: <metric>  <metric>(v/t)
    header_parts = ["label", "experiment", "checkpoint", "eval_mode", "step", "n_samples"]
    for mc in metric_cols:
        header_parts.append(mc)
        header_parts.append(f"{mc}(v/t)")

    with open(outpath, "w") as f:
        f.write("\t".join(header_parts) + "\n")

        for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
            parts = [
                r["label"],
                r["experiment"],
                r["checkpoint"],
                r["eval_mode"],
                str(r["step"]),
                str(r["n_samples"]),
            ]
            for mc in metric_cols:
                val = r.get(mc)
                parts.append(f"{val:.4f}" if val is not None else "")
                v = r.get(f"{mc}__valid", 0)
                t = r.get(f"{mc}__total", 0)
                parts.append(f"{v}/{t}")
            f.write("\t".join(parts) + "\n")

    return outpath


# ============================================================================
# Console summary
# ============================================================================

def print_dataset_table(dataset: str, rows: list[dict]):
    """Pretty-print one dataset table to stdout."""
    if not rows:
        return

    metric_cols = _metric_columns(rows)

    label_w = max(max(len(r["label"]) for r in rows), 6)
    col_w = 22

    n_samples_str = "/".join(sorted(set(str(r["n_samples"]) for r in rows)))

    print(f"\n{'=' * 100}")
    print(f"  DATASET: {dataset}  (n_samples: {n_samples_str})")
    print(f"{'=' * 100}")

    hdr = f"  {'label':<{label_w}}"
    for mc in metric_cols:
        hdr += f"  {mc:>{col_w}}"
    print(hdr)
    print("  " + "-" * (label_w + (col_w + 2) * len(metric_cols)))

    for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
        line = f"  {r['label']:<{label_w}}"
        for mc in metric_cols:
            val = r.get(mc)
            v = r.get(f"{mc}__valid", 0)
            t = r.get(f"{mc}__total", 0)
            if val is not None:
                cell = f"{val:.4f} ({v}/{t})"
            else:
                cell = f"-- ({v}/{t})"
            line += f"  {cell:>{col_w}}"
        print(line)

    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect speech-edit evaluation from .results JSONL files. "
                    "Produces one TSV per dataset."
    )
    parser.add_argument(
        "--input_globs", nargs="+", default=[
            "exp/ct-100k-default-mt/inference/*/eval-test_clean-v1-*",
            "exp/ct-100k-default-c2a/inference/*/eval-test_clean-v1-*",
            "exp/minguniaudioedit/test_clean/speech_edit",
            # "exp/minguniaudioedit/test_clean/speech_edit-short",
            "exp/stepaudiox/test_clean/speech_edit",
            # "exp/stepaudiox/test_clean/speech_edit-short",
            "exp/cv3/test_clean/speech_edit",
            # "exp/cv3/test_clean/speech_edit-short",
        ],
        help="Glob patterns matching directories that contain .results files"
    )
    parser.add_argument(
        "--wer_filter", type=float, default=0,
        help="Exclude samples with WER > this value (0 = disabled). "
             "Exclusion applies to ALL metrics for fairness."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/summary",
        help="Output directory for per-dataset TSV files"
    )
    args = parser.parse_args()
    wer_filter = args.wer_filter if args.wer_filter > 0 else None

    print(f"Input globs: {args.input_globs}")
    print(f"WER filter:  {'>' + str(wer_filter) + ' excluded' if wer_filter else 'disabled'}")
    print(f"Output dir:  {args.output_dir}")

    tables = collect_all(args.input_globs, wer_filter=wer_filter)

    if not tables:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(tables)} datasets, "
          f"{sum(len(v) for v in tables.values())} total rows")

    for dataset in sorted(tables):
        rows = tables[dataset]
        outpath = write_dataset_tsv(dataset, rows, args.output_dir)
        print_dataset_table(dataset, rows)
        print(f"  >> {outpath}")


if __name__ == "__main__":
    main()
