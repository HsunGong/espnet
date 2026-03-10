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

NOCOLOR = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[32m"
RED     = "\033[31m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

# Per-bin color for WER histogram bars (index matches WER_BINS order)
# green=low WER, yellow=medium, red=high
WER_BIN_COLORS: list[str] = [
    GREEN,   # 0           (exact match)
    GREEN,   # (0, 0.1]
    YELLOW,  # (0.1, 0.2]
    YELLOW,  # (0.2, 0.3]
    RED,     # (0.3, 0.5]
    RED,     # (0.5, 0.8]
    RED,     # (0.8, 1.0]
    RED,     # >1.0
]

# ============================================================================
# Dataset grouping: merge related datasets for aggregate reporting
# ============================================================================
# key = original dataset name  →  value = group name
# Datasets not listed here are kept as-is (no merging).
DATASET_GROUP: dict[str, str] = {
    # transcription edit operations → "transcription_edit"
    "transcription_del": "transcription_edit",
    "transcription_ins": "transcription_edit",
    # "transcription_sub": "transcription_edit", # -> ignore this
    # style variations → "style"
    "style_emotion": "style",
    "style_whisper": "style",
    # audio effects → "audio_effect"
    "audio_effect_dereverb": "audio_effect",
    "audio_effect_pitch": "audio_effect",
    "audio_effect_reverb": "audio_effect",
    "audio_effect_speed": "audio_effect",
    "audio_effect_volume": "audio_effect",

    "speech_remove_mix": "speech_mix",
    "speech_add_mix": "speech_mix",
    "sound_remove_mix": "sound_mix",
    "sound_add_mix": "sound_mix",
    "music_remove_mix": "music_mix",
    "music_add_mix": "music_mix",
    "sing_remove_mix": "sing_mix",
    "sing_add_mix": "sing_mix",

    "music_creative_edit": "creative_mix",
    "sound_creative_edit": "creative_mix",
    "speech_creative_edit": "creative_mix",
}


# ============================================================================
# Custom WER line-chart: experiments to plot & display names
# ============================================================================
# key = label produced by parse_eval_dir  →  value = display name in legend
# Plot order follows insertion order.
WER_LINE_CHART_LABELS: dict[str, str] = {
    "ct-c2a_v2-1000k/s359k/cat2split1":                  "Bagpiper-Edit (ST)",
    "ct-mt-t2a_v2-1000k/s356k/t2a_t2a":                  "Bagpiper-Edit (MT)",
    "cv3/speech_edit":                                     "Cosyvoice-3.0",
    "minguniaudioedit/speech_edit":                        "Ming-UniAudio-Edit",
    "stepaudiox/speech_edit":                              "Step-Audio-EditX",
    "opuslm_v2_stage2_pretrain_base/s350k/tgt2audio":     "Bagpiper-Base",
}
WER_LINE_CHART_DATASET: str = "transcription_replace_sentence"

# ============================================================================
# Custom WER histogram comparison: experiments to compare side-by-side
# ============================================================================
# key = label produced by parse_eval_dir  →  value = display name
WER_HIST_CMP_LABELS: dict[str, str] = {
    "opuslm_v2_stage2_pretrain_base/s350k/t2a_t2a":      "Bagpiper-Base",
    "ct-c2a_v2-1000k/s359k/cat2split1":                  "Bagpiper-Edit (ST)",
    "ct-mt-t2a_v2-1000k/s356k/t2a_t2a":                  "Bagpiper-Edit (MT)",
}
WER_HIST_CMP_DATASET: str = "transcription_replace_sentence"


# ============================================================================
# WER distribution bins for histogram
# ============================================================================
# Each bin: (label, lower_inclusive, upper_exclusive) — last bin is open-ended
WER_BINS: list[tuple[str, float, float]] = [
    ("0.0",       0.0, 0.001),   # exact match (WER == 0)
    ("(0.0,0.2]", 0.001, 0.1001),
    ("(0.2,0.4]", 0.2001, 0.4001),
    ("(0.4,0.6]", 0.4001, 0.6001),
    ("(0.6,0.8]", 0.6001, 0.8001),
    ("(0.8,1.0]", 0.8001, 1.0001),
    # ("(0,0.1]", 0.001, 0.1001),
    # ("(0.1,0.2]", 0.1001, 0.2001),
    # ("(0.2,0.3]", 0.2001, 0.3001),
    # ("(0.3,0.4]", 0.3001, 0.4001),
    # ("(0.4,0.5]", 0.4001, 0.5001),
    # ("(0.5,0.6]", 0.5001, 0.6001),
    # ("(0.6,0.7]", 0.6001, 0.7001),
    # ("(0.7,0.8]", 0.7001, 0.8001),
    # ("(0.8,0.9]", 0.8001, 0.9001),
    # ("(0.9,1.0]", 0.9001, 1.0001),
    (">1.0",    1.0001, float("inf")),
]

# ============================================================================
# WER graph: which datasets to render in the WER histogram sections.
# Names match dataset file stems (e.g. "transcription_ins") or group names
# (e.g. "transcription_edit", "audio_effect", "style").
# Empty list  →  plot ALL datasets (no filtering).
# Override at runtime with  --wer_datasets.
# ============================================================================
WER_GRAPH_DATASETS: list[str] = [
    # "transcription_ins",
    # "transcription_del",
    # "transcription_sub",
    "transcription_edit",   # grouped
    "transcription_replace_sentence",
    "style",                # grouped
    "audio_effect",         # grouped
    "speech_mix",           # grouped
    "sound_mix",
    "music_mix",
    "sing_mix",
    "creative_mix",
]


# ============================================================================
# Extra sub-metrics worth extracting as separate columns
# ============================================================================
# key = metric name, value = list of (extra_field, column_suffix)
EXTRA_FIELDS: dict[str, list[tuple[str, str]]] = {
    "asr_wer": [("edit_acc", "edit_acc")],
    # "speaker_similarity_wespeaker": [("sim", "sim")],
    "speaker_similarity_wavlm": [("sim", "sim")],
    "pseudo_mos": [
        ("utmos", "utmos"),
        ("dns_overall", "dns_overall"),
        ("dns_p808", "dns_p808"),
    ],
    "llm_judge_openai": [
        ("audio_quality", "audio_quality"),
        ("change_quality", "change_quality"),
        ("coherence", "coherence"),
        ("preservation", "preservation"),
        ("creativity", "creativity"),
        # new dimensions (may coexist with old ones across different tasks)
        ("generation_quality", "generation_quality"),
        ("main_consistency", "main_consistency"),
        ("operation_effect", "operation_effect"),
    ],
    "llm_judge_gemini": [
        ("generation_quality", "generation_quality"),
        ("main_consistency", "main_consistency"),
        ("operation_effect", "operation_effect"),
        # legacy dimensions (for tasks still using old prompts)
        ("edit_fidelity", "edit_fidelity"),
        ("audio_quality", "audio_quality"),
        ("coherence", "coherence"),
        ("preservation", "preservation"),
        ("creativity", "creativity"),
    ],
    "clap_similarity": [
        ("audio_sim", "audio_sim"),
        ("main_text_src_sim", "main_text_src_sim"),
        ("main_text_gen_sim", "main_text_gen_sim"),
        ("main_text_delta", "main_text_delta"),
        ("y_text_sim", "y_text_sim"),
        ("x_text_sim", "x_text_sim"),
    ],
    "audio_event_flam": [
        ("audio_sim", "audio_sim"),
        ("main_text_src_sim", "main_text_src_sim"),
        ("main_text_gen_sim", "main_text_gen_sim"),
        ("main_text_delta", "main_text_delta"),
        ("y_text_sim", "y_text_sim"),
        ("x_text_sim", "x_text_sim"),
    ],
    "llm_judge_caption_llm": [
        ("caption_similarity", "caption_similarity"),
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
# WER distribution analysis
# ============================================================================

def compute_wer_distribution(samples: list[dict]) -> dict[str, int]:
    """
    Bin sample WER scores into WER_BINS categories.
    Returns {bin_label: count}.
    """
    counts = {label: 0 for label, _, _ in WER_BINS}
    n_valid = 0
    n_missing = 0

    for s in samples:
        wer_data = s.get("metrics", {}).get("asr_wer", {})
        if not wer_data.get("valid") or wer_data.get("score") is None:
            n_missing += 1
            continue
        score = wer_data["score"]
        n_valid += 1
        for label, lo, hi in WER_BINS:
            if lo <= score < hi:
                counts[label] += 1
                break

    return counts


def _wer_dist_row(info: dict, samples: list[dict]) -> dict:
    """Build one WER-distribution row from raw samples."""
    bin_counts = compute_wer_distribution(samples)
    return {
        "label": info["label"],
        "experiment": info["experiment"],
        "step": info["step"],
        "eval_mode": info["eval_mode"],
        "n_total": len(samples),
        "n_with_wer": sum(bin_counts.values()),
        "bin_counts": bin_counts,
    }


def collect_wer_distributions(
    input_globs: list[str],
) -> dict[str, list[dict]]:
    """
    Collect WER histogram data from all directories.
    Returns: { dataset_name: [ {label, bin_counts, n_total, ...} ] }
    """
    raw = _collect_raw_samples(input_globs)
    tables: dict[str, list[dict]] = defaultdict(list)

    for info, dataset, samples in raw:
        tables[dataset].append(_wer_dist_row(info, samples))

    return dict(tables)


def collect_wer_distributions_grouped(
    input_globs: list[str],
    group_map: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """
    Collect WER histogram data, merged by DATASET_GROUP.
    """
    if group_map is None:
        group_map = DATASET_GROUP

    raw = _collect_raw_samples(input_globs)

    merged: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    group_members: dict[str, set[str]] = defaultdict(set)

    for info, dataset, samples in raw:
        group = group_map.get(dataset)
        if group is None:
            continue
        group_members[group].add(dataset)
        key = (group, info["label"])
        if key not in merged:
            merged[key] = (info, [])
        merged[key][1].extend(samples)

    active_groups = {g for g, m in group_members.items() if len(m) >= 2}

    tables: dict[str, list[dict]] = defaultdict(list)
    for (group, label), (info, samples) in merged.items():
        if group not in active_groups:
            continue
        tables[group].append(_wer_dist_row(info, samples))

    return dict(tables)


def print_wer_histogram(dataset: str, rows: list[dict]):
    """Pretty-print WER distribution as an ASCII bar chart."""
    if not rows:
        return

    bin_labels = [label for label, _, _ in WER_BINS]
    label_w = max(max(len(r["label"]) for r in rows), 6)
    bin_w = max(len(b) for b in bin_labels)
    bar_max = 40  # max bar width in chars

    sep = BLUE + "=" * 100 + NOCOLOR
    print(f"\n{sep}")
    print(f"  {BOLD}{BLUE}WER DISTRIBUTION:{NOCOLOR} {YELLOW}{dataset}{NOCOLOR}")
    print(sep)

    for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
        bc = r["bin_counts"]
        n = r["n_with_wer"]
        if n == 0:
            continue

        print(f"\n  {CYAN}{r['label']}{NOCOLOR}  (n={n})")
        print(BLUE + f"  {'-' * 70}" + NOCOLOR)

        max_count = max(bc.values()) if bc else 1
        for bi, bl in enumerate(bin_labels):
            c = bc.get(bl, 0)
            pct = 100.0 * c / n if n else 0
            bar_len = int(bar_max * c / max_count) if max_count > 0 else 0
            bar_color = WER_BIN_COLORS[bi]
            bar = bar_color + '\u2588' * bar_len + NOCOLOR
            pct_color = WER_BIN_COLORS[bi]
            print(f"    WER {YELLOW}{bl:>{bin_w}}{NOCOLOR}  "
                  f"{bar}{' ' * (bar_max - bar_len)}  "
                  f"{c:>4} ({pct_color}{pct:5.1f}%{NOCOLOR})")

    print()


def write_wer_distribution_tsv(
    dataset: str,
    rows: list[dict],
    output_dir: str,
) -> str:
    """Write WER distribution to a TSV file. Returns output path."""
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{dataset}_wer_dist.tsv")

    bin_labels = [label for label, _, _ in WER_BINS]

    header = ["label", "n_total", "n_with_wer"]
    for bl in bin_labels:
        header.append(f"wer_{bl}_count")
        header.append(f"wer_{bl}_pct")

    with open(outpath, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
            bc = r["bin_counts"]
            n = r["n_with_wer"]
            parts = [r["label"], str(r["n_total"]), str(n)]
            for bl in bin_labels:
                c = bc.get(bl, 0)
                pct = 100.0 * c / n if n else 0
                parts.append(str(c))
                parts.append(f"{pct:.1f}%")
            f.write("\t".join(parts) + "\n")

    return outpath


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

def _collect_raw_samples(
    input_globs: list[str],
) -> list[tuple[dict, str, list[dict]]]:
    """
    Read all .results files and return (dir_info, dataset, samples) triples.
    Shared by collect_all and collect_all_grouped.
    """
    dirs = discover_eval_dirs(input_globs)
    result = []
    for d in dirs:
        info = parse_eval_dir(d)
        results_files = sorted(
            f for f in os.listdir(d) if f.endswith(".results")
        )
        for rf in results_files:
            dataset = rf[:-8]
            samples = read_results_file(os.path.join(d, rf))
            if samples:
                result.append((info, dataset, samples))
    return result


def _build_row(
    info: dict, samples: list[dict], wer_filter: float | None,
) -> dict:
    """Build one summary row from raw samples."""
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
    return row


def collect_all(
    input_globs: list[str],
    wer_filter: float | None = None,
) -> dict[str, list[dict]]:
    """
    Collect from all directories.
    Returns: { dataset_name: [ {label, experiment, ..., <metric>: avg, ...} ] }
    """
    raw = _collect_raw_samples(input_globs)
    tables: dict[str, list[dict]] = defaultdict(list)

    for info, dataset, samples in raw:
        tables[dataset].append(_build_row(info, samples, wer_filter))

    return dict(tables)


def collect_all_grouped(
    input_globs: list[str],
    wer_filter: float | None = None,
    group_map: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """
    Collect and merge datasets according to DATASET_GROUP.

    Raw samples from datasets mapped to the same group are concatenated
    *per experiment directory*, then metrics are recomputed on the merged
    sample set — so averages are exact, not averages-of-averages.

    Returns: { group_name: [ {label, experiment, ..., <metric>: avg, ...} ] }
    Only groups that actually have ≥2 member datasets are returned.
    """
    if group_map is None:
        group_map = DATASET_GROUP

    raw = _collect_raw_samples(input_globs)

    # key = (group_name, label)  →  merged samples list
    merged: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    # Track which individual datasets contributed to each group
    group_members: dict[str, set[str]] = defaultdict(set)

    for info, dataset, samples in raw:
        group = group_map.get(dataset)
        if group is None:
            continue  # not part of any group
        group_members[group].add(dataset)
        key = (group, info["label"])
        if key not in merged:
            merged[key] = (info, [])
        merged[key][1].extend(samples)

    # Only keep groups with ≥2 contributing datasets
    active_groups = {g for g, members in group_members.items() if len(members) >= 2}

    tables: dict[str, list[dict]] = defaultdict(list)
    for (group, label), (info, samples) in merged.items():
        if group not in active_groups:
            continue
        tables[group].append(_build_row(info, samples, wer_filter))

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

    sep = BLUE + "=" * 100 + NOCOLOR
    print(f"\n{sep}")
    print(f"  {BOLD}{BLUE}DATASET:{NOCOLOR} {YELLOW}{dataset}{NOCOLOR}  "
          f"({CYAN}n_samples: {n_samples_str}{NOCOLOR})")
    print(sep)

    hdr = f"  {BOLD}{'label':<{label_w}}{NOCOLOR}"
    for mc in metric_cols:
        hdr += f"  {MAGENTA}{mc:>{col_w}}{NOCOLOR}"
    print(hdr)
    print(BLUE + "  " + "-" * (label_w + (col_w + 2) * len(metric_cols)) + NOCOLOR)

    for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
        line = f"  {CYAN}{r['label']:<{label_w}}{NOCOLOR}"
        for mc in metric_cols:
            val = r.get(mc)
            v = r.get(f"{mc}__valid", 0)
            t = r.get(f"{mc}__total", 0)
            if val is not None:
                cell_text = f"{val:.4f} ({v}/{t})"
                cell = f"{GREEN}{cell_text}{NOCOLOR}"
            else:
                cell_text = f"-- ({v}/{t})"
                cell = f"{RED}{cell_text}{NOCOLOR}"
            # Right-align by visible width (color codes are invisible)
            pad = max(0, col_w - len(cell_text))
            line += "  " + " " * pad + cell
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
            'exp/*/inference/*/eval-*',
            'exp/*/test_clean/*',
            # "exp/*-1000k/inference/*/eval-test_clean-v1-*",
            # "exp/opuslm_v2_stage2_pretrain_base/inference/inference_audio_step_350000/eval-test_clean-v1-*",
            # "exp/ct-100k-default-mt/inference/*/eval-test_clean-v1-*",
            # "exp/ct-100k-default-c2a/inference/*/eval-test_clean-v1-*",
            # "exp/minguniaudioedit/test_clean/speech_edit",
            # "exp/minguniaudioedit/test_clean/speech_edit-short",
            # "exp/stepaudiox/test_clean/speech_edit",
            # "exp/stepaudiox/test_clean/speech_edit-short",
            # "exp/cv3/test_clean/speech_edit",
            # "exp/cv3/test_clean/speech_edit-short",
        ],
        help="Glob patterns matching directories that contain .results files"
    )
    parser.add_argument(
        "--wer_filter", type=float, default=0.,
        help="Exclude samples with WER > this value (0 = disabled). "
             "Exclusion applies to ALL metrics for fairness."
    )
    parser.add_argument(
        "--wer_datasets", nargs="*", default=None,
        metavar="DATASET",
        help="Dataset names to include in WER histogram display. "
             "Pass no value (--wer_datasets) to show ALL. "
             f"Default: {WER_GRAPH_DATASETS}"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/summary",
        help="Output directory for per-dataset TSV files"
    )
    args = parser.parse_args()
    wer_filter = args.wer_filter if args.wer_filter > 0 else None
    # --wer_datasets not passed → use module default; --wer_datasets (no values) → all
    wer_datasets: list[str] = (
        WER_GRAPH_DATASETS if args.wer_datasets is None else args.wer_datasets
    )

    wer_info = (f"{RED}>{wer_filter} excluded{NOCOLOR}" if wer_filter
                else f"{GREEN}disabled{NOCOLOR}")
    wer_ds_display = (f"{GREEN}all{NOCOLOR}" if not wer_datasets
                      else CYAN + ", ".join(wer_datasets) + NOCOLOR)
    print(f"{CYAN}Input globs:{NOCOLOR} {args.input_globs}")
    print(f"{CYAN}WER filter: {NOCOLOR} {wer_info}")
    print(f"{CYAN}WER datasets:{NOCOLOR} {wer_ds_display}")
    print(f"{CYAN}Output dir: {NOCOLOR} {YELLOW}{args.output_dir}{NOCOLOR}")

    tables = collect_all(args.input_globs, wer_filter=wer_filter)

    if not tables:
        print(f"{RED}ERROR: No results found.{NOCOLOR}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{GREEN}Found {len(tables)} datasets, "
          f"{sum(len(v) for v in tables.values())} total rows{NOCOLOR}")

    for dataset in sorted(tables):
        rows = tables[dataset]
        outpath = write_dataset_tsv(dataset, rows, args.output_dir)
        print_dataset_table(dataset, rows)
        print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{outpath}{NOCOLOR}")

    # ---- Grouped (merged) datasets ----
    grouped_tables = collect_all_grouped(
        args.input_globs, wer_filter=wer_filter
    )
    if grouped_tables:
        print(MAGENTA + f"\n{'#' * 100}" + NOCOLOR)
        print(f"  {BOLD}{YELLOW}GROUPED DATASETS{NOCOLOR}  "
              f"(merged by DATASET_GROUP)")
        print(MAGENTA + f"{'#' * 100}" + NOCOLOR)
        grouped_dir = os.path.join(args.output_dir, "grouped")
        for group in sorted(grouped_tables):
            rows = grouped_tables[group]
            outpath = write_dataset_tsv(group, rows, grouped_dir)
            print_dataset_table(f"[GROUP] {group}", rows)
            print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{outpath}{NOCOLOR}")

    # ---- WER distribution histogram ----
    wer_tables = collect_wer_distributions(args.input_globs)
    if wer_tables:
        print(MAGENTA + f"\n{'#' * 100}" + NOCOLOR)
        print(f"  {BOLD}{YELLOW}WER DISTRIBUTION HISTOGRAM{NOCOLOR}")
        print(MAGENTA + f"{'#' * 100}" + NOCOLOR)
        for dataset in sorted(wer_tables):
            if wer_datasets and dataset not in wer_datasets:
                continue
            wer_rows = wer_tables[dataset]
            print_wer_histogram(dataset, wer_rows)
            wer_tsv = write_wer_distribution_tsv(
                dataset, wer_rows, args.output_dir
            )
            print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{wer_tsv}{NOCOLOR}")

    # ---- WER distribution histogram (grouped) ----
    wer_grouped = collect_wer_distributions_grouped(args.input_globs)
    if wer_grouped:
        print(MAGENTA + f"\n{'#' * 100}" + NOCOLOR)
        print(f"  {BOLD}{YELLOW}WER DISTRIBUTION HISTOGRAM (GROUPED){NOCOLOR}")
        print(MAGENTA + f"{'#' * 100}" + NOCOLOR)
        grouped_dir = os.path.join(args.output_dir, "grouped")
        for group in sorted(wer_grouped):
            if wer_datasets and group not in wer_datasets:
                continue
            wer_rows = wer_grouped[group]
            print_wer_histogram(f"[GROUP] {group}", wer_rows)
            wer_tsv = write_wer_distribution_tsv(
                group, wer_rows, grouped_dir
            )
            print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{wer_tsv}{NOCOLOR}")

    # ---- WER line chart (matplotlib) ----
    all_wer = {**wer_tables, **wer_grouped}   # merged + ungrouped
    chart_dataset = WER_LINE_CHART_DATASET
    chart_rows = all_wer.get(chart_dataset, [])
    if chart_rows and WER_LINE_CHART_LABELS:
        print(MAGENTA + f"\n{'#' * 100}" + NOCOLOR)
        print(f"  {BOLD}{YELLOW}WER LINE CHART{NOCOLOR}  "
              f"(dataset: {CYAN}{chart_dataset}{NOCOLOR})")
        print(MAGENTA + f"{'#' * 100}" + NOCOLOR)
        chart_path = plot_wer_line_chart(
            chart_rows, chart_dataset,
            WER_LINE_CHART_LABELS, args.output_dir,
        )
        if chart_path:
            print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{chart_path}{NOCOLOR}")

    # ---- WER histogram comparison subplots (matplotlib) ----
    if WER_HIST_CMP_LABELS:
        hist_dataset = WER_HIST_CMP_DATASET
        raw_wer_grouped = collect_raw_wer_scores(args.input_globs)
        raw_scores = raw_wer_grouped.get(hist_dataset, {})
        if raw_scores:
            print(MAGENTA + f"\n{'#' * 100}" + NOCOLOR)
            print(f"  {BOLD}{YELLOW}WER HISTOGRAM COMPARISON{NOCOLOR}  "
                  f"(dataset: {CYAN}{hist_dataset}{NOCOLOR})")
            print(MAGENTA + f"{'#' * 100}" + NOCOLOR)
            hist_path = plot_wer_hist_comparison(
                raw_scores, hist_dataset,
                WER_HIST_CMP_LABELS, args.output_dir,
            )
            if hist_path:
                print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{hist_path}{NOCOLOR}")
        else:
            print(f"  {RED}No raw WER scores found for dataset '{hist_dataset}' "
                  f"— cannot plot histogram{NOCOLOR}", file=sys.stderr)

# ============================================================================
# WER histogram comparison (matplotlib)
# ============================================================================

def collect_raw_wer_scores(
    input_globs: list[str],
    group_map: dict[str, str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Collect raw per-sample WER scores for every dataset (ungrouped)
    AND for every merged group (grouped).

    Returns: { dataset_or_group_name: { dir_label: [wer_score, ...] } }
    """
    if group_map is None:
        group_map = DATASET_GROUP

    raw = _collect_raw_samples(input_globs)

    def _extract_wer(samples: list[dict]) -> list[float]:
        out: list[float] = []
        for s in samples:
            wer_data = s.get("metrics", {}).get("asr_wer", {})
            if wer_data.get("valid") and wer_data.get("score") is not None:
                out.append(float(wer_data["score"]))
        return out

    # ---- ungrouped (per raw dataset) ----
    ungrouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # ---- grouped ----
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_members: dict[str, set[str]] = defaultdict(set)

    for info, dataset, samples in raw:
        scores = _extract_wer(samples)
        if not scores:
            continue
        # ungrouped: always store under original dataset name
        ungrouped[dataset][info["label"]].extend(scores)
        # grouped: merge into group if mapped
        group = group_map.get(dataset)
        if group is not None:
            group_members[group].add(dataset)
            grouped[group][info["label"]].extend(scores)

    # Only keep groups with ≥2 member datasets
    active_groups = {g for g, m in group_members.items() if len(m) >= 2}

    result: dict[str, dict[str, list[float]]] = dict(ungrouped)
    for group, label_scores in grouped.items():
        if group in active_groups:
            result[group] = dict(label_scores)

    return result

def _bin_wer_scores(scores: list[float]) -> list[int]:
    """Bin a list of WER scores according to WER_BINS. Returns counts per bin."""
    counts = [0] * len(WER_BINS)
    for s in scores:
        for bi, (_label, lo, hi) in enumerate(WER_BINS):
            if lo <= s < hi:
                counts[bi] += 1
                break
    return counts


def plot_wer_hist_comparison(
    raw_scores: dict[str, list[float]],
    dataset: str,
    label_map: dict[str, str],
    output_dir: str,
) -> str | None:
    """Plot N side-by-side subplots, one histogram per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping histogram", file=sys.stderr)
        return None

    names: list[str] = []
    binned: list[list[int]] = []

    for orig_label, display_name in label_map.items():
        scores = raw_scores.get(orig_label)
        if not scores:
            continue
        names.append(display_name)
        binned.append(_bin_wer_scores(scores))

    if not binned:
        return None

    n_models = len(binned)
    n_bins = len(WER_BINS)
    bin_labels = [b[0] for b in WER_BINS]

    # Shared y-max across all subplots
    global_max = max(max(c) for c in binned)
    y_max = int(global_max * 1.15) + 1

    model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for mi, ax in enumerate(axes):
        counts = binned[mi]
        color = model_colors[mi % len(model_colors)]
        bars = ax.bar(range(n_bins), counts, color=color, edgecolor="black", linewidth=0.2)

        # percentage labels on top of each bar
        total = sum(counts)
        for bi, bar in enumerate(bars):
            h = bar.get_height()
            pct = h / total * 100 if total else 0
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_max * 0.01,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=12)

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, fontsize=16, rotation=24, ha="right")
        ax.set_yticks([])
        ax.set_ylim(0, y_max * 1.01)
        # ax.tick_params(axis="y", which="major", labelsize=16, rotation=30)
        ax.set_title(names[mi], fontsize=30)
        ax.grid(True, axis="y", alpha=0.2, linestyle=":")

        if mi == 0:
            ax.set_ylabel("#Samples", fontsize=30)

    # fig.suptitle(f"WER Distribution — {dataset}", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{dataset}_wer_hist_cmp.png")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ============================================================================
# WER line chart (matplotlib)
# ============================================================================

def plot_wer_line_chart(
    wer_rows: list[dict],
    dataset: str,
    label_map: dict[str, str],
    output_dir: str,
) -> str | None:
    """
    Draw a line chart of WER-bin percentages for selected experiments.

    Parameters
    ----------
    wer_rows   : rows from collect_wer_distributions_grouped[dataset]
    dataset    : dataset / group name (used in title & filename)
    label_map  : {parse_eval_dir label → display name}  — also controls plot order
    output_dir : directory to save the .png

    Returns the path to the saved figure, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {RED}matplotlib not installed — skipping line chart{NOCOLOR}",
              file=sys.stderr)
        return None

    if not wer_rows:
        print(f"  {RED}No WER data for {dataset} — skipping line chart{NOCOLOR}")
        return None

    bin_labels = [label for label, _, _ in WER_BINS]

    # Build a quick lookup:  label_str → row
    row_by_label: dict[str, dict] = {r["label"]: r for r in wer_rows}

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    plotted = 0
    for idx, (orig_label, display_name) in enumerate(label_map.items()):
        row = row_by_label.get(orig_label)
        if row is None:
            print(f"  {YELLOW}WARN: label '{orig_label}' not found in "
                  f"{dataset} WER data — skipping{NOCOLOR}")
            continue
        n = row["n_with_wer"]
        if n == 0:
            continue
        pcts = [100.0 * row["bin_counts"].get(bl, 0) / n for bl in bin_labels]
        marker = markers[idx % len(markers)]
        ax.plot(bin_labels, pcts, marker=marker, label=display_name,
                linewidth=2, markersize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"  {RED}No matching experiments for line chart{NOCOLOR}")
        return None

    ax.set_xlabel("WER Range", fontsize=13)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_title(f"WER Distribution — {dataset}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{dataset}_wer_line_chart.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    main()
