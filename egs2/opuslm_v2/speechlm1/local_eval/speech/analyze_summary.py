#!/usr/bin/env python3
"""
Analyze per-dataset TSV files produced by collect_summary.py.
Output in Markdown format (tables), optionally written to a .md file.

Usage:
    python3 local_eval/speech/analyze_summary.py \\
        --input_dir results/summary

    python3 local_eval/speech/analyze_summary.py \\
        --input_dir results/summary \\
        --output results/summary/report.md \\
        --datasets transcription_ins audio_effect_speed
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================================
# Read TSV
# ============================================================================

def read_dataset_tsv(filepath: str) -> tuple[list[str], list[dict]]:
    """
    Read a dataset TSV produced by collect_summary.py.

    Returns:
        metric_cols: list of metric column names (excluding (v/t) columns)
        rows: list of row dicts with keys: label, experiment, checkpoint,
              eval_mode, step, n_samples, + metric values as floats
    """
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []

        meta = {"label", "experiment", "checkpoint", "eval_mode", "step", "n_samples"}
        metric_cols = [h for h in header if h not in meta and not h.endswith("(v/t)")]

        rows = []
        for raw in reader:
            row: dict = {
                "label": raw["label"],
                "experiment": raw["experiment"],
                "checkpoint": raw.get("checkpoint", ""),
                "eval_mode": raw.get("eval_mode", ""),
                "step": raw.get("step", "0"),
                "n_samples": raw.get("n_samples", "0"),
            }
            for mc in metric_cols:
                val_str = raw.get(mc, "")
                row[mc] = float(val_str) if val_str else None
                row[f"{mc}(v/t)"] = raw.get(f"{mc}(v/t)", "")
            rows.append(row)

    return metric_cols, rows


# ============================================================================
# Markdown helpers
# ============================================================================

def _md_row(cells: list[str]) -> str:
    """Build a markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _md_sep(n: int) -> str:
    """Build a markdown separator row: first col left-aligned, rest right."""
    return "| :--- | " + " | ".join(["---:"] * (n - 1)) + " |"


# ============================================================================
# Per-dataset table (markdown)
# ============================================================================

def format_table(dataset: str, metric_cols: list[str], rows: list[dict]) -> list[str]:
    """Format one dataset as markdown lines."""
    if not rows:
        return []

    lines: list[str] = []
    lines.append(f"## {dataset}")
    lines.append("")

    # Find best values per metric for bold highlighting
    best: dict[str, tuple[float, bool]] = {}
    for mc in metric_cols:
        vals = [r[mc] for r in rows if r[mc] is not None]
        if not vals:
            continue
        lower_better = "wer" in mc.lower() or "distance" in mc.lower()
        best[mc] = (min(vals) if lower_better else max(vals), lower_better)

    # Header row
    header_cells = ["label"] + metric_cols
    lines.append(_md_row(header_cells))
    lines.append(_md_sep(len(header_cells)))

    # Data rows
    for r in sorted(rows, key=lambda x: (x["experiment"], x["step"], x["eval_mode"])):
        cells = [r["label"]]
        for mc in metric_cols:
            val = r[mc]
            vt = r.get(f"{mc}(v/t)", "")
            if val is not None:
                is_best = mc in best and val == best[mc][0]
                num = f"{val:.4f}"
                if is_best:
                    num = f"**{num}**"
                cells.append(f"{num} ({vt})")
            else:
                cells.append(f"-- ({vt})")
        lines.append(_md_row(cells))

    lines.append("")
    lines.append("> **bold** = best in column")
    lines.append("")
    return lines


# ============================================================================
# Ranking
# ============================================================================

DEFAULT_RANK_METRICS = {
    "asr_wer":                       (0.30, True),
    "speaker_similarity_wespeaker":  (0.00, False),
    "speaker_similarity_wespeaker.sim":  (0.00, False),
    "speaker_similarity_wavlm":     (0.00, False),
    "speaker_similarity_wavlm.sim": (0.05, False),
    "pseudo_mos":                    (0.15, False),
    "pseudo_mos.utmos":              (0.02, False),
    "pseudo_mos.dns_overall":       (0.02, False),
    "llm_judge_caption_llm":        (0.10, False),
    "emotion_modelscope":           (0.05, False),
    "asr_wer.edit_acc":             (0.05, False),
    "speed_duration":               (0.00, False),
    "volume_loudness":              (0.00, False),
    "pitch_shift":                  (0.00, False),
}


def _normalize_score(val: float | None, ascending: bool) -> float:
    if val is None:
        return 0.0
    if ascending:
        return 1.0 / (1.0 + val / 100.0)
    else:
        if val > 1.0:
            return val / 10.0
        return val


def format_ranking(metric_cols: list[str], rows: list[dict]) -> list[str]:
    """Compute composite ranking and return markdown lines."""
    active_metrics = {}
    total_weight = 0.0
    for mc in metric_cols:
        if mc in DEFAULT_RANK_METRICS:
            w, asc = DEFAULT_RANK_METRICS[mc]
            if w > 0:
                active_metrics[mc] = (w, asc)
                total_weight += w

    if not active_metrics:
        return []

    for mc in active_metrics:
        w, asc = active_metrics[mc]
        active_metrics[mc] = (w / total_weight if total_weight > 0 else 0, asc)

    scored_rows = []
    for r in rows:
        composite = 0.0
        for mc, (w, asc) in active_metrics.items():
            composite += w * _normalize_score(r[mc], asc)
        scored_rows.append((composite, r))
    scored_rows.sort(key=lambda x: -x[0])

    lines: list[str] = []

    metric_str = ", ".join(
        f"`{mc}`({'↑' if not asc else '↓'})={w:.0%}"
        for mc, (w, asc) in sorted(active_metrics.items(), key=lambda x: -x[1][0])
    )
    lines.append(f"<details><summary>Ranking weights: {metric_str}</summary>")
    lines.append("")

    sorted_mc = sorted(active_metrics.keys())
    header_cells = ["Rank", "label", "composite"] + sorted_mc
    lines.append(_md_row(header_cells))
    lines.append(_md_sep(len(header_cells)))

    for i, (comp, r) in enumerate(scored_rows, 1):
        rank_str = f"**{i}**" if i <= 3 else str(i)
        cells = [rank_str, r["label"], f"{comp:.4f}"]
        for mc in sorted_mc:
            val = r[mc]
            cells.append(f"{val:.4f}" if val is not None else "--")
        lines.append(_md_row(cells))

    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


# ============================================================================
# Global cross-dataset summary (markdown)
# ============================================================================

def format_global_summary(
    global_scores: dict[str, dict[str, list[float]]],
) -> list[str]:
    """Format the global cross-dataset summary as markdown."""
    if not global_scores:
        return []

    lines: list[str] = []
    lines.append("---")
    lines.append("")
    lines.append("## Global Cross-Dataset Summary")
    lines.append("")
    lines.append("Values are averaged across all datasets. `(Nds)` = number of datasets.")
    lines.append("")

    all_global_metrics: set[str] = set()
    for label_data in global_scores.values():
        all_global_metrics.update(label_data.keys())
    global_metric_cols = sorted(all_global_metrics)

    labels = sorted(global_scores.keys())

    # Find best values for bold highlighting
    best: dict[str, tuple[float, bool]] = {}
    for mc in global_metric_cols:
        vals = []
        for lb in labels:
            v_list = global_scores[lb].get(mc, [])
            if v_list:
                vals.append(sum(v_list) / len(v_list))
        if not vals:
            continue
        lower_better = "wer" in mc.lower() or "distance" in mc.lower()
        best[mc] = (min(vals) if lower_better else max(vals), lower_better)

    header_cells = ["label"] + global_metric_cols
    lines.append(_md_row(header_cells))
    lines.append(_md_sep(len(header_cells)))

    global_rows = []
    for lb in labels:
        row_data: dict = {"label": lb}
        cells = [lb]
        for mc in global_metric_cols:
            vals = global_scores[lb].get(mc, [])
            if vals:
                avg = sum(vals) / len(vals)
                row_data[mc] = avg
                is_best = mc in best and avg == best[mc][0]
                num = f"{avg:.4f}"
                if is_best:
                    num = f"**{num}**"
                cells.append(f"{num} ({len(vals)}ds)")
            else:
                row_data[mc] = None
                cells.append("--")
        global_rows.append(row_data)
        lines.append(_md_row(cells))

    lines.append("")

    # Global ranking
    active_metrics = {}
    total_weight = 0.0
    for mc in global_metric_cols:
        if mc in DEFAULT_RANK_METRICS:
            w, asc = DEFAULT_RANK_METRICS[mc]
            if w > 0:
                active_metrics[mc] = (w, asc)
                total_weight += w

    if active_metrics:
        for mc in active_metrics:
            w, asc = active_metrics[mc]
            active_metrics[mc] = (w / total_weight if total_weight > 0 else 0, asc)

        lines.append("### Global Ranking")
        lines.append("")

        scored = []
        for rd in global_rows:
            composite = 0.0
            for mc, (w, asc) in active_metrics.items():
                composite += w * _normalize_score(rd[mc], asc)
            scored.append((composite, rd))
        scored.sort(key=lambda x: -x[0])

        header_cells = ["Rank", "label", "composite"]
        lines.append(_md_row(header_cells))
        lines.append(_md_sep(len(header_cells)))
        for i, (comp, rd) in enumerate(scored, 1):
            rank_str = f"**{i}**" if i <= 3 else str(i)
            lines.append(_md_row([rank_str, rd["label"], f"{comp:.4f}"]))
        lines.append("")

    return lines


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-dataset TSV files from collect_summary.py. "
                    "Output in Markdown format."
    )
    parser.add_argument(
        "--input_dir", type=str, default="results/summary",
        help="Directory containing per-dataset TSV files"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific datasets to analyze (default: all found TSVs)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write markdown output to this file (default: stdout only)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    tsv_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".tsv")
    )
    if not tsv_files:
        print(f"ERROR: No TSV files in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.datasets:
        allowed = set(args.datasets)
        tsv_files = [f for f in tsv_files if f[:-4] in allowed]

    # Collect all markdown output
    md_lines: list[str] = []
    md_lines.append(f"# Speech-Edit Evaluation Report")
    md_lines.append("")
    md_lines.append(f"Source: `{args.input_dir}/` — {len(tsv_files)} dataset(s)")
    md_lines.append("")

    global_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for tsv_file in tsv_files:
        dataset = tsv_file[:-4]
        filepath = os.path.join(args.input_dir, tsv_file)
        metric_cols, rows = read_dataset_tsv(filepath)

        md_lines.extend(format_table(dataset, metric_cols, rows))
        md_lines.extend(format_ranking(metric_cols, rows))

        for r in rows:
            for mc in metric_cols:
                if r[mc] is not None:
                    global_scores[r["label"]][mc].append(r[mc])

    if len(tsv_files) > 1 and global_scores:
        md_lines.extend(format_global_summary(global_scores))

    output_text = "\n".join(md_lines)

    # Print to stdout
    print(output_text)

    # Optionally write to file
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
        print(f"\n>> Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
