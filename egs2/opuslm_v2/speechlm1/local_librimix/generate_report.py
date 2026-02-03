#!/usr/bin/env python3
"""
Generate evaluation report in various formats.

This script combines evaluation results and generates:
- Markdown report
- LaTeX table
- CSV for spreadsheet import

Usage:
    python generate_report.py --results_dir eval_results/ --output_dir reports/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_summary(results_dir: str) -> Dict:
    """Load summary from results directory."""
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def load_wer_summary(results_dir: str) -> Dict:
    """Load WER summary if available."""
    wer_path = os.path.join(results_dir, "wer_summary.json")
    if os.path.exists(wer_path):
        with open(wer_path, 'r') as f:
            return json.load(f)
    return {}


def generate_markdown_report(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "report.md",
) -> None:
    """Generate a Markdown report."""
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    with open(output_path, 'w') as f:
        f.write("# Speech Enhancement Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Collect all data
        all_summaries = {}
        all_metrics = set()
        for label, results_dir in zip(labels, results_dirs):
            summary = load_summary(results_dir)
            wer_summary = load_wer_summary(results_dir)
            summary.update(wer_summary)
            all_summaries[label] = summary
            all_metrics.update(summary.keys())
        
        # Summary table
        f.write("## Summary\n\n")
        
        # Header
        header = "| Metric |"
        separator = "|--------|"
        for label in labels:
            header += f" {label} |"
            separator += "--------|"
        f.write(header + "\n")
        f.write(separator + "\n")
        
        # Rows
        for metric in sorted(all_metrics):
            row = f"| {metric} |"
            for label in labels:
                if label in all_summaries and metric in all_summaries[label]:
                    mean = all_summaries[label][metric]["mean"]
                    std = all_summaries[label][metric]["std"]
                    row += f" {mean:.4f} ± {std:.4f} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        
        # Individual experiment details
        f.write("\n## Experiment Details\n\n")
        for label, results_dir in zip(labels, results_dirs):
            f.write(f"### {label}\n\n")
            f.write(f"- **Path**: `{results_dir}`\n")
            
            summary = all_summaries.get(label, {})
            if summary:
                f.write("\n| Metric | Mean | Std | Min | Max | Count |\n")
                f.write("|--------|------|-----|-----|-----|-------|\n")
                for metric in sorted(summary.keys()):
                    stats = summary[metric]
                    f.write(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                           f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n")
            f.write("\n")
    
    print(f"Markdown report saved to: {output_path}")


def generate_latex_table(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "table.tex",
    metrics: Optional[List[str]] = None,
) -> None:
    """Generate a LaTeX table."""
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # Collect data
    all_summaries = {}
    all_metrics = set()
    for label, results_dir in zip(labels, results_dirs):
        summary = load_summary(results_dir)
        wer_summary = load_wer_summary(results_dir)
        summary.update(wer_summary)
        all_summaries[label] = summary
        all_metrics.update(summary.keys())
    
    if metrics:
        all_metrics = set(metrics) & all_metrics
    
    with open(output_path, 'w') as f:
        # Table header
        num_cols = len(labels) + 1
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{l{'c' * len(labels)}}}\n")
        f.write("\\toprule\n")
        
        # Header row
        f.write("Metric")
        for label in labels:
            f.write(f" & {label}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows
        for metric in sorted(all_metrics):
            row = metric.replace("_", "\\_")
            for label in labels:
                if label in all_summaries and metric in all_summaries[label]:
                    mean = all_summaries[label][metric]["mean"]
                    std = all_summaries[label][metric]["std"]
                    row += f" & {mean:.3f}$\\pm${std:.3f}"
                else:
                    row += " & -"
            f.write(row + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Speech Enhancement Evaluation Results}\n")
        f.write("\\label{tab:enh_results}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {output_path}")


def generate_csv(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "results.csv",
) -> None:
    """Generate a CSV file."""
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # Collect data
    all_summaries = {}
    all_metrics = set()
    for label, results_dir in zip(labels, results_dirs):
        summary = load_summary(results_dir)
        wer_summary = load_wer_summary(results_dir)
        summary.update(wer_summary)
        all_summaries[label] = summary
        all_metrics.update(summary.keys())
    
    with open(output_path, 'w') as f:
        # Header
        header = "experiment,metric,mean,std,min,max,count"
        f.write(header + "\n")
        
        # Data rows
        for label in labels:
            summary = all_summaries.get(label, {})
            for metric in sorted(summary.keys()):
                stats = summary[metric]
                f.write(f"{label},{metric},{stats['mean']},{stats['std']},"
                       f"{stats['min']},{stats['max']},{stats['count']}\n")
    
    print(f"CSV file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation report in various formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        required=True,
        help="Result directory/directories",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each result directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["markdown", "csv"],
        choices=["markdown", "latex", "csv", "all"],
        help="Output formats",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to include (default: all)",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    formats = args.format
    if "all" in formats:
        formats = ["markdown", "latex", "csv"]
    
    if "markdown" in formats:
        generate_markdown_report(
            args.results_dir,
            args.labels,
            os.path.join(args.output_dir, "report.md"),
        )
    
    if "latex" in formats:
        generate_latex_table(
            args.results_dir,
            args.labels,
            os.path.join(args.output_dir, "table.tex"),
            args.metrics,
        )
    
    if "csv" in formats:
        generate_csv(
            args.results_dir,
            args.labels,
            os.path.join(args.output_dir, "results.csv"),
        )


if __name__ == "__main__":
    main()
