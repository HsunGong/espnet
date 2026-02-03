#!/usr/bin/env python3
"""
Visualize and analyze enhancement evaluation results.

This script provides:
1. Summary statistics
2. Per-metric histograms
3. Improvement analysis
4. Comparison between different checkpoints

Usage:
    python analyze_results.py --results_dir eval_results/ --output_dir plots/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_results(results_dir: str) -> Dict:
    """Load results from a results.json file."""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def load_summary(results_dir: str) -> Dict:
    """Load summary from a summary.json file."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def compare_checkpoints(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Compare evaluation results across multiple checkpoints/experiments.
    
    Args:
        results_dirs: List of result directories to compare
        labels: Labels for each result directory
        metrics: Metrics to compare (default: all common metrics)
    """
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # Load all summaries
    summaries = {}
    for label, results_dir in zip(labels, results_dirs):
        try:
            summaries[label] = load_summary(results_dir)
        except FileNotFoundError as e:
            print(f"Warning: Skipping {label}: {e}")
    
    if not summaries:
        print("No valid results found")
        return
    
    # Find common metrics
    all_metrics = set()
    for summary in summaries.values():
        all_metrics.update(summary.keys())
    
    if metrics:
        all_metrics = set(metrics) & all_metrics
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    
    # Header
    header = f"{'Metric':<25}"
    for label in labels:
        if label in summaries:
            header += f" {label[:15]:>15}"
    print(header)
    print("-" * 80)
    
    # Rows
    for metric in sorted(all_metrics):
        row = f"{metric:<25}"
        for label in labels:
            if label in summaries and metric in summaries[label]:
                mean = summaries[label][metric]["mean"]
                std = summaries[label][metric]["std"]
                row += f" {mean:>7.4f}±{std:<5.3f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    
    print("=" * 80)


def analyze_improvements(results: Dict) -> None:
    """Analyze improvement patterns in the results."""
    improvement_metrics = ["stoi_improvement", "si_snr_improvement"]
    
    for metric in improvement_metrics:
        values = []
        for wav_id, result in results.items():
            if metric in result and not np.isnan(result[metric]):
                values.append((wav_id, result[metric]))
        
        if values:
            values.sort(key=lambda x: x[1])
            print(f"\n{metric} Analysis:")
            print(f"  Total samples: {len(values)}")
            print(f"  Improved: {sum(1 for _, v in values if v > 0)} "
                  f"({100*sum(1 for _, v in values if v > 0)/len(values):.1f}%)")
            print(f"  Degraded: {sum(1 for _, v in values if v < 0)} "
                  f"({100*sum(1 for _, v in values if v < 0)/len(values):.1f}%)")
            
            if len(values) >= 5:
                print(f"  Bottom 5:")
                for wav_id, val in values[:5]:
                    print(f"    {wav_id}: {val:.4f}")
                print(f"  Top 5:")
                for wav_id, val in values[-5:]:
                    print(f"    {wav_id}: {val:.4f}")


def plot_histograms(
    results: Dict,
    output_dir: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """Plot histograms for each metric."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect values for each metric
    metric_values = {}
    for wav_id, result in results.items():
        if isinstance(result, dict):
            for metric, value in result.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if metric not in metric_values:
                        metric_values[metric] = []
                    metric_values[metric].append(value)
    
    if metrics:
        metric_values = {k: v for k, v in metric_values.items() if k in metrics}
    
    # Create histograms
    for metric, values in metric_values.items():
        if len(values) < 10:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(values):.4f}')
        ax.axvline(np.median(values), color='g', linestyle='--', 
                   label=f'Median: {np.median(values):.4f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_hist.png'), dpi=150)
        plt.close()
    
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze enhancement evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        required=True,
        help="Result directory/directories to analyze",
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
        default=None,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple result directories",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate histogram plots",
    )
    
    args = parser.parse_args()
    
    if args.compare and len(args.results_dir) > 1:
        compare_checkpoints(args.results_dir, args.labels)
    
    # Analyze each directory
    for i, results_dir in enumerate(args.results_dir):
        label = args.labels[i] if args.labels and i < len(args.labels) else results_dir
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print(f"{'='*60}")
        
        try:
            results = load_results(results_dir)
            summary = load_summary(results_dir)
            
            # Print summary
            print("\nSummary Statistics:")
            print("-" * 40)
            for metric in sorted(summary.keys()):
                stats = summary[metric]
                print(f"  {metric:<20}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            # Analyze improvements
            analyze_improvements(results)
            
            # Generate plots
            if args.plot:
                output_dir = args.output_dir or os.path.join(results_dir, "plots")
                plot_histograms(results, output_dir)
                
        except Exception as e:
            print(f"Error analyzing {results_dir}: {e}")


if __name__ == "__main__":
    main()
