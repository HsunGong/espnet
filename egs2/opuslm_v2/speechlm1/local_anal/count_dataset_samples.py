#!/usr/bin/env python3
"""
Read all key~json-path pairs from ESPNET_DATASET_REGISTRY (colon-separated list of
YAML files), then for each dataset JSON count the number of available samples
(data["samples"]) and print the results.

Usage:
    # Make sure ESPNET_DATASET_REGISTRY is set, e.g. by sourcing path.sh first:
    source path.sh && python local_anal/count_dataset_samples.py
    source path.sh && python local_anal/count_dataset_samples.py --min-samples 1000
    source path.sh && python local_anal/count_dataset_samples.py --min-samples 100 --max-samples 500000
"""

import argparse
import json
import os
import sys

import yaml


def load_registry(registry_env: str):
    """Parse ESPNET_DATASET_REGISTRY into {dataset_key: json_path} dict."""
    datasets = {}
    yaml_paths = [p.strip() for p in registry_env.split(":") if p.strip()]
    for yaml_path in yaml_paths:
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(os.getcwd(), yaml_path)
        if not os.path.isfile(yaml_path):
            print(f"[WARN] YAML not found: {yaml_path}", file=sys.stderr)
            continue
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            print(f"[WARN] Unexpected YAML format in {yaml_path}", file=sys.stderr)
            continue
        for key, val in content.items():
            if isinstance(val, dict):
                json_path = val.get("path", None)
            elif isinstance(val, str):
                json_path = val
            else:
                json_path = None
            if json_path:
                datasets[key] = json_path
    return datasets


def count_samples(json_path: str):
    """Return number of samples in a dataset JSON, or None on error."""
    if not os.path.isfile(json_path):
        return None, f"file not found: {json_path}"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        return len(samples), None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Count samples per dataset from ESPNET_DATASET_REGISTRY."
    )
    parser.add_argument(
        "--min-samples", type=int, default=None, metavar="N",
        help="Ignore datasets with fewer than N samples.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Ignore datasets with more than N samples.",
    )
    args = parser.parse_args()

    registry_env = os.environ.get("ESPNET_DATASET_REGISTRY", "")
    if not registry_env:
        print("ERROR: ESPNET_DATASET_REGISTRY is not set.", file=sys.stderr)
        print("Please source path.sh first:", file=sys.stderr)
        print("  source path.sh && python local_anal/count_dataset_samples.py", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing ESPNET_DATASET_REGISTRY ({registry_env.count(':') + 1} YAML files) ...\n")
    if args.min_samples is not None or args.max_samples is not None:
        lo = args.min_samples if args.min_samples is not None else 0
        hi = args.max_samples if args.max_samples is not None else float('inf')
        print(f"Filter: keeping datasets with sample count in [{lo}, {hi}]\n")

    datasets = load_registry(registry_env)
    print(f"Total datasets found: {len(datasets)}")

    # Count samples per dataset
    results = []
    errors = []
    skipped = []
    for key, json_path in datasets.items():
        count, err = count_samples(json_path)
        if err:
            errors.append((key, json_path, err))
            results.append((key, json_path, -1))
        else:
            # Apply min/max filter
            if args.min_samples is not None and count < args.min_samples:
                skipped.append((key, count, "below min"))
                continue
            if args.max_samples is not None and count > args.max_samples:
                skipped.append((key, count, "above max"))
                continue
            results.append((key, json_path, count))

    if skipped:
        print(f"Skipped (out-of-range): {len(skipped)} datasets\n")
    else:
        print()

    # Sort by count descending (errors last)
    results.sort(key=lambda x: x[2] if x[2] >= 0 else -2, reverse=True)

    # Print header
    col_w = 80
    print(f"{'Dataset Key':<{col_w}}  {'#Samples':>10}  {'JSON Path'}")
    print("-" * (col_w + 15 + 80))

    total = 0
    for key, json_path, count in results:
        if count >= 0:
            total += count
            count_str = f"{count:>10,}"
        else:
            count_str = f"{'ERROR':>10}"
        print(f"{key:<{col_w}}  {count_str}  {json_path}")

    print("-" * (col_w + 15 + 80))
    print(f"{'TOTAL':<{col_w}}  {total:>10,}")

    if errors:
        print(f"\n[WARN] {len(errors)} datasets had errors:")
        for key, json_path, err in errors:
            print(f"  {key}: {err}")


if __name__ == "__main__":
    main()
