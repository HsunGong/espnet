#!/usr/bin/env python3
"""Generate a filtered test script based on YAML keys and keywords."""

import argparse
import yaml
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate filtered test script from YAML keys.")
    parser.add_argument("--input-yaml", type=Path, default="/mnt/home/xungong-andr-1766e0/opuslm_sft/egs2/opuslm_v2/speechlm1/data/part2_4/debug/data.yaml",
                        help="Path to the input YAML file.")
    parser.add_argument("--include", nargs='*', default=[],
                        help="Keywords to filter keys (space-separated). If empty, include all keys.")
    parser.add_argument("--exclude", nargs='*', default=[],
                        help="Keywords to filter keys (space-separated). If empty, include all keys.")
    parser.add_argument("--output-sh", type=Path, required=True,
                        help="Path to the output shell script.")
    parser.add_argument("--template-sh", type=Path, default="runs/run-pretrain-audio.sh",
                        help="Path to the template shell script.")
    args = parser.parse_args()

    # Read YAML and get keys
    with open(args.input_yaml, 'r') as f:
        data = yaml.safe_load(f)
    all_keys = list(data.keys())

    # Filter keys
    if args.include:
        all_keys = [key for key in all_keys if any(kw in key for kw in args.include)]
    if args.exclude:
        all_keys = [key for key in all_keys if not any(kw in key for kw in args.exclude)]

    print(f"Total keys: {len(all_keys)}, Filtered: {len(all_keys)}")
    print("Filtered keys:", all_keys[:10], "..." if len(all_keys) > 10 else "")

    # Read template script
    with open(args.template_sh, 'r') as f:
        template_content = f.read()

    # Generate new test_sets
    test_sets_lines = []
    for key in all_keys:
        test_sets_lines.append(f'test_sets+="dialogue:{key} "')

    new_test_sets = '\n'.join(test_sets_lines)

    # Replace test_sets in template
    # Find the test_sets block
    pattern = r'(test_sets=""\n)(.*?)(\n\nbash)'
    replacement = r'\1' + new_test_sets + r'\n\3'
    new_content = re.sub(pattern, replacement, template_content, flags=re.DOTALL)

    # Write to output
    args.output_sh.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_sh, 'w') as f:
        f.write(new_content)

    print(f"Generated script: {args.output_sh}")

if __name__ == "__main__":
    main()