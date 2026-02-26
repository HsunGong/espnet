#!/usr/bin/env python3
"""Quick check of .results format."""
import json, sys

fpath = sys.argv[1]
with open(fpath) as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        d = json.loads(line)
        print(f"Sample {i}: id={d['id']}, metrics={list(d['metrics'].keys())}")
        for k, v in d['metrics'].items():
            print(f"  {k}: score={v.get('score')}, valid={v.get('valid')}, error={v.get('error')}")
