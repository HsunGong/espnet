#!/usr/bin/env python3
"""
Equivalent of run_all.sh, but launches each bash local_split/run.sh call
in parallel via multiprocessing.

Usage (same positional args as run_all.sh, extra args forwarded to run.sh):
    python local_split/run_all.py <expdir> <name_prefix> [extra run.sh args ...]

Example:
    python local_split/run_all.py data/part2_4/full part2_4
"""

import argparse
import subprocess
import sys
import multiprocessing
from pathlib import Path


def run_bash(cmd: list[str]) -> int:
    """Run a bash command, stream output, and return exit code."""
    label = " ".join(cmd[2:4])  # show run.sh + first flag for identification
    print(f"[START] {label}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[FAIL rc={result.returncode}] {label}", flush=True)
    else:
        print(f"[DONE] {label}", flush=True)
    return result.returncode


def build_cmd(expdir: str, yaml_path: str, name_prefix: str,
              data_type: str, vad_tag: str, run_stages: str,
              extra_args: list[str]) -> list[str]:
    tag = f"{vad_tag}-{data_type}"
    return [
        "bash", "local_split/run.sh",
        "--input_jsonl_raw", f"data/part2_4/metadata.{data_type}.jsonl",
        "--expdir", f"{expdir}/{tag}",
        "--yaml_path", yaml_path,
        "--name_prefix", f"{name_prefix}-{tag}",
        "--run_stages", run_stages,
        "--k", "100000",
    ] + extra_args


def run_group(cmds: list[list[str]], group_name: str, nproc: int) -> None:
    print(f"\n=== Running group: {group_name} ({len(cmds)} jobs, nproc={nproc}) ===",
          flush=True)
    with multiprocessing.Pool(processes=nproc) as pool:
        results = pool.map(run_bash, cmds)
    failed = [i for i, rc in enumerate(results) if rc != 0]
    if failed:
        raise RuntimeError(
            f"Group '{group_name}': {len(failed)} job(s) failed: "
            + ", ".join(str(i) for i in failed)
        )
    print(f"=== Group '{group_name}' finished OK ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel version of run_all.sh"
    )
    parser.add_argument("expdir", help="Experiment base directory (e.g. data/part2_4/full)")
    parser.add_argument("name_prefix", help="Name prefix for dataset entries (e.g. part2_4)")
    parser.add_argument(
        "--nproc",
        type=int,
        default=0,
        help="Max parallel jobs per group (0 = number of data_types in that group)",
    )
    args, extra_args = parser.parse_known_args()

    expdir = args.expdir
    name_prefix = args.name_prefix
    yaml_path = f"{expdir}/data.yaml"

    print(f"yaml at {yaml_path}")

    # ---- group 1: no-VAD ------------------------------------------------
    novad_types = [
        "music.min_10",
        "music.min_0.max_10",
        "sound.min_0.max_7",
        "sound.min_7.max_10",
        "sound.min_10",
        "speech.min_3.max_5",
        "speech.min_5.max_8",
        "speech.min_8.max_20",
        "speech.min_20.max_25",
    ]
    novad_cmds = [
        build_cmd(expdir, yaml_path, name_prefix, dt, "novad", "0,4,5", extra_args)
        for dt in novad_types
    ]
    nproc1 = args.nproc or len(novad_cmds)
    run_group(novad_cmds, "novad", nproc1)

    # ---- group 2: with VAD ----------------------------------------------
    vad_types = [
        "music.min_10",
        "sound.min_7.max_10",
        "sound.min_10",
        "speech.min_5.max_8",
        "speech.min_20.max_25",
        "speech.min_25",
    ]
    vad_cmds = [
        build_cmd(expdir, yaml_path, name_prefix, dt, "vad", "1,2,4,5", extra_args)
        for dt in vad_types
    ]
    nproc2 = args.nproc or len(vad_cmds)
    run_group(vad_cmds, "vad", nproc2)

    # ---- post-processing Python scripts ---------------------------------
    print("=== Running post-processing scripts ===", flush=True)
    subprocess.run([
        sys.executable, "local_split/generate_filtered_test_script.py",
        "--template-sh", "runs/run-pretrain-audio-continue.sh",
        "--include", "split1",
        "--output-sh", "runs/test_all_split1.sh",
    ], check=True)
    subprocess.run([
        sys.executable, "local_split/generate_filtered_test_script.py",
        "--template-sh", "runs/run-pretrain-audio.sh",
        "--exclude", "split1",
        "--output-sh", "runs/test_all_normal.sh",
    ], check=True)
    print("=== All done ===", flush=True)


if __name__ == "__main__":
    main()
