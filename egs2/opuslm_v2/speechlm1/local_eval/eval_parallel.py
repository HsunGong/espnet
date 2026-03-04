#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU parallel runner for local_eval.eval.

--metadata is expanded to individual *.jsonl files (via directory listing or
glob).  Each (metadata-jsonl, data-dir) pair becomes a separate parallel task,
so the parallelism is  metadata-files x data-dirs  rather than just data-dirs.
Pairs where the data-dir has no matching .scp file are automatically skipped.

Usage example:
  python local_eval/eval_parallel.py \\
      --gpus 0,1,2,3 --max-workers-per-gpu 1 \\
      --config local_eval/eval/eval.yaml \\
      --metadata data/test_clean/speech_edit \\
      --data-dirs \\
          exp/stepaudiox/test_clean/speech_edit \\
          exp/cv3/test_clean/speech_edit \\
          exp/minguniaudioedit/test_clean/speech_edit

Summary files are written to  <data-dir>.<metadata-stem>.summary  by default.
Use --summary-dir <dir> to redirect all summary files into a single directory.
"""

import argparse
import glob
import os
import shlex
import subprocess
import sys
import threading
import time
from multiprocessing import Process, Queue, current_process


# ---------------------------------------------------------------------------
# GPU list parsing (identical to infer_parallel.py)
# ---------------------------------------------------------------------------

def parse_gpus(gpu_str: str) -> list[str]:
    """
    Supports:
      "0,1,2"
      "0-3"      -> 0,1,2,3
      "0,2-4,7"
    Returns a de-duplicated, order-preserving list of GPU id strings.
    """
    gpu_str = gpu_str.strip()
    if not gpu_str:
        raise ValueError("Empty --gpus")

    parts = [p.strip() for p in gpu_str.split(",") if p.strip()]
    gpus: list[str] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i, b_i = int(a), int(b)
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            gpus.extend([str(i) for i in range(a_i, b_i + 1)])
        else:
            gpus.append(str(int(p)))
    seen: set[str] = set()
    out: list[str] = []
    for g in gpus:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Tee helper
# ---------------------------------------------------------------------------

def _tee_reader(src, sink_file, sink_lock, console_prefix: str) -> None:
    """
    Read lines from *src* (a binary pipe), write each line both to
    stdout (with *console_prefix*) and to *sink_file*.
    Runs in its own daemon thread.
    """
    for raw in iter(src.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        with sink_lock:
            sys.stdout.write(console_prefix + line)
            sys.stdout.flush()
        sink_file.write(line)
        sink_file.flush()


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def worker_loop(
    gpu_id: str,
    task_q: "Queue[tuple[int, str, str] | None]",
    config: str,
    python_bin: str,
    extra_args: list[str],
    summary_dir: str | None,
    no_resume: bool,
    dry_run: bool,
) -> None:
    """Fixed to one GPU; pops tasks from *task_q* until it receives None."""
    proc_name = current_process().name
    # shared lock so lines from different workers don't interleave on stdout
    console_lock = threading.Lock()

    while True:
        item = task_q.get()
        if item is None:
            return

        idx, metadata_file, data_dir = item

        # ---- derive summary file path --------------------------------
        data_dir_clean = data_dir.rstrip("/")
        if not os.path.isdir(data_dir):
            continue

        metadata_stem = os.path.splitext(os.path.basename(metadata_file))[0]

        if summary_dir:
            base_name = os.path.basename(data_dir_clean)
            summary_path = os.path.join(summary_dir, f"{base_name}.{metadata_stem}.summary")
        else:
            summary_path = f"{data_dir_clean}.{metadata_stem}.summary"

        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)

        # ---- build command ------------------------------------------
        cmd = [
            python_bin, "-m", "local_eval.eval",
            "--config", config,
            "--metadata", metadata_file,
            "--data-dir", data_dir,
        ]
        if not no_resume:
            cmd.append("--resume")
        cmd.extend(extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] [GPU {gpu_id}] [{proc_name}] START #{idx}: {metadata_stem} @ {data_dir}\n"
            f"  CMD    : {shlex.join(cmd)}\n"
            f"  SUMMARY: {summary_path}",
            flush=True,
        )

        if dry_run:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[GPU {gpu_id}] [{proc_name}] DRY-RUN DONE #{idx}",
                flush=True,
            )
            continue

        # ---- run with tee -------------------------------------------
        console_prefix = f"[GPU {gpu_id}][#{idx}] "
        ret = _run_with_tee(cmd, env, summary_path, console_prefix, console_lock)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if ret == 0:
            print(
                f"[{ts}] [GPU {gpu_id}] [{proc_name}] DONE #{idx}: {metadata_stem} @ {data_dir}",
                flush=True,
            )
        else:
            print(
                f"[{ts}] [GPU {gpu_id}] [{proc_name}] FAIL (code={ret}) #{idx}: {metadata_stem} @ {data_dir}",
                flush=True,
            )
            # To abort everything on first failure, replace the pass below
            # with: import os; os._exit(1)


def _run_with_tee(
    cmd: list[str],
    env: dict,
    summary_path: str,
    console_prefix: str,
    console_lock: threading.Lock,
) -> int:
    """
    Run *cmd*, piping merged stdout+stderr to both the console and
    *summary_path*.  Returns the process exit code.
    """
    with open(summary_path, "w", encoding="utf-8") as sf:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
        )
        t = threading.Thread(
            target=_tee_reader,
            args=(proc.stdout, sf, console_lock, console_prefix),
            daemon=True,
        )
        t.start()
        t.join()
        return proc.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-GPU parallel runner for local_eval.eval. "
            "Evaluates multiple --data-dirs concurrently, teeing each job's "
            "output to a .summary file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # GPU / concurrency
    ap.add_argument(
        "--gpus", "-g",
        required=True,
        help='GPU ids, e.g. "7,6,5" or "0-3" or "0,2-4,7"',
    )
    ap.add_argument(
        "--max-workers-per-gpu", "-n",
        type=int,
        default=1,
        help="Max parallel workers per GPU (default: 1)",
    )

    # eval arguments
    ap.add_argument(
        "--config",
        required=True,
        help="Path to eval yaml config, e.g. local_eval/eval/eval.yaml",
    )
    ap.add_argument(
        "--metadata",
        required=True,
        help=(
            "Metadata directory, file, or glob pattern. "
            "A directory is expanded to individual *.jsonl files; "
            "each file becomes a separate parallel task per data-dir. "
            "e.g. data/test_clean/speech_edit  or  data/test_clean/speech_edit/*.jsonl"
        ),
    )
    ap.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        metavar="DATA_DIR",
        help="One or more --data-dir paths to evaluate (space-separated)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Do NOT pass --resume to local_eval.eval (resume is on by default)",
    )

    # output
    ap.add_argument(
        "--summary-dir",
        default=None,
        help=(
            "Directory to collect all .summary files. "
            "Default: <data-dir>.summary next to each data-dir."
        ),
    )
    ap.add_argument(
        "--extra-args",
        default="",
        help="Extra arguments appended verbatim, e.g. '--foo 1 --bar baz'",
    )

    # misc
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable (default: current interpreter)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )

    args = ap.parse_args()

    gpus = parse_gpus(args.gpus)
    if args.max_workers_per_gpu <= 0:
        raise ValueError("--max-workers-per-gpu must be > 0")

    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    if args.summary_dir:
        os.makedirs(args.summary_dir, exist_ok=True)

    # ---- resolve metadata to individual JSONL files -----------------
    metadata_input = args.metadata.rstrip("/")
    if os.path.isdir(metadata_input):
        metadata_files = sorted(glob.glob(os.path.join(metadata_input, "*.jsonl")))
    elif os.path.isfile(metadata_input):
        metadata_files = [metadata_input]
    else:
        # Treat as glob pattern (e.g. data/test_clean/speech_edit/*.jsonl)
        metadata_files = sorted(glob.glob(metadata_input))

    if not metadata_files:
        print(f"ERROR: No metadata JSONL files found for: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    print(f"Metadata files ({len(metadata_files)}):")
    for mf in metadata_files:
        print(f"  {mf}")

    # ---- build task list: (metadata_file, data_dir) pairs -----------
    # Pre-filter: skip pairs where data_dir has no matching .scp
    tasks: list[tuple[str, str]] = []
    for data_dir in args.data_dirs:
        if not os.path.isdir(data_dir):
            print(f"  Warning: data-dir not found, skipping: {data_dir}")
            continue
        for meta_file in metadata_files:
            stem = os.path.splitext(os.path.basename(meta_file))[0]
            scp_path = os.path.join(data_dir, f"{stem}.scp")
            if os.path.isfile(scp_path):
                tasks.append((meta_file, data_dir))
            else:
                print(f"  Skip: {stem}.jsonl x {os.path.basename(data_dir.rstrip('/'))} (no {stem}.scp)")

    if not tasks:
        print("No valid (metadata, data-dir) pairs found.", file=sys.stderr)
        sys.exit(1)

    # ---- fill task queue --------------------------------------------
    task_q: "Queue[tuple[int, str, str] | None]" = Queue()
    for i, (meta_file, data_dir) in enumerate(tasks):
        task_q.put((i, meta_file, data_dir))

    total_workers = len(gpus) * args.max_workers_per_gpu
    print(
        f"Tasks : {len(tasks)} ({len(metadata_files)} metadata x {len(args.data_dirs)} data-dirs, after filtering)\n"
        f"GPUs  : {gpus}\n"
        f"Workers: {len(gpus)} GPU(s) x {args.max_workers_per_gpu} = {total_workers}",
        flush=True,
    )

    # ---- spawn workers ----------------------------------------------
    workers: list[Process] = []
    for gpu_id in gpus:
        for wi in range(args.max_workers_per_gpu):
            p = Process(
                target=worker_loop,
                args=(
                    gpu_id,
                    task_q,
                    args.config,
                    args.python,
                    extra_args,
                    args.summary_dir,
                    args.no_resume,
                    args.dry_run,
                ),
                name=f"worker-gpu{gpu_id}-{wi}",
            )
            p.daemon = False
            p.start()
            workers.append(p)

    # send termination sentinel for each worker
    for _ in workers:
        task_q.put(None)

    # wait for all workers
    exit_code = 0
    for p in workers:
        p.join()
        if p.exitcode not in (0, None):
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
