#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import queue
import shlex
import subprocess
import sys
import time
from multiprocessing import Process, Queue, current_process


def parse_gpus(gpu_str: str) -> list[str]:
    """
    支持:
      "0,1,2"
      "0-3"  -> 0,1,2,3
      "0,2-4,7"
    返回 GPU id 的字符串列表，比如 ["0","2","3","4","7"]
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
    # 去重但保持顺序
    seen = set()
    out = []
    for g in gpus:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def worker_loop(
    gpu_id: str,
    task_q: "Queue[tuple[int, str]]",
    infer_script: str,
    output_dir: str,
    python_bin: str,
    extra_args: list[str],
    dry_run: bool,
) -> None:
    """
    每个 worker 固定绑定一个 gpu_id，循环取任务执行。
    task: (idx, jsonl_path)
    """
    proc_name = current_process().name
    while True:
        try:
            item = task_q.get()
        except Exception:
            return

        if item is None:
            # 结束信号
            return

        idx, jsonl_path = item
        cmd = [
            python_bin,
            infer_script,
            "--jsonl-files",
            jsonl_path,
            "--output-dir",
            output_dir,
            *extra_args,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        # 简单的前后提示，方便你在 console 看哪个任务跑在哪个 GPU
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[GPU {gpu_id}] [{proc_name}] START #{idx}: {jsonl_path}\n"
            f"  CMD: {shlex.join(cmd)}",
            flush=True,
        )

        if dry_run:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[GPU {gpu_id}] [{proc_name}] DRY-RUN DONE #{idx}",
                flush=True,
            )
            continue

        # stdout/stderr 直接输出到当前 console（不 capture）
        ret = subprocess.run(cmd, env=env).returncode

        if ret == 0:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[GPU {gpu_id}] [{proc_name}] DONE #{idx}: {jsonl_path}",
                flush=True,
            )
        else:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[GPU {gpu_id}] [{proc_name}] FAIL (code={ret}) #{idx}: {jsonl_path}",
                flush=True,
            )
            # 这里选择“失败继续跑后续任务”。如果你想失败就立刻全停，可改成 os._exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-GPU, multi-worker scheduler for running inference jobs over jsonl files."
    )
    ap.add_argument(
        "--gpus",
        required=True,
        help='可用 GPU id 列表，比如 "7,6,5" 或 "0-3" 或 "0,2-4,7"',
    )
    ap.add_argument(
        "--max-workers-per-gpu",
        type=int,
        required=True,
        help="每张 GPU 最多并行 worker 数",
    )
    ap.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="N 个 jsonl 路径（空格分隔）",
    )
    ap.add_argument(
        "--infer-script",
        required=True,
        help="infer python 文件路径，比如 local_eval/speech/infer_stepaudiox.py",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，比如 exp/stepaudiox/test_clean/speech_edit",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help='python 可执行文件（默认用当前 python）。比如 "python" 或 "python3"',
    )
    ap.add_argument(
        "--extra-args",
        default="",
        help='额外参数字符串，会原样追加到命令末尾，例如: \'--foo 1 --bar "x y"\'',
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不实际执行",
    )

    args = ap.parse_args()

    gpus = parse_gpus(args.gpus)
    if args.max_workers_per_gpu <= 0:
        raise ValueError("--max-workers-per-gpu must be > 0")

    jsonl_files = args.jsonl
    infer_script = args.infer_script
    output_dir = args.output_dir
    python_bin = args.python

    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    # 任务队列
    task_q: "Queue[tuple[int, str] | None]" = Queue()
    for i, jf in enumerate(jsonl_files):
        task_q.put((i, jf))

    # 启动 worker：每个 GPU 起 max_workers_per_gpu 个进程
    workers: list[Process] = []
    for gpu_id in gpus:
        for wi in range(args.max_workers_per_gpu):
            p = Process(
                target=worker_loop,
                args=(gpu_id, task_q, infer_script, output_dir, python_bin, extra_args, args.dry_run),
                name=f"worker-gpu{gpu_id}-{wi}",
            )
            p.daemon = False
            p.start()
            workers.append(p)

    # 给每个 worker 塞一个结束信号
    for _ in workers:
        task_q.put(None)

    # 等待
    exit_code = 0
    for p in workers:
        p.join()
        if p.exitcode not in (0, None):
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
