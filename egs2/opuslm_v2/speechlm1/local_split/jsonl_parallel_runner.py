#!/usr/bin/env python3

import json
from typing import Callable, Any

from joblib import Parallel, delayed
from tqdm import tqdm


class JsonlParallelRunner:
    def __init__(
        self,
        input_jsonl: str,
        output_jsonl: str,
        process_fn: Callable[[int, str], dict | None],
        n_jobs: int,
        backend: str = "threading",
        desc: str = "Processing",
        resume: bool = False,
        resume_key_fn: Callable[[dict], str] | None = None,
    ) -> None:
        self.input_jsonl = input_jsonl
        self.output_jsonl = output_jsonl
        self.process_fn = process_fn
        self.n_jobs = n_jobs
        self.backend = backend
        self.desc = desc
        self.resume = resume
        self.resume_key_fn = resume_key_fn

    def _load_done_keys(self) -> set[str]:
        done_keys: set[str] = set()
        if not self.resume or self.resume_key_fn is None:
            return done_keys

        try:
            with open(self.output_jsonl, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done_keys.add(self.resume_key_fn(rec))
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

        return done_keys

    def _filter_input_lines(self, done_keys: set[str]) -> tuple[list[str], int]:
        with open(self.input_jsonl, "r", encoding="utf-8") as fin:
            all_lines = fin.readlines()

        if not self.resume or self.resume_key_fn is None:
            return all_lines, 0

        lines: list[str] = []
        skipped_done = 0
        for line in all_lines:
            try:
                rec = json.loads(line)
                key = self.resume_key_fn(rec)
                if key in done_keys:
                    skipped_done += 1
                    continue
                lines.append(line)
            except Exception:
                continue

        return lines, skipped_done

    def run(self) -> None:
        done_keys = self._load_done_keys()
        write_mode = "a" if (self.resume and len(done_keys) > 0) else "w"

        if self.resume:
            print(f"Resume enabled: found {len(done_keys)} existing samples")

        lines, skipped_done = self._filter_input_lines(done_keys)

        if self.resume:
            print(f"Resume skip count: {skipped_done}")

        pbar = tqdm(total=len(lines), desc=self.desc)
        with open(self.output_jsonl, write_mode, encoding="utf-8") as fout:
            for ret in Parallel(n_jobs=self.n_jobs, backend=self.backend, return_as="generator")(
                delayed(self.process_fn)(idx, line) for idx, line in enumerate(lines)
            ):
                if ret is not None:
                    fout.write(json.dumps(ret, ensure_ascii=False) + "\n")
                pbar.update(1)
        pbar.close()
