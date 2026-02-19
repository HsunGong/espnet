#!/usr/bin/env python3
"""Step 4.3: Generate main-audio caption with the Captioner model.

Input:  metadata.step4_repeat_main.audio_only.jsonl  (from step4_repeat_gen.py)
          Each record has `main.audio_path` pointing to the repeated audio but
          may lack `main.audio_caption`.
Output: metadata.step4_repeat_caption.jsonl
          Same records with `main.audio_caption` filled by the Captioner.

This is a direct-captioning alternative to step4_repeat_rewrite_main.py, which
uses a text-only LLM rewriter instead.  Here we feed the repeated audio to the
same Captioner used in step2_caption.py so the caption is grounded in the
actual audio signal.
"""
import argparse
import json
import logging
import os
from typing import Any

from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner


def get_caption(llm_client: VLLMClient, audio_path: str) -> str | None:
    """Call the Captioner for a single audio file, return the caption string."""
    try:
        resp = llm_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{audio_path}"},
                        }
                    ],
                }
            ],
            temperature=0.65,
            max_tokens=1024,
        )
    except Exception as e:
        logging.warning(f"Captioner call failed for {audio_path}: {e}")
        return None

    if isinstance(resp, str):
        return resp.strip() or None
    return None


def process_one(idx: int, line: str, llm_client: VLLMClient) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
    except Exception as e:
        logging.warning(f"Line {idx}: JSON parse error: {e}")
        return None

    main_audio_path = data.get("main", {}).get("audio_path", "")
    if not main_audio_path:
        logging.warning(f"Line {idx}: missing main.audio_path, skipping")
        return None
    if not os.path.exists(main_audio_path):
        logging.warning(f"Line {idx}: main audio not found: {main_audio_path}, skipping")
        return None

    caption = get_caption(llm_client, main_audio_path)
    if caption is None:
        logging.warning(f"Line {idx}: captioner returned None for {main_audio_path}")
        return None

    data["main"]["audio_caption"] = caption
    data["main"]["caption_source"] = "step4_captioner"
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4.3: Caption the repeated main-audio with the Captioner model "
            "(ground-truth audio signal, no LLM text rewriting)."
        )
    )
    parser.add_argument(
        "-i",
        "--input_jsonl",
        required=True,
        help="Path to input metadata.step4_repeat_main.audio_only.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output_jsonl",
        required=True,
        help="Path to output metadata.step4_repeat_caption.jsonl",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API URL for the Captioner server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        help="Captioner model name served by vLLM",
    )
    parser.add_argument("--nj", type=int, default=32, help="Number of parallel workers")
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default="threading",
        choices=["threading", "loky"],
        help="joblib backend",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already written in output_jsonl using split1.audio_path as key",
    )
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step4_repeat_caption")

    llm_client = VLLMClient(
        base_url=args.vllm_url,
        model=args.model,
        max_concurrent=args.nj * 2,
        timeout=1200,
    )

    def _process(idx: int, line: str) -> dict[str, Any] | None:
        return process_one(idx, line, llm_client)

    runner = JsonlParallelRunner(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        process_fn=_process,
        n_jobs=args.nj,
        backend=args.parallel_backend,
        desc="Step4.3 Caption repeated audio",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["split1"]["audio_path"],
    )
    runner.run()


if __name__ == "__main__":
    main()
