#!/usr/bin/env python3
import argparse
import json
import os
import sys
import logging
import soundfile as sf

from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

def get_resp(llm_client: VLLMClient, target_audio_path):
    resp = None
    try:
        resp = llm_client.chat_completion(messages=[{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"file://{target_audio_path}"}}
                ], 
            }],
            temperature=0.65,
            # top_k=20,
            max_tokens=1024,
        )
    except Exception as e:
        logging.warning(f"Error calling LLM for {target_audio_path}: {e}")
        return None
    return resp

def process_one(line: str, llm_client):
    try:
        data = json.loads(line)
        
        # Expecting input from step1_vad.jsonl
        # Structure: {"main": ..., "split1": {"audio_path": ...}, "split2": {"audio_path": ...}}
        
        # Process split1
        if "split1" in data and "audio_path" in data["split1"]:
            path1 = data["split1"]["audio_path"]
            if os.path.exists(path1):
                cap1 = get_resp(llm_client, path1)
                if cap1:
                    data["split1"]["audio_caption"] = cap1
                else:
                    logging.warning(f"Failed to get caption for {path1}")
        
        # Process split2
        if "split2" in data and "audio_path" in data["split2"]:
            path2 = data["split2"]["audio_path"]
            if os.path.exists(path2):
                cap2 = get_resp(llm_client, path2)
                if cap2:
                    data["split2"]["audio_caption"] = cap2
                else:
                    logging.warning(f"Failed to get caption for {path2}")
                    
        return data
    except Exception as e:
        logging.warning(f"Error processing line: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate captions for split audios.")
    parser.add_argument("--input_jsonl", required=True, help="Path to input metadata.step1_vad.jsonl")
    parser.add_argument("--output_jsonl", required=True, help="Path to output metadata.step2_caption.jsonl")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Captioner", help="Model name for vLLM")
    parser.add_argument("--nj", type=int, default=32, help="Number of parallel workers")
    parser.add_argument(
        "--parallel_backend",
        type=str,
        default="threading",
        choices=["threading", "loky"],
        help="joblib backend",
    )
    parser.add_argument("--config_path", type=str, default=None)

    args = parser.parse_args()

    if args.config_path:
        args, _ = apply_step_config(args, "step2_caption")

    llm_client = VLLMClient(base_url=args.vllm_url, model=args.model, max_concurrent=args.nj*2, timeout=1200)

    def _process(idx: int, line: str) -> dict | None:
        return process_one(line, llm_client)

    runner = JsonlParallelRunner(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        process_fn=_process,
        n_jobs=args.nj,
        backend=args.parallel_backend,
        desc="Processing Captions",
        resume=False,
    )
    runner.run()

if __name__ == "__main__":
    main()
