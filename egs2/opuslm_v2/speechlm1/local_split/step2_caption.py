#!/usr/bin/env python3
import argparse
import json
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import soundfile as sf

from local_split.sft_vllm_client import VLLMClient

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

    args = parser.parse_args()

    llm_client = VLLMClient(base_url=args.vllm_url, model=args.model, max_concurrent=args.nj*2, timeout=1200)

    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    success_pbar = tqdm(total=len(lines), desc="Processing Captions")
    
    with open(args.output_jsonl, 'w', encoding='utf-8') as fout:
        # Use threading backend for I/O bound tasks (API calls)
        for ret in Parallel(n_jobs=args.nj, backend="threading", return_as="generator")(
            delayed(process_one)(line, llm_client) for line in lines
        ):
            if ret:
                fout.write(json.dumps(ret, ensure_ascii=False) + '\n')
            success_pbar.update(1)

if __name__ == "__main__":
    main()
