#!/usr/bin/env python3
import argparse
import json
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import requests
from jinja2 import Template

# Assuming VLLMClient is in local_split.sft_vllm_client but we need a text-only client for Instruct model
# Or we can reuse VLLMClient if it supports text-only chat completion.
# Let's check sft_vllm_client content first or assume it's standard OpenAI-compatible client wrapper.
from local_split.sft_vllm_client import VLLMClient

system_prompt = """You are an expert audio caption consistency editor.

There are THREE captions. The main caption describes the full original audio. split1 and split2 describe two consecutive temporal segments of that audio.

Your task is to refine two split audio captions (split1 and split2) using the main audio caption as ground truth for terminology, style, and overall context.

Core Objective:
Ensure each split caption is fully consistent with the main caption while accurately describing only the events within that split segment.

Editing Principles:

1. Treat the main caption as authoritative for global context (environment, speaker identity, tone, setting, narrative framing).

2. Resolve inconsistencies by applying one of the following operations when necessary:
   - Modify: Adjust wording to remove contradictions.
   - Delete: Remove only the conflicting portion.
   - Add: Insert missing details ONLY if:
       • the event clearly belongs to that split’s time span, and
       • it is explicitly stated in the main caption.

3. Do NOT:
   - Introduce new events not mentioned in the main caption.
   - Add new technical, numerical, institutional, or interpretive details.
   - Expand beyond what is needed to ensure consistency.
   - Describe events outside the split’s temporal scope.

4. Prefer minimal edits. Preserve valid local details from the split caption if they do not contradict the main caption.

Change Reporting Rules:
- Be concise.
- List only what was changed (no full rewrites in the explanation).
- If no changes are required, output exactly: "No changes needed."

Output valid JSON only.
"""

user_prompt_template = Template("""
## Main Caption:
{{ main_caption }}
## Split1 Caption:
{{ split1_caption }}
## Split2 Caption: 
{{ split2_caption }}

## Output the result in valid JSON format with the following keys:
{
    "split1": {
        "refined_caption": "...",
        "changes": "..."
    },
    "split2": {
        "refined_caption": "...",
        "changes": "..."
    }
}
""")

def get_refined_captions(llm_client: VLLMClient, main_caption, split1_caption, split2_caption) -> dict | None:
    user_prompt = user_prompt_template.render(
        main_caption=main_caption,
        split1_caption=split1_caption,
        split2_caption=split2_caption
    )

    resp = None
    try:
        resp: dict = llm_client.chat_completion(messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=8192,
            json_mode=True
        )
    except Exception as e:
        logging.warning(f"Error calling LLM: {e}")
        return None
    
    return resp

def process_one(line: str, llm_client):
    try:
        data = json.loads(line)
        
        main_cap = data.get("main", {}).get("audio_caption", "")
        split1_cap = data.get("split1", {}).get("audio_caption", "")
        split2_cap = data.get("split2", {}).get("audio_caption", "")

        if not main_cap or (not split1_cap and not split2_cap):
            return data # Nothing to refine

        # Call LLM
        refined_json = get_refined_captions(llm_client, main_cap, split1_cap, split2_cap)
        assert refined_json is not None

        if "split1" in refined_json:
            if "split1" in data:
                data["split1"]["caption_changes"] = refined_json["split1"].get("changes", "")
                # The user asked to "cover" the audio-caption (overwrite possibly, but let's keep original safe or strictly follow overwrite instruction)
                # User said: "覆盖的 audio-caption 覆盖到 split1 和 split2 上" (Overwrite audio-caption)
                if refined_json["split1"].get("refined_caption"):
                    data["split1"]["audio_caption"] = refined_json["split1"].get("refined_caption")

        if "split2" in refined_json:
            if "split2" in data:
                data["split2"]["caption_changes"] = refined_json["split2"].get("changes", "")
                if refined_json["split2"].get("refined_caption"):
                    data["split2"]["audio_caption"] = refined_json["split2"].get("refined_caption")

        return data
    except Exception as e:
        logging.warning(f"Error processing line: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Refine captions for split audios using Main caption.")
    parser.add_argument("-i", "--input_jsonl", required=True, help="Path to input metadata.step2_caption.jsonl")
    parser.add_argument("-o", "--output_jsonl", required=True, help="Path to output metadata.step3_refine.jsonl")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    # Updated default model as requested
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", help="Model name for vLLM")
    parser.add_argument("--nj", type=int, default=32, help="Number of parallel workers")

    args = parser.parse_args()

    # VLLMClient usually handles the API details
    llm_client = VLLMClient(base_url=args.vllm_url, model=args.model, max_concurrent=args.nj*2, timeout=1200)

    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    success_pbar = tqdm(total=len(lines), desc="Refining Captions")
    
    with open(args.output_jsonl, 'w', encoding='utf-8') as fout:
        for ret in Parallel(n_jobs=args.nj, backend="threading", return_as="generator")(
            delayed(process_one)(line, llm_client) for line in lines
        ):
            if ret:
                fout.write(json.dumps(ret, ensure_ascii=False) + '\n')
            success_pbar.update(1)

if __name__ == "__main__":
    main()
