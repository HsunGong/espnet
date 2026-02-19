#!/usr/bin/env python3
import argparse
import json
import os
import sys
import logging
import requests
from jinja2 import Template

# Assuming VLLMClient is in local_split.sft_vllm_client but we need a text-only client for Instruct model
# Or we can reuse VLLMClient if it supports text-only chat completion.
# Let's check sft_vllm_client content first or assume it's standard OpenAI-compatible client wrapper.
from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

system_prompt = """You are an expert audio caption consistency editor.

You are given THREE captions describing the same original audio:
- A full-audio caption (authoritative reference for global context)
- Caption A (split1): first consecutive temporal segment
- Caption B (split2): second consecutive temporal segment

Your task is to refine Caption A and Caption B using the full-audio caption as the ground truth for terminology, speaker identity, environment, and overall framing.

Core Objective:
Each refined segment caption must be fully consistent with the full-audio caption while describing ONLY the audio events that occur within that segment.

Editing Principles:

1) Resolve inconsistencies using ONLY these operations when necessary:
   - Modify: minimal wording adjustments to remove contradictions
   - Delete: remove only the conflicting portion
   - Add: insert missing details ONLY if the detail is explicitly stated in the full-audio caption, AND it clearly belongs inside that segment’s time span

2) Strict limits:
   - Do NOT introduce any new events not mentioned in the full-audio caption.
   - Do NOT add new technical, numerical, institutional, or interpretive details.
   - Do NOT expand content beyond what is needed for consistency.
   - Do NOT describe events outside the segment’s temporal scope.
   - Do NOT rewrite for style or fluency. Preserve the original writing style and phrasing whenever possible.
   - Prefer the smallest possible edit that fixes the inconsistency.

4) Naming constraint:
   - The refined captions MUST NOT mention the words: "main", "split1", "split2", "Caption A", "Caption B", "segment", or any label that reveals segmentation.
   - Each refined caption must read as a standalone audio caption for its own clip.

Change Reporting Rules:
- Be concise.
- List only what was changed (do not restate the full caption).
- If no changes are required, output exactly: "No changes needed."

Output valid JSON only. No extra text.
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
    global system_prompt, user_prompt_template

    parser = argparse.ArgumentParser(description="Refine captions for split audios using Main caption.")
    parser.add_argument("-i", "--input_jsonl", required=True, help="Path to input metadata.step2_caption.jsonl")
    parser.add_argument("-o", "--output_jsonl", required=True, help="Path to output metadata.step3_refine.jsonl")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    # Updated default model as requested
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", help="Model name for vLLM")
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
        args, config = apply_step_config(args, "step3_refine_caption")
        step_cfg = config["step3_refine_caption"]
        system_prompt = step_cfg["system_prompt"]
        user_prompt_template = Template(step_cfg["user_prompt"])

    # VLLMClient usually handles the API details
    llm_client = VLLMClient(base_url=args.vllm_url, model=args.model, max_concurrent=args.nj*2, timeout=1200)

    def _process(idx: int, line: str) -> dict | None:
        return process_one(line, llm_client)

    runner = JsonlParallelRunner(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        process_fn=_process,
        n_jobs=args.nj,
        backend=args.parallel_backend,
        desc="Refining Captions",
        resume=False,
    )
    runner.run()

if __name__ == "__main__":
    main()
