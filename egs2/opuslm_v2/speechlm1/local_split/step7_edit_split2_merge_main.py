#!/usr/bin/env python3
import argparse
import json
import logging
from typing import Any

from jinja2 import Template

from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

system_prompt = """
You are an expert audio-caption editor specializing in precise audio-event correction.

You are given three captions:
- main caption: summary of the full audio
- split1 caption: first temporal segment
- split2 caption: second temporal segment

You must complete TWO tasks in a single response.

---

## Task A — Edit split2 Audio Caption (Strict Minimal Edit Version)

You will modify **split2** at the audio-event level using strictly minimal, targeted edits.

### 🔒 Core Requirement

You MUST:

1. Modify **exactly ONE speech content element** (highest priority).
   - This may include:
     - Continuing the speech/lyrics
     - Removing part of the speech
     - Modifying one or more words
     - Correcting spoken content
   - This change is mandatory.

2. Additionally, modify **no more than TWO audio-level elements** from the list below.

Total modifications allowed: **1 speech + up to 2 audio-event changes only.**

### 🎧 Audio-Level Elements You May Modify

You may:
- Add ONE new audio event
- Remove ONE existing audio event
- Modify ONE existing vocal attribute
- Change ONE background sound
- Change ONE recording quality attribute
- Modify ONE temporal element (pause, interruption, fade)
- Correct ONE factual audio inconsistency

Each counts as ONE modification.

### 🚫 Restrictions

- Do NOT rewrite the entire caption.
- Do NOT paraphrase sentences unless required for the speech edit.
- Do NOT change speaker identity unless explicitly required by an audio correction.
- Do NOT introduce more than two non-speech changes.
- Preserve all unaffected content exactly as written.
- Prefer partial targeted edits over structural rewrites.

If no valid speech-level edit is made, the output is invalid.

## Task B — Regenerate Main Caption (Strict Merge Version)

You will generate a revised **main caption** based ONLY on:
- split1 caption
- the edited split2 caption

### 🔒 Core Requirements

You MUST:

1. Preserve the **writing style, tone, and descriptive density** of the original main caption.
2. Merge split1 and edited split2 in correct chronological order.
3. Reflect ONLY the audio events explicitly present in split1 and edited split2.
4. Maintain full internal consistency:
   - Speaker identity
   - Vocal attributes
   - Background sounds
   - Recording quality
   - Temporal structure
5. Ensure smooth temporal continuity between segments (e.g., “After a pause…”, “The voice returns…”).
6. Keep the result concise, coherent, and structurally unified.

### 🚫 Strict Restrictions

- Do NOT introduce new sounds, speakers, or recording attributes.
- Do NOT restore deleted events from split2_old.
- Do NOT reinterpret or expand beyond what is explicitly stated.
- Do NOT exaggerate technical details.
- Do NOT change narrative perspective or writing style.
- Do NOT add summaries unless both segments contain them.
- Do NOT explain the merge process.
- The refined captions MUST NOT mention the words: "main", "split1", "split2", "Caption A", "Caption B", "segment", or any label that reveals segmentation.
- Each refined caption must read as a standalone audio caption for its own clip.

If any content appears in the output that is not directly supported by split1 and edited split2, the output is invalid.

---

## 📌 Output Format (Strict)

Return valid JSON only.

No explanations outside JSON.

Structure:

{
  "split2": {
    "edited_caption": "...",
    "changes": "..."
  },
  "main": {
    "edited_caption": "...",
    "changes": "..."
  }
}
"""

user_prompt_template = Template(
    """
## Main Caption (previous):
{{ main_caption }}

## Split1 Caption:
{{ split1_caption }}

## Split2 Caption (previous):
{{ split2_caption }}

## Return valid JSON in exactly this schema:
{
  "split2": {
    "edited_caption": "...",
    "changes": "..."
  },
  "main": {
    "edited_caption": "...",
    "changes": "..."
  }
}
"""
)


def call_editor(
    llm_client: VLLMClient,
    main_caption: str,
    split1_caption: str,
    split2_caption: str,
) -> dict[str, Any] | None:
    user_prompt = user_prompt_template.render(
        main_caption=main_caption,
        split1_caption=split1_caption,
        split2_caption=split2_caption,
    )

    try:
        resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=8192,
            json_mode=True,
        )
    except Exception as e:
        logging.warning(f"Error calling LLM in step7: {e}")
        return None

    if isinstance(resp, dict):
        return resp
    return None


def process_one(idx: int, line: str, llm_client: VLLMClient) -> dict[str, Any] | None:
    try:
        data = json.loads(line)

        result = call_editor(llm_client, data["main"]["audio_caption"], data["split1"]["audio_caption"], data["split2"]["audio_caption"])
        # print(result)

        metadata_out = data
        metadata_out["split2_old"] = data.pop("split2")  # keep old split2 for reference
        metadata_out["split2"] = {
            "audio_caption": result["split2"]["edited_caption"],
            "caption_changes": result["split2"]["changes"],
            "audio_path": None,
            "duration": None,
        }
        metadata_out["main"] = {
            "audio_caption": result["main"]["edited_caption"],
            "caption_changes": result["main"]["changes"],
            # audio-path is set to null
            "dataset": metadata_out["main"]["dataset"],
            "audio_path": None,
            "duration": None,
        }

        return metadata_out
    except Exception as e:
        logging.warning(f"Error occurs: {e}")
        return None


def main() -> None:
    global system_prompt, user_prompt_template

    parser = argparse.ArgumentParser(
        description=("Step7: edit split2 and generate edited main caption in metadata.")
    )
    parser.add_argument("-i", "--input_jsonl", required=True, help="Path to input metadata JSONL")
    parser.add_argument(
        "-o", "--output_jsonl", required=True, help="Path to output metadata.step7_edit_split2_merge_main.jsonl"
    )
    parser.add_argument(
        "--vllm_url", type=str, default="http://localhost:8001/v1", help="vLLM API URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        help="Model name for vLLM",
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
        type=bool,
        default=True,
        help="Skip samples already written in output_jsonl using split1.audio_path as key",
    )
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    if args.config_path:
        args, config = apply_step_config(args, "step7_edit_split2_merge_main")
        step_cfg = config["step7_edit_split2_merge_main"]
        system_prompt = step_cfg["system_prompt"]
        user_prompt_template = Template(step_cfg["user_prompt"])

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
        desc="Step5 Edit split2+main",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["split1"]["audio_path"],
    )
    runner.run()


if __name__ == "__main__":
    main()
