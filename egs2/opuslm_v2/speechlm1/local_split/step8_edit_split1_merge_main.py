#!/usr/bin/env python3
"""Step 8: edit split1 caption, then regenerate main caption for
the audio composition split1_original + split1_edited.

The underlying audio is the same split1 file played twice (identical to the
step4 repeat arrangement), but the two passes are described with slightly
different captions:
  - First pass  → split1 original caption
  - Second pass → split1 edited caption  (Task A output)
  - Main        → merged caption covering both passes  (Task B output)

Input:  metadata.step3_refine.jsonl
Output: metadata.step8_edit_split1_merge_main.jsonl
"""
import argparse
import copy
import json
import logging
from typing import Any

from jinja2 import Template

from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

system_prompt = """
You are an expert audio-caption editor specializing in precise audio-event correction.

You are given ONE caption:
- split1 caption: the single segment that is played twice to form the full audio

You must complete TWO tasks in a single response.

---

## Task A — Edit split1 Audio Caption (Strict Minimal Edit Version)

You will produce a lightly modified version of the split1 caption.
This edited caption will represent the **second consecutive rendering** of the same audio.

### 🔒 Core Requirement

You MUST:

1. Modify **exactly ONE speech content element** (highest priority).
   - This may include:
     - Continuing the speech/lyrics
     - Removing part of the speech
     - Modifying one or more words
     - Correcting or varying spoken content
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

For the `changes` field, use a bullet list — one bullet per distinct change made, e.g.:
- Speech: changed "hello" → "hi there"
- Vocal: removed mention of slight echo

---

## Task B — Generate Main Caption (Strict Merge Version)

You will generate a **main caption** that covers the full audio:
**split1_original (first pass) followed immediately by split1_edited (second pass)**.

### 🔒 Core Requirements

You MUST:

1. Write in natural, descriptive audio-caption prose (no external style reference is given).
2. Describe both passes in correct chronological order (first pass → second pass).
3. Reflect ONLY the audio events explicitly present in split1_original and split1_edited.
4. **Speech/lyrics completeness (critical):** The complete speech content and lyrics from
   split1_edited MUST be written out in FULL during the second pass. Do NOT summarize,
   abbreviate, or omit any spoken lines or lyrics.
5. Maintain full internal consistency:
   - Speaker identity
   - Vocal attributes
   - Background sounds
   - Recording quality
   - Temporal structure
6. Read as one continuous recording — do NOT reintroduce the scene for the second pass.
7. Use natural continuation wording (e.g., "After a brief pause…", "The voice continues…",
   "He then says…") WITHOUT any explicit repetition markers.
8. Global/persistent attributes (environment, mic quality, background noise, acoustics) must
   be stated ONCE near the beginning and NOT restated during the second pass.
9. Keep the result concise, coherent, and structurally unified.

### 🚫 Strict Restrictions

- Do NOT introduce new sounds, speakers, or recording attributes not present in either pass.
- Do NOT reinterpret or expand beyond what is explicitly stated.
- Do NOT summarize or skip any spoken lines from split1_edited in the second pass.
- Do NOT exaggerate technical details.
- Do NOT change narrative perspective or writing style.
- Do NOT use words that reveal the segmentation: "main", "split1", "split2", "Caption A",
  "Caption B", "segment", "first pass", "second pass", "repeat", "again".
- The result must read as a standalone audio caption for a single continuous clip.

If any content appears in the output that is not directly supported by split1_original or
split1_edited, the output is invalid.

For the `changes` field, use a bullet list — one bullet per notable decision, e.g.:
- First pass: based on split1_original; global attributes stated here
- Second pass: full speech from split1_edited written out verbatim/in full
- Global attributes (e.g., room acoustics) omitted from second pass

---

## 📌 Output Format (Strict)

Return valid JSON only. No explanations outside JSON.

{
  "split1": {
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
## Split1 Caption:
{{ split1_caption }}

## Audio Composition
split1_original (first pass) + split1_edited (second pass)

## Return valid JSON in exactly this schema:
{
  "split1": {
    "edited_caption": "...",
    "changes": "- <change 1>\\n- <change 2>\\n..."
  },
  "main": {
    "edited_caption": "...",
    "changes": "- <decision 1>\\n- <decision 2>\\n..."
  }
}
"""
)


def call_editor(
    llm_client: VLLMClient,
    split1_caption: str,
) -> dict[str, Any] | None:
    user_prompt = user_prompt_template.render(
        split1_caption=split1_caption,
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
        logging.warning(f"Error calling LLM in step8: {e}")
        return None

    if isinstance(resp, dict):
        return resp
    return None


def process_one(idx: int, line: str, llm_client: VLLMClient) -> dict[str, Any] | None:
    try:
        data = json.loads(line)

        result = call_editor(
            llm_client,
            split1_caption=data["split1"]["audio_caption"],
        )
        if result is None:
            logging.warning(f"Line {idx}: LLM returned None")
            return None

        metadata_out = data

        # Preserve original split1 for reference
        original_split1 = copy.deepcopy(metadata_out["split1"])

        # split2 = split1_edited (same audio file played as second pass)
        metadata_out["split2"] = {
            "audio_caption": result["split1"]["edited_caption"],
            "caption_changes": result["split1"]["changes"],
            "audio_path": None,
            "duration": None,
        }

        # main → merged caption for split1_orig + split1_edited
        metadata_out["main"] = {
            "audio_caption": result["main"]["edited_caption"],
            "caption_changes": result["main"]["changes"],
            "dataset": data["main"].get("dataset"),
            # audio_path is null: the repeated audio is produced by step4_repeat_gen
            "audio_path": None,
            "duration": None,
        }

        return metadata_out
    except Exception as e:
        logging.warning(f"Error occurs at line {idx}: {e}")
        return None


def main() -> None:
    global system_prompt, user_prompt_template

    parser = argparse.ArgumentParser(
        description=(
            "Step6: edit split1 caption and regenerate main caption for "
            "audio composition split1_original + split1_edited."
        )
    )
    parser.add_argument("-i", "--input_jsonl", required=True, help="Path to input metadata JSONL")
    parser.add_argument(
        "-o",
        "--output_jsonl",
        required=True,
        help="Path to output metadata.step8_edit_split1_merge_main.jsonl",
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
        args, config = apply_step_config(args, "step8_edit_split1_merge_main")
        step_cfg = config["step8_edit_split1_merge_main"]
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
        desc="Step6 Edit split1+main",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["split1_old"]["audio_path"],
    )
    runner.run()


if __name__ == "__main__":
    main()
