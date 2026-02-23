#!/usr/bin/env python3
import argparse
import json
import logging
from typing import Any

import jinja2

from local_split.sft_vllm_client import VLLMClient
from local_split.local_config import apply_step_config
from local_split.jsonl_parallel_runner import JsonlParallelRunner

SYSTEM_PROMPT = """You are an expert audio-caption rewriter.

Task:
Write a NEW main caption that matches the real audio composition:
the Split1 audio played twice consecutively (Split1 + Split1).

Sources and Priority:
- Main Caption is ONLY as the **WRITING STYLE guide** (paragraphing, descriptive depth, vocabulary).
- Use Split1 Caption as the CONTENT source (speech and audible events).
- If any factual conflict exists, follow Split1.

Core Requirement — Two sequential renderings of speech:
- The spoken content must be described twice in sequence within the caption.
- The second rendering must fully re-describe the spoken lines (by quotation or close paraphrase).
- Do NOT mechanically copy-paste the Split1 text; rewrite organically.

Continuity Constraint (prevents “new clip start” errors):
- The caption must read as one continuous recording.
- Do NOT reintroduce the scene in the second rendering.
- Avoid “opens/begins/starts” phrasing for the second rendering.
- Use natural continuation wording (e.g., “After a brief pause…”, “He continues…”, “The voice returns…”) WITHOUT any explicit repetition markers.

Global Attributes Rule (strict):
- Persistent/global attributes (environment, mic/recording quality, background hiss/hum, acoustics, mixing) may be introduced ONCE near the beginning.
- After introducing them, do NOT restate them in later paragraphs, including during the second rendering.
- If Split1 contains repeated global-attribute sentences, keep them only in the first rendering and omit them from the second rendering.

Content Constraints:
- Use ONLY information explicitly present in Split1 (plus non-conflicting stylistic phrasing from Main Caption).
- Do NOT introduce new sounds, new events, new technical details (numbers/frequencies), or new interpretations.
- Do NOT add or change speaker identity, emotion, or vocal behaviors in Split1.
- Keep the caption concise, coherent, and aligned with the Main Caption’s style.

Output:
Return valid JSON only in the required format.

{
  "main": {
    "rewritten_caption": "...",
    "changes": "Briefly note that the spoken lines are rendered twice in sequence and that global recording attributes are stated once. Do not use repetition meta-words."
  }
}
"""

user_prompt_tmpl = jinja2.Template("""
## Main Caption (not repeated one, just as a reference)
{{ main_caption }}

## Split1 Caption
{{split1_caption}}

## Real Audio Composition
split1 + split1 (repeat once)

## Output format
{
  "main": {
    "rewritten_caption": "...",
    "changes": "..."
  }
}
""")


def rewrite_main_caption(
    llm_client: VLLMClient,
    split1_caption: str,
    main_caption: str,
) -> dict[str, Any] | None:
    user_prompt = user_prompt_tmpl.render(split1_caption=split1_caption, main_caption=main_caption)

    try:
        resp = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            json_mode=True,
        )
    except Exception as e:
        logging.warning(f"LLM rewrite failed: {e}")
        return None

    if isinstance(resp, dict):
        return resp
    return None

def process_one(idx: int, line: str, llm_client: VLLMClient) -> dict[str, Any] | None:
    try:
        data = json.loads(line)

        llm_out = rewrite_main_caption(
            llm_client=llm_client,
            main_caption=data["main"]["audio_caption"],
            split1_caption=data["split1"]["audio_caption"],
        )

        main_out = llm_out["main"]
        rewritten_caption = main_out["rewritten_caption"].strip()
        caption_changes = main_out["changes"]

        data["main"]["audio_caption"] = rewritten_caption
        data["main"]["caption_changes"] = caption_changes
        data["main"]["caption_source"] = "step5_rewrite_from_repeat_audio"
    except Exception as e:
        logging.warning(f"Line {idx}: invalid json: {e}")
        return None

    return data


def main() -> None:
    global SYSTEM_PROMPT, user_prompt_tmpl

    parser = argparse.ArgumentParser(
        description=(
            "Step5 metadata generation: rewrite main caption with LLM based on Step4 (repeat-gen) metadata."
        )
    )
    parser.add_argument("-i", "--input_jsonl", required=True, help="Path to input metadata.step4_repeat_main.jsonl")
    parser.add_argument("-o", "--output_jsonl", required=True, help="Path to output metadata.step5_repeat_rewrite_main.jsonl")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8001/v1", help="vLLM API URL")
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
        args, config = apply_step_config(args, "step5_repeat_rewrite_main")
        step_cfg = config["step5_repeat_rewrite_main"]
        SYSTEM_PROMPT = step_cfg["system_prompt"]
        user_prompt_tmpl = jinja2.Template(step_cfg["user_prompt"])

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
        desc="Step4.2 Repeat+Rewrite",
        resume=args.resume,
        resume_key_fn=lambda rec: rec["split1"]["audio_path"],
    )
    runner.run()


if __name__ == "__main__":
    main()
