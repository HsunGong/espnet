#!/usr/bin/env python3
"""Export first K lines from metadata.jsonl into chat-format JSONL and dataset JSON.

Outputs:
  - <output_basename>.jsonl
  - <output_basename>.json

Format of .jsonl:
  {"messages": [
      {"role": "user", "content": [...]},
      {"role": "assistant", "content": [...]}
  ]}

If --inverse is NOT set (default):
  User: text
  Assistant: audio

If --inverse is set:
  User: audio
  Assistant: text
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export top-K metadata rows to chat-format JSONL and dataset .json"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("/mnt/home/xungong-andr-1766e0/prep/data/audio_edit/part3/metadata.jsonl"),
        help="Input metadata.jsonl path",
    )
    parser.add_argument(
        "-k",
        type=int,
        help="Take first K non-empty lines",
    )
    parser.add_argument(
        "-o", "--output-basename",
        type=Path,
        required=True,
        help="Output basename (without extension), e.g. data/debug/part3_debug",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="qwen_caption",
        help="Field used for text content (e.g., qwen_caption or gemini_caption)",
    )
    parser.add_argument(
        "--utt-prefix",
        type=str,
        default="debug_",
        help="Utterance id prefix (if needed for ID generation)",
    )
    parser.add_argument(
        "-inv", "--inverse",
        action="store_true",
        help="If set, User=Audio and Assistant=Text. Default: User=Text, Assistant=Audio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    out_base = args.output_basename
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Output files: .jsonl and .json
    jsonl_path = out_base.with_suffix(out_base.suffix + ".jsonl")
    json_path = out_base.with_suffix(out_base.suffix + ".json")

    samples = []
    kept = 0

    with args.input.open("r", encoding="utf-8") as f_in, \
            jsonl_path.open("w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if args.k and kept >= args.k:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Need at least audio_path and text_field
            if "audio_path" not in obj:
                # Try fallback keys if needed or skip
                continue
                
            # If text field not present, skip? Or use empty?
            # Assuming strict requirement for now
            if args.text_field not in obj:
                continue

            kept += 1
            
            # Extract content
            audio_path = str(obj["audio_path"]).strip()
            text_content = str(obj[args.text_field]).strip()
            
            # Construct messages
            # content structure: list of dicts {"type": "text", "text": ...} or {"type": "audio", "audio_url": ...}
            
            if args.inverse:
                # User = Audio, Assistant = Text
                messages = [
                    ("user", "text", text_content),
                    ("assistant", "audio", audio_path)
                ]
            else:
                # User = Text, Assistant = Audio
                messages = [
                    ("user", "audio", audio_path),
                    ("assistant", "text", text_content)
                ]
            # Create output object
            idx = obj.get("id", kept)
            utt_id = f"{args.utt_prefix}{obj.get('dataset', 'unknown')}_{idx}"
            out_obj = {"messages": messages, "example_id": utt_id}

            # Write to jsonl
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            
            # ID generation for samples list
            # Use 'id' from source if available, else generate
            samples.append(utt_id)

    # valid_jsonl_path = jsonl_path.resolve()
    # It seems for these configs, absolute path is preferred or relative?
    # Usually absolute to avoid CWD issues.
    
    dataset_json = {
        "data_entry": [
            {
                "name": "dialogue", # Generic name
                "path": str(jsonl_path.resolve()),
                "reader": "dialogue", # Proposed reader name, or just "jsonl"
            }
        ],
        "samples": samples,
    }

    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(dataset_json, f_json, ensure_ascii=False, indent=2)
        f_json.write("\n")

    print(
        f"Done. kept={kept} | jsonl = {jsonl_path} | json = {json_path}"
    )


if __name__ == "__main__":
    main()
