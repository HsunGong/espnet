#!/usr/bin/env python3
"""Assemble dialogue dataset (.jsonl + .json) from speech editing metadata JSONL.

Supported modes
---------------
Single-turn:

  cat2target
      User: "Transcription: <C1>\nEdit Prompt: <C2>" → Asst: target audio

Multi-turn:

  t2a_t2a
      Turn 1: User: <C1>        → Asst: source audio
      Turn 2: User: <C2>        → Asst: target audio

  a2t_t2a
      Turn 1: User: source audio → Asst: <C1> text
      Turn 2: User: <C2>         → Asst: target audio

Records that do not have the required audio path(s) are silently skipped
(a count is printed at the end).
"""

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

random.seed(7)

# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

MODE_CHOICES = [
    "cat2split1",  # Transcription+Edit Prompt -> Target Audio (as prefill)
    "t2a_t2a",     # Transcription -> Source Audio (turn1) ; Edit Prompt -> Target Audio (turn2)
    "a2t_t2a",     # Source Audio -> Transcription (turn1) ; Edit Prompt -> Target Audio (turn2)
]

# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(record: dict, mode: str) -> list | None:
    """Return the messages list for one record + mode, or None to skip."""
    source_audio = str(record.get("audio_path") or "")
    target_audio = str(record.get("target_audio_path") or "")

    source_text = str(record.get("text") or "")
    edit_prompt = str(record.get("edit_prompt") or "")

    if not edit_prompt:
        print("ERROR: mode requires edit prompt but it's missing; skipping record.")
        return None
    if not source_text:
        print("ERROR: mode requires source text but it's missing; skipping record.")
        return None

    concat_caption = f"Transcription: {source_text}\nEdit Prompt: {edit_prompt}"

    messages = None
    last_audio = None
    
    if mode == "cat2split1":
        messages = [["user", "text", concat_caption]]
        last_audio = target_audio

    elif mode == "t2a_t2a":
        messages = [
            ["user", "text", source_text],
            ["assistant", "audio", source_audio],
            ["user", "text", edit_prompt],
        ]
        last_audio = target_audio

    elif mode == "a2t_t2a":
        messages = [
            ["user", "audio", source_audio],
            ["assistant", "text", source_text],
            ["user", "text", edit_prompt],
        ]
        last_audio = target_audio
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if last_audio and not os.path.exists(last_audio):
        return messages
    else:
        return messages + [["assistant", "audio", last_audio]]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble dialogue dataset (.jsonl + .json) from speech editing metadata JSONL."
    )
    parser.add_argument(
        "-i",
        "--input_jsonl",
        type=Path,
        required=True,
        help="Input metadata JSONL.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True, help="Output directory for dataset JSONL."
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=None,
        choices=MODE_CHOICES,
        required=True,
        help="Dialogue assembly mode(s) (see module docstring). Multiple modes can be specified.",
    )
    parser.add_argument("-k", "--k", type=int, default=-1, help="Max samples to keep (-1 = all).")
    parser.add_argument("--yaml-path", type=Path, help="Path to the YAML file to append to.")
    parser.add_argument("--name-prefix", type=str, help="Prefix for the name in YAML.")
    args = parser.parse_args()

    modes = args.mode
    basename = args.input_jsonl.stem

    # Read records once, reuse across modes
    records: list[dict] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading input JSONL", unit=" lines"):
            records.append(json.loads(line))

    if args.k != -1:
        print(f"Shuffling {len(records)} records and selecting top {args.k}…")
        random.shuffle(records)
        records = records[: args.k]

    for mode in modes:
        output_jsonl = args.output_dir / f"dialogue.{mode}.{basename}.jsonl"
        print(
            f"[assemble_dialogue] mode={mode} input = {args.input_jsonl} output = {output_jsonl} k={args.k}"
        )
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_json = output_jsonl.with_suffix(".json")

        sample_ids: list[str] = []
        n_written = 0
        n_skipped = 0
        n_trimmed = 0

        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for idx, record in tqdm(
                enumerate(records), total=len(records), desc="Writing output JSONL"
            ):
                messages = build_messages(record, mode)
                if messages is None:
                    n_skipped += 1
                    continue

                # If the last assistant audio doesn't exist on disk, drop that turn
                # (instead of skipping the whole record)
                if (
                    messages
                    and messages[-1][0] == "assistant"
                    and messages[-1][1] == "audio"
                    and not os.path.exists(messages[-1][2])
                ):
                    messages = messages[:-1]
                    n_trimmed += 1

                # Derive stable example_id from the first audio path in messages
                audio_paths = [m[2] for m in messages if m[1] == "audio"]
                ref_path = audio_paths[0] if audio_paths else ""
                stem = os.path.splitext(os.path.basename(ref_path))[0] if ref_path else f"record_{idx}"
                example_id = record["id"]

                entry = {
                    "messages": messages,
                    "example_id": example_id,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                sample_ids.append(example_id)
                n_written += 1

        print(
            f"[assemble_dialogue] mode={mode}  written={n_written}  skipped={n_skipped}  trimmed={n_trimmed}"
        )

        output_json_data = {
            "data_entry": [
                {
                    "name": "dialogue",
                    "path": output_jsonl.absolute().as_posix(),
                    "reader": "dialogue",
                }
            ],
            "metadata": args.input_jsonl.absolute().as_posix(),
            "dialogue": output_jsonl.absolute().as_posix(),
            "samples": sample_ids,
        }
        with open(output_json, "w", encoding="utf-8") as fjson:
            json.dump(output_json_data, fjson, indent=2, ensure_ascii=False)

        # Append to YAML if specified
        if args.yaml_path and args.name_prefix:
            lock_file = args.yaml_path.with_suffix(".lock")
            retries = 3
            import time

            while retries > 0:
                if lock_file.exists():
                    retries -= 1
                    print(f"Lock file {lock_file} exists")
                    time.sleep(1 + random.random() * 2)  # Sleep a random time before
                    continue
                lock_file.touch()
                name = f"{args.name_prefix}-{basename}-{mode}"
                abs_path = output_json.absolute().as_posix()
                with open(args.yaml_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{name}:\n  path: {abs_path}\n")
                print(f"Appended to {args.yaml_path}: {name}")
                lock_file.unlink()
                break

if __name__ == "__main__":
    main()
