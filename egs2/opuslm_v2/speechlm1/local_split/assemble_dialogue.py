#!/usr/bin/env python3
"""Assemble dialogue dataset (.jsonl + .json) from nested metadata JSONL.

Supported modes
---------------
Single-turn:

  cat2split1   (formerly concat_split)
      User: "Audio Clip1: <C1>\\nAudio Clip2: <C2>" → Asst: split1 audio

  main2split1     (formerly main)
      User: <C-main> → Asst: split1 audio

  cat2main     *needs main audio path*
      User: "Audio Clip1: <C1>\\nAudio Clip2: <C2>" → Asst: main audio

  main2main       *needs main audio path*
      User: <C-main> → Asst: main audio

Multi-turn (needs split2):

  t2a_t2a
      Turn 1: User: <C-main>    → Asst: split1 audio
      Turn 2: User: <C2>        → Asst: split2 audio

  a2t_t2a
      Turn 1: User: split1 audio → Asst: <C1> text
      Turn 2: User: <C2>         → Asst: split2 audio

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
    "cat2split1",  # Caption1+Caption2 -> Audio1 (as prefill)
    "main2split1",  # Caption-Main -> Audio1 (as prefill)
    "cat2main",  # Caption1+Caption2 -> Main Audio (total)
    "main2main",  # Caption-Main -> Main Audio (total)
    "t2a_t2a",  # Caption1 -> Audio1 (turn1) ; Caption2 -> Audio2 (turn2)
    "a2t_t2a",  # Audio1 -> Caption1 (turn1) ; Caption2 -> Audio2 (turn2)
]

# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_messages(record: dict, mode: str) -> list | None:
    """Return the messages list for one record + mode, or None to skip."""
    main = record.get("main", {}) or {}
    split1 = record.get("split1", {}) or {}
    split2 = record.get("split2", {}) or {}

    main_audio = str(main.get("audio_path") or "")
    split1_audio = str(split1.get("audio_path") or "")
    split2_audio = str(split2.get("audio_path") or "")

    main_caption = str(main.get("audio_caption") or "")
    split1_caption = str(split1.get("audio_caption") or "")
    split2_caption = str(split2.get("audio_caption") or "")

    if "cat2" in mode and not split2_caption:
        print("ERROR: mode requires split2 caption but it's missing; skipping record.")
    if "main2" in mode and not main_caption:
        print("ERROR: mode requires main caption but it's missing; skipping record.")

    concat_caption = f"Audio Clip1: {split1_caption}\nAudio Clip2: {split2_caption}"

    messages = None
    last_audio = None
    if mode == "cat2split1":
        messages = [["user", "text", concat_caption]]
        last_audio = split1_audio

    elif mode == "main2split1":
        messages = [["user", "text", main_caption]]
        last_audio = split1_audio
    elif mode == "cat2main":
        messages = [["user", "text", concat_caption]]
        last_audio = main_audio
    elif mode == "main2main":
        messages = [["user", "text", main_caption]]
        last_audio = main_audio

    elif mode == "t2a_t2a":
        messages = [
            ["user", "text", main_caption],
            ["assistant", "audio", split1_audio],
            ["user", "text", split2_caption],
        ]
        last_audio = split2_audio

    elif mode == "a2t_t2a":
        messages = [
            ["user", "audio", split1_audio],
            ["assistant", "text", split1_caption],
            ["user", "text", split2_caption],
        ]
        last_audio = split2_audio
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
        description="Assemble dialogue dataset (.jsonl + .json) from metadata JSONL."
    )
    parser.add_argument(
        "-i",
        "--input_jsonl",
        type=Path,
        required=True,
        help="Input nested metadata JSONL (main / split1 / split2).",
    )
    parser.add_argument(
        "-o", "--output_jsonl", type=Path, required=True, help="Output dataset JSONL."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=MODE_CHOICES,
        help="Dialogue assembly mode (see module docstring).",
    )
    parser.add_argument("-k", "--k", type=int, default=-1, help="Max samples to keep (-1 = all).")
    parser.add_argument("--yaml-path", type=Path, help="Path to the YAML file to append to.")
    parser.add_argument("--name-prefix", type=str, help="Prefix for the name in YAML.")
    args = parser.parse_args()
    print(
        f"[assemble_dialogue] mode={args.mode} input = {args.input_jsonl} output = {args.output_jsonl} k={args.k}"
    )

    # Resolve mode
    mode = args.mode
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_jsonl.with_suffix(".json")

    # Read records
    records: list[tuple[int, dict]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading input JSONL", unit=" lines"):
            records.append(json.loads(line))

    if args.k != -1:
        print(f"Shuffling {len(records)} records and selecting top {args.k}…")
        random.shuffle(records)
        records = records[: args.k]

    sample_ids: list[str] = []
    n_written = 0
    n_skipped = 0
    n_trimmed = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
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
            example_id = f"{stem}_{idx}"

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
    # print(f"[assemble_dialogue] Output: {args.output_jsonl}")

    output_json_data = {
        "data_entry": [
            {
                "name": "dialogue",
                "path": args.output_jsonl.absolute().as_posix(),
                "reader": "dialogue",
            }
        ],
        "metadata": args.input_jsonl.absolute().as_posix(),
        "dialogue": args.output_jsonl.absolute().as_posix(),
        "samples": sample_ids,
    }
    with open(output_json, "w", encoding="utf-8") as fjson:
        json.dump(output_json_data, fjson, indent=2, ensure_ascii=False)

    # Append to YAML if specified
    if args.yaml_path and args.name_prefix:
        lock_file = args.yaml_path.with_suffix(".lock")
        retries = 3
        import time
        import random

        while retries > 0:
            if lock_file.exists():
                retries -= 1
                print(f"Lock file {lock_file} exists")
                time.sleep(1 + random.random() * 2)  # Sleep a random time before
                continue
            lock_file.touch()
            basename = args.output_jsonl.stem
            name = f"{args.name_prefix}-{basename}"
            abs_path = output_json.absolute().as_posix()
            with open(args.yaml_path, "a", encoding="utf-8") as f:
                f.write(f"\n{name}:\n  path: {abs_path}\n")
            print(f"Appended to {args.yaml_path}: {name}")
            lock_file.unlink()
            break


if __name__ == "__main__":
    main()
