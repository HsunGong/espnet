#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

random.seed(7)

def build_user_text(record: dict, caption_mode: str) -> str:
    main_caption = record.get("main", {}).get("audio_caption", "")
    split1_caption = record.get("split1", {}).get("audio_caption", "")
    split2_caption = record.get("split2", {}).get("audio_caption", "")

    if caption_mode == "main":
        return main_caption
    return f"Audio Clip1: {split1_caption}\nAudio Clip2: {split2_caption}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble dialogue dataset (.jsonl + .json) from metadata JSONL. "
            "Supports main-caption mode or concat(split1+split2) mode."
        )
    )
    parser.add_argument("-i", "--input_jsonl", type=Path, required=True, help="Input metadata JSONL")
    parser.add_argument("-o", "--output_jsonl", type=Path, required=True, help="Output dataset JSONL")
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="main",
        choices=["main", "concat_split"],
        help="main: use main.audio_caption; concat_split: use split1+split2 captions",
    )
    parser.add_argument(
        "--audio_mode",
        type=str,
        default="split1",
        choices=["main", "split1"],
        help="Which audio to use as assistant response: main, split1",
    )
    parser.add_argument(
        "-k", "--k",
        type=int,
        default=-1,
        help="Number of samples to keep. -1 means all samples; otherwise random.shuffle then take top-k.",
    )
    args = parser.parse_args()

    output_json = args.output_jsonl.with_suffix(".json")
    sample_ids: list[str] = []

    records: list[tuple[int, dict]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            record = json.loads(line)
            records.append((idx, record))

    if args.k != -1:
        print(f"Shuffling {len(records)} records and selecting top {args.k}...")
        random.shuffle(records)
        if args.k >= 0:
            records = records[: args.k]

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for idx, record in records:
            if args.audio_mode == "split1":
                audio_path = record["split1"]["audio_path"]
            else:
                audio_path = record["main"]["audio_path"]

            user_text = build_user_text(record, args.caption_mode)
            basename = os.path.basename(audio_path)
            stem, _ = os.path.splitext(basename)
            example_id = f"{stem}_{idx}"

            entry = {
                "messages": [
                    ["user", "text", user_text],
                    ["assistant", "audio", audio_path],
                ],
                "example_id": example_id,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sample_ids.append(example_id)

    output_json_data = {
        "data_entry": [
            {
                "name": "dialogue",
                "path": os.path.abspath(args.output_jsonl),
                "reader": "dialogue",
            }
        ],
        "samples": sample_ids,
    }
    with open(output_json, "w", encoding="utf-8") as fjson:
        json.dump(output_json_data, fjson, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
