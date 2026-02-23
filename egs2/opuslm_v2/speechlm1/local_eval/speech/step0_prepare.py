#!/usr/bin/env python3
"""Prepare metadata.jsonl for LibriSpeech test-clean from Kaldi-style files.

By default this script reads:
  - data/test_clean/kaldi/wav.scp
  - data/test_clean/kaldi/text
  - data/test_clean/kaldi/utt2spk
  - data/test_clean/kaldi/spk2gender

and writes:
  - data/test_clean/metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_kaldi_two_column(path: Path) -> Dict[str, str]:
    """Read a Kaldi file with at least 2 columns as key/value."""
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid format in {path} line {line_no}: {raw!r}")
            key, value = parts
            out[key] = value
    return out


def read_wav_scp_with_order(path: Path) -> Tuple[Dict[str, str], List[str]]:
    """Read wav.scp while preserving utterance order."""
    mapping: Dict[str, str] = {}
    order: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid format in {path} line {line_no}: {raw!r}")
            utt_id, wav_path = parts
            mapping[utt_id] = wav_path
            order.append(utt_id)
    return mapping, order


def validate_keys(wav: Dict[str, str], text: Dict[str, str], utt2spk: Dict[str, str]) -> None:
    wav_keys = set(wav)
    text_keys = set(text)
    utt2spk_keys = set(utt2spk)

    if wav_keys != text_keys or wav_keys != utt2spk_keys:
        missing_in_text = sorted(wav_keys - text_keys)[:10]
        missing_in_utt2spk = sorted(wav_keys - utt2spk_keys)[:10]
        extra_text = sorted(text_keys - wav_keys)[:10]
        extra_utt2spk = sorted(utt2spk_keys - wav_keys)[:10]
        raise ValueError(
            "Inconsistent utterance ids across wav.scp/text/utt2spk. "
            f"missing_in_text={missing_in_text}, "
            f"missing_in_utt2spk={missing_in_utt2spk}, "
            f"extra_text={extra_text}, "
            f"extra_utt2spk={extra_utt2spk}"
        )


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item["audio_caption"] = item.pop("caption", item.pop("qwen_caption", None))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row must be object in {path} line {line_no}")
            rows.append(item)
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaldi_dir",
        type=Path,
        default=repo_root / "data" / "test_clean" / "kaldi",
        help="Directory containing wav.scp/text/utt2spk/spk2gender",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "test_clean" / "metadata.jsonl",
        help="Output metadata jsonl path",
    )
    parser.add_argument(
        "--ori",
        type=Path,
        default=None,
        help=(
            "Optional original metadata jsonl. If provided (or default file exists), "
            "rows are aligned by utt_id and updated with Kaldi fields while preserving "
            "other existing fields."
        ),
    )
    args = parser.parse_args()

    kaldi_dir: Path = args.kaldi_dir
    output_path: Path = args.output
    ori_path: Optional[Path] = args.ori

    # Auto-enable ori merge if default ori file exists and user didn't pass --ori.
    if ori_path is None:
        auto_ori = repo_root / "data" / "test_clean" / "metadata.ori.jsonl"
        if auto_ori.exists():
            ori_path = auto_ori

    wav_scp = kaldi_dir / "wav.scp"
    text_path = kaldi_dir / "text"
    utt2spk_path = kaldi_dir / "utt2spk"
    spk2gender_path = kaldi_dir / "spk2gender"

    for p in (wav_scp, text_path, utt2spk_path, spk2gender_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    wav_map, utt_order = read_wav_scp_with_order(wav_scp)
    text_map = read_kaldi_two_column(text_path)
    utt2spk_map = read_kaldi_two_column(utt2spk_path)
    spk2gender_map = read_kaldi_two_column(spk2gender_path)

    validate_keys(wav_map, text_map, utt2spk_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_missing_gender = 0
    written = 0

    if not ori_path.exists():
        raise FileNotFoundError(f"--ori file does not exist: {ori_path}")

    ori_rows = read_jsonl(ori_path)
    kaldi_updates: Dict[str, dict] = {}
    for utt_id in utt_order:
        speaker = utt2spk_map[utt_id]
        gender = spk2gender_map.get(speaker, "")
        if gender == "":
            num_missing_gender += 1
        kaldi_updates[utt_id] = {
            "dataset": "librispeech_test_clean",
            "id": utt_id,
            "utt_id": utt_id,
            "text": text_map[utt_id].lower(),
            "speaker": speaker,
            "gender": "male" if gender == "m" else "female",
            "audio_path": os.path.abspath(wav_map[utt_id]),
        }

    used_ids = set()
    with output_path.open("w", encoding="utf-8") as f:
        for row in ori_rows:
            utt_id = str(row.get("utt_id", ""))
            if utt_id and utt_id in kaldi_updates:
                row.update(kaldi_updates[utt_id])
                used_ids.add(utt_id)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

        # Append Kaldi utterances not found in ori to avoid accidental data loss.
        missing_from_ori = [u for u in utt_order if u not in used_ids]
        for utt_id in missing_from_ori:
            f.write(json.dumps(kaldi_updates[utt_id], ensure_ascii=False) + "\n")
            written += 1

    print(f"Merged ori metadata: {ori_path}")
    print(f"Wrote {written} samples to: {output_path}")
    print(f"Aligned utt_id count: {len(used_ids)}")
    if len(used_ids) != len(utt_order):
        print(f"Appended missing Kaldi utt_id count: {len(utt_order) - len(used_ids)}")

    

if __name__ == "__main__":
    main()
