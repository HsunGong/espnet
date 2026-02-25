#!/usr/bin/env python3
"""Convert inference results to SCP files for evaluation.

Directory naming convention in exp-inference-dir:
    dialogue_{name_prefix}-{category}-{mode}/
       results_0.jsonl, results_1.jsonl, ...

Each results_*.jsonl line:
    {"example_id": "...", "messages": [["assistant", "audio", "/abs/path.wav"]]}

Output:
    {exp_inference_dir}/{name_prefix}-{mode}/{category}.scp

SCP format:
    <id> <absolute-wav-path>

For cat2split1 mode with --metadata-dir:
    Metadata file {metadata_dir}/{category}.jsonl contains records with
    "example_id" and "source_audio_path".
    The generated audio is truncated to the duration of source_audio_path
    and written to {wav_stem}.chunked.wav before being added to the SCP.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# Known dialogue modes (used to split the directory name)
KNOWN_MODES = ["a2t_t2a", "t2a_t2a", "cat2split1"]


def parse_dialogue_dir(dirname: str, name_prefix: str):
    """Return (category, mode) parsed from directory name, or None if it doesn't match."""
    # Expected: dialogue_{name_prefix}-{category}-{mode}
    expected_prefix = f"dialogue_{name_prefix}-"
    if not dirname.startswith(expected_prefix):
        return None
    rest = dirname[len(expected_prefix):]  # "{category}-{mode}"
    for mode in KNOWN_MODES:
        suffix = f"-{mode}"
        if rest.endswith(suffix):
            category = rest[: -len(suffix)]
            return category, mode
    return None


def read_results_jsonl_files(dialogue_dir: Path):
    """Yield (example_id, wav_path) from all results_*.jsonl in dialogue_dir."""
    jsonl_files = sorted(dialogue_dir.glob("results_*.jsonl"))
    if not jsonl_files:
        # Fall back to results.jsonl if it exists
        fallback = dialogue_dir / "results.jsonl"
        if fallback.exists():
            jsonl_files = [fallback]
    for jf in jsonl_files:
        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                example_id = record.get("example_id", "")
                messages = record.get("messages", [])
                # Find the first assistant-audio message
                wav_path = None
                for msg in messages:
                    if len(msg) >= 3 and msg[0] == "assistant" and msg[1] == "audio":
                        wav_path = msg[2]
                        break
                if example_id and wav_path:
                    yield example_id, wav_path


def load_metadata(metadata_path: Path) -> dict:
    """Load metadata from a jsonl file.

    Returns a dict mapping example_id -> record (with at least source_audio_path).
    """
    meta = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example_id = record["id"]
            meta[example_id] = record
    # print("load metadata with", len(meta), "at", metadata_path)
    return meta


def get_audio_duration_samples(wav_path: str) -> tuple[int, int]:
    """Return (num_frames, samplerate) for a wav file without loading all data."""
    info = sf.info(wav_path)
    return info.frames, info.samplerate


def truncate_and_save_wav(src_wav: str, n_frames: int, out_path: str) -> str:
    """Read src_wav, keep only the first n_frames samples, write to out_path.

    Returns out_path.
    """
    data, sr = sf.read(src_wav, always_2d=False)
    if data.ndim == 1:
        truncated = data[:n_frames]
    else:
        truncated = data[:n_frames, :]
    sf.write(out_path, truncated, sr, subtype="PCM_16")
    return out_path


def apply_cat2split1_chunking(
    entries: list[tuple[str, str]],
    metadata: dict,
) -> list[tuple[str, str]]:
    """For cat2split1 mode: truncate each generated wav to its source duration.

    Args:
        entries:  list of (example_id, wav_path)
        metadata: dict mapping example_id -> record (must have source_audio_path)

    Returns:
        New list of (example_id, chunked_wav_path).
        Entries without metadata are passed through unchanged with a warning.
    """
    result = []
    for example_id, wav_path in entries:
        record = metadata[example_id]

        source_audio_path = record["audio_path"]

        n_frames = sf.info(source_audio_path).frames

        # Write alongside original wav: foo.wav -> foo.chunked.wav
        out_path = str(Path(wav_path).with_suffix(".chunked.flac"))
        truncate_and_save_wav(wav_path, n_frames, out_path)
        result.append((example_id, out_path))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert inference results to SCP files."
    )
    parser.add_argument(
        "--exp-inference-dir",
        type=Path,
        required=True,
        help="Inference directory, e.g. exp/ct-100k-default-mt/inference/inference_audio_step_380000",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        required=True,
        help="Prefix used to filter dialogue directories, e.g. eval-test_clean-v1",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing metadata jsonl files named {category}.jsonl. "
            "Required for cat2split1 mode: each record must have 'example_id' "
            "and 'source_audio_path'. The generated audio is truncated to the "
            "duration of source_audio_path and saved as {wav_stem}.chunked.wav."
        ),
    )
    args = parser.parse_args()

    exp_dir: Path = args.exp_inference_dir.resolve()
    name_prefix: str = args.name_prefix
    metadata_dir: Path | None = args.metadata_dir.resolve() if args.metadata_dir else None

    if not exp_dir.is_dir():
        raise FileNotFoundError(f"exp-inference-dir not found: {exp_dir}")

    if metadata_dir is not None and not metadata_dir.is_dir():
        raise FileNotFoundError(f"--metadata-dir not found: {metadata_dir}")

    # Collect: (category, mode) -> list of (example_id, wav_path)
    collected: dict[tuple, list] = defaultdict(list)

    for subdir in sorted(exp_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_dialogue_dir(subdir.name, name_prefix)
        if parsed is None:
            continue
        category, mode = parsed
        entries = list(read_results_jsonl_files(subdir))
        collected[(category, mode)].extend(entries)

    if not collected:
        print(f"[convert_results] No matching dialogue directories found under {exp_dir}")
        print(f"  name_prefix = {name_prefix!r}")
        return

    exported_dirs = set()

    for (category, mode), entries in sorted(collected.items()):
        # For cat2split1 mode, truncate generated audio to source duration
        if mode == "cat2split1":
            assert metadata_dir is not None, "metadata-dir is required for cat2split1 mode"
            meta_path = metadata_dir / f"{category}.jsonl"
            metadata = load_metadata(meta_path)
            entries = apply_cat2split1_chunking(entries, metadata)

        out_dir = exp_dir / f"{name_prefix}-{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)

        scp_path = out_dir / f"{category}.scp"
        with open(scp_path, "w", encoding="utf-8") as f:
            for example_id, wav_path in sorted(entries):
                f.write(f"{example_id} {os.path.abspath(wav_path)}\n")

        exported_dirs.add(out_dir)

        print(
            f"  {scp_path.relative_to(exp_dir)}  ({len(entries)} entries)"
        )

    print()
    print("[convert_results] Exported directories:")
    for d in sorted(exported_dirs):
        print(f"  {d}")


if __name__ == "__main__":
    main()
