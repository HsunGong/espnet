#!/usr/bin/env python3
"""
Analyze inference data validity by comparing inferred audio durations
against ground-truth dataset audio durations.

Dataset index is read automatically from the ESPNET_DATASET_REGISTRY
environment variable (colon-separated list of yaml paths).

Usage:
    python analysis_infer_data.py \\
        --infer_dir exp/.../inference_speech_continue_step_350000 [dir2 ...] \\
        [--include KEYWORD1 KEYWORD2] \\
        [--exclude KEYWORD1 KEYWORD2]

The --include / --exclude keywords are matched against the inference
subfolder names (e.g. dialogue_part2_4_debug-novad-speech.min_0...).

Output:
    For each matched dialogue_NAME/ subfolder in the inference dir,
    writes a valid.jsonl alongside results.jsonl with extended fields.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    sf = None
    _HAS_SOUNDFILE = False

try:
    import yaml as _yaml
    def load_yaml(path: str):
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    import json as _json_fallback
    def load_yaml(path: str):  # type: ignore[misc]
        # minimal fallback: try json
        with open(path) as f:
            return _json_fallback.load(f)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> Optional[float]:
    """Return duration of an audio file in seconds, or None on failure."""
    if not audio_path:
        return None
    if not os.path.exists(audio_path):
        return None
    if _HAS_SOUNDFILE:
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            pass
    # Fallback: scipy wave for .wav
    if audio_path.endswith(".wav"):
        try:
            from scipy.io import wavfile
            rate, data = wavfile.read(audio_path)
            return data.shape[0] / rate
        except Exception:
            pass
    return None


def audio_exists_and_valid(audio_path: Optional[str]) -> bool:
    """True if the path is non-None, file exists, and is readable."""
    if not audio_path:
        return False
    if not os.path.exists(audio_path):
        return False
    dur = get_audio_duration(audio_path)
    return dur is not None and dur > 0


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_dataset_json(json_path: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Load a dataset .json file.

    Returns:
        (dialogue_jsonl_path, metadata_jsonl_path, samples_list)
    """
    with open(json_path) as f:
        data = json.load(f)

    dialogue_path = None
    for entry in data.get("data_entry", []):
        if entry.get("reader") == "dialogue" or entry.get("name") == "dialogue":
            dialogue_path = entry["path"]
            break

    metadata_path = data.get("metadata", None)
    samples = data.get("samples", [])
    return dialogue_path, metadata_path, samples


def load_dialogue_jsonl(dialogue_path: str) -> Dict[str, List]:
    """
    Load a dialogue .jsonl file.

    Returns:
        dict: example_id -> messages list
    """
    entries: Dict[str, List] = {}
    corrupt = 0
    with open(dialogue_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                eid = d.get("example_id", "")
                if eid:
                    entries[eid] = d.get("messages", [])
            except json.JSONDecodeError:
                corrupt += 1
                print(f"  [WARN] Corrupt line {lineno} in {dialogue_path}")
    if corrupt:
        print(f"  [WARN] {corrupt} corrupt lines in {dialogue_path}")
    return entries


def load_metadata_jsonl(metadata_path: str, samples: List[str]) -> Dict[str, dict]:
    """
    Load a metadata .jsonl file; entries are indexed by position matching `samples`.

    Returns:
        dict: example_id -> metadata dict  (keys: main, split1, split2, meta, ...)
    """
    result: Dict[str, dict] = {}
    with open(metadata_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if i < len(samples):
                    result[samples[i]] = d
                else:
                    # More metadata lines than samples – index unknown
                    pass
            except json.JSONDecodeError:
                print(f"  [WARN] Corrupt metadata line {i+1} in {metadata_path}")
    return result


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def get_last_assistant_audio(messages: List) -> Optional[str]:
    """Return the path of the last assistant/audio message, or None."""
    for msg in reversed(messages):
        if len(msg) >= 3 and msg[0] == "assistant" and msg[1] == "audio":
            return msg[2]
    return None


# ---------------------------------------------------------------------------
# Validity logic
# ---------------------------------------------------------------------------

def compute_validity(
    infer_duration: Optional[float],
    gt_last_audio_path: Optional[str],
    gt_last_duration: Optional[float],
    split1_duration: Optional[float],
    is_split1_type: bool,
) -> Tuple[bool, Optional[float]]:
    """
    Determine if the inferred audio is valid and return (valid, duration_rate).

    Rules:
    - If gt_last_audio does not exist (path missing or unreadable):
        valid = infer_duration >= 0.5 * split1_duration
    - Else if dataset key contains '2split1':
        valid = infer_duration >= 1.8 * gt_last_duration
    - Else (non-split1 type):
        valid = 0.8 * gt_last_duration <= infer_duration <= 1.5 * gt_last_duration
    """
    if infer_duration is None or infer_duration <= 0:
        return False, None

    gt_exists = audio_exists_and_valid(gt_last_audio_path)

    if not gt_exists:
        # Fallback to split1 duration criterion
        if split1_duration is not None and split1_duration > 0:
            rate = infer_duration / split1_duration
            valid = infer_duration >= 0.5 * split1_duration
            return valid, rate
        # Cannot determine – mark invalid
        return False, None

    # gt_last_duration should be valid here
    if gt_last_duration is None or gt_last_duration <= 0:
        return False, None

    rate = infer_duration / gt_last_duration

    if is_split1_type:
        return rate >= 1.8, rate
    else:
        return 0.8 <= rate <= 1.5, rate


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _load_dataset_info(key: str, val: dict) -> Optional[dict]:
    """Load and return dataset info for one key, or None on fatal error."""
    json_path = val["path"]
    try:
        dialogue_path, metadata_path, samples = load_dataset_json(json_path)
    except Exception as e:
        print(f"  [WARN] Cannot load json for '{key}': {e}")
        return None

    dialogue_data: Dict[str, List] = {}
    if dialogue_path:
        if os.path.exists(dialogue_path):
            try:
                dialogue_data = load_dialogue_jsonl(dialogue_path)
            except Exception as e:
                print(f"  [WARN] Cannot load dialogue jsonl for '{key}': {e}")
        else:
            print(f"  [WARN] Dialogue jsonl not found: {dialogue_path}")

    metadata_data: Dict[str, dict] = {}
    if metadata_path:
        if os.path.exists(metadata_path):
            try:
                metadata_data = load_metadata_jsonl(metadata_path, samples)
            except Exception as e:
                print(f"  [WARN] Cannot load metadata jsonl for '{key}': {e}")
        else:
            print(f"  [WARN] Metadata jsonl not found: {metadata_path}")

    return {
        "dialogue": dialogue_data,
        "metadata": metadata_data,
        "samples": samples,
    }


def process_infer_dir(
    infer_dir: Path,
    all_datasets: Dict[str, dict],   # merged registry: key -> {path: ...}
    include: List[str],
    exclude: List[str],
) -> None:
    """Process one inference directory."""
    print(f"\n{'='*60}")
    print(f"Inference dir: {infer_dir}")
    print(f"{'='*60}")

    # Collect dialogue_* subdirs, applying include/exclude on the subdir name
    all_subdirs = sorted(
        d for d in infer_dir.iterdir()
        if d.is_dir() and d.name.startswith("dialogue_")
    )
    subdirs = []
    for d in all_subdirs:
        name = d.name
        if include and not all(kw in name for kw in include):
            continue
        if exclude and any(kw in name for kw in exclude):
            continue
        subdirs.append(d)

    print(f"  Selected {len(subdirs)} / {len(all_subdirs)} subdirs")

    # Lazy-load dataset info only for subdirs we will actually process
    dataset_infos: Dict[str, dict] = {}  # dataset_key -> info

    dataset_stats: Dict[str, dict] = {}

    for subdir in subdirs:
        dataset_key = subdir.name[len("dialogue_"):]

        # Lazy load
        if dataset_key not in dataset_infos:
            if dataset_key not in all_datasets:
                print(f"  [SKIP] Dataset key not found in registry: {dataset_key}")
                continue
            info = _load_dataset_info(dataset_key, all_datasets[dataset_key])
            if info is None:
                continue
            dataset_infos[dataset_key] = info

        results_path = subdir / "results.jsonl"
        if not results_path.exists():
            print(f"  [SKIP] No results.jsonl in {subdir.name}")
            continue

        dinfo = dataset_infos[dataset_key]
        dialogue_data: Dict[str, List] = dinfo["dialogue"]
        metadata_data: Dict[str, dict] = dinfo["metadata"]
        is_split1_type = "2split1" in dataset_key

        valid_count = 0
        total_count = 0
        corrupt_detected: List[str] = []  # example_ids with corrupt files
        output_records: List[dict] = []

        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result_entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [WARN] Corrupt line in results.jsonl for {dataset_key}")
                    continue

                total_count += 1
                example_id = result_entry.get("example_id", "")

                # ---- Inferred audio ----
                infer_messages = result_entry.get("messages", [])
                infer_audio_path = get_last_assistant_audio(infer_messages)
                infer_duration = get_audio_duration(infer_audio_path) if infer_audio_path else None
                if infer_audio_path and infer_duration is None:
                    corrupt_detected.append(example_id)

                # ---- Ground-truth messages ----
                gt_messages = dialogue_data.get(example_id, [])

                # ---- Metadata (if available) ----
                meta = metadata_data.get(example_id, {})

                main_entry: Optional[dict] = meta.get("main") if meta else None
                split1_entry: Optional[dict] = meta.get("split1") if meta else None
                split2_entry: Optional[dict] = meta.get("split2") if meta else None

                main_audio_path: Optional[str] = main_entry.get("audio_path") if main_entry else None
                split1_audio_path: Optional[str] = split1_entry.get("audio_path") if split1_entry else None
                split2_audio_path: Optional[str] = split2_entry.get("audio_path") if split2_entry else None

                # If no metadata split1, fall back to last GT audio
                if not split1_audio_path:
                    split1_audio_path = get_last_assistant_audio(gt_messages)

                # ---- Durations ----
                # Prefer metadata duration to avoid re-reading audio
                split1_duration: Optional[float] = None
                if split1_entry and split1_entry.get("duration") is not None:
                    split1_duration = split1_entry["duration"]
                elif split1_audio_path:
                    split1_duration = get_audio_duration(split1_audio_path)
                    if split1_audio_path and os.path.exists(split1_audio_path) and split1_duration is None:
                        corrupt_detected.append(f"{example_id}/split1")

                gt_last_audio_path = get_last_assistant_audio(gt_messages)
                gt_last_duration: Optional[float] = None
                if gt_last_audio_path and os.path.exists(gt_last_audio_path):
                    # Try metadata first
                    if split1_entry and gt_last_audio_path == split1_audio_path and split1_entry.get("duration") is not None:
                        gt_last_duration = split1_entry["duration"]
                    else:
                        gt_last_duration = get_audio_duration(gt_last_audio_path)
                        if gt_last_duration is None:
                            corrupt_detected.append(f"{example_id}/gt_last")

                # ---- Validity ----
                valid, duration_rate = compute_validity(
                    infer_duration=infer_duration,
                    gt_last_audio_path=gt_last_audio_path,
                    gt_last_duration=gt_last_duration,
                    split1_duration=split1_duration,
                    is_split1_type=is_split1_type,
                )
                if valid:
                    valid_count += 1

                # ---- Build output record ----
                out: dict = dict(result_entry)
                out["gt_messages"] = gt_messages

                if main_audio_path is not None:
                    out["main"] = main_audio_path

                out["split1"] = split1_audio_path  # always included (may be None)

                if split2_audio_path is not None:
                    out["split2"] = split2_audio_path

                out["valid"] = valid
                out["valid_duration_rate"] = round(duration_rate, 4) if duration_rate is not None else None

                output_records.append(out)

        # Write valid.jsonl
        valid_path = subdir / "valid.jsonl"
        with open(valid_path, "w") as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        dataset_stats[dataset_key] = {
            "valid": valid_count,
            "valid_path": valid_path,
            "total": total_count,
            "corrupt": list(set(corrupt_detected)),
        }

    # ---- Summary ----
    print(f"\n{'Dataset':80s}  {'Valid':>6}  {'Total':>6}  {'Rate':>6}  Corrupt  Valid Path")
    print("-" * 110)
    for dk, st in sorted(dataset_stats.items()):
        rate_str = (
            f"{st['valid']/st['total']*100:.1f}%"
            if st["total"] > 0 else "  N/A"
        )
        corrupt_str = str(len(st["corrupt"])) if st["corrupt"] else "0"
        print(f"  {dk[:78]:78s}  {st['valid']:>6}  {st['total']:>6}  {rate_str:>6}  {corrupt_str} {st['valid_path']}")
        if st["corrupt"]:
            for cid in st["corrupt"][:5]:
                print(f"      [CORRUPT] {cid}")
            if len(st["corrupt"]) > 5:
                print(f"      ... and {len(st['corrupt'])-5} more")
    print()


def load_registry() -> Dict[str, dict]:
    """
    Read ESPNET_DATASET_REGISTRY (colon-separated yaml paths) and merge all
    datasets into a single dict {dataset_key: {"path": ...}}.
    Later entries silently overwrite earlier ones with the same key.
    """
    registry_env = os.environ.get("ESPNET_DATASET_REGISTRY", "")
    if not registry_env:
        print("[ERROR] ESPNET_DATASET_REGISTRY is not set or empty.")
        sys.exit(1)

    yaml_paths = [p.strip() for p in registry_env.split(":") if p.strip()]
    print(f"ESPNET_DATASET_REGISTRY: {len(yaml_paths)} yaml file(s)")

    merged: Dict[str, dict] = {}
    for ypath in yaml_paths:
        if not os.path.exists(ypath):
            print(f"  [WARN] Registry yaml not found, skipping: {ypath}")
            continue
        try:
            data = load_yaml(ypath)
            if not isinstance(data, dict):
                print(f"  [WARN] Unexpected format in {ypath}, skipping")
                continue
            merged.update(data)
            print(f"  Loaded {len(data)} entries from {ypath}")
        except Exception as e:
            print(f"  [WARN] Cannot load {ypath}: {e}")

    print(f"  Total merged: {len(merged)} dataset entries")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze inference data validity vs ground-truth dataset.\n"
            "Dataset registry is read from the ESPNET_DATASET_REGISTRY env var."
        )
    )
    parser.add_argument(
        "--infer_dir",
        required=True,
        nargs="+",
        metavar="DIR",
        help="One or more inference directories to process.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        metavar="KEYWORD",
        help="Only process inference subdirs whose name contains ALL of these keywords.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="KEYWORD",
        help="Skip inference subdirs whose name contains ANY of these keywords.",
    )
    args = parser.parse_args()

    if not _HAS_SOUNDFILE:
        print("[WARN] soundfile not available; will attempt scipy fallback for .wav")

    # ---- Load dataset registry ----
    all_datasets = load_registry()

    if args.include:
        print(f"Include keywords (on subdir name): {args.include}")
    if args.exclude:
        print(f"Exclude keywords (on subdir name): {args.exclude}")

    # ---- Process each inference directory ----
    for infer_dir_str in args.infer_dir:
        infer_dir = Path(infer_dir_str)
        if not infer_dir.exists():
            print(f"\n[ERROR] Inference dir not found: {infer_dir}")
            continue
        process_infer_dir(infer_dir, all_datasets, args.include or [], args.exclude or [])


if __name__ == "__main__":
    main()
