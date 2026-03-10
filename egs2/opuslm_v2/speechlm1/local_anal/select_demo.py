#!/usr/bin/env python3
"""
Select high-quality demo samples from evaluation results.

Workflow:
  1. Discover eval directories via input_globs (same as collect_summary.py).
  2. For each dataset type, read per-sample .results JSONL files.
  3. Apply per-metric lambda filters from a YAML config.
     Only samples that pass ALL filters simultaneously are selected.
  4. Load <metadata_dir>/<dataset>.jsonl for each dataset type and index by id.
  5. For each selected sample, enrich the metadata item with:
       - model_key → {audio_path: <generated_audio>, **metrics}
  6. Write one output JSONL per dataset type into output_dir.

Usage:
    python3 local_anal/select_demo.py \\
        --input_globs \\
            "exp/ct-mt-t2a_v2-1000k/inference/*/eval-test_clean-v1-*" \\
            "exp/cv3/test_clean/speech_edit" \\
        --filter_yaml local_anal/select_demo_filter.yaml \\
        --metadata_dir data/test_clean/speech_edit \\
        --output_dir local_anal/demo_selected

Output (local_anal/demo_selected/):
    transcription_ins.jsonl
    style_emotion.jsonl
    ...
  Each line is the original metadata item, augmented with a "models" dict:
    {
      ...<original metadata fields>...,
      "models": {
        "ct-mt-t2a_v2-1000k/s356k/t2a_t2a": {
          "audio_path": "/path/to/generated.wav",
          "asr_wer": 0.05,
          "speaker_similarity_wavlm.sim": 0.82,
          ...
        },
        "cv3/speech_edit": { ... },
      }
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

# ============================================================================
# ANSI colors (for terminal output)
# ============================================================================
NOCOLOR = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[32m"
RED     = "\033[31m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

# ============================================================================
# Extra sub-fields: metric_name → [(extra_key, dotted_suffix)]
# Mirror of collect_summary.py — used to extract sub-scores.
# ============================================================================
EXTRA_FIELDS: dict[str, list[tuple[str, str]]] = {
    "asr_wer": [("edit_acc", "edit_acc")],
    "speaker_similarity_wavlm": [("sim", "sim")],
    "pseudo_mos": [
        ("utmos", "utmos"),
        ("dns_overall", "dns_overall"),
        ("dns_p808", "dns_p808"),
    ],
    "llm_judge_openai": [
        ("audio_quality", "audio_quality"),
        ("change_quality", "change_quality"),
        ("coherence", "coherence"),
        ("preservation", "preservation"),
        ("creativity", "creativity"),
        ("generation_quality", "generation_quality"),
        ("main_consistency", "main_consistency"),
        ("operation_effect", "operation_effect"),
        ("score", "score"),
        ("consistency", "consistency"),
    ],
    "llm_judge_gemini": [
        ("generation_quality", "generation_quality"),
        ("main_consistency", "main_consistency"),
        ("operation_effect", "operation_effect"),
        ("edit_fidelity", "edit_fidelity"),
        ("audio_quality", "audio_quality"),
        ("coherence", "coherence"),
        ("preservation", "preservation"),
        ("creativity", "creativity"),
        ("score", "score"),
    ],
    "clap_similarity": [
        ("audio_sim", "audio_sim"),
        ("main_text_src_sim", "main_text_src_sim"),
        ("main_text_gen_sim", "main_text_gen_sim"),
        ("main_text_delta", "main_text_delta"),
        ("y_text_sim", "y_text_sim"),
        ("x_text_sim", "x_text_sim"),
    ],
    "audio_event_flam": [
        ("audio_sim", "audio_sim"),
        ("main_text_src_sim", "main_text_src_sim"),
        ("main_text_gen_sim", "main_text_gen_sim"),
        ("main_text_delta", "main_text_delta"),
        ("y_text_sim", "y_text_sim"),
        ("x_text_sim", "x_text_sim"),
    ],
    "llm_judge_caption_llm": [
        ("caption_similarity", "caption_similarity"),
    ],
    "emotion_modelscope": [],
    "speed_duration": [],
    "volume_loudness": [],
    "pitch_shift": [],
    "fad": [],
}


# ============================================================================
# Path Parsing  (reused from collect_summary.py)
# ============================================================================

def parse_eval_dir(dirpath: str) -> dict[str, str]:
    """
    Extract a human-readable row identity from any eval directory.

    Handles:
        exp/<model>/inference/<ckpt>/eval-<testset>-<mode>
        exp/<model>/test_clean/<suite>
        exp/<model>/test_clean/<suite>-short
    """
    parts = Path(dirpath).parts
    try:
        exp_idx = list(parts).index("exp")
    except ValueError:
        exp_idx = 0

    experiment = parts[exp_idx + 1] if exp_idx + 1 < len(parts) else "unknown"

    if "inference" in parts:
        inf_idx = list(parts).index("inference")
        checkpoint = parts[inf_idx + 1] if inf_idx + 1 < len(parts) else ""
        eval_dir_name = parts[inf_idx + 2] if inf_idx + 2 < len(parts) else ""

        step_m = re.search(r"step_(\d+)", checkpoint)
        step = step_m.group(1) if step_m else "0"

        mode_m = re.search(r"eval-[^-]+-[^-]+-(.+)$", eval_dir_name)
        eval_mode = mode_m.group(1) if mode_m else eval_dir_name

        label = f"{experiment}/s{int(step) // 1000}k/{eval_mode}"
    else:
        checkpoint = ""
        step = "0"
        eval_mode = parts[-1] if len(parts) > exp_idx + 1 else ""
        suffix = f"/{eval_mode}" if eval_mode not in (experiment, "") else ""
        label = f"{experiment}{suffix}"

    return {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "eval_mode": eval_mode,
        "step": step,
        "label": label,
    }


# ============================================================================
# Filter config loading
# ============================================================================

def load_filter_config(yaml_path: str) -> dict[str, dict[str, Callable[[float], bool]]]:
    """
    Load the YAML filter config and compile lambda strings.

    Returns:
        { dataset_type: { metric_key: compiled_lambda_func } }
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        print(f"{RED}ERROR: Empty or invalid filter config: {yaml_path}{NOCOLOR}",
              file=sys.stderr)
        sys.exit(1)

    config: dict[str, dict[str, Callable]] = {}
    for dataset, filters in raw.items():
        if not isinstance(filters, dict):
            continue
        compiled: dict[str, Callable] = {}
        for metric_key, lambda_str in filters.items():
            try:
                fn = eval(lambda_str)  # noqa: S307 — intentional lambda eval
                if not callable(fn):
                    raise ValueError(f"Not callable: {lambda_str}")
                compiled[metric_key] = fn
            except Exception as e:
                print(f"{RED}ERROR: Cannot compile filter "
                      f"'{dataset}.{metric_key}': {lambda_str!r} → {e}{NOCOLOR}",
                      file=sys.stderr)
                sys.exit(1)
        config[dataset] = compiled  # empty dict means "accept all samples"

    return config


# ============================================================================
# Extract per-sample flat metrics dict from a .results sample
# ============================================================================

def extract_flat_metrics(sample: dict) -> dict[str, float | None]:
    """
    From a .results sample, produce a flat dict:
        { "asr_wer": 0.05, "asr_wer.edit_acc": 0.0, "pseudo_mos": 0.58,
          "pseudo_mos.utmos": 3.44, ... }
    None values indicate the metric existed but was invalid/missing.
    """
    result: dict[str, float | None] = {}
    metrics = sample.get("metrics", {})

    for metric_name, m_data in metrics.items():
        if not isinstance(m_data, dict):
            continue

        # Main score
        if m_data.get("valid") and m_data.get("score") is not None:
            result[metric_name] = float(m_data["score"])
        else:
            result[metric_name] = None

        # Extra sub-fields
        extra = m_data.get("extra") or {}
        extra_defs = EXTRA_FIELDS.get(metric_name, [])
        for ef_field, ef_suffix in extra_defs:
            v = extra.get(ef_field)
            key = f"{metric_name}.{ef_suffix}"
            if v is not None and isinstance(v, (int, float)):
                result[key] = float(v)
            else:
                result[key] = None

    return result


# ============================================================================
# Apply filters to a single sample
# ============================================================================

def sample_passes_filters(
    flat_metrics: dict[str, float | None],
    filters: dict[str, Callable[[float], bool]],
) -> bool:
    """
    Return True only if ALL filter conditions pass.
    If any required metric is None (invalid/missing), return False.
    """
    for metric_key, fn in filters.items():
        val = flat_metrics.get(metric_key)
        if val is None:
            return False  # skip invalid / missing
        try:
            if not fn(val):
                return False
        except Exception:
            return False
    return True


# ============================================================================
# Discover eval directories (same as collect_summary.py)
# ============================================================================

def discover_eval_dirs(input_globs: list[str]) -> list[str]:
    """Expand globs and return deduplicated list of directories."""
    dirs: list[str] = []
    seen: set[str] = set()
    for pattern in input_globs:
        for path in sorted(glob.glob(pattern)):
            rp = os.path.realpath(path)
            if os.path.isdir(rp) and rp not in seen:
                seen.add(rp)
                dirs.append(path)
    return dirs


# ============================================================================
# Read .results JSONL
# ============================================================================

def read_results_file(filepath: str) -> list[dict]:
    """Read a .results JSONL file, return list of parsed dicts."""
    samples: list[dict] = []
    with open(filepath) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  {YELLOW}WARN: JSON error in {filepath}:{line_no}: {e}{NOCOLOR}",
                      file=sys.stderr)
    return samples


# ============================================================================
# Load metadata JSONL → index by id
# ============================================================================

def load_metadata(metadata_dir: str, dataset: str) -> dict[str, dict]:
    """
    Load <metadata_dir>/<dataset>.jsonl and return {id: metadata_dict}.
    """
    filepath = os.path.join(metadata_dir, f"{dataset}.jsonl")
    if os.path.isfile(filepath):
        return _read_jsonl_index(filepath)
    return {}


def _read_jsonl_index(filepath: str) -> dict[str, dict]:
    """Read a JSONL file and index by 'id' field."""
    index: dict[str, dict] = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                uid = item.get("id") or item.get("utt_id")
                if uid:
                    index[uid] = item
            except json.JSONDecodeError:
                pass
    return index


# ============================================================================
# Main selection logic
# ============================================================================

def select_demos(
    input_globs: list[str],
    filter_config: dict[str, dict[str, Callable]],
    metadata_dir: str,
    model_keys: list[str] | None = None,
    or_model_keys: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Select demo samples.

    Args:
        model_keys:    AND mode — a sample is selected only when **every**
                       specified key has at least one matching model that
                       passes all filters for that sample.
        or_model_keys: OR mode — a sample is selected when **any one** of
                       the specified keys has a matching model that passes
                       all filters.
        If both are given, a sample must satisfy both conditions.

    Returns:
        { dataset_type: {
            sample_id: {
                "metadata": { ...original metadata fields... },
                "models": {
                    model_label: {
                        "audio_path": str,
                        **flat_metrics,
                    }
                }
            }
          }
        }
    """
    dirs = discover_eval_dirs(input_globs)

    # Combine both key lists for the filter-model check.
    # ALL models' data is always collected; filters only apply to matching ones.
    all_filter_keys = (model_keys or []) + (or_model_keys or [])

    def _is_filter_model(label: str) -> bool:
        if not all_filter_keys:
            return True  # no restriction — every model is a filter model
        return any(mk in label for mk in all_filter_keys)

    def _matching_filter_keys(label: str) -> list[str]:
        """Return which filter keys (AND + OR) are matched by *label*."""
        if not all_filter_keys:
            return ["__all__"]
        return [mk for mk in all_filter_keys if mk in label]

    # ---- Pass 1: Collect ALL models' data; apply filters only for filter models ----
    # all_data: dataset → sample_id → model_label → model_entry
    all_data: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    # Track per-sample which filter-keys have at least one passing model
    # passed_filter_keys[dataset][sid] = set of filter-key strings that passed
    passed_filter_keys: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    # selected_ids: filled after pass 1 based on AND / OR logic
    selected_ids: dict[str, set[str]] = defaultdict(set)

    # Counters for reporting (only counted for filter models)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "passed": 0, "skipped_invalid": 0, "skipped_filter": 0,
    })

    for d in dirs:
        info = parse_eval_dir(d)
        model_label = info["label"]
        is_filter = _is_filter_model(model_label)

        results_files = sorted(
            f for f in os.listdir(d) if f.endswith(".results")
        )

        for rf in results_files:
            dataset = rf[:-8]  # strip .results

            # Only process datasets that have filters defined
            if dataset not in filter_config:
                continue

            filters = filter_config[dataset]
            samples = read_results_file(os.path.join(d, rf))

            for sample in samples:
                sid = sample.get("id")
                if not sid:
                    continue

                flat_metrics = extract_flat_metrics(sample)

                # Build model entry (always, for all models)
                model_entry: dict[str, Any] = {
                    "audio_path": sample.get("eval_audio_path", ""),
                }
                for mk, mv in flat_metrics.items():
                    if mv is not None:
                        model_entry[mk] = mv
                all_data[dataset][sid][model_label] = model_entry

                # Apply filters only for filter models
                if is_filter:
                    stats[dataset]["total"] += 1

                    has_required = all(
                        flat_metrics.get(k) is not None for k in filters
                    )
                    if not has_required:
                        stats[dataset]["skipped_invalid"] += 1
                        continue

                    if not sample_passes_filters(flat_metrics, filters):
                        stats[dataset]["skipped_filter"] += 1
                        continue

                    stats[dataset]["passed"] += 1
                    # Record which filter-keys this passing model satisfies
                    for fk in _matching_filter_keys(model_label):
                        passed_filter_keys[dataset][sid].add(fk)

    # ---- Determine selected_ids from AND / OR logic ----
    all_datasets = set(passed_filter_keys.keys())
    for dataset in all_datasets:
        for sid, fk_set in passed_filter_keys[dataset].items():
            # AND: every model_key must have at least one passing model
            and_ok = True
            if model_keys:
                and_ok = all(mk in fk_set for mk in model_keys)
            # OR: at least one or_model_key must have a passing model
            or_ok = True
            if or_model_keys:
                or_ok = any(mk in fk_set for mk in or_model_keys)
            if and_ok and or_ok:
                selected_ids[dataset].add(sid)

    # ---- Pass 2: Build selected dict with ALL models' data for passing samples ----
    selected: dict[str, dict[str, dict]] = {}
    for dataset, sids in selected_ids.items():
        selected[dataset] = {}
        for sid in sids:
            selected[dataset][sid] = {
                "metadata": None,
                "models": all_data[dataset].get(sid, {}),
            }

    # Load metadata and populate
    for dataset in list(selected.keys()):
        meta_index = load_metadata(metadata_dir, dataset)
        if not meta_index:
            print(f"  {YELLOW}WARN: No metadata found for dataset '{dataset}' "
                  f"in {metadata_dir} — skipping entire dataset{NOCOLOR}",
                  file=sys.stderr)
            del selected[dataset]
            continue

        missing_sids = []
        for sid in list(selected[dataset].keys()):
            if sid in meta_index:
                selected[dataset][sid]["metadata"] = meta_index[sid]
            else:
                missing_sids.append(sid)
                del selected[dataset][sid]
        if missing_sids:
            print(f"  {YELLOW}WARN: {len(missing_sids)} sample(s) in "
                  f"'{dataset}' not found in metadata — skipped: "
                  f"{missing_sids[:5]}{'...' if len(missing_sids) > 5 else ''}"
                  f"{NOCOLOR}", file=sys.stderr)

    # Print summary
    print(f"\n{BOLD}{BLUE}{'=' * 80}{NOCOLOR}")
    print(f"  {BOLD}DEMO SELECTION SUMMARY{NOCOLOR}")
    print(f"{BOLD}{BLUE}{'=' * 80}{NOCOLOR}")

    for dataset in sorted(stats):
        s = stats[dataset]
        n_unique = len(selected.get(dataset, {}))
        print(f"\n  {CYAN}{dataset}{NOCOLOR}:")
        print(f"    Total samples checked : {s['total']}")
        print(f"    Skipped (invalid)     : {YELLOW}{s['skipped_invalid']}{NOCOLOR}")
        print(f"    Skipped (filter)      : {RED}{s['skipped_filter']}{NOCOLOR}")
        print(f"    Passed                : {GREEN}{s['passed']}{NOCOLOR}")
        print(f"    Unique sample IDs     : {GREEN}{n_unique}{NOCOLOR}")

        # Show filter thresholds
        if dataset in filter_config:
            print(f"    Filters:")
            for mk, fn in filter_config[dataset].items():
                # Try to recover the lambda source for display
                print(f"      {MAGENTA}{mk}{NOCOLOR}: {fn.__qualname__ if hasattr(fn, '__qualname__') else '?'}")

    print()
    return dict(selected)


# ============================================================================
# Write output
# ============================================================================

def write_output(
    selected: dict[str, dict[str, dict]],
    output_dir: str,
) -> list[str]:
    """
    Write one JSONL per dataset type.
    Each line: original metadata + "models" dict.
    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    for dataset in sorted(selected):
        entries = selected[dataset]
        if not entries:
            continue

        outpath = os.path.join(output_dir, f"{dataset}.jsonl")

        with open(outpath, "w") as f:
            for sid in sorted(entries):
                entry = entries[sid]
                # Merge metadata with models
                out_item = dict(entry["metadata"]) if entry["metadata"] else {"id": sid}
                out_item["models"] = entry["models"]
                f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

        paths.append(outpath)
        print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{outpath}{NOCOLOR}  "
              f"({GREEN}{len(entries)} samples{NOCOLOR})")

    return paths


# ============================================================================
# Also write a summary TSV for quick inspection
# ============================================================================

def write_summary_tsv(
    selected: dict[str, dict[str, dict]],
    filter_config: dict[str, dict[str, Callable]],
    output_dir: str,
) -> list[str]:
    """Write one TSV per dataset for quick metric overview."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    for dataset in sorted(selected):
        entries = selected[dataset]
        if not entries:
            continue

        # Collect all metric keys across all models
        metric_keys: set[str] = set()
        model_labels: set[str] = set()
        for sid, entry in entries.items():
            for mlabel, mdata in entry["models"].items():
                model_labels.add(mlabel)
                for k, v in mdata.items():
                    if k != "audio_path" and isinstance(v, (int, float)):
                        metric_keys.add(k)

        metric_keys_sorted = sorted(metric_keys)
        model_labels_sorted = sorted(model_labels)

        outpath = os.path.join(output_dir, f"{dataset}_summary.tsv")

        with open(outpath, "w") as f:
            # Header
            header = ["id"]
            for ml in model_labels_sorted:
                for mk in metric_keys_sorted:
                    header.append(f"{ml}|{mk}")
            f.write("\t".join(header) + "\n")

            for sid in sorted(entries):
                parts = [sid]
                for ml in model_labels_sorted:
                    mdata = entries[sid]["models"].get(ml, {})
                    for mk in metric_keys_sorted:
                        val = mdata.get(mk)
                        parts.append(f"{val:.4f}" if val is not None else "")
                f.write("\t".join(parts) + "\n")

        paths.append(outpath)
        print(f"  {YELLOW}>>{NOCOLOR} {CYAN}{outpath}{NOCOLOR}")

    return paths


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Select high-quality demo samples from eval results "
                    "using per-dataset metric filters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_globs", nargs="+", default=[
            "exp/*-1000k/inference/*/*", # infer_step/dialgoue_catgory
            "exp/opuslm_v2_stage2_pretrain_base/inference/*/*",
            "exp/*",
        ],
        help="Base glob patterns; metadata_key is automatically appended to each.",
    )
    parser.add_argument(
        "--metadata_key", type=str,
        default="test_clean/speech_edit",
        help="Relative key used to: (1) derive metadata_dir=data/<metadata_key>, "
             "(2) append to every input_glob to restrict to matching eval dirs.",
    )
    parser.add_argument(
        "--filter_yaml", type=str,
        default="local_anal/select_demo_filter.yaml",
        help="YAML file with per-dataset metric filter lambdas",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="demo_selected",
        help="Output directory for selected demo JSONL + summary TSV files",
    )
    parser.add_argument(
        "--model_keys", nargs="+", default=None,
        help='AND mode: a sample is selected only when ALL specified model '
             'keys have at least one matching model that passes every filter. '
             'E.g. --model_keys "cv3/speech_edit" "ct-mt-t2a_v2-1000k/s356k". '
             'If neither --model_keys nor --or_model_keys is given, '
             'all discovered models are used (OR semantics).',
    )
    parser.add_argument(
        "--or_model_keys", nargs="+", default=None,
        help='OR mode: a sample is selected when ANY ONE of the specified '
             'model keys has a matching model that passes every filter. '
             'E.g. --or_model_keys "cv3/speech_edit" "ct-mt-t2a_v2-1000k/s356k". '
             'Can be combined with --model_keys (both conditions must hold).',
    )
    args = parser.parse_args()

    # ---- Derive metadata_dir and effective input_globs from metadata_key ----
    metadata_dir = os.path.join("data", args.metadata_key)
    effective_globs = [
        glob_p.rstrip("/") + "/" + args.metadata_key
        for glob_p in args.input_globs
    ]

    # ---- Print config ----
    print(f"{CYAN}Metadata key:  {NOCOLOR}{YELLOW}{args.metadata_key}{NOCOLOR}")
    print(f"{CYAN}Metadata dir:  {NOCOLOR}{YELLOW}{metadata_dir}{NOCOLOR}")
    print(f"{CYAN}Input globs:   {NOCOLOR}{effective_globs}")
    print(f"{CYAN}Filter YAML:   {NOCOLOR}{YELLOW}{args.filter_yaml}{NOCOLOR}")
    print(f"{CYAN}Output dir:    {NOCOLOR}{YELLOW}{args.output_dir}{NOCOLOR}")
    if args.model_keys:
        print(f"{CYAN}Model keys (AND): {NOCOLOR}{YELLOW}{args.model_keys}{NOCOLOR}")
    else:
        print(f"{CYAN}Model keys (AND): {NOCOLOR}(none)")
    if args.or_model_keys:
        print(f"{CYAN}Model keys (OR):  {NOCOLOR}{YELLOW}{args.or_model_keys}{NOCOLOR}")
    else:
        print(f"{CYAN}Model keys (OR):  {NOCOLOR}(none)")
    if not args.model_keys and not args.or_model_keys:
        print(f"{CYAN}  → using all discovered models (OR){NOCOLOR}")

    # ---- Load filter config ----
    filter_config = load_filter_config(args.filter_yaml)
    print(f"\n{GREEN}Loaded filters for {len(filter_config)} dataset types:{NOCOLOR}")
    for ds in sorted(filter_config):
        metric_names = ", ".join(sorted(filter_config[ds].keys()))
        print(f"  {CYAN}{ds}{NOCOLOR}: {metric_names}")

    # ---- Run selection ----
    selected = select_demos(effective_globs, filter_config, metadata_dir,
                            model_keys=args.model_keys,
                            or_model_keys=args.or_model_keys)

    if not selected:
        print(f"\n{RED}ERROR: No samples selected. "
              f"Check input_globs and filter thresholds.{NOCOLOR}",
              file=sys.stderr)
        sys.exit(1)

    # ---- Write outputs ----
    total_selected = sum(len(v) for v in selected.values())
    print(f"\n{BOLD}{BLUE}{'=' * 80}{NOCOLOR}")
    print(f"  {BOLD}WRITING OUTPUTS{NOCOLOR}  "
          f"({GREEN}{total_selected} total samples{NOCOLOR})")
    print(f"{BOLD}{BLUE}{'=' * 80}{NOCOLOR}\n")

    write_output(selected, args.output_dir)
    print()
    write_summary_tsv(selected, filter_config, args.output_dir)

    print(f"\n{GREEN}Done! Selected {total_selected} demo samples "
          f"across {len(selected)} datasets.{NOCOLOR}")


if __name__ == "__main__":
    main()
