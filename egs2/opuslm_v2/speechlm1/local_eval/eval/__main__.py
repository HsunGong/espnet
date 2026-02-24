from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json
import os
import re
from tqdm import tqdm

from joblib import Parallel, delayed

from .scorers.asr_wer import ASRWERScorer
from .scorers.llm_judge_caption_llm import LLMJudgeCaptionLLMScorer
from .scorers.llm_judge_gemini import LLMJudgeGeminiScorer
from .scorers.speed_duration import SpeedDurationScorer
from .scorers.volume_loudness import VolumeLoudnessScorer
from .scorers.pitch_shift import PitchShiftScorer
from .scorers.pseudo_mos import PseudoMOSScorer
from .scorers.emotion_modelscope import EmotionModelscopeScorer
from .scorers.speaker_similarity_wespeaker import SpeakerSimilarityWespeakerScorer

SCORER_CLASSES = {
    "asr_wer": ASRWERScorer,
    "llm_judge_caption_llm": LLMJudgeCaptionLLMScorer,
    "llm_judge_gemini": LLMJudgeGeminiScorer,
    "speed_duration": SpeedDurationScorer,
    "volume_loudness": VolumeLoudnessScorer,
    "pitch_shift": PitchShiftScorer,
    "pseudo_mos": PseudoMOSScorer,
    "emotion_modelscope": EmotionModelscopeScorer,
    "speaker_similarity_wespeaker": SpeakerSimilarityWespeakerScorer,
}

def run_scorer(scorer, task_cfg, samples):
    scorer.configure_task(task_cfg)
    rows, summary = scorer.run(samples)
    return scorer.name, rows, summary

DEFAULT_TASK_SCORERS: dict[str, list[str]] = {
    "transcription_ins": ["asr_wer", "speaker_similarity_wespeaker"],
    "transcription_del": ["asr_wer", "speaker_similarity_wespeaker"],
    "transcription_sub": ["asr_wer", "speaker_similarity_wespeaker"],
    "transcription_replace_sentence": ["asr_wer", "speaker_similarity_wespeaker"],
    "transcription_add_paralinguistic": ["asr_wer", "speaker_similarity_wespeaker"],
    "style_whisper": ["llm_judge_caption_llm", "asr_wer"],
    "style_emotion": ["llm_judge_caption_llm", "asr_wer"],
    "audio_effect_speed": ["speed_duration"],
    "audio_effect_volume": ["volume_loudness"],
    "audio_effect_pitch": ["pitch_shift"],
    "audio_effect_reverb": ["llm_judge_gemini", "llm_judge_caption_llm"],
    "audio_effect_dereverb": ["asr_wer"],
}

# region: utils
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _expand_env_in_string(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")
    return _ENV_PATTERN.sub(repl, value)

def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _expand_env_in_string(value)
    return value

def load_config(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return _expand_env(raw)

def infer_sample_id(record: dict[str, Any]) -> str:
    if "id" in record and record["id"]:
        return str(record["id"])
    if "utt_id" in record and record["utt_id"]:
        return str(record["utt_id"])
    return ""

def load_metadata_by_type(metadata_input: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = Path(metadata_input)
    files: list[Path]
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"metadata path not found: {path}")

    by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for file_path in files:
        fallback_type = file_path.stem
        for row in read_jsonl(file_path):
            sample_id = infer_sample_id(row)
            if not sample_id:
                continue
            row["id"] = sample_id
            if "edit_type" in row and row["edit_type"]:
                task_type = str(row["edit_type"])
            else:
                task_type = fallback_type
            by_type.setdefault(task_type, {})[sample_id] = row
    return by_type

def load_scp_by_type(data_dir: str | Path) -> dict[str, dict[str, str]]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"data-dir is not a directory: {data_path}")

    result: dict[str, dict[str, str]] = {}
    for scp_file in sorted(data_path.glob("*.scp")):
        task_type = scp_file.stem
        result[task_type] = read_scp(scp_file)
    return result

def build_samples(
    task_type: str,
    metadata_by_type: dict[str, dict[str, dict[str, Any]]],
    scp_by_type: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    meta = metadata_by_type[task_type]
    scp = scp_by_type[task_type]

    samples: list[dict[str, Any]] = []
    for sample_id, eval_audio_path in scp.items():
        if sample_id not in meta:
            continue
        sample = dict(meta[sample_id])
        sample["sample_id"] = sample_id
        sample["id"] = sample_id
        sample["edit_type"] = task_type
        sample["eval_audio_path"] = eval_audio_path
        samples.append(sample)

    stats = {
        "metadata_count": len(meta),
        "scp_count": len(scp),
        "matched_count": len(samples),
    }
    return samples, stats

def print_task_summary(task_type: str, summaries: dict[str, dict[str, Any]], output_path: str) -> None:
    print(f"\n[{task_type}] -> {output_path}")
    for scorer_name, summary in summaries.items():
        avg_score = summary["avg_score"] if "avg_score" in summary else None
        avg_str = "N/A" if avg_score is None else f"{avg_score:.4f}"
        valid = int(summary["valid"]) if "valid" in summary else 0
        total = int(summary["total"]) if "total" in summary else 0
        errors = int(summary["errors"]) if "errors" in summary else 0
        print(f"  - {scorer_name}: avg={avg_str} valid={valid}/{total} errors={errors}")
        if "high_accuracy" in summary or "low_accuracy" in summary:
            high = summary["high_accuracy"] if "high_accuracy" in summary else None
            low = summary["low_accuracy"] if "low_accuracy" in summary else None
            high_str = "N/A" if high is None else f"{float(high):.4f}"
            low_str = "N/A" if low is None else f"{float(low):.4f}"
            print(f"    emotion: high_acc={high_str} low_acc={low_str}")
        if "avg_hyp_similarity" in summary or "avg_ref_similarity" in summary:
            hyp = summary["avg_hyp_similarity"] if "avg_hyp_similarity" in summary else None
            ref = summary["avg_ref_similarity"] if "avg_ref_similarity" in summary else None
            hyp_str = "N/A" if hyp is None else f"{float(hyp):.4f}"
            ref_str = "N/A" if ref is None else f"{float(ref):.4f}"
            print(f"    speaker: hyp_sim={hyp_str} ref_sim={ref_str}")
        if "submetric_avg" in summary and isinstance(summary["submetric_avg"], dict):
            submetric_avg = summary["submetric_avg"]
            if submetric_avg:
                items = []
                for k, v in sorted(submetric_avg.items()):
                    if v is None:
                        items.append(f"{k}=N/A")
                    else:
                        items.append(f"{k}={float(v):.3f}")
                print(f"    submetrics: {' '.join(items)}")

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_scp(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid SCP line at {path}:{ln}: {line}")
            utt_id, wav_path = parts
            mapping[utt_id] = wav_path
    return mapping

def normalize_scorer_entries(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    entries: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        raise ValueError("tasks.<type>.scorers must be a list")
    for item in raw:
        if isinstance(item, str):
            entries.append({"name": item})
        elif isinstance(item, dict) and "name" in item and item["name"]:
            entries.append(dict(item))
        else:
            raise ValueError(f"Invalid scorer entry: {item}")
    return entries

def get_task_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "tasks" not in config:
        return {}
    tasks_cfg = config["tasks"]
    if not tasks_cfg:
        return {}
    if not isinstance(tasks_cfg, dict):
        raise ValueError("tasks must be a mapping")
    return {str(k): dict(v or {}) for k, v in tasks_cfg.items()}

def build_default_task_configs(
    metadata_by_type: dict[str, dict[str, dict[str, Any]]],
    scp_by_type: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    available = sorted(set(metadata_by_type).intersection(set(scp_by_type)))
    task_cfg: dict[str, dict[str, Any]] = {}
    for task_type in available:
        if task_type in DEFAULT_TASK_SCORERS:
            scorers = DEFAULT_TASK_SCORERS[task_type]
        else:
            scorers = []
        if scorers:
            task_cfg[task_type] = {"scorers": [{"name": name} for name in scorers]}
    return task_cfg

def collect_used_scorers(tasks_cfg: dict[str, dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for task_cfg in tasks_cfg.values():
        if "scorers" not in task_cfg:
            continue
        entries = normalize_scorer_entries(task_cfg["scorers"])
        for entry in entries:
            name = str(entry["name"])
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names

# endregion

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated data with modular scorers.")
    parser.add_argument("--metadata", required=True, help="Path to metadata jsonl file or directory of jsonl files")
    parser.add_argument("--data-dir", required=True, help="Path to data directory containing *.scp")
    parser.add_argument("--config", required=True, help="Path to eval yaml config")
    parser.add_argument("--output-dir", default=None, help="Output directory for *.results (default: data-dir)")
    parser.add_argument(
        "--types",
        default="",
        help="Comma-separated task types to evaluate; default evaluates all configured tasks.",
    )
    args = parser.parse_args()


    config = load_config(args.config)
    if "runtime" in config:
        runtime_cfg = config["runtime"]
    else:
        runtime_cfg = {}
    if "use_default_task_mapping" in runtime_cfg:
        use_default_task_mapping = bool(runtime_cfg["use_default_task_mapping"])
    else:
        use_default_task_mapping = False

    metadata_by_type = load_metadata_by_type(args.metadata)
    scp_by_type = load_scp_by_type(args.data_dir)

    tasks_cfg = get_task_configs(config)
    if not tasks_cfg and use_default_task_mapping:
        tasks_cfg = build_default_task_configs(metadata_by_type, scp_by_type)
    if not tasks_cfg:
        raise RuntimeError("No tasks configured. Please set `tasks` in YAML.")

    selected_types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else []
    if selected_types:
        tasks_cfg = {k: v for k, v in tasks_cfg.items() if k in set(selected_types)}
        if not tasks_cfg:
            raise RuntimeError(f"No tasks matched --types={args.types}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_scorers = collect_used_scorers(tasks_cfg)
    scorers_dict = {}
    for scorer_name in used_scorers:
        print(f"Initializing scorer: {scorer_name}")
        scorer_cls = SCORER_CLASSES[scorer_name]
        scorer_cfg = config.get("scorers", {}).get(scorer_name, {})
        scorer_kwargs = dict(scorer_cfg)

        if "num_workers" in runtime_cfg:
            scorer_kwargs["num_workers"] = runtime_cfg["num_workers"]
        if "use_gpu" in runtime_cfg:
            scorer_kwargs["use_gpu"] = runtime_cfg["use_gpu"]

        scorer_kwargs["global_models"] = config.get("models", {})
        scorers_dict[scorer_name] = scorer_cls(name=scorer_name, **scorer_kwargs)
        print(f"Initialized scorer: {scorer_name} with args: {scorer_kwargs}")


    type_list = list(tasks_cfg.keys())
    print(f"Configured tasks: {', '.join(type_list)}")

    for task_type in tqdm(type_list, desc="tasks"):
        task_entry = tasks_cfg[task_type]
        if "disabled" in task_entry and bool(task_entry["disabled"]):
            print(f"Skip task {task_type}: disabled=true")
            continue
        if task_type not in metadata_by_type:
            print(f"Skip task {task_type}: missing metadata")
            continue
        if task_type not in scp_by_type:
            print(f"Skip task {task_type}: missing {task_type}.scp")
            continue

        samples, stats = build_samples(task_type, metadata_by_type, scp_by_type)
        if not samples:
            print(
                f"Skip task {task_type}: no matched samples "
                f"(metadata={stats['metadata_count']} scp={stats['scp_count']})"
            )
            continue

        if "scorers" in task_entry:
            scorer_entries = normalize_scorer_entries(task_entry["scorers"])
        else:
            scorer_entries = []
        if not scorer_entries:
            print(f"Skip task {task_type}: no scorers configured")
            continue

        per_sample: dict[str, dict[str, Any]] = {}
        for sample in samples:
            sid = str(sample["sample_id"])
            per_sample[sid] = {
                "id": sid,
                "edit_type": task_type,
                "eval_audio_path": str(sample["eval_audio_path"]),
                "metrics": {},
            }

        task_summaries: dict[str, dict[str, Any]] = {}
        
        scorer_instances = []
        for scorer_entry in scorer_entries:
            scorer_instances.append((scorers_dict[scorer_entry["name"]], scorer_entry))

        results = Parallel(n_jobs=runtime_cfg.get("num_workers", 4), prefer="threads")(
            delayed(run_scorer)(inst, entry, samples)
            for inst, entry in scorer_instances
        )

        for scorer_name, rows, summary in results:
            task_summaries[scorer_name] = summary
            for row in rows:
                sid = str(row["sample_id"])
                metric_payload = dict(row)
                metric_payload.pop("sample_id", None)
                metric_payload.pop("scorer", None)
                if sid in per_sample:
                    per_sample[sid]["metrics"][scorer_name] = metric_payload

        result_rows = [per_sample[str(sample["sample_id"])] for sample in samples]
        output_path = output_dir / f"{task_type}.results"
        write_jsonl(output_path, result_rows)
        print_task_summary(task_type, task_summaries, str(output_path))


if __name__ == "__main__":
    main()
