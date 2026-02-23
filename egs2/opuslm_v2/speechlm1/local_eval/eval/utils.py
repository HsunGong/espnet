from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

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


def render_template(template: str, context: dict[str, Any]) -> str:
    try:
        from jinja2 import Template
    except ModuleNotFoundError:
        return template
    return Template(template).render(**context)


def infer_sample_id(record: dict[str, Any]) -> str:
    if "id" in record and record["id"]:
        return str(record["id"])
    if "utt_id" in record and record["utt_id"]:
        return str(record["utt_id"])
    return ""


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


def extract_first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "valid"}:
        return True
    if text in {"false", "no", "n", "0", "invalid"}:
        return False
    return None



build_samples,
load_metadata_by_type,
load_scp_by_type,
print_task_summary,
write_jsonl,