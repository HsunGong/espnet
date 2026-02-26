from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import json
import re

def safe_mean(values: list[float]) -> float | None:
    if not values or len(values) == 0:
        return None
    return float(sum(values) / len(values))


def summarize_metric_rows(
    rows: list[dict[str, Any]],
    score_keys: list[str] = ["score"],
) -> dict[str, Any]:

    key_scores: dict[str, list[float]] = {k: [] for k in score_keys}
    errors = 0
    valid = 0

    for row in rows:
        if "error" in row and row["error"]:
            errors += 1
        if "valid" in row and bool(row["valid"]):
            valid += 1
            extra = row.get("extra") or {}
            for key in score_keys:
                val = row.get("score") if key == "score" else extra.get(key)
                if val is not None:
                    try:
                        key_scores[key].append(float(val))
                    except (TypeError, ValueError):
                        pass

    result: dict[str, Any] = {
        "valid": valid,
        "total": len(rows),
        "errors": errors,
    }
    for key in score_keys:
        result[f"avg_{key}"] = safe_mean(key_scores[key])
    return result


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


class BaseScorer(ABC):
    score_keys = ["score"]

    def __init__(self, *, name: str, resume: bool = True, **kwargs: Any) -> None:
        self.name = name
        self.task_cfg: dict[str, Any] = {}
        self.resume: bool = resume

    def configure_task(self, task_cfg: dict[str, Any] | None) -> None:
        self.task_cfg = dict(task_cfg or {})
        # Allow any task config to override score_keys (useful for LLM scorers)
        raw_keys = self.task_cfg.get("score_keys")
        if isinstance(raw_keys, list) and raw_keys:
            self.score_keys = [str(k) for k in raw_keys]

    @abstractmethod
    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def make_result(
        self,
        *,
        sample_id: str,
        score: float | None,
        valid: bool,
        error: str | None = None,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "scorer": self.name,
            "score": score,
            "valid": valid,
            "error": error,
            "reason": reason,
            "extra": extra or {},
        }

    def finalize(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return rows, summarize_metric_rows(rows, score_keys=self.score_keys)
