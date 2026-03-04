from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import json
import re

def safe_mean(values: list[float]) -> float | None:
    if not values or len(values) == 0:
        return None
    return float(sum(values) / len(values))


def compute_aspect_avg(
    judge_resp: dict[str, Any],
    exclude_keys: tuple[str, ...] = ("reason",),
) -> tuple[float, dict[str, float]]:
    """Extract all numeric aspect scores from an LLM judge response and compute their mean.

    Parameters
    ----------
    judge_resp : dict
        The parsed JSON response from the LLM judge.
    exclude_keys : tuple of str
        Keys to exclude from aspect extraction (e.g., "reason").

    Returns
    -------
    avg_score : float
        Mean of all extracted numeric aspect scores, or 0.0 if none found.
    aspect_scores : dict[str, float]
        Mapping of aspect name to its numeric score.
    """
    aspect_scores: dict[str, float] = {}
    for k, v in judge_resp.items():
        if k in exclude_keys:
            continue
        if isinstance(v, (int, float)):
            aspect_scores[k] = float(v)
    if not aspect_scores:
        return 0.0, aspect_scores
    avg = sum(aspect_scores.values()) / len(aspect_scores)
    return round(avg, 4), aspect_scores


def auto_detect_score_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Detect numeric aspect keys stored in the ``extra`` dict of result rows.

    Returns ``["score"] + sorted(detected_aspect_keys)`` so that
    ``summarize_metric_rows`` tracks both the overall avg and each aspect.
    """
    aspect_keys: set[str] = set()
    for row in rows:
        extra = row.get("extra") or {}
        for k, v in extra.items():
            if k not in ("judge_resp",) and isinstance(v, (int, float)):
                aspect_keys.add(k)
    if aspect_keys:
        return ["score"] + sorted(aspect_keys)
    return ["score"]


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


# ---------------------------------------------------------------------------
# vLLM caption rewriter (shared by CLAP / FLAM scorers)
# ---------------------------------------------------------------------------

_CAPTION_REWRITE_SYSTEM = (
    "You are a concise audio caption editor. "
    "Rewrite audio descriptions to only one sentence, "
    "preserving the key audio content faithfully."
)


def rewrite_caption_vllm(caption: str, client: Any) -> str:
    """Rewrite *caption* to ≤20 words using a ``VLLMClient`` instance.

    Parameters
    ----------
    caption : str
        The original caption text.
    client : VLLMClient
        An initialised vLLM client (from ``llm_judge_caption_llm``).

    Returns
    -------
    str
        The rewritten caption, or the original *caption* on failure.
    """
    result = client.chat_completion(
        messages=[
            {"role": "system", "content": _CAPTION_REWRITE_SYSTEM},
            {"role": "user", "content": caption},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    if not result or not str(result).strip():
        return caption
    return str(result).strip()


# ---------------------------------------------------------------------------
# Combined caption + human-labels rewriter (for CLAP / FLAM scorers)
# ---------------------------------------------------------------------------

_LABELS_REWRITE_SYSTEM = (
    "You are a concise audio caption editor. "
    "Given an audio caption and event labels, compress them into ONE "
    "descriptive sentence (no more than 20 words) that captures the key "
    "audio content faithfully."
)


def rewrite_labels_vllm(
    caption: str | None,
    human_labels: list[str] | None,
    client: Any,
) -> str | None:
    """Rewrite *caption* + *human_labels* to ≤20 words via a VLLMClient.

    Parameters
    ----------
    caption : str | None
        The audio caption text (e.g. ``metadata.main.audio_caption``).
    human_labels : list[str] | None
        Event labels (e.g. ``["Music", "Opera"]``).
    client : VLLMClient
        An initialised vLLM client.

    Returns
    -------
    str | None
        The rewritten text, or *None* when both inputs are empty.
    """
    parts: list[str] = []
    if caption and isinstance(caption, str) and caption.strip():
        parts.append(f"Caption: {caption.strip()}")
    if human_labels:
        joined = ", ".join(str(l) for l in human_labels if str(l).strip())
        if joined:
            parts.append(f"Labels: {joined}")
    if not parts:
        return None

    user_msg = "\n".join(parts)
    result = client.chat_completion(
        messages=[
            {"role": "system", "content": _LABELS_REWRITE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    if not result or not str(result).strip():
        # fallback: join available info
        if human_labels:
            return ", ".join(str(l) for l in human_labels if str(l).strip())
        return caption.strip() if caption else None
    return str(result).strip()


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
