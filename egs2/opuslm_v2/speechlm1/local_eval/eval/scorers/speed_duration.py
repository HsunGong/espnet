from __future__ import annotations

from typing import Any

import soundfile as sf
from tqdm import tqdm

from .base import extract_first_float
from .base import BaseScorer


class SpeedDurationScorer(BaseScorer):
    score_keys = ["score", "rel_error", "pred_duration", "expected_duration"]

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def __init__(self, *, name: str, relative_tolerance: float = 0.15, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.relative_tolerance = relative_tolerance

    def _extract_speed_factor(self, sample: dict[str, Any]) -> float | None:
        candidates = []
        edit_kwargs = sample.get("edit_kwargs")
        if isinstance(edit_kwargs, dict):
            for key in ("speed", "rate", "effect_param", "param", "value"):
                if key in edit_kwargs:
                    candidates.append(edit_kwargs[key])
        for key in ("effect_param", "edit_prompt", "edit_operation", "speed_factor", "param"):
            if key in sample:
                candidates.append(sample[key])
        for value in candidates:
            parsed = extract_first_float(value)
            if parsed is not None and parsed > 0:
                return parsed
        return None

    def _duration(self, audio_path: str) -> float:
        info = sf.info(audio_path)
        if info.samplerate <= 0:
            raise RuntimeError(f"invalid sample rate: {audio_path}")
        return float(info.frames / info.samplerate)

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tol = self.relative_tolerance

        # Phase 1: read durations for all samples
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [measure]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                speed_factor = self._extract_speed_factor(sample)
                if speed_factor is None:
                    raise RuntimeError("unable to parse speed factor")
                orig_duration = float(sample.get("duration") or sample.get("original_duration") or 0.0)
                if orig_duration <= 0:
                    raise RuntimeError("missing/invalid original duration")
                pred_duration = self._duration(str(sample["eval_audio_path"]))
                pending.append({
                    "sample_id": sample_id,
                    "speed_factor": speed_factor,
                    "orig_duration": orig_duration,
                    "pred_duration": pred_duration,
                })
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="speed_duration_failed",
                        reason=str(exc),
                    )
                )

        # Phase 2: compute scores
        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            try:
                speed_factor = item["speed_factor"]
                orig_duration = item["orig_duration"]
                pred_duration = item["pred_duration"]
                expected = orig_duration / speed_factor
                rel_error = abs(pred_duration - expected) / max(expected, 1e-6)
                score = self._clamp01(1.0 - rel_error / max(tol, 1e-6))
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"rel_error={rel_error:.4f}",
                        extra={
                            "speed_factor": speed_factor,
                            "expected_duration": expected,
                            "pred_duration": pred_duration,
                            "rel_error": rel_error,
                        },
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="speed_duration_failed",
                        reason=str(exc),
                    )
                )
        return self.finalize(rows)
