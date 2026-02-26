from __future__ import annotations

import math
from typing import Any

import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

from .base import extract_first_float
from .base import BaseScorer


class VolumeLoudnessScorer(BaseScorer):
    score_keys = ["score", "delta_err_db", "orig_lufs", "pred_lufs"]

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def __init__(self, *, name: str, delta_tolerance_db: float = 3.0, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.delta_tolerance_db = delta_tolerance_db

    def _extract_volume_factor(self, sample: dict[str, Any]) -> float | None:
        candidates = []
        edit_kwargs = sample.get("edit_kwargs")
        if isinstance(edit_kwargs, dict):
            for key in ("volume", "gain", "effect_param", "param", "value"):
                if key in edit_kwargs:
                    candidates.append(edit_kwargs[key])
        for key in ("effect_param", "edit_prompt", "edit_operation", "volume_factor", "param"):
            if key in sample:
                candidates.append(sample[key])
        for value in candidates:
            parsed = extract_first_float(value)
            if parsed is not None and parsed > 0:
                return parsed
        return None

    def _lufs(self, audio_path: str) -> float:
        audio, sr = sf.read(audio_path)
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(audio))

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tol_db = self.delta_tolerance_db

        # Phase 1: read LUFS measurements for all samples
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [measure]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                factor = self._extract_volume_factor(sample)
                if factor is None:
                    raise RuntimeError("unable to parse volume factor")
                orig_audio = str(sample.get("audio_path") or "")
                pred_audio = str(sample.get("eval_audio_path") or "")
                if not orig_audio or not pred_audio:
                    raise RuntimeError("missing audio paths")
                orig_lufs = self._lufs(orig_audio)
                pred_lufs = self._lufs(pred_audio)
                pending.append({
                    "sample_id": sample_id,
                    "factor": factor,
                    "orig_lufs": orig_lufs,
                    "pred_lufs": pred_lufs,
                })
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="volume_loudness_failed",
                        reason=str(exc),
                    )
                )

        # Phase 2: compute scores
        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            try:
                factor = item["factor"]
                orig_lufs = item["orig_lufs"]
                pred_lufs = item["pred_lufs"]
                observed_delta = pred_lufs - orig_lufs
                expected_delta = 20.0 * math.log10(factor)
                err = abs(observed_delta - expected_delta)
                score = self._clamp01(1.0 - err / max(tol_db, 1e-6))
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"delta_err_db={err:.3f}",
                        extra={
                            "volume_factor": factor,
                            "orig_lufs": orig_lufs,
                            "pred_lufs": pred_lufs,
                            "expected_delta_db": expected_delta,
                            "observed_delta_db": observed_delta,
                            "delta_err_db": err,
                        },
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="volume_loudness_failed",
                        reason=str(exc),
                    )
                )
        return self.finalize(rows)
