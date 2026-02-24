from __future__ import annotations

import numpy as np
from typing import Any

import librosa
from tqdm import tqdm

from .base import extract_first_float
from .base import BaseScorer


class PitchShiftScorer(BaseScorer):
    score_keys = ["score", "semitone_err", "f0_orig", "f0_pred"]

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def _semitone_diff(f0_a: float, f0_b: float) -> float:
        if f0_a <= 0 or f0_b <= 0:
            return 0.0
        return float(12.0 * np.log2(f0_b / f0_a))

    def __init__(self, *, name: str, semitone_tolerance: float = 1.0, **kwargs: Any) -> None:
        super().__init__(name=name)
        self.semitone_tolerance = semitone_tolerance

    def _extract_semitones(self, sample: dict[str, Any]) -> float | None:
        candidates = []
        edit_kwargs = sample.get("edit_kwargs")
        if isinstance(edit_kwargs, dict):
            for key in ("n_steps", "steps", "semitones", "effect_param", "param", "value"):
                if key in edit_kwargs:
                    candidates.append(edit_kwargs[key])
        for key in ("effect_param", "edit_prompt", "edit_operation", "semitones", "param"):
            if key in sample:
                candidates.append(sample[key])
        for value in candidates:
            parsed = extract_first_float(value)
            if parsed is not None:
                return parsed
        return None

    def _get_f0_median(self, audio_path: str) -> float:
        wav, sr = librosa.load(audio_path, sr=None, mono=True)
        f0, _, _ = librosa.pyin(wav, sr=sr, fmin=50, fmax=500)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) == 0:
            raise RuntimeError(f"no valid F0 found in {audio_path}")
        return float(np.median(valid_f0))

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tol_st = self.semitone_tolerance

        # Phase 1: extract F0 for all samples
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [measure]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                target_steps = self._extract_semitones(sample)
                if target_steps is None:
                    raise RuntimeError("unable to parse semitone steps")
                orig_audio = str(sample.get("audio_path") or "")
                pred_audio = str(sample.get("eval_audio_path") or "")
                if not orig_audio or not pred_audio:
                    raise RuntimeError("missing audio paths")
                orig_f0 = self._get_f0_median(orig_audio)
                pred_f0 = self._get_f0_median(pred_audio)
                pending.append({
                    "sample_id": sample_id,
                    "target_steps": target_steps,
                    "orig_f0": orig_f0,
                    "pred_f0": pred_f0,
                })
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="pitch_shift_failed",
                        reason=str(exc),
                    )
                )

        # Phase 2: compute scores
        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            try:
                target_steps = item["target_steps"]
                orig_f0 = item["orig_f0"]
                pred_f0 = item["pred_f0"]
                observed_steps = self._semitone_diff(orig_f0, pred_f0)
                err = abs(observed_steps - target_steps)
                score = self._clamp01(1.0 - err / max(tol_st, 1e-6))
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"semitone_err={err:.2f}",
                        extra={
                            "target_semitones": target_steps,
                            "observed_semitones": observed_steps,
                            "semitone_err": err,
                            "f0_orig": orig_f0,
                            "f0_pred": pred_f0,
                        },
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="pitch_shift_failed",
                        reason=str(exc),
                    )
                )
        return self.finalize(rows)
