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
            raise ValueError("f0 must be positive")
        return 12.0 * np.log2(f0_b / f0_a)

    def _extract_pitch_shift(self, sample: dict[str, Any]) -> float | None:
        candidates = []
        edit_kwargs = sample.get("edit_kwargs")
        if isinstance(edit_kwargs, dict):
            for key in ("pitch", "semitone", "effect_param", "param", "value"):
                if key in edit_kwargs:
                    candidates.append(edit_kwargs[key])
        for key in ("effect_param", "edit_prompt", "edit_operation", "pitch_shift", "param"):
            if key in sample:
                candidates.append(sample[key])
        for value in candidates:
            parsed = extract_first_float(value)
            if parsed is not None:
                return parsed
        return None

    def _median_f0(self, audio_path: str) -> float:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        voiced = f0[~np.isnan(f0)]
        if voiced.size == 0:
            raise RuntimeError("no voiced frame detected")
        return float(np.median(voiced))

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tol = float(self.cfg.get("semitone_tolerance", 1.5))

        # ------------------------------------------------------------------
        # Phase 1: extract F0 from all audio files
        # ------------------------------------------------------------------
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [f0]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                expected_shift = self._extract_pitch_shift(sample)
                if expected_shift is None:
                    raise RuntimeError("unable to parse pitch shift")
                orig_audio = str(sample.get("audio_path") or "")
                pred_audio = str(sample.get("eval_audio_path") or "")
                if not orig_audio or not pred_audio:
                    raise RuntimeError("missing audio paths")
                f0_orig = self._median_f0(orig_audio)
                f0_pred = self._median_f0(pred_audio)
                pending.append({
                    "sample_id": sample_id,
                    "expected_shift": expected_shift,
                    "f0_orig": f0_orig,
                    "f0_pred": f0_pred,
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

        # ------------------------------------------------------------------
        # Phase 2: compute scores
        # ------------------------------------------------------------------
        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            try:
                f0_orig = item["f0_orig"]
                f0_pred = item["f0_pred"]
                expected_shift = item["expected_shift"]
                observed_shift = self._semitone_diff(f0_orig, f0_pred)
                err = abs(observed_shift - expected_shift)
                score = self._clamp01(1.0 - err / max(tol, 1e-6))
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"semitone_err={err:.3f}",
                        extra={
                            "expected_shift": expected_shift,
                            "observed_shift": observed_shift,
                            "semitone_err": err,
                            "f0_orig": f0_orig,
                            "f0_pred": f0_pred,
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
