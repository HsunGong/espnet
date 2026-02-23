from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from .base import safe_mean
from .base import BaseScorer


DEFAULT_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unk",
]


class EmotionModelscopeScorer(BaseScorer):
    """Emotion classification scorer based on modelscope emotion2vec pipeline."""
    score_keys = ["score", "confidence"]

    def __init__(self, *, name: str, cfg: dict[str, Any], runtime: dict[str, Any], global_config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, cfg=cfg, runtime=runtime, global_config=global_config)
        self.labels = [str(x).lower() for x in (self.cfg.get("labels") or DEFAULT_LABELS)]
        
        model_id = str(self.cfg.get("model", "iic/emotion2vec_plus_large"))
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline = pipeline(task=Tasks.emotion_recognition, model=model_id)

    def _extract_reference(self, sample: dict[str, Any]) -> tuple[str | None, str]:
        label_field = str(self.task_cfg.get("label_field") or self.cfg.get("label_field") or "emotion_label")
        level_field = str(self.task_cfg.get("level_field") or self.cfg.get("level_field") or "emotion_level")
        ref = sample.get(label_field)
        level = sample.get(level_field)

        if ref:
            ref_label = str(ref).lower()
            return ref_label, str(level or "unk").lower()

        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        parts = sample_id.split("_")
        if len(parts) >= 2:
            return parts[0].lower(), parts[1].lower()
        if parts:
            return parts[0].lower(), "unk"
        return None, "unk"

    def _predict_emotion(self, wav: np.ndarray) -> tuple[str, float]:
        result = self.pipeline(wav, granularity="utterance", extract_embedding=False)
        if isinstance(result, dict):
            result = [result]
        if not isinstance(result, list) or not result:
            raise RuntimeError(f"invalid emotion output: {result}")
        scores = result[0].get("scores")
        if not isinstance(scores, list) or not scores:
            raise RuntimeError(f"missing emotion scores: {result}")
        idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        if idx >= len(self.labels):
            raise RuntimeError(f"emotion index {idx} out of label range {len(self.labels)}")
        return self.labels[idx], float(scores[idx])

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import librosa

        sr = int(self.cfg.get("audio_sr", 16000))

        # Phase 1: load audio and run emotion predictor for all samples
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [predict]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                audio_path = str(sample.get("eval_audio_path") or "")
                if not audio_path:
                    raise RuntimeError("missing eval_audio_path")
                wav, _ = librosa.load(audio_path, sr=sr, mono=True)
                hyp_label, conf = self._predict_emotion(wav)
                pending.append({
                    "sample_id": sample_id,
                    "hyp_label": hyp_label,
                    "confidence": conf,
                    "sample": sample,
                })
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="emotion_eval_failed",
                        reason=str(exc),
                    )
                )

        # Phase 2: compute scores
        high_scores: list[float] = []
        low_scores: list[float] = []

        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            hyp_label = item["hyp_label"]
            conf = item["confidence"]
            sample = item["sample"]
            try:
                ref_label, level = self._extract_reference(sample)
                if not ref_label:
                    raise RuntimeError("missing emotion reference label")
                matched = hyp_label == ref_label
                score = 1.0 if matched else 0.0
                if level == "high":
                    high_scores.append(score)
                elif level == "low":
                    low_scores.append(score)
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"ref={ref_label}, hyp={hyp_label}",
                        extra={
                            "ref_label": ref_label,
                            "hyp_label": hyp_label,
                            "confidence": conf,
                            "level": level,
                        },
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="emotion_eval_failed",
                        reason=str(exc),
                    )
                )

        rows, summary = self.finalize(rows)
        summary["high_accuracy"] = safe_mean(high_scores)
        summary["low_accuracy"] = safe_mean(low_scores)
        return rows, summary
