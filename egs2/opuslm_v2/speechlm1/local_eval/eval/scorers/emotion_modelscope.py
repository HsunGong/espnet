from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from .base import BaseScorer

DEFAULT_LABELS: list[str] = [
    "happy", "angry", "sad", "humour", "confusion",
    "disgusted", "empathy", "embarrass", "fear",
    "surprised", "excited", "depressed", "coldness", "admiration",
]


class EmotionModelscopeScorer(BaseScorer):
    """Emotion accuracy scorer using funasr emotion2vec (batch inference)."""

    score_keys = ["score", "confidence"]

    def __init__(
        self,
        *,
        name: str,
        model: str = "iic/emotion2vec_plus_large",
        gen_kwargs: dict = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(name=model)
        self.gen_kwargs = gen_kwargs

        ourlabels =  ['happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted', 'empathy', 'embarrass', 'fear', 'surprised', 'excited', 'depressed', 'coldness', 'admiration']
        emotionlabels = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']

        self.label_mapping = {
            # our labels : emotion2vec-label
            "happy": "开心/happy",
            "angry": "生气/angry",
            "sad": "难过/sad",
            "humour": "<unk>",
            "confusion": "<unk>",
            "disgusted": "厌恶/disgusted",
            "empathy": "<unk>",
            "embarrass": "<unk>",
            "fear": "恐惧/fearful",
            "surprised": "吃惊/surprised",
            "excited": "<unk>",
            "depressed": "难过/sad",
            "coldness": "<unk>",
            "admiration": "<unk>",
        }

        from funasr import AutoModel
        self.model = AutoModel(model=model, hub="hf", trust_remote_code=True)

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # ── Phase 1: validate inputs ─────────────────────────────────────────
        error_rows: list[dict[str, Any]] = []
        valid_items: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [validate]", leave=False):
            sample_id = str(sample["sample_id"])
            audio_path = str(sample.get("eval_audio_path") or "")
            ref = sample["edit_kwargs"]["style"]
            emotion2vec_ref = self.label_mapping.get(ref, "<unk>")

            if not audio_path:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="missing_eval_audio_path",
                ))
                continue
            if not ref:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="missing_reference_label",
                ))
                continue

            valid_items.append({"sample_id": sample_id, "audio_path": audio_path, "ref": ref, "emotion2vec_ref": emotion2vec_ref})

        # ── Phase 2: batch generate ──────────────────────────────────────────
        audio_paths = [item["audio_path"] for item in valid_items]
        predictions: list[dict[str, Any]] = []  # {hyp_label, confidence} per valid item

        try:
            assert len(audio_paths), "No valid items to process"

            results = self.model.generate(audio_paths, granularity="utterance", extract_embedding=False, disable_pbar=True, **self.gen_kwargs)
            for idx in range(len(audio_paths)):
                max_idx = np.argmax(results[idx]["scores"])
                predictions.append({"hyp_label": results[idx]["labels"][max_idx], "confidence": float(results[idx]["scores"][max_idx]), "conf_dict": dict(zip(results[idx]["labels"], results[idx]["scores"]))})
        except Exception as exc:
            for item in valid_items:
                error_rows.append(self.make_result(
                    sample_id=item["sample_id"], score=None, valid=False,
                    error="emotion_generate_failed", reason=str(exc),
                ))
            return self.finalize(error_rows)

        # ── Phase 3: per-sample accuracy scoring ─────────────────────────────
        scored_rows: list[dict[str, Any]] = []

        for item, pred in tqdm(
            zip(valid_items, predictions),
            desc=f"{self.name} [score]",
            total=len(valid_items),
            leave=False,
        ):
            sample_id = item["sample_id"]
            ref = item["emotion2vec_ref"]
            hyp = pred["hyp_label"]
            conf = pred["confidence"]

            if hyp is None:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="empty_prediction",
                ))
                continue

            score = 1.0 if hyp == ref else 0.0
            scored_rows.append(self.make_result(
                sample_id=sample_id,
                score=score,
                valid=True,
                reason=f"ref={ref}, hyp={hyp}",
                extra={"ref_label": ref, "hyp_label": hyp, "confidence": conf},
            ))

        return self.finalize(error_rows + scored_rows)
