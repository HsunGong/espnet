from __future__ import annotations

from typing import Any

from tqdm import tqdm
import re

from .base import BaseScorer

import whisper
from jiwer import process_words as jiwer_process_words
from whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    return english_normalizer(text.lower().strip())


class ASRWERScorer(BaseScorer):
    score_keys = ["score", "wer", "hits", "substitutions", "deletions", "insertions"]

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def __init__(self, *, name: str, cfg: dict[str, Any], runtime: dict[str, Any], global_config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, cfg=cfg, runtime=runtime)
        if whisper is None:
            raise RuntimeError("whisper module is not installed")
            
        wcfg = self.cfg.get("whisper", {})
        model_name = str(wcfg.get("model_name", "base"))
        device = str(wcfg.get("device", "cpu"))
        self.model = whisper.load_model(model_name, device=device)

    def _ref_text(self, sample: dict[str, Any]) -> str:
        return str(sample.get("target_text") or sample.get("text") or "").strip()

    def _transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Whisper model is unavailable")
        wcfg = self.cfg.get("whisper", {})
        language = wcfg.get("language")
        compute_type = str(wcfg.get("compute_type", "float16")).lower()
        fp16 = "16" in compute_type
        out = self.model.transcribe(audio_path, language=language, fp16=fp16)
        return str(out.get("text", "")).strip()

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Phase 1: validate inputs and transcribe all audio up-front
        pending: list[dict[str, Any]] = []   # samples ready for scoring
        rows: list[dict[str, Any]] = []      # results for invalid samples

        for sample in tqdm(samples, desc=f"{self.name} [transcribe]", leave=False):
            sample_id = str(sample["sample_id"])
            audio_path = str(sample.get("eval_audio_path") or "")
            ref = self._ref_text(sample)

            if not audio_path:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="missing_eval_audio_path",
                    )
                )
                continue
            if not ref:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="missing_reference_text",
                    )
                )
                continue

            try:
                hyp = self._transcribe(audio_path)
            except Exception as exc:  # pragma: no cover
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="transcription_failed",
                        reason=str(exc),
                    )
                )
                continue

            pending.append({
                "sample_id": sample_id,
                "ref_norm": normalize_text(ref),
                "hyp_norm": normalize_text(hyp),
            })

        # Phase 2: per-sentence CSDI via jiwer.process_words + ave-score
        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            ref_norm = item["ref_norm"]
            hyp_norm = item["hyp_norm"]
            try:
                word_out = jiwer_process_words(ref_norm, hyp_norm)
                C = word_out.hits           # Correct
                S = word_out.substitutions  # Substitutions
                D = word_out.deletions      # Deletions
                I = word_out.insertions     # Insertions
                wer_value = word_out.wer
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=wer_value * 100,
                        valid=True,
                        reason=f"WER={wer_value*100:.2f} C={C} S={S} D={D} I={I}",
                        extra={
                            "wer": wer_value,
                            "hits": C,
                            "substitutions": S,
                            "deletions": D,
                            "insertions": I,
                            "hyp_text": hyp_norm,
                            "ref_text": ref_norm,
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="asr_wer_failed",
                        reason=str(exc),
                    )
                )

        return self.finalize(rows)
