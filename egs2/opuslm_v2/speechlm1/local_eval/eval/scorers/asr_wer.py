from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from .base import BaseScorer, safe_mean

import torch
from jiwer import process_words as jiwer_process_words
from whisper_normalizer.english import EnglishTextNormalizer
import soundfile as sf
import librosa

english_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    return english_normalizer(text.lower().strip())


def _load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


class ASRWERScorer(BaseScorer):
    score_keys = ["wer", "hits", "substitutions", "deletions", "insertions"]

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "openai/whisper-large-v3",
        # assistant_model_name: str | None = "distil-whisper/distil-large-v2",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        language: str = "english",
        batch_size: int = 8,
        gen_kwargs: dict = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)

        from transformers import (
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        self.language = language
        self.batch_size = batch_size
        self.gen_kwargs = gen_kwargs


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True, attn_implementation="flash_attention_2"
        ).to(self.device).eval()

        # self.assistant_model = None
        # if assistant_model_name:
        #     self.assistant_model = WhisperForCausalLM.from_pretrained(
        #         assistant_model_name, torch_dtype=torch_dtype, trust_remote_code=True, attn_implementation="flash_attention_2"
        #     ).to(self.device).eval()

    @torch.inference_mode()
    def _batch_transcribe(self, audio_arrays: list[np.ndarray]) -> list[str]:
        inputs = self.processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        ).input_features.to(self.device, dtype=torch.bfloat16)

        # if self.assistant_model is not None:
        #     gen_kwargs["assistant_model"] = self.assistant_model

        predicted_ids = self.model.generate(inputs, language=self.language, **self.gen_kwargs)

        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # ── Phase 1: validate & load audio ──────────────────────────────────
        error_rows: list[dict[str, Any]] = []
        valid_items: list[dict[str, Any]] = []  # {sample_id, audio, ref}

        for sample in tqdm(samples, desc=f"{self.name} [load]", leave=False):
            sample_id = sample["sample_id"]
            audio_path = sample.get("eval_audio_path")
            ref = sample["target_text"]

            if not audio_path:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="missing_eval_audio_path",
                ))
                continue
            if not ref:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="missing_reference_text",
                ))
                continue
            try:
                audio = _load_audio(audio_path)
            except Exception as exc:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="audio_load_failed", reason=str(exc),
                ))
                continue

            valid_items.append({"sample_id": sample_id, "audio": audio, "ref": ref})

        # ── Phase 2: batch transcription ─────────────────────────────────────
        transcripts: dict[str, str] = {}  # sample_id -> hypothesis text

        for i in tqdm(
            range(0, len(valid_items), self.batch_size),
            desc=f"{self.name} [transcribe]",
            leave=False,
        ):
            batch = valid_items[i : i + self.batch_size]
            try:
                hyps = self._batch_transcribe([b["audio"] for b in batch])
                for item, hyp in zip(batch, hyps):
                    transcripts[item["sample_id"]] = hyp
            except Exception as exc:
                for item in batch:
                    error_rows.append(self.make_result(
                        sample_id=item["sample_id"], score=None, valid=False,
                        error="transcription_failed", reason=str(exc),
                    ))

        # ── Phase 3: per-sample WER scoring ──────────────────────────────────
        scored_rows: list[dict[str, Any]] = []

        for item in tqdm(valid_items, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            if sample_id not in transcripts:
                continue  # transcription already recorded as error

            ref_norm = normalize_text(item["ref"])
            hyp_norm = normalize_text(transcripts[sample_id])
            try:
                word_out = jiwer_process_words(ref_norm, hyp_norm)
                C, S, D, I = (
                    word_out.hits,
                    word_out.substitutions,
                    word_out.deletions,
                    word_out.insertions,
                )
                wer_value = word_out.wer
                scored_rows.append(self.make_result(
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
                ))
            except Exception as exc:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="asr_wer_failed", reason=str(exc),
                ))

        rows, summary = self.finalize(error_rows + scored_rows, score_keys=["wer"])

        summary["submetric_avg"] = {}
        for k in self.score_keys:
            all_values = [row["extra"][k] for row in scored_rows if k in row["extra"]]
            summary["submetric_avg"][k] = sum(all_values)

        # recompute wer (S + D + I) / (H + S + D)
        summary["submetric_avg"]["real-wer"] = 100 * (summary["submetric_avg"]["substitutions"] + summary["submetric_avg"]["deletions"] + summary["submetric_avg"]["insertions"]) / (summary["submetric_avg"]["hits"] + summary["submetric_avg"]["substitutions"] + summary["submetric_avg"]["deletions"])

        return rows, summary
