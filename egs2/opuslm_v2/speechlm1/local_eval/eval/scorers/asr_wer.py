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

global_tags = ["[sigh]", "[laugh]", "[exhale]", "[snort]", "[cough]", "[uhm]", "[surprise-oh]", "[surprise-wa]", "[dissatisfaction-hnn]", "[question-ah]", "[question-yi]"]

def normalize_text(text: str) -> str:
    norm_text = text.lower().strip()
    # also remove some flags (which is not necessary in ASR evaluation)
    for tag in global_tags:
        norm_text = norm_text.replace(tag, "")
    norm_text = english_normalizer(norm_text)
    return norm_text


def _load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


class ASRWERScorer(BaseScorer):
    score_keys = ["wer", "edit_acc", "hits", "substitutions", "deletions", "insertions"]

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "openai/whisper-large-v3",
        # assistant_model_name: str | None = "distil-whisper/distil-large-v2",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        language: str = "english",
        batch_size: int = 8,
        gen_kwargs: dict = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        from transformers import (
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        self.language = language
        self.batch_size = batch_size
        self.gen_kwargs = gen_kwargs

        self.device = device
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
            sample_id = sample["id"]
            audio_path = sample.get("eval_audio_path")
            origin_text = sample["text"]
            ref_text = sample["target_text"]

            if not audio_path:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="missing_eval_audio_path",
                ))
                continue
            if not ref_text:
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

            valid_items.append({"sample_id": sample_id, "audio": audio, "ref_text": ref_text, "origin_text": origin_text})

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

            ref_norm = normalize_text(item["ref_text"])
            origin_norm = normalize_text(item["origin_text"])

            # find differ phase between ref_norm and origin_norm?
            word_out = jiwer_process_words(origin_norm, ref_norm)
            edit_words = []
            for ali in word_out.alignments[0]:
                ori_word = " ".join(word_out.references[0][ali.ref_start_idx:ali.ref_end_idx])
                ref_word = " ".join(word_out.hypotheses[0][ali.hyp_start_idx:ali.hyp_end_idx])

                if ali.type == "substitute":
                    edit_words.append((ori_word, ref_word))
                elif ali.type == "delete":
                    edit_words.append((ori_word, None))
                elif ali.type == "insert":
                    edit_words.append((None, ref_word))

            # print(edit_words)
            # print(word_out.alignments)
            # print(word_out.references)

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

                edit_acc = []
                for ori_word, ref_word in edit_words:
                    if ori_word and ref_word:
                        edit_acc.append(ref_word in hyp_norm and ori_word not in hyp_norm)
                    elif ref_word:
                        edit_acc.append(ref_word in hyp_norm)
                    elif ori_word:
                        edit_acc.append(ori_word not in hyp_norm)

                scored_rows.append(self.make_result(
                    sample_id=sample_id,
                    score=wer_value,
                    valid=True,
                    reason=f"WER={wer_value:.2f}% C={C} S={S} D={D} I={I}",
                    extra={
                        "wer": wer_value,
                        "hits": C,
                        "substitutions": S,
                        "deletions": D,
                        "insertions": I,
                        "hyp_text": hyp_norm,
                        "ref_text": ref_norm,
                        "origin_text": origin_norm,
                        "edit_words": edit_words,
                        "edit_acc": safe_mean(edit_acc) or 1.0,
                    },
                ))
            except Exception as exc:
                error_rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="asr_wer_failed", reason=str(exc),
                ))

        rows, summary = self.finalize(error_rows + scored_rows)

        summary["submetric_avg"] = {}
        for k in self.score_keys:
            all_values = [row["extra"][k] for row in scored_rows if k in row["extra"]]
            summary["submetric_avg"][k] = sum(v for v in all_values if v)

        # recompute wer (S + D + I) / (H + S + D)
        try:
            summary["submetric_avg"]["wer"] = 100 * (summary["submetric_avg"]["substitutions"] + summary["submetric_avg"]["deletions"] + summary["submetric_avg"]["insertions"]) / (summary["submetric_avg"]["hits"] + summary["submetric_avg"]["substitutions"] + summary["submetric_avg"]["deletions"])
        except:
            summary["submetric_avg"]["wer"] = "N/A (no samples)"
    
        summary["avg_wer"] = summary["submetric_avg"]["wer"]
        summary["avg_edit_acc"] = summary["submetric_avg"]["edit_acc"] / len(scored_rows) if scored_rows else "N/A"
        return rows, summary
