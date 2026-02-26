from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from .base import safe_mean, BaseScorer

# Path to UniSpeech speaker-verification directory (contains verification.py / models/)
sys.path.insert(0, "seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification")


class SpeakerSimilarityWavlmScorer(BaseScorer):
    """Speaker similarity scorer using WavLM-Large (UniSpeech ECAPA-TDNN) embeddings.

    The model accepts raw 16 kHz waveforms and produces 256-dim embeddings.
    Similarity is cosine distance in the same range [-1, 1] as WeSpeaker.
    """

    score_keys = ["score", "sim"]

    def __init__(
        self,
        *,
        name: str,
        checkpoint: str,
        model_name: str = "wavlm_large",
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.batch_size = batch_size

        # Import lazily so the sys.path insertion above is guaranteed to have
        # happened before the import (works even when this module is imported
        # at the top-level of a larger package).
        from verification import init_model  # type: ignore[import]

        self.model = init_model(model_name, checkpoint)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load_wav(self, wav_path: str) -> torch.Tensor:
        """Load audio, resample to 16 kHz, return mono waveform (1, T)."""
        pcm, sr = torchaudio.load(wav_path)
        pcm = pcm.mean(dim=0, keepdim=True).float()  # stereo -> mono
        if sr != 16000:
            pcm = torchaudio.functional.resample(pcm, sr, 16000)
        return pcm  # (1, T)

    # ------------------------------------------------------------------
    # Embedding extraction (one file at a time – variable-length safe)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_embeddings(self, paths: list[str]) -> dict[str, torch.Tensor]:
        """Extract one embedding per unique path using padded batch inference."""
        unique_paths = list(dict.fromkeys(paths))  # preserve order, deduplicate

        # Pre-load all waveforms (CPU)
        wavs: list[torch.Tensor] = [
            self._load_wav(p).squeeze(0)  # (T,)
            for p in tqdm(unique_paths, desc=f"{self.name} [load]", leave=False)
        ]

        embeddings: dict[str, torch.Tensor] = {}

        for i in tqdm(
            range(0, len(unique_paths), self.batch_size),
            desc=f"{self.name} [embed]",
            leave=False,
        ):
            batch_paths = unique_paths[i : i + self.batch_size]
            batch_wavs = wavs[i : i + self.batch_size]

            # # Pad all waveforms to the longest one in the batch
            # max_len = max(w.shape[0] for w in batch_wavs)
            # padded = torch.zeros(len(batch_wavs), max_len)
            # # padding_mask = torch.zeros(len(batch_wavs), max_len, dtype=torch.bool)
            # for j, w in enumerate(batch_wavs):
            #     padded[j, : w.shape[0]] = w
            #     # padding_mask[j, w.shape[0] :] = True

            # padded = padded.to(self.device)  # (B, T)
            # embs = self.model(padded) # , padding_mask=padding_mask)         # (B, emb_dim)

            # there is no mask support, so we can not use batch-decode
            for path, wav in zip(batch_paths, batch_wavs):
                wav = wav.unsqueeze(0)  # (1, T)
                emb = self.model(wav.to(self.device))  # (1, emb_dim)
                embeddings[path] = emb.squeeze(0)  # (emb_dim,)

        return embeddings

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        unique_paths: list[str] = []
        seen: set[str] = set()
        for sample in samples:
            for key in ("audio_path", "eval_audio_path"):
                p = sample.get(key)
                if p and p not in seen:
                    unique_paths.append(p)
                    seen.add(p)

        embeddings = self._extract_embeddings(unique_paths)

        rows: list[dict[str, Any]] = []
        sims: list[float] = []

        for sample in tqdm(samples, desc=f"{self.name} [score]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                assert sample["eval_audio_path"] is not None, "missing eval_audio_path"
                sim = self._cosine(
                    embeddings[sample["audio_path"]],
                    embeddings[sample["eval_audio_path"]],
                )
                cos_score = (sim + 1.0) / 2  # [-1, 1] -> [0, 1]

                sims.append(sim)
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=cos_score,
                        valid=True,
                        reason=f"sim={sim:.4f}",
                        extra={"sim": sim},
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="speaker_similarity_failed",
                        reason=str(exc),
                    )
                )

        rows, summary = self.finalize(rows)
        summary["avg_sim"] = safe_mean(sims)
        return rows, summary
