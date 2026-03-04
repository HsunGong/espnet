from __future__ import annotations

from typing import Any

import torch
import torchaudio
from tqdm import tqdm
from torch.nn import functional as F

from .base import safe_mean, BaseScorer


class SpeakerSimilarityWespeakerScorer(BaseScorer):
    """Speaker similarity scorer using WeSpeaker batch embedding extraction."""

    score_keys = ["score", "sim"]

    def __init__(
        self,
        *,
        name: str,
        model_id: str = "english",
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.batch_size = batch_size

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        import wespeaker
        self.we = wespeaker.load_model(model_id)
        self.we.set_device(device)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load_feat(self, wav_path: str) -> torch.Tensor:
        """Load audio, resample to model rate, compute fbank features -> (T, F)."""
        try:
            pcm, sr = torchaudio.load(wav_path)
            pcm = pcm.to(torch.float)
            assert pcm.shape[-1] / sr > 0.5, "audio too short for embedding"
            if sr != self.we.resample_rate:
                pcm = torchaudio.functional.resample(pcm, sr, self.we.resample_rate)
            feats = self.we.compute_features(pcm, sample_rate=self.we.resample_rate, cmn=True)
            return feats.squeeze(0)  # (T, F)
        except:
            return None

    # ------------------------------------------------------------------
    # Batch embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_embeddings(self, paths: list[str]) -> dict[str, torch.Tensor]:
        """Extract one embedding per unique path via padded batch inference."""
        feats_map: dict[str, torch.Tensor] = {
            path: self._load_feat(path)
            for path in tqdm(paths, desc=f"{self.name} [load]", leave=False)
        }

        path_list = list(feats_map)
        embeddings: dict[str, torch.Tensor] = {}

        for path in tqdm(path_list, desc=f"{self.name} [paths]", leave=False):
            try:
                feat = self._load_feat(path)
                outputs = self.we.model(feat.unsqueeze(0).to(self.device))  # (1, D) or tuple
                outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = outputs.detach().cpu()
                embed = outputs.squeeze(0)
            except Exception as exc:
                print(f"Failed to extract embedding for {path}: {exc}")
                embed = None
            embeddings[path] = embed

        return embeddings

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a, b).item()

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Fail fast: KeyError if a sample is missing the required field
        unique_paths: set[str] = set()
        for sample in samples:
            if sample["audio_path"]:
                unique_paths.add(sample["audio_path"])
            if sample["eval_audio_path"]:
                unique_paths.add(sample["eval_audio_path"])

        embeddings = self._extract_embeddings(list(unique_paths))

        rows: list[dict[str, Any]] = []
        sims: list[float] = []

        for sample in tqdm(samples, desc=f"{self.name} [score]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                assert sample["eval_audio_path"] is not None, "missing eval_audio_path"
                try:
                    sim = self._cosine(embeddings[sample["audio_path"]], embeddings[sample["eval_audio_path"]])
                except:
                    sim = -1.0  # treat missing embedding as maximally dissimilar

                cos_score = (sim + 1.0) / 2  # normalize: [-1, 1] => [0, 1]

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
