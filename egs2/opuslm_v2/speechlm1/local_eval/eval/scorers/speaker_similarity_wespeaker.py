from __future__ import annotations

from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from .base import safe_mean, BaseScorer


class SpeakerSimilarityWespeakerScorer(BaseScorer):
    """Speaker similarity scorer using WeSpeaker batch embedding extraction."""

    score_keys = ["score", "similarity"]

    def __init__(
        self,
        *,
        name: str,
        model_id: str = "english",
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
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
        pcm, sr = torchaudio.load(wav_path)
        pcm = pcm.to(torch.float)
        if sr != self.we.resample_rate:
            pcm = torchaudio.functional.resample(pcm, sr, self.we.resample_rate)
        feats = self.we.compute_features(pcm, sample_rate=self.we.resample_rate, cmn=True)
        return feats.squeeze(0)  # (T, F)

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

        for i in tqdm(
            range(0, len(path_list), self.batch_size),
            desc=f"{self.name} [embed]",
            leave=False,
        ):
            batch_paths = path_list[i : i + self.batch_size]
            batch_feats = [feats_map[p] for p in batch_paths]

            # Pad along T to max length in batch
            max_t = max(f.shape[0] for f in batch_feats)
            padded = torch.zeros(len(batch_feats), max_t, batch_feats[0].shape[1])
            for j, f in enumerate(batch_feats):
                padded[j, : f.shape[0]] = f

            outputs = self.we.model(padded.to(self.device))
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            for path, emb in zip(batch_paths, outputs.detach().cpu().unbind(0)):
                embeddings[path] = emb

        return embeddings

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        cosine_score = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        return cosine_score.item()

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
                sim = self._cosine(embeddings[sample["audio_path"]], embeddings[sample["eval_audio_path"]])
                cos_score = (sim + 1.0) / 2  # normalize: [-1, 1] => [0, 1]

                sims.append(sim)
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=cos_score,
                        valid=True,
                        reason=f"sim={sim:.4f}",
                        extra={"similarity": sim},
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
        summary["avg_similarity"] = safe_mean(sims)
        return rows, summary
