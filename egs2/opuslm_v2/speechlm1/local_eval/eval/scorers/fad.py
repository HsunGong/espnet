from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from .base import BaseScorer


class FADScorer(BaseScorer):
    """Fréchet Audio Distance scorer (fully in-memory, no tmp files).

    FAD is a *corpus-level* metric; per-sample rows carry score=None.
    The FAD value is reported in summary["submetric_avg"]["fad"].

    Parameters
    ----------
    embedding : str
        fadtk model identifier, e.g. "encodec-emb", "clap-2023", "vggish", etc.
        See `fadtk.model_loader.get_model` for supported names.
    device : str
        Torch device string.  Falls back to "cpu" when CUDA is unavailable.
    """

    score_keys = ["fad"]

    def __init__(
        self,
        *,
        name: str,
        embedding: str = "encodec-emb",
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        try:
            from fadtk.model_loader import get_model
            from fadtk.fad import FrechetAudioDistance
        except ImportError as exc:
            raise ImportError(
                "fadtk is required for FADScorer. "
                "Install it following `tools/install_fadtk.sh`."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device_str = device

        self.ml = get_model(embedding)
        # Override device so the model lands on the right GPU/CPU
        self.ml.device = torch.device(device)
        self.ml.load_model()

        # Keep a reference to the fadtk helpers (no FrechetAudioDistance
        # instance – we call the free functions directly to avoid any file IO).
        from fadtk.fad import calc_embd_statistics, calc_frechet_distance
        self._calc_stats = calc_embd_statistics
        self._calc_fd = calc_frechet_distance

    # ------------------------------------------------------------------
    # In-memory audio → embedding
    # ------------------------------------------------------------------

    def _wav_to_numpy(self, path: str) -> np.ndarray:
        """Load audio, resample to model SR, convert to mono float64 ndarray."""
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True).float()  # -> (1, T)
        if sr != self.ml.sr:
            wav = torchaudio.functional.resample(wav, sr, self.ml.sr)
        # soundfile convention: float64 in [-1, 1]
        return wav.squeeze(0).numpy().astype(np.float64)

    def _embed(self, path: str) -> np.ndarray:
        """Return embedding array (n_frames, n_features) for one audio file."""
        wav_np = self._wav_to_numpy(path)
        return self.ml.get_embedding(wav_np).astype(np.float32)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ref_embeds: list[np.ndarray] = []
        hyp_embeds: list[np.ndarray] = []

        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [embed & score]", leave=False):
            sample_id = str(sample["sample_id"])
            ref_path = sample["audio_path"]
            hyp_path = sample["eval_audio_path"]

            errors: list[str] = []

            ref_emb: np.ndarray | None = None
            hyp_emb: np.ndarray | None = None

            if ref_path:
                try:
                    ref_emb = self._embed(ref_path)
                except Exception as exc:
                    errors.append(f"ref_embed_failed: {exc}")
            else:
                errors.append("missing audio_path")

            if hyp_path:
                try:
                    hyp_emb = self._embed(hyp_path)
                except Exception as exc:
                    errors.append(f"hyp_embed_failed: {exc}")
            else:
                errors.append("missing eval_audio_path")

            valid = len(errors) == 0 and ref_emb is not None and hyp_emb is not None

            if valid:
                ref_embeds.append(ref_emb)
                hyp_embeds.append(hyp_emb)

            rows.append(
                self.make_result(
                    sample_id=sample_id,
                    score=fad_score if valid else None,          # FAD is corpus-level, not per-sample
                    valid=valid,
                    error="; ".join(errors) if errors else None,
                    reason="; ".join(errors) if errors else "ok",
                )
            )

        fad_score: float | None = None
        if ref_embeds and hyp_embeds:
            ref_all = np.concatenate(ref_embeds, axis=0)
            hyp_all = np.concatenate(hyp_embeds, axis=0)
            mu_ref, cov_ref = self._calc_stats(ref_all)
            mu_hyp, cov_hyp = self._calc_stats(hyp_all)
            fad_score = float(self._calc_fd(mu_ref, cov_ref, mu_hyp, cov_hyp))

        # Build summary via finalize (gives valid/total/errors counts)
        rows, summary = self.finalize(rows)
        summary["submetric_avg"] = {"fad": fad_score}
        # Also surface at top level for convenience
        summary["avg_fad"] = fad_score
        return rows, summary
