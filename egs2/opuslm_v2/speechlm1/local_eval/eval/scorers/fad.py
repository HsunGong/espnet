"""Robust Fréchet Audio Distance (FAD) scorer.

FAD is a *corpus-level* metric: every sample receives the same FAD value.
The score is reported in ``summary["submetric_avg"]["fad"]`` and
``summary["avg_fad"]``.

Ground-truth resolution order:
  1. ``target_audio_path`` (preferred – the reference-edited target)
  2. ``audio_path`` (fallback – the original source audio)

Embedding extraction is done **per-file** (not via
``FrechetAudioDistance.score()``) so that:
  - batch-concatenation shape issues are avoided
  - individual failures are logged cleanly
  - embeddings are normalised to a consistent 2-D ``(frames, D)`` shape

Install:
  pip install frechet-audio-distance
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import resampy
import soundfile as sf
import torch
from tqdm import tqdm

from .base import BaseScorer


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def _load_audio(
    path: str,
    sample_rate: int,
    channels: int = 1,
) -> np.ndarray:
    """Read *path*, resample, and convert to mono/stereo float32."""
    wav, sr = sf.read(path, dtype="float32")

    # mono down-mix
    if channels == 1 and wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    # keep stereo as-is when channels==2

    if sr != sample_rate:
        if wav.ndim == 1:
            wav = resampy.resample(wav, sr, sample_rate)
        else:
            cols = []
            for c in range(wav.shape[1]):
                cols.append(resampy.resample(wav[:, c], sr, sample_rate))
            min_len = min(len(x) for x in cols)
            wav = np.stack([x[:min_len] for x in cols], axis=1)

    return wav.astype(np.float32, copy=False)


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_2d(emb: np.ndarray) -> np.ndarray:
    return np.atleast_2d(emb)


# ---------------------------------------------------------------------------
# FADScorer
# ---------------------------------------------------------------------------

class FADScorer(BaseScorer):
    """Fréchet Audio Distance scorer backed by *frechet_audio_distance*.

    Parameters
    ----------
    model : str
        One of ``"vggish"``, ``"pann"``, ``"clap"``, ``"encodec"``.
    sample_rate : int | None
        Override the default sample-rate for the chosen model.
    use_pca / use_activation : bool
        VGGish-specific flags.
    submodel_name / enable_fusion : str / bool
        CLAP-specific flags.
    channels : int | None
        Audio channel count (encodec-48k defaults to 2).
    on_problem : str
        ``"skip"`` to silently skip bad files; ``"error"`` to raise.
    """

    score_keys = ["fad"]

    def __init__(
        self,
        *,
        name: str,
        model: str = "vggish",
        sample_rate: int | None = None,
        channels: int | None = None,
        use_pca: bool = False,
        use_activation: bool = False,
        submodel_name: str = "630k-audioset",
        enable_fusion: bool = False,
        on_problem: str = "skip",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        try:
            from frechet_audio_distance import FrechetAudioDistance
        except ImportError as exc:
            raise ImportError(
                "frechet_audio_distance is required for FADScorer. "
                "Install it with: pip install frechet-audio-distance"
            ) from exc

        self.on_problem = on_problem

        # Build model config following package conventions
        cfg: dict[str, Any] = {"model_name": model, "verbose": verbose}

        if model == "vggish":
            cfg["sample_rate"] = sample_rate or 16000
            cfg["use_pca"] = use_pca
            cfg["use_activation"] = use_activation
            cfg["channels"] = 1

        elif model == "pann":
            cfg["sample_rate"] = sample_rate or 16000
            cfg["use_pca"] = False
            cfg["use_activation"] = False
            cfg["channels"] = 1

        elif model == "clap":
            cfg["sample_rate"] = sample_rate or 48000
            cfg["submodel_name"] = submodel_name
            cfg["enable_fusion"] = enable_fusion
            cfg["channels"] = 1

        elif model == "encodec":
            sr = sample_rate or 48000
            cfg["sample_rate"] = sr
            cfg["channels"] = channels if channels is not None else (2 if sr == 48000 else 1)

        else:
            raise ValueError(f"Unsupported FAD model: {model}")

        self.frechet = FrechetAudioDistance(**cfg)
        self.model_name = model

    # ------------------------------------------------------------------
    # Per-file embedding extraction (robust)
    # ------------------------------------------------------------------

    def _embed_one(self, path: str) -> np.ndarray:
        """Return embedding of shape ``(frames, D)`` for a single audio file."""
        f = self.frechet
        wav = _load_audio(path, f.sample_rate, f.channels)

        model = self.model_name

        if model == "vggish":
            emb = f.model.forward(wav, f.sample_rate)
            emb = _to_numpy(emb)

        elif model == "pann":
            with torch.no_grad():
                wav_m = np.mean(wav, axis=1) if wav.ndim == 2 else wav
                t = torch.tensor(wav_m).float().unsqueeze(0)
                if hasattr(f, "device"):
                    t = t.to(f.device)
                out = f.model(t, None)
                emb = _to_numpy(out["embedding"])

        elif model == "clap":
            with torch.no_grad():
                wav_m = np.mean(wav, axis=1) if wav.ndim == 2 else wav
                t = torch.tensor(wav_m).float().unsqueeze(0)
                emb = _to_numpy(
                    f.model.get_audio_embedding_from_data(t, use_tensor=True)
                )

        elif model == "encodec":
            with torch.no_grad():
                if f.sample_rate == 48000:
                    if wav.ndim == 1:
                        wav = np.stack([wav, wav], axis=1)
                    elif wav.ndim == 2 and wav.shape[1] != 2:
                        wav = wav[:, :2]
                    t = torch.tensor(wav).float().unsqueeze(0).transpose(1, 2)
                else:
                    if wav.ndim == 2:
                        wav = np.mean(wav, axis=1)
                    t = torch.tensor(wav).float().unsqueeze(0).unsqueeze(0)

                if hasattr(f, "device"):
                    t = t.to(f.device)
                emb = f.model.encoder(t)     # (1, D, frames)
                emb = emb.squeeze(0).transpose(0, 1)  # (frames, D)
                emb = _to_numpy(emb)
        else:
            raise ValueError(f"Unsupported model_name: {model}")

        emb = _ensure_2d(emb)

        if emb.size == 0:
            raise RuntimeError(f"Empty embedding for: {path}")
        if not np.isfinite(emb).all():
            raise RuntimeError(f"Non-finite embedding values for: {path}")

        return emb.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Scoring (corpus-level)
    # ------------------------------------------------------------------

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ref_embeds: list[np.ndarray] = []
        hyp_embeds: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [embed]", leave=False):
            sample_id = str(sample["sample_id"])

            # GT resolution: prefer target_audio_path, fall back to audio_path
            ref_path = sample.get("target_audio_path") or sample.get("audio_path")
            hyp_path = sample.get("eval_audio_path")

            errors: list[str] = []
            ref_emb: np.ndarray | None = None
            hyp_emb: np.ndarray | None = None

            # --- reference embedding ---
            if ref_path:
                try:
                    ref_emb = self._embed_one(ref_path)
                except Exception as exc:
                    if self.on_problem == "skip":
                        errors.append(f"ref_embed_skipped: {exc}")
                    else:
                        raise
            else:
                errors.append("missing ref path (no target_audio_path or audio_path)")

            # --- hypothesis embedding ---
            if hyp_path:
                try:
                    hyp_emb = self._embed_one(hyp_path)
                except Exception as exc:
                    if self.on_problem == "skip":
                        errors.append(f"hyp_embed_skipped: {exc}")
                    else:
                        raise
            else:
                errors.append("missing eval_audio_path")

            valid = (not errors) and ref_emb is not None and hyp_emb is not None
            if valid and ref_emb is not None and hyp_emb is not None:
                ref_embeds.append(ref_emb)
                hyp_embeds.append(hyp_emb)

            rows.append(
                self.make_result(
                    sample_id=sample_id,
                    score=None,  # FAD is corpus-level; filled in below
                    valid=valid,
                    error="; ".join(errors) if errors else None,
                    reason="; ".join(errors) if errors else "ok",
                )
            )

        # ---- compute corpus-level FAD ----
        fad_score: float | None = None
        if ref_embeds and hyp_embeds:
            ref_all = np.concatenate(ref_embeds, axis=0)
            hyp_all = np.concatenate(hyp_embeds, axis=0)

            mu_ref, sigma_ref = self.frechet.calculate_embd_statistics(ref_all)
            mu_hyp, sigma_hyp = self.frechet.calculate_embd_statistics(hyp_all)
            fad_score = float(
                self.frechet.calculate_frechet_distance(
                    mu_ref, sigma_ref, mu_hyp, sigma_hyp,
                )
            )

            # Back-fill every valid row with the corpus-level FAD
            for row in rows:
                if row["valid"]:
                    row["score"] = fad_score

        # Build summary via finalize (gives valid/total/errors counts)
        rows, summary = self.finalize(rows)
        summary["submetric_avg"] = {"fad": fad_score}
        summary["avg_fad"] = fad_score
        return rows, summary
