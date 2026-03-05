"""CLAP audio-text & audio-audio cosine similarity scorer.

Computes per-sample metrics for audio editing evaluation using ``laion_clap``:

1. **Consistency metrics** (main content preservation):
   - ``audio_sim``          — cosine(audio_emb(source), audio_emb(generated))
   - ``main_text_src_sim``  — cosine(audio_emb(source), text_emb(main_label))
   - ``main_text_gen_sim``  — cosine(audio_emb(generated), text_emb(main_label))
   - ``main_text_delta``    — main_text_gen_sim − main_text_src_sim

2. **Operation-specific metrics**:
   - ``y_text_sim`` — cosine(audio_emb(generated), text_emb(y_label))  [ADD / REPLACE]
   - ``x_text_sim`` — cosine(audio_emb(generated), text_emb(x_label))  [REMOVE / REPLACE]

Labels are extracted from ``metadata`` sub-dicts (``main`` / ``y`` / ``x``),
each having ``audio_caption`` and ``human_labels``.  When ``vllm_client`` is
configured the caption + human_labels are compressed to ≤ 20 words via LLM.

Audio files used:
    source = ``audio_path``      — original audio (before editing)
    eval   = ``eval_audio_path`` — model-generated audio

Install:
  pip install laion-clap
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from torch.nn import functional as F
from .base import BaseScorer, safe_mean, rewrite_labels_vllm

# CLAP expects 48 kHz
_CLAP_SR = 48_000

# ---------------------------------------------------------------------------
# Lazy AudioSet loader & UID-based audio resolver
# ---------------------------------------------------------------------------
_AUDIOSET_CACHE_DIR = Path("/tmp/clap_audioset_cache")
_AUDIOSET_DS = None
_AUDIOSET_VID_MAP: dict[str, int] | None = None


def _ensure_audioset() -> None:
    """Lazily load AudioSet balanced train+test and build video_id -> index map."""
    global _AUDIOSET_DS, _AUDIOSET_VID_MAP
    if _AUDIOSET_DS is not None:
        return
    from datasets import load_dataset, concatenate_datasets

    _AUDIOSET_DS = concatenate_datasets([
        load_dataset("agkphysics/AudioSet", "balanced", split="train", trust_remote_code=True),
        load_dataset("agkphysics/AudioSet", "balanced", split="test", trust_remote_code=True),
    ])
    _AUDIOSET_VID_MAP = {vid: i for i, vid in enumerate(_AUDIOSET_DS["video_id"])}


def _resolve_audioset_audio(uid: str) -> str | None:
    """Look up AudioSet audio by *uid* (video_id), cache WAV to ``/tmp``, return path."""
    cache_path = _AUDIOSET_CACHE_DIR / f"{uid}.wav"
    if cache_path.exists():
        return str(cache_path)

    _ensure_audioset()
    idx = _AUDIOSET_VID_MAP.get(uid)  # type: ignore[union-attr]
    if idx is None:
        return None

    row = _AUDIOSET_DS[idx]  # type: ignore[index]
    audio_obj = row["audio"]
    y = np.asarray(audio_obj["array"], dtype=np.float32)
    orig_sr = audio_obj["sampling_rate"]
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    _AUDIOSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(cache_path), y, orig_sr)
    return str(cache_path)


def _resolve_meta_audio_path(meta_sub: dict[str, Any] | None) -> str | None:
    """Resolve audio path from a metadata sub-dict.

    Falls back to AudioSet UID lookup when ``audio_path`` is missing or invalid.
    """
    if not meta_sub:
        return None
    # 1. Try direct audio_path
    audio_path = meta_sub.get("audio_path")
    if audio_path and Path(str(audio_path)).exists():
        return str(audio_path)
    # 2. Fallback: resolve from AudioSet via uid / video_id
    uid = meta_sub.get("uid") or meta_sub.get("video_id")
    if uid:
        return _resolve_audioset_audio(uid)
    return None


def _extract_labels(d: dict[str, Any] | None) -> list[str]:
    """Extract ``human_labels`` list from a metadata sub-dict."""
    if not d:
        return []
    raw = d.get("human_labels")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [str(l) for l in raw if str(l).strip()]


class CLAPSimilarityScorer(BaseScorer):
    """Per-sample CLAP audio-text & audio-audio cosine similarity scorer.

    Parameters
    ----------
    enable_fusion : bool
        Whether to enable CLAP's fusion mode.
    device : str
        Torch device string.  Falls back to ``"cpu"`` when CUDA unavailable.
    vllm_client : dict, optional
        If provided, kwargs forwarded to ``VLLMClient`` (must contain at least
        ``base_url`` and ``model``).  When set, caption + human_labels from
        each metadata sub-dict are compressed to ≤ 20 words via LLM before
        computing CLAP similarity.
    """

    score_keys = [
        "score",
        "audio_sim",
        "main_text_src_sim",
        "main_text_gen_sim",
        "main_text_delta",
        "y_text_sim",
        "x_text_sim",
    ]

    def __init__(
        self,
        *,
        name: str,
        enable_fusion: bool = False,
        device: str = "cuda",
        vllm_client: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        try:
            import laion_clap
        except ImportError as exc:
            raise ImportError(
                "laion_clap is required for CLAPSimilarityScorer. "
                "Install it with: pip install laion-clap"
            ) from exc

        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt()  # downloads default pretrained checkpoint

        # Optional vLLM client for rewriting labels
        self._vllm = None
        if vllm_client is not None:
            from .llm_judge_caption_llm import VLLMClient
            self._vllm = VLLMClient(**vllm_client)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _int16_to_float32(x: np.ndarray) -> np.ndarray:
        return (x / 32767.0).astype("float32")

    @staticmethod
    def _float32_to_int16(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, a_min=-1.0, a_max=1.0)
        return (x * 32767.0).astype("int16")

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio from *path*, resample to 48 kHz, return ``(1, T)`` float32."""
        import librosa

        audio, _ = librosa.load(path, sr=_CLAP_SR)  # mono float32
        # Quantize to int16 then back to float32 (matches CLAP training pipeline)
        audio = self._int16_to_float32(self._float32_to_int16(audio))
        return audio.reshape(1, -1)  # (1, T)

    @torch.inference_mode()
    def _audio_embedding(self, path: str) -> np.ndarray:
        """Return audio embedding ``(1, D)`` as numpy array."""
        audio_data = self._load_audio(path)
        audio_data = torch.from_numpy(audio_data).float()
        emb = self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        return np.asarray(emb)  # (1, D)

    @torch.inference_mode()
    def _text_embedding(self, text: str) -> np.ndarray:
        """Return text embedding ``(1, D)`` as numpy array."""
        emb = self.model.get_text_embedding([text], use_tensor=True)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        return np.asarray(emb)  # (1, D)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two (1, D) vectors."""
        return F.cosine_similarity(torch.from_numpy(a), torch.from_numpy(b), dim=0).item()

    def _get_rewritten_label(self, meta_sub: dict[str, Any] | None) -> str | None:
        """Extract caption + human_labels from *meta_sub* and optionally
        compress via LLM.  Returns *None* when no usable text is found."""
        if not meta_sub:
            return None
        caption = meta_sub.get("audio_caption", "")
        human_labels = _extract_labels(meta_sub)

        if self._vllm is not None:
            return rewrite_labels_vllm(caption, human_labels, self._vllm)

        # Fallback: join human_labels or use caption
        if human_labels:
            return ", ".join(human_labels)
        if caption and isinstance(caption, str) and caption.strip():
            return caption.strip()
        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [clap]", leave=False):
            sample_id = str(sample.get("sample_id", sample.get("id", "")))
            operation = str(sample.get("operation", "")).upper()
            src_path = sample.get("audio_path")
            gen_path = sample.get("eval_audio_path")
            meta = sample["metadata"]
            main_audio_path = _resolve_meta_audio_path(meta.get("main"))

            # --- extract & rewrite labels for each metadata sub-dict ------
            main_label = self._get_rewritten_label(meta.get("main"))
            y_label = (
                self._get_rewritten_label(meta.get("y"))
                if operation in ("ADD", "REPLACE")
                else None
            )
            x_label = (
                self._get_rewritten_label(meta.get("x"))
                if operation in ("REMOVE", "REPLACE")
                else None
            )

            # --- validation -----------------------------------------------
            errors: list[str] = []
            if not src_path or not Path(str(src_path)).exists():
                errors.append(f"missing/invalid audio_path: {src_path}")
            if not gen_path or not Path(str(gen_path)).exists():
                errors.append(f"missing/invalid eval_audio_path: {gen_path}")
            if not main_audio_path or not Path(str(main_audio_path)).exists():
                errors.append(f"missing/invalid metadata.main.audio_path: {main_audio_path}")
            if not main_label:
                errors.append("no main label (metadata.main missing caption/labels)")
            if errors:
                rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error="; ".join(errors),
                ))
                continue

            try:
                # --- compute embeddings -----------------------------------
                main_audio_emb = self._audio_embedding(str(main_audio_path))
                gen_audio_emb = self._audio_embedding(str(gen_path))
                main_text_emb = self._text_embedding(main_label)  # type: ignore[arg-type]

                # 1. Consistency metrics
                audio_sim = self._cosine_similarity(main_audio_emb, gen_audio_emb)
                main_text_src_sim = self._cosine_similarity(main_audio_emb, main_text_emb)
                main_text_gen_sim = self._cosine_similarity(gen_audio_emb, main_text_emb)
                main_text_delta = main_text_gen_sim - main_text_src_sim

                # 2. Operation-specific metrics
                y_text_sim_val: float | None = None
                x_text_sim_val: float | None = None
                if y_label:
                    y_text_emb = self._text_embedding(y_label)
                    y_text_sim_val = self._cosine_similarity(gen_audio_emb, y_text_emb)
                if x_label:
                    x_text_emb = self._text_embedding(x_label)
                    x_text_sim_val = self._cosine_similarity(gen_audio_emb, x_text_emb)
            except Exception as exc:
                rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error=f"clap_failed: {exc}",
                ))
                continue

            extra: dict[str, Any] = {
                "audio_sim": audio_sim,
                "main_text_src_sim": main_text_src_sim,
                "main_text_gen_sim": main_text_gen_sim,
                "main_text_delta": main_text_delta,
                "main_label": main_label[:200] if main_label else "",
            }
            if y_text_sim_val is not None:
                extra["y_text_sim"] = y_text_sim_val
                extra["y_label"] = y_label[:200] if y_label else ""
            if x_text_sim_val is not None:
                extra["x_text_sim"] = x_text_sim_val
                extra["x_label"] = x_label[:200] if x_label else ""

            rows.append(self.make_result(
                sample_id=sample_id,
                score=audio_sim,
                valid=True,
                reason=f"op={operation}",
                extra=extra,
            ))

        return self.finalize(rows)
