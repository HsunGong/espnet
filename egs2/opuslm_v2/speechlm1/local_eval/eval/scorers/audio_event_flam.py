"""Audio-event scorer based on OpenFLAM **global** similarity.

Computes per-sample metrics for audio editing evaluation using OpenFLAM:

1. **Consistency metrics** (main content preservation):
   - ``audio_sim``          — cosine(audio_feat(source), audio_feat(generated))
   - ``main_text_src_sim``  — cosine(audio_feat(source), text_feat(main_label))
   - ``main_text_gen_sim``  — cosine(audio_feat(generated), text_feat(main_label))
   - ``main_text_delta``    — main_text_gen_sim − main_text_src_sim

2. **Operation-specific metrics**:
   - ``y_text_sim`` — cosine(audio_feat(generated), text_feat(y_label))  [ADD / REPLACE]
   - ``x_text_sim`` — cosine(audio_feat(generated), text_feat(x_label))  [REMOVE / REPLACE]

Labels are extracted from ``metadata`` sub-dicts (``main`` / ``y`` / ``x``),
each having ``audio_caption`` and ``human_labels``.  When ``vllm_client`` is
configured the caption + human_labels are compressed to ≤ 20 words via LLM.

Audio files used:
    source = ``audio_path``      — original audio (before editing)
    eval   = ``eval_audio_path`` — model-generated audio

Reference — OpenFLAM:
    Paper : https://arxiv.org/abs/2505.05335
    Code  : https://github.com/AdobeResearch/OpenFLAM
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .base import BaseScorer, safe_mean, rewrite_labels_vllm

# OpenFLAM expects 48 kHz mono audio
_FLAM_SR = 48_000


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


class AudioEventFLAMScorer(BaseScorer):
    """Audio-event scorer using OpenFLAM global cosine similarity.

    Parameters
    ----------
    model_name : str
        OpenFLAM model variant, passed to ``openflam.OpenFLAM(model_name=...)``.
    ckpt_path : str
        Path to cache / download the OpenFLAM checkpoint.
    device : str
        Torch device string.
    vllm_client : dict, optional
        If provided, kwargs forwarded to ``VLLMClient`` (must contain at least
        ``base_url`` and ``model``).  When set, caption + human_labels from
        each metadata sub-dict are compressed to ≤ 20 words via LLM before
        computing FLAM similarity.
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
        model_name: str = "v1-base",
        ckpt_path: str = "/tmp/openflam",
        device: str = "cuda",
        vllm_client: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        try:
            import openflam
        except ImportError as exc:
            raise ImportError(
                "openflam is required for AudioEventFLAMScorer.  "
                "Install via: pip install openflam"
            ) from exc

        self.flam = openflam.OpenFLAM(
            model_name=model_name,
            default_ckpt_path=ckpt_path,
        ).to(self.device)
        self.flam.eval()

        # Optional vLLM client for rewriting labels
        self._vllm = None
        if vllm_client is not None:
            from .llm_judge_caption_llm import VLLMClient
            self._vllm = VLLMClient(**vllm_client)

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    def _load_wav(self, wav_path: str) -> torch.Tensor:
        """Load audio, resample to 48 kHz, return mono tensor ``(1, T)``."""
        pcm, sr = torchaudio.load(wav_path)
        pcm = pcm.mean(dim=0, keepdim=True).float()  # stereo → mono
        if sr != _FLAM_SR:
            pcm = torchaudio.functional.resample(pcm, sr, _FLAM_SR)
        return pcm.to(self.device)  # (1, T)

    # ------------------------------------------------------------------
    # Core: similarity computation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _audio_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Return global audio features ``(1, dim)``."""
        return self.flam.get_global_audio_features(wav)

    @torch.inference_mode()
    def _text_features(self, text: str) -> torch.Tensor:
        """Return text features ``(1, dim)``."""
        return self.flam.get_text_features([text])

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two ``(1, dim)`` tensors."""
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a @ b.T).squeeze().item()

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
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [flam]", leave=False):
            sample_id = str(sample.get("sample_id", sample.get("id", "")))
            operation = str(sample.get("operation", "")).upper()
            src_path = sample.get("audio_path")          # source (original)
            gen_path = sample.get("eval_audio_path")     # generated output
            meta = sample.get("metadata") or {}
            main_audio_path = (meta.get("main") or {}).get("audio_path")

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
            if not src_path or not Path(src_path).exists():
                errors.append(f"missing/invalid audio_path: {src_path}")
            if not gen_path or not Path(gen_path).exists():
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
                # --- load audio -------------------------------------------
                wav_main = self._load_wav(str(main_audio_path))
                wav_gen = self._load_wav(str(gen_path))

                # --- compute features -------------------------------------
                main_audio_feat = self._audio_features(wav_main)
                gen_audio_feat = self._audio_features(wav_gen)
                main_text_feat = self._text_features(main_label)  # type: ignore[arg-type]

                # 1. Consistency metrics
                audio_sim = self._cosine_sim(main_audio_feat, gen_audio_feat)
                main_text_src_sim = self._cosine_sim(main_audio_feat, main_text_feat)
                main_text_gen_sim = self._cosine_sim(gen_audio_feat, main_text_feat)
                main_text_delta = main_text_gen_sim - main_text_src_sim

                # 2. Operation-specific metrics
                y_text_sim_val: float | None = None
                x_text_sim_val: float | None = None
                if y_label:
                    y_text_feat = self._text_features(y_label)
                    y_text_sim_val = self._cosine_sim(gen_audio_feat, y_text_feat)
                if x_label:
                    x_text_feat = self._text_features(x_label)
                    x_text_sim_val = self._cosine_sim(gen_audio_feat, x_text_feat)

            except Exception as exc:
                rows.append(self.make_result(
                    sample_id=sample_id, score=None, valid=False,
                    error=f"flam_failed: {exc}",
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
