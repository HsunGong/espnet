from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .base import safe_mean
from .base import BaseScorer


CAMPPLUS_VOX = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {"feat_dim": 80, "embedding_size": 512},
}
CAMPPLUS_COMMON = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {"feat_dim": 80, "embedding_size": 192},
}
ERES2NET_VOX = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {"feat_dim": 80, "embedding_size": 192},
}
ERES2NET_COMMON = {
    "obj": "speakerlab.models.eres2net.ResNet_aug.ERes2Net",
    "args": {"feat_dim": 80, "embedding_size": 192},
}
ERES2NET_BASE_3D = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {"feat_dim": 80, "embedding_size": 512, "m_channels": 32},
}
ERES2NET_LARGE_3D = {
    "obj": "speakerlab.models.eres2net.ResNet.ERes2Net",
    "args": {"feat_dim": 80, "embedding_size": 512, "m_channels": 64},
}

SUPPORTED_MODEL_IDS: dict[str, dict[str, Any]] = {
    "damo/speech_campplus_sv_en_voxceleb_16k": {"model": CAMPPLUS_VOX, "model_pt": "campplus_voxceleb.bin"},
    "damo/speech_campplus_sv_zh-cn_16k-common": {"model": CAMPPLUS_COMMON, "model_pt": "campplus_cn_common.bin"},
    "damo/speech_eres2net_sv_en_voxceleb_16k": {"model": ERES2NET_VOX, "model_pt": "pretrained_eres2net.ckpt"},
    "damo/speech_eres2net_sv_zh-cn_16k-common": {"model": ERES2NET_COMMON, "model_pt": "pretrained_eres2net_aug.ckpt"},
    "damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k": {"model": ERES2NET_BASE_3D, "model_pt": "eres2net_base_model.ckpt"},
    "damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k": {"model": ERES2NET_LARGE_3D, "model_pt": "eres2net_large_model.ckpt"},
}


class SpeakerSimilarity3DScorer(BaseScorer):
    """Speaker similarity scorer inspired by 3D-Speaker embedding extraction."""
    score_keys = ["score", "hyp_similarity", "ref_similarity"]

    def __init__(self, *, name: str, cfg: dict[str, Any], runtime: dict[str, Any], global_config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, cfg=cfg, runtime=runtime, global_config=global_config)
        self.model = None
        self.feature_extractor = None
        self.device = "cpu"
        self.similarity = None
        self.sample_rate = int(self.cfg.get("sample_rate", 16000))
        self.embedding_cache: dict[str, Any] = {}
        self._torch = None
        self._torchaudio = None
        self._init_error: str | None = None
        

    def _resolve_model_conf(self) -> tuple[dict[str, Any], str]:
        model_id = str(self.cfg.get("model_id") or "")
        model_cfg = self.cfg.get("model")
        checkpoint_path = str(self.cfg.get("checkpoint_path") or "")

        if model_id:
            if model_id not in SUPPORTED_MODEL_IDS:
                raise RuntimeError(f"unsupported model_id: {model_id}")
            model_info = SUPPORTED_MODEL_IDS[model_id]
            if model_cfg is None:
                model_cfg = model_info["model"]
            if not checkpoint_path:
                local_model_dir = str(self.cfg.get("local_model_dir", "pretrained"))
                checkpoint_path = str(Path(local_model_dir) / model_id.split("/", 1)[1] / model_info["model_pt"])

        if model_cfg is None:
            raise RuntimeError("speaker model config is missing (set `model_id` or `model`)")
        if not checkpoint_path:
            raise RuntimeError("speaker checkpoint path is missing")
        return dict(model_cfg), checkpoint_path



    def _load_wav(self, wav_file: str) -> Any:
        assert self._torch is not None
        assert self._torchaudio is not None
        torch = self._torch
        torchaudio = self._torchaudio

        if ".ark:" in wav_file:
            try:
                import kaldiio
            except Exception as exc:
                raise RuntimeError(f"kaldiio required for ark input: {exc}") from exc
            retval = kaldiio.load_mat(wav_file)
            if isinstance(retval, tuple):
                if isinstance(retval[0], int):
                    fs, wav = retval
                else:
                    wav, fs = retval
            else:
                wav, fs = retval, self.sample_rate
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 1:
                wav = wav[None, :]
            wav_tensor = torch.tensor(wav, dtype=torch.float32)
        else:
            wav_tensor, fs = torchaudio.load(wav_file)
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor[:1, :]

        if int(fs) != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(wav_tensor, int(fs), self.sample_rate)
        return wav_tensor

    def _embedding(self, wav_path: str) -> Any:
        if wav_path in self.embedding_cache:
            return self.embedding_cache[wav_path]

        if self.model is None or self.feature_extractor is None or self._torch is None:
            raise RuntimeError(self._init_error or "speaker model not initialized")
        torch = self._torch

        wav = self._load_wav(wav_path)
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(feat).detach().cpu()
        self.embedding_cache[wav_path] = emb
        return emb

    def _cosine(self, emb1: Any, emb2: Any) -> float:
        if self.similarity is None:
            raise RuntimeError("similarity op not initialized")
        return float(self.similarity(emb1, emb2).item())

    def _score(self, hyp_sim: float, ref_sim: float | None) -> float:
        mode = str(self.task_cfg.get("score_mode") or self.cfg.get("score_mode") or "normalized_hyp")
        if mode == "relative_to_ref" and ref_sim is not None and math.isfinite(ref_sim):
            return self._clamp01(hyp_sim / max(ref_sim, 1e-6))
        return self._clamp01((hyp_sim + 1.0) / 2.0)

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        


        prompt_key = str(self.task_cfg.get("prompt_audio_key") or self.cfg.get("prompt_audio_key") or "prompt_audio_path")
        ref_key = str(self.task_cfg.get("reference_audio_key") or self.cfg.get("reference_audio_key") or "audio_path")
        hyp_key = str(self.task_cfg.get("hypothesis_audio_key") or self.cfg.get("hypothesis_audio_key") or "eval_audio_path")

        # ------------------------------------------------------------------
        # Phase 1: extract embeddings for all samples
        # ------------------------------------------------------------------
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [embed]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                prompt_wav = str(sample.get(prompt_key) or "")
                hyp_wav = str(sample.get(hyp_key) or "")
                ref_wav = str(sample.get(ref_key) or "")
                if not prompt_wav:
                    raise RuntimeError(f"missing prompt wav from `{prompt_key}`")
                if not hyp_wav:
                    raise RuntimeError(f"missing hypothesis wav from `{hyp_key}`")

                prompt_emb = self._embedding(prompt_wav)
                hyp_emb = self._embedding(hyp_wav)
                ref_emb = self._embedding(ref_wav) if ref_wav else None
                pending.append({
                    "sample_id": sample_id,
                    "prompt_emb": prompt_emb,
                    "hyp_emb": hyp_emb,
                    "ref_emb": ref_emb,
                })
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

        # ------------------------------------------------------------------
        # Phase 2: compute cosine similarities and scores
        # ------------------------------------------------------------------
        ref_sims: list[float] = []
        hyp_sims: list[float] = []

        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            try:
                hyp_sim = self._cosine(item["prompt_emb"], item["hyp_emb"])
                ref_sim: float | None = None
                if item["ref_emb"] is not None:
                    ref_sim = self._cosine(item["prompt_emb"], item["ref_emb"])
                    if math.isfinite(ref_sim):
                        ref_sims.append(ref_sim)

                if not math.isfinite(hyp_sim):
                    raise RuntimeError(f"invalid hyp similarity: {hyp_sim}")
                hyp_sims.append(hyp_sim)
                score = self._score(hyp_sim, ref_sim)
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason=f"hyp_sim={hyp_sim:.4f}",
                        extra={
                            "hyp_similarity": hyp_sim,
                            "ref_similarity": ref_sim,
                            "score_mode": str(self.task_cfg.get("score_mode") or self.cfg.get("score_mode") or "normalized_hyp"),
                        },
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
        summary["avg_hyp_similarity"] = safe_mean(hyp_sims)
        summary["avg_ref_similarity"] = safe_mean(ref_sims)
        return rows, summary
    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
