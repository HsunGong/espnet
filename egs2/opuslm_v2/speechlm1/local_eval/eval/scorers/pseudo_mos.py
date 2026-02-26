from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .base import safe_mean
from .base import BaseScorer


DEFAULT_MOS_BOUNDS: dict[str, tuple[float, float]] = {
    "utmos": (1.0, 5.0),
    "utmosv2": (1.0, 5.0),
    "dns_overall": (1.0, 5.0),
    "dns_p808": (1.0, 5.0),
    "plcmos": (1.0, 5.0),
    "singmos_v1": (1.0, 5.0),
    "singmos_pro": (1.0, 5.0),
}
# https://github.com/inclusionAI/Ming-Freeform-Audio-Edit/blob/main/eval_scripts/pyscripts/calculate_dnsmos_metrics.py
# https://github.com/wavlab-speech/versa/blob/main/versa/utterance_metrics/pseudo_mos.py

def _stft_for_dnsmos_pro(
    samples: np.ndarray,
    *,
    win_length: int = 320,
    hop_length: int = 160,
    n_fft: int = 320,
) -> np.ndarray:
    import librosa

    spec = librosa.stft(y=samples, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(spec).T
    spec = np.clip(spec, 10 ** (-7), 10**7)
    return np.log10(spec)


class PseudoMOSScorer(BaseScorer):
    """
    Pseudo-MOS scorer supporting:
    - utmos
    - utmosv2
    - dnsmos
    - plcmos
    - singmos_v1 / singmos_pro
    - dnsmos_pro_{variant}
    """
    score_keys = ["score"]

    def __init__(
        self,
        *,
        name: str,
        use_gpu: bool = False,
        cache_dir: str = "versa_cache",
        predictor_types: list[str] | None = None,
        predictor_args: dict[str, Any] | None = None,
        aggregate_keys: list[str] | None = None,
        aggregate_weights: dict[str, float] | None = None,
        metric_bounds: dict[str, list[float]] | None = None,
        normalize_aggregate: bool = True,
        allow_download: bool = False,
        utmos_hub_repo: str = "ftshijt/SpeechMOS:main",
        utmos_hub_entry: str = "utmos22_strong",
        singmos_hub_repo: str = "South-Twilight/SingMOS:v1.1.1",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.predictor_types_default = predictor_types or ["utmosv2"]
        self.predictor_args_default = predictor_args or {}
        self.aggregate_keys_default = aggregate_keys
        self.aggregate_weights_default = aggregate_weights or {}
        self.metric_bounds = metric_bounds or {}
        self.normalize_aggregate = normalize_aggregate
        self.allow_download = allow_download
        self.utmos_hub_repo = utmos_hub_repo
        self.utmos_hub_entry = utmos_hub_entry
        self.singmos_hub_repo = singmos_hub_repo
        
        self.predictor_dict: dict[str, Any] = {}
        self.predictor_fs: dict[str, int] = {}
        self.device = "cpu"
        self._torch = None
        self._librosa = None
        self._utmosv2 = None
        self._process_audio_only_versa = None
        self._setup_predictors()

    def _predictor_types(self) -> list[str]:
        return list(self.task_cfg.get("predictor_types") or self.predictor_types_default)

    def _predictor_args(self) -> dict[str, Any]:
        return dict(self.task_cfg.get("predictor_args") or self.predictor_args_default)

    def _setup_predictors(self) -> None:
        import torch
        import librosa
        self._torch = torch
        self._librosa = librosa

        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

        predictor_types = self._predictor_types()
        predictor_args = self._predictor_args()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        if "utmos" in predictor_types:
            torch.hub.set_dir(self.cache_dir)
            utmos = torch.hub.load(self.utmos_hub_repo, self.utmos_hub_entry).to(self.device)
            self.predictor_dict["utmos"] = utmos.float()
            self.predictor_fs["utmos"] = 16000

        if any(x in predictor_types for x in ("dnsmos", "plcmos")):
            import onnxruntime  # noqa: F401
            from speechmos import dnsmos, plcmos
        else:
            dnsmos = None
            plcmos = None

        for predictor in predictor_types:
            if predictor == "dnsmos":
                self.predictor_dict["dnsmos"] = dnsmos
                self.predictor_fs["dnsmos"] = int(predictor_args.get("dnsmos", {}).get("fs", 16000))
            elif predictor == "plcmos":
                self.predictor_dict["plcmos"] = plcmos
                self.predictor_fs["plcmos"] = int(predictor_args.get("plcmos", {}).get("fs", 16000))
            elif predictor in {"utmos", "utmosv2"}:
                continue
            elif predictor == "singmos_v1":
                torch.hub.set_dir(self.cache_dir)
                singmos = torch.hub.load(
                    self.singmos_hub_repo,
                    "singmos_v1",
                    trust_repo=True,
                ).to(self.device)
                self.predictor_dict["singmos_v1"] = singmos.float()
                self.predictor_fs["singmos_v1"] = 44100
            elif predictor == "singmos_pro":
                torch.hub.set_dir(self.cache_dir)
                singmos = torch.hub.load(
                    self.singmos_hub_repo,
                    "singmos_pro",
                    trust_repo=True,
                ).to(self.device)
                self.predictor_dict["singmos_pro"] = singmos.float()
                self.predictor_fs["singmos_pro"] = 44100
            elif predictor.startswith("dnsmos_pro_"):
                from speechmos import dnsmos_pro
                variant = predictor[len("dnsmos_pro_") :]
                model = dnsmos_pro.DNSMOS_PRO(variant)
                if self.device.startswith("cuda"):
                    model.cuda()
                self.predictor_dict[predictor] = model
                self.predictor_fs[predictor] = 16000
            else:
                raise NotImplementedError(f"Unsupported pseudo_mos predictor: {predictor}")

    def _normalize_audio(self, wav: np.ndarray) -> np.ndarray:
        return wav / (np.max(np.abs(wav)) + 1e-9)

    def _predict_one(self, wav: np.ndarray, fs: int) -> dict[str, float]:
        import librosa
        torch = self._torch
        use_gpu = self.device.startswith("cuda")
        scores: dict[str, float] = {}

        for predictor in self._predictor_types():
            if predictor == "utmos":
                target_fs = self.predictor_fs["utmos"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                if use_gpu:
                    pred_tensor = pred_tensor.to(self.device)
                score = self.predictor_dict["utmos"](pred_tensor.float(), target_fs).item()
                scores["utmos"] = float(score)
            elif predictor == "utmosv2":
                target_fs = self.predictor_fs["utmosv2"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                if use_gpu:
                    pred_tensor = pred_tensor.to(self.device)
                with torch.no_grad():
                    spec_tensor, domain = self._process_audio_only_versa(pred_tensor, target_fs)
                    if use_gpu:
                        spec_tensor = spec_tensor.to(self.device)
                    # repeat for robustness if short
                    repeat_n = 4
                    vals = []
                    d = torch.tensor([domain]).to(self.device) if use_gpu else torch.tensor([domain])
                    for _ in range(max(1, repeat_n)):
                        vals.append(
                            float(
                                self.predictor_dict["utmosv2"](pred_tensor.float(), spec_tensor, d)
                                .squeeze(1)
                                .detach()
                                .cpu()
                                .numpy()[0]
                            )
                        )
                scores["utmosv2"] = float(sum(vals) / len(vals))
            elif predictor == "dnsmos":
                target_fs = self.predictor_fs["dnsmos"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred = self._normalize_audio(pred)
                score = self.predictor_dict["dnsmos"].run(pred, sr=target_fs)
                scores["dns_overall"] = float(score["ovrl_mos"])
                scores["dns_p808"] = float(score["p808_mos"])
            elif predictor == "plcmos":
                target_fs = self.predictor_fs["plcmos"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred = self._normalize_audio(pred)
                score = self.predictor_dict["plcmos"].run(pred, sr=target_fs)
                scores["plcmos"] = float(score["plcmos"])
            elif predictor == "singmos_v1":
                target_fs = self.predictor_fs["singmos_v1"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                length_tensor = torch.tensor([pred_tensor.size(1)]).int()
                if use_gpu:
                    pred_tensor = pred_tensor.to(self.device)
                    length_tensor = length_tensor.to(self.device)
                score = self.predictor_dict["singmos_v1"](pred_tensor.float(), length_tensor)[0].item()
                scores["singmos_v1"] = float(score)
            elif predictor == "singmos_pro":
                target_fs = self.predictor_fs["singmos_pro"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                length_tensor = torch.tensor([pred_tensor.size(1)]).int()
                if use_gpu:
                    pred_tensor = pred_tensor.to(self.device)
                    length_tensor = length_tensor.to(self.device)
                score = self.predictor_dict["singmos_pro"](pred_tensor.float(), length_tensor)[0].item()
                scores["singmos_pro"] = float(score)
            elif predictor.startswith("dnsmos_pro_"):
                target_fs = self.predictor_fs[predictor]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                spec = torch.FloatTensor(_stft_for_dnsmos_pro(pred))
                if use_gpu:
                    spec = spec.to(self.device)
                with torch.no_grad():
                    prediction = self.predictor_dict[predictor](spec[None, None, ...])
                scores[predictor] = float(prediction[0, 0].item())
            else:
                raise NotImplementedError(f"Not supported predictor: {predictor}")
        return scores

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _aggregate_score(self, metrics: dict[str, float]) -> float | None:
        if not metrics:
            return None
        bounds_cfg = self.metric_bounds
        weights_cfg = dict(self.task_cfg.get("aggregate_weights") or self.aggregate_weights_default)
        agg_keys = list(self.task_cfg.get("aggregate_keys") or self.aggregate_keys_default or metrics.keys())
        normalize = self.normalize_aggregate

        values: list[float] = []
        weights: list[float] = []
        for key in agg_keys:
            if key not in metrics:
                continue
            val = float(metrics[key])
            if normalize:
                lo, hi = bounds_cfg.get(key, DEFAULT_MOS_BOUNDS.get(key, [0.0, 5.0]))
                lo, hi = float(lo), float(hi)
                denom = hi - lo
                if denom <= 0:
                    norm = 0.0
                else:
                    norm = self._clamp01((val - lo) / denom)
                val = norm
            w = float(weights_cfg.get(key, 1.0))
            values.append(val)
            weights.append(max(0.0, w))
        if not values:
            return None
        if sum(weights) <= 0:
            return safe_mean(values)
        weighted = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        return float(weighted)

    def configure_task(self, task_cfg: dict[str, Any] | None) -> None:
        super().configure_task(task_cfg)
        predictor_types = self._predictor_types()
        extra_keys: list[str] = []
        for pt in predictor_types:
            if pt == "dnsmos":
                extra_keys += ["dns_overall", "dns_p808"]
            elif pt in ("utmos", "utmosv2", "plcmos", "singmos_v1", "singmos_pro"):
                extra_keys.append(pt if pt != "plcmos" else "plcmos")
            elif pt.startswith("dnsmos_pro_"):
                extra_keys.append(pt)
        if extra_keys:
            self.score_keys = ["score"] + extra_keys

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import librosa  # local import to keep module optional
        
        # Phase 1: load audio and run predictors for all samples
        pending: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for sample in tqdm(samples, desc=f"{self.name} [predict]", leave=False):
            sample_id = str(sample["sample_id"])
            try:
                audio_path = str(sample.get("eval_audio_path") or "")
                if not audio_path:
                    raise RuntimeError("missing eval_audio_path")
                wav, fs = librosa.load(audio_path, sr=None, mono=True)
                wav = wav.astype(np.float32)
                metrics = self._predict_one(wav, int(fs))
                pending.append({"sample_id": sample_id, "metrics": metrics})
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="pseudo_mos_failed",
                        reason=str(exc),
                    )
                )

        # Phase 2: aggregate scores
        metric_collect: dict[str, list[float]] = {}

        for item in tqdm(pending, desc=f"{self.name} [score]", leave=False):
            sample_id = item["sample_id"]
            metrics = item["metrics"]
            try:
                score = self._aggregate_score(metrics)
                if score is None:
                    raise RuntimeError("empty pseudo_mos metrics")
                for k, v in metrics.items():
                    if math.isfinite(v):
                        metric_collect.setdefault(k, []).append(float(v))
                extra: dict[str, Any] = {"metrics": metrics}
                extra.update({k: v for k, v in metrics.items()})
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=score,
                        valid=True,
                        reason="pseudo_mos computed",
                        extra=extra,
                    )
                )
            except Exception as exc:
                rows.append(
                    self.make_result(
                        sample_id=sample_id,
                        score=None,
                        valid=False,
                        error="pseudo_mos_failed",
                        reason=str(exc),
                    )
                )

        rows, summary = self.finalize(rows)
        summary["submetric_avg"] = {k: safe_mean(vs) for k, vs in metric_collect.items()}
        return rows, summary
