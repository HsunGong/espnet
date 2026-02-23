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
    # Score keys are dynamic (depend on predictor_types); override in configure_task
    # or set to cover common sub-metrics.
    score_keys = ["score"]

    def __init__(self, *, name: str, cfg: dict[str, Any], runtime: dict[str, Any], global_config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, cfg=cfg, runtime=runtime, global_config=global_config)
        self.predictor_dict: dict[str, Any] = {}
        self.predictor_fs: dict[str, int] = {}
        self.device = "cpu"
        self._torch = None
        self._librosa = None
        self._utmosv2 = None
        self._process_audio_only_versa = None
        self._setup_predictors()

    def _predictor_types(self) -> list[str]:
        return list(self.task_cfg.get("predictor_types") or self.cfg.get("predictor_types") or ["utmosv2"])

    def _predictor_args(self) -> dict[str, Any]:
        return dict(self.task_cfg.get("predictor_args") or self.cfg.get("predictor_args") or {})

    def _setup_predictors(self) -> None:
        if self._initialized:
            return
        try:
            import torch
            import librosa
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"missing pseudo_mos dependency: {exc}") from exc

        self._torch = torch
        self._librosa = librosa

        use_gpu = bool(self.cfg.get("use_gpu", self.runtime.get("use_gpu", False)))
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        predictor_types = self._predictor_types()
        predictor_args = self._predictor_args()
        cache_dir = str(self.cfg.get("cache_dir", "versa_cache"))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if "utmos" in predictor_types:
            torch.hub.set_dir(cache_dir)
            utmos_repo = str(self.cfg.get("utmos_hub_repo", "ftshijt/SpeechMOS:main"))
            utmos_entry = str(self.cfg.get("utmos_hub_entry", "utmos22_strong"))
            utmos = torch.hub.load(utmos_repo, utmos_entry).to(self.device)
            self.predictor_dict["utmos"] = utmos.float()
            self.predictor_fs["utmos"] = 16000

        if "utmosv2" in predictor_types:
            try:
                import utmosv2
                from utmosv2.dataset.multi_spec import process_audio_only_versa
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(f"utmosv2 is not installed: {exc}") from exc
            self._utmosv2 = utmosv2
            self._process_audio_only_versa = process_audio_only_versa
            utmos_v2 = utmosv2.create_model(pretrained=bool(self.cfg.get("utmosv2_pretrained", True)))
            self.predictor_dict["utmosv2"] = utmos_v2.to(self.device)
            self.predictor_fs["utmosv2"] = 16000

        if any(x in predictor_types for x in ("dnsmos", "plcmos")):
            try:
                import onnxruntime  # noqa: F401
                from speechmos import dnsmos, plcmos
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError(f"speechmos/onnxruntime is required for dnsmos/plcmos: {exc}") from exc
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
                torch.hub.set_dir(cache_dir)
                singmos = torch.hub.load(
                    str(self.cfg.get("singmos_hub_repo", "South-Twilight/SingMOS:v1.1.1")),
                    "singmos_v1",
                    trust_repo=True,
                ).to(self.device)
                self.predictor_dict["singmos_v1"] = singmos
                self.predictor_fs["singmos_v1"] = 16000
            elif predictor == "singmos_pro":
                torch.hub.set_dir(cache_dir)
                singmos = torch.hub.load(
                    str(self.cfg.get("singmos_hub_repo", "South-Twilight/SingMOS:v1.1.1")),
                    "singmos_pro",
                    trust_repo=True,
                ).to(self.device)
                self.predictor_dict["singmos_pro"] = singmos
                self.predictor_fs["singmos_pro"] = 16000
            elif predictor.startswith("dnsmos_pro_"):
                variant = predictor[len("dnsmos_pro_") :]
                model_path = Path(cache_dir) / f"dnsmos_pro_{variant}.pt"
                if not model_path.exists():
                    if not bool(self.cfg.get("allow_download", False)):
                        raise RuntimeError(f"{model_path} not found and allow_download=false")
                    import requests

                    url = (
                        "https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/"
                        f"runs/{variant.upper()}/model_best.pt"
                    )
                    response = requests.get(url, timeout=int(self.runtime.get("timeout_sec", 90)))
                    response.raise_for_status()
                    model_path.write_bytes(response.content)
                self.predictor_dict[predictor] = torch.jit.load(str(model_path), map_location=self.device)
                self.predictor_fs[predictor] = 16000
            else:
                raise NotImplementedError(f"Not supported predictor type: {predictor}")

        self._initialized = True

    def _normalize_audio(self, wav: np.ndarray) -> np.ndarray:
        max_val = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
        if max_val > 0:
            return wav / max_val
        return wav

    def _predict_one(self, wav: np.ndarray, fs: int) -> dict[str, float]:
        if not self._initialized:
            self._setup_predictors()
        assert self._torch is not None
        assert self._librosa is not None

        scores: dict[str, float] = {}
        torch = self._torch
        librosa = self._librosa
        use_gpu = self.device.startswith("cuda")

        for predictor in self.predictor_dict.keys():
            if predictor == "utmos":
                target_fs = self.predictor_fs["utmos"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                if use_gpu:
                    pred_tensor = pred_tensor.to(self.device)
                score = self.predictor_dict["utmos"](pred_tensor.float(), target_fs)[0].item()
                scores["utmos"] = float(score)
            elif predictor == "utmosv2":
                target_fs = self.predictor_fs["utmosv2"]
                pred = librosa.resample(wav, orig_sr=fs, target_sr=target_fs) if fs != target_fs else wav
                if self._utmosv2 is None or self._process_audio_only_versa is None:
                    raise RuntimeError("utmosv2 is unavailable")
                cfg = self.predictor_dict["utmosv2"].cfg
                spec_info = self._process_audio_only_versa(pred, cfg)
                spec_tensor = torch.tensor(spec_info).float().unsqueeze(0)
                data_type = np.zeros(int(self.cfg.get("utmosv2_data_type_len", 10)), dtype=np.float32)
                data_type_idx = int(self.cfg.get("utmosv2_data_type_index", 1))
                if 0 <= data_type_idx < data_type.shape[0]:
                    data_type[data_type_idx] = float(self.cfg.get("utmosv2_data_type_value", 0.0))
                d = torch.tensor(data_type).unsqueeze(0)
                pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                if use_gpu:
                    spec_tensor = spec_tensor.to(self.device)
                    d = d.to(self.device)
                    pred_tensor = pred_tensor.to(self.device)

                repeat_n = int(self.cfg.get("utmosv2_num_repetitions", 5))
                vals: list[float] = []
                with torch.no_grad():
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

    def _aggregate_score(self, metrics: dict[str, float]) -> float | None:
        if not metrics:
            return None
        bounds_cfg = dict(self.cfg.get("metric_bounds") or {})
        weights_cfg = dict(self.task_cfg.get("aggregate_weights") or self.cfg.get("aggregate_weights") or {})
        agg_keys = list(self.task_cfg.get("aggregate_keys") or self.cfg.get("aggregate_keys") or metrics.keys())
        normalize = bool(self.cfg.get("normalize_aggregate", True))

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
        # Build score_keys from predictor types so submetrics are aggregated
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


        try:
            import librosa  # local import to keep module optional
        except Exception as exc:
            rows = [
                self.make_result(
                    sample_id=str(s["sample_id"]),
                    score=None,
                    valid=False,
                    error="pseudo_mos_dependency_missing",
                    reason=str(exc),
                )
                for s in samples
            ]
            return self.finalize(rows)

        # ------------------------------------------------------------------
        # Phase 1: load audio and run predictors for all samples
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Phase 2: aggregate scores
        # ------------------------------------------------------------------
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
                # Flatten sub-metrics into extra so score_keys can find them
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
    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
