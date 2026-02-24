from __future__ import annotations

import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm

from .base import coerce_bool, render_template, try_parse_json
from .base import BaseScorer


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _require_key(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing required key `{path}.{key}`")
    return mapping[key]


class LLMJudgeGeminiScorer(BaseScorer):
    def __init__(
        self,
        *,
        name: str,
        model_ref: str = "",
        global_models: dict[str, Any] | None = None,
        decode_kwargs: dict[str, Any] | None = None,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self.num_workers = num_workers
        self.decode_kwargs = dict(decode_kwargs or {})
        
        global_models = global_models or {}
        self.model_cfg = dict(global_models.get(model_ref, {}))
        
        provider = str(self.model_cfg.get("provider", ""))
        if provider != "gemini":
            raise ValueError(f"expected provider 'gemini', got '{provider}'")

    def _resolve_task(self) -> tuple[str, str, str, str, dict[str, Any]]:
        prompts = _require_key(self.task_cfg, "prompts", "tasks.<type>.scorers[]")
        system_prompt = str(_require_key(prompts, "system", "tasks.<type>.scorers[].prompts"))
        user_prompt = str(_require_key(prompts, "user", "tasks.<type>.scorers[].prompts"))
        audio_a_key = str(_require_key(self.task_cfg, "audio_a_key", "tasks.<type>.scorers[]"))
        audio_b_key = str(_require_key(self.task_cfg, "audio_b_key", "tasks.<type>.scorers[]"))
        task_decode_kwargs: dict[str, Any] = {}
        if "decode_kwargs" in self.task_cfg:
            task_decode_kwargs = dict(self.task_cfg["decode_kwargs"])
        return system_prompt, user_prompt, audio_a_key, audio_b_key, task_decode_kwargs

    def _infer_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample["sample_id"])
        system_tmpl, user_tmpl, audio_a_key, audio_b_key, task_decode_kwargs = self._resolve_task()
        audio_a_path = str(sample[audio_a_key])
        audio_b_path = str(sample[audio_b_key])

        context = dict(sample)
        context["sample_json"] = json.dumps(sample, ensure_ascii=False)
        user_prompt = render_template(user_tmpl, context)
        system_prompt = render_template(system_tmpl, context)

        decode = dict(self.model_cfg.get("decode_kwargs", {}))
        decode.update(self.decode_kwargs)
        decode.update(task_decode_kwargs)
        
        api_key = self.model_cfg.get("api_key", "")
        base_url = self.model_cfg.get("base_url", "")
        model = self.model_cfg.get("model", "")
        headers = dict(self.model_cfg.get("headers", {}))
        
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for audio_path in [audio_a_path, audio_b_path]:
            mime, _ = mimetypes.guess_type(audio_path)
            mime = mime or "audio/wav"
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({
                "type": "input_audio",
                "input_audio": {"format": mime, "data": b64},
            })
        messages.append({"role": "user", "content": user_content})

        payload = {"model": model, "messages": messages, "stream": False}
        payload.update(decode)

        resp = requests.post(
            base_url,
            headers=headers,
            json=payload,
            timeout=self.model_cfg.get("timeout_sec", 60),
        )
        resp.raise_for_status()
        data = resp.json()
        
        raw_response = ""
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                    raw_response = str(first["message"].get("content", ""))
            elif "content" in data:
                raw_response = str(data["content"])
            elif "text" in data:
                raw_response = str(data["text"])
        if not raw_response:
            raw_response = json.dumps(data, ensure_ascii=False)

        return {"sample_id": sample_id, "raw_response": raw_response}

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        workers = int(self.task_cfg.get("num_workers", self.num_workers))
            
        # Phase 1: Inference
        inferences: dict[str, str] = {}
        errors: dict[str, str] = {}
        
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._infer_one, sample): sample for sample in samples}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{self.name} [infer]", leave=False):
                sample = futures[future]
                sample_id = str(sample["sample_id"])
                try:
                    res = future.result()
                    inferences[sample_id] = res["raw_response"]
                except Exception as exc:
                    errors[sample_id] = str(exc)

        # Phase 2: Aggregation / Scoring
        rows: list[dict[str, Any]] = []
        for sample in tqdm(samples, desc=f"{self.name} [score]", leave=False):
            sample_id = str(sample["sample_id"])
            if sample_id in errors:
                rows.append(self.make_result(
                    sample_id=sample_id,
                    score=None,
                    valid=False,
                    error="llm_judge_gemini_infer_failed",
                    reason=errors[sample_id],
                ))
                continue
                
            raw = inferences[sample_id]
            try:
                parsed = try_parse_json(raw)
                if parsed is None:
                    raise RuntimeError(f"judge did not return valid JSON: {raw}")
                valid_raw = _require_key(parsed, "valid", "judge_json")
                valid = coerce_bool(valid_raw)
                if valid is None:
                    raise RuntimeError(f"cannot coerce `valid` to bool: {valid_raw}")
                reason = str(_require_key(parsed, "reason", "judge_json"))
                if "score" in parsed:
                    score = _clamp01(float(parsed["score"]))
                else:
                    score = 1.0 if valid else 0.0
                
                extra: dict[str, Any] = {"judge_raw": raw}
                for key in self.score_keys:
                    if key != "score" and key in parsed:
                        try:
                            extra[key] = _clamp01(float(parsed[key]))
                        except (TypeError, ValueError):
                            pass
                rows.append(self.make_result(
                    sample_id=sample_id,
                    score=score,
                    valid=bool(valid),
                    reason=reason,
                    extra=extra,
                ))
            except Exception as exc:
                rows.append(self.make_result(
                    sample_id=sample_id,
                    score=None,
                    valid=False,
                    error="llm_judge_gemini_score_failed",
                    reason=str(exc),
                ))

        return self.finalize(rows)
