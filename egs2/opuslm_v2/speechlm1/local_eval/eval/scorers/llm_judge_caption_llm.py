from __future__ import annotations

import json
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


class LLMJudgeCaptionLLMScorer(BaseScorer):
    def __init__(
        self,
        *,
        name: str,
        captioner_model_ref: str = "",
        judge_model_ref: str = "",
        global_models: dict[str, Any] | None = None,
        caption_decode_kwargs: dict[str, Any] | None = None,
        judge_decode_kwargs: dict[str, Any] | None = None,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self.num_workers = num_workers
        self.caption_decode_kwargs = dict(caption_decode_kwargs or {})
        self.judge_decode_kwargs = dict(judge_decode_kwargs or {})
        
        global_models = global_models or {}
        self.captioner_model_cfg = dict(global_models.get(captioner_model_ref, {}))
        self.judge_model_cfg = dict(global_models.get(judge_model_ref, {}))

    def _resolve_task(self) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
        prompts = _require_key(self.task_cfg, "prompts", "tasks.<type>.scorers[]")
        system_prompt = str(_require_key(prompts, "system", "tasks.<type>.scorers[].prompts"))
        user_prompt = str(_require_key(prompts, "user", "tasks.<type>.scorers[].prompts"))
        caption_prompt = str(_require_key(self.task_cfg, "caption_prompt", "tasks.<type>.scorers[]"))
        audio_a_key = str(_require_key(self.task_cfg, "audio_a_key", "tasks.<type>.scorers[]"))
        audio_b_key = str(_require_key(self.task_cfg, "audio_b_key", "tasks.<type>.scorers[]"))
        caption_decode_kwargs: dict[str, Any] = {}
        judge_decode_kwargs: dict[str, Any] = {}
        if "caption_decode_kwargs" in self.task_cfg:
            caption_decode_kwargs = dict(self.task_cfg["caption_decode_kwargs"])
        if "judge_decode_kwargs" in self.task_cfg:
            judge_decode_kwargs = dict(self.task_cfg["judge_decode_kwargs"])
        return (
            system_prompt,
            user_prompt,
            caption_prompt,
            audio_a_key,
            audio_b_key,
            caption_decode_kwargs,
            judge_decode_kwargs,
        )

    def _call_qwen(self, model_cfg: dict[str, Any], messages: list[dict[str, Any]], decode_kwargs: dict[str, Any]) -> str:
        api_key = model_cfg.get("api_key", "")
        base_url = model_cfg.get("base_url", "").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        model = model_cfg.get("model", "")
        headers = dict(model_cfg.get("headers", {}))
        
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
            
        payload = {"model": model, "messages": messages, "stream": False}
        payload.update(model_cfg.get("decode_kwargs", {}))
        payload.update(decode_kwargs)
        
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=model_cfg.get("timeout_sec", 60))
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                    return str(first["message"].get("content", ""))
            if "content" in data:
                return str(data["content"])
            if "text" in data:
                return str(data["text"])
        return json.dumps(data, ensure_ascii=False)

    def _caption_audio(
        self,
        *,
        audio_path: str,
        sample: dict[str, Any],
        caption_prompt_template: str,
        caption_decode_kwargs: dict[str, Any],
    ) -> str:
        prompt_context = dict(sample)
        prompt_context["sample_json"] = json.dumps(sample, ensure_ascii=False)
        caption_prompt = render_template(caption_prompt_template, prompt_context)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption_prompt},
                    {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                ],
            }
        ]
        decode = dict(self.caption_decode_kwargs)
        decode.update(caption_decode_kwargs)
        return self._call_qwen(self.captioner_model_cfg, messages, decode).strip()

    def _infer_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample["sample_id"])
        (
            system_tmpl,
            user_tmpl,
            caption_prompt_tmpl,
            audio_a_key,
            audio_b_key,
            task_caption_decode_kwargs,
            task_judge_decode_kwargs,
        ) = self._resolve_task()
        
        audio_a_path = str(sample[audio_a_key])
        audio_b_path = str(sample[audio_b_key])

        caption_a = self._caption_audio(
            audio_path=audio_a_path,
            sample=sample,
            caption_prompt_template=caption_prompt_tmpl,
            caption_decode_kwargs=task_caption_decode_kwargs,
        )
        caption_b = self._caption_audio(
            audio_path=audio_b_path,
            sample=sample,
            caption_prompt_template=caption_prompt_tmpl,
            caption_decode_kwargs=task_caption_decode_kwargs,
        )

        context = dict(sample)
        context["sample_json"] = json.dumps(sample, ensure_ascii=False)
        context["caption_a"] = caption_a
        context["caption_b"] = caption_b
        
        judge_messages = [
            {"role": "system", "content": render_template(system_tmpl, context)},
            {"role": "user", "content": render_template(user_tmpl, context)},
        ]
        judge_decode = dict(self.judge_decode_kwargs)
        judge_decode.update(task_judge_decode_kwargs)
        judge_raw = self._call_qwen(self.judge_model_cfg, judge_messages, judge_decode)
        
        return {
            "sample_id": sample_id,
            "caption_a": caption_a,
            "caption_b": caption_b,
            "judge_raw": judge_raw,
        }

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        workers = int(self.task_cfg.get("num_workers", self.num_workers))
            
        # Phase 1: Inference
        inferences: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._infer_one, sample): sample for sample in samples}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{self.name} [infer]", leave=False):
                sample = futures[future]
                sample_id = str(sample["sample_id"])
                try:
                    res = future.result()
                    inferences[sample_id] = res
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
                    error="llm_judge_caption_llm_infer_failed",
                    reason=errors[sample_id],
                ))
                continue
                
            res = inferences[sample_id]
            judge_raw = res["judge_raw"]
            try:
                parsed = try_parse_json(judge_raw)
                if parsed is None:
                    raise RuntimeError(f"judge did not return valid JSON: {judge_raw}")
                valid_raw = _require_key(parsed, "valid", "judge_json")
                valid = coerce_bool(valid_raw)
                if valid is None:
                    raise RuntimeError(f"cannot coerce `valid` to bool: {valid_raw}")
                reason = str(_require_key(parsed, "reason", "judge_json"))
                if "score" in parsed:
                    score = _clamp01(float(parsed["score"]))
                else:
                    score = 1.0 if valid else 0.0
                    
                extra: dict[str, Any] = {
                    "caption_a": res["caption_a"],
                    "caption_b": res["caption_b"],
                    "judge_raw": judge_raw,
                }
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
                    error="llm_judge_caption_llm_score_failed",
                    reason=str(exc),
                ))

        return self.finalize(rows)
