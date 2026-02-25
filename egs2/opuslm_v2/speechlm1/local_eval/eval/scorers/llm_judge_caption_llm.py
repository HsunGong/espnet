from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm

from .base import coerce_bool, render_template, try_parse_json
from .base import BaseScorer


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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

        models = global_models or {}

        cap = dict(models.get(captioner_model_ref, {}))
        self.captioner_endpoint: str = str(cap.get("base_url", "")).rstrip("/") + "/chat/completions"
        self.captioner_model: str = str(cap.get("model", ""))
        self.captioner_headers: dict[str, str] = dict(cap.get("headers", {}))
        self.captioner_model_decode_kwargs: dict[str, Any] = dict(cap.get("decode_kwargs", {}))
        _cap_key = str(cap.get("api_key", ""))
        if _cap_key and "Authorization" not in self.captioner_headers:
            self.captioner_headers["Authorization"] = f"Bearer {_cap_key}"
        self.captioner_headers.setdefault("Content-Type", "application/json")

        jdg = dict(models.get(judge_model_ref, {}))
        self.judge_endpoint: str = str(jdg.get("base_url", "")).rstrip("/") + "/chat/completions"
        self.judge_model: str = str(jdg.get("model", ""))
        self.judge_headers: dict[str, str] = dict(jdg.get("headers", {}))
        self.judge_model_decode_kwargs: dict[str, Any] = dict(jdg.get("decode_kwargs", {}))
        _jdg_key = str(jdg.get("api_key", ""))
        if _jdg_key and "Authorization" not in self.judge_headers:
            self.judge_headers["Authorization"] = f"Bearer {_jdg_key}"
        self.judge_headers.setdefault("Content-Type", "application/json")

    def _resolve_task(self) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        prompts = _require_key(self.task_cfg, "prompts", "tasks.<type>.scorers[]")
        system_prompt = str(_require_key(prompts, "system", "tasks.<type>.scorers[].prompts"))
        user_prompt = str(_require_key(prompts, "user", "tasks.<type>.scorers[].prompts"))
        caption_prompt = str(_require_key(self.task_cfg, "caption_prompt", "tasks.<type>.scorers[]"))
        caption_decode_kwargs = dict(self.task_cfg.get("caption_decode_kwargs") or {})
        judge_decode_kwargs = dict(self.task_cfg.get("judge_decode_kwargs") or {})
        return system_prompt, user_prompt, caption_prompt, caption_decode_kwargs, judge_decode_kwargs

    def _call_api(self, endpoint: str, model: str, headers: dict, base_decode: dict, extra_decode: dict, messages: list) -> str:
        payload = {"model": model, "messages": messages, "stream": False, **base_decode, **extra_decode}
        resp = requests.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                return str(data["choices"][0].get("message", {}).get("content", ""))
            if "content" in data:
                return str(data["content"])
            if "text" in data:
                return str(data["text"])
        return json.dumps(data, ensure_ascii=False)

    def _caption_audio(self, audio_path: str, sample: dict[str, Any], caption_prompt_template: str, extra_decode: dict[str, Any]) -> str:
        context = {**sample, "sample_json": json.dumps(sample, ensure_ascii=False)}
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": render_template(caption_prompt_template, context)},
                {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
            ],
        }]
        return self._call_api(
            self.captioner_endpoint, self.captioner_model, self.captioner_headers,
            self.captioner_model_decode_kwargs, {**self.caption_decode_kwargs, **extra_decode}, messages,
        ).strip()

    def _infer_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        sid = str(sample["sample_id"])
        system_tmpl, user_tmpl, caption_tmpl, task_cap_decode, task_jdg_decode = self._resolve_task()

        caption_a = self._caption_audio(str(sample["audio_path"]), sample, caption_tmpl, task_cap_decode)
        caption_b = self._caption_audio(str(sample["eval_audio_path"]), sample, caption_tmpl, task_cap_decode)

        context = {**sample, "sample_json": json.dumps(sample, ensure_ascii=False), "caption_a": caption_a, "caption_b": caption_b}
        judge_messages = [
            {"role": "system", "content": render_template(system_tmpl, context)},
            {"role": "user", "content": render_template(user_tmpl, context)},
        ]
        judge_raw = self._call_api(
            self.judge_endpoint, self.judge_model, self.judge_headers,
            self.judge_model_decode_kwargs, {**self.judge_decode_kwargs, **task_jdg_decode}, judge_messages,
        )
        return {"sample_id": sid, "caption_a": caption_a, "caption_b": caption_b, "judge_raw": judge_raw}

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        workers = int(self.task_cfg.get("num_workers", self.num_workers))

        inferences: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._infer_one, s): s for s in samples}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{self.name} [infer]", leave=False):
                sid = str(futures[future]["sample_id"])
                try:
                    inferences[sid] = future.result()
                except Exception as exc:
                    errors[sid] = str(exc)

        rows: list[dict[str, Any]] = []
        for sample in tqdm(samples, desc=f"{self.name} [score]", leave=False):
            sid = str(sample["sample_id"])
            if sid in errors:
                rows.append(self.make_result(sample_id=sid, score=None, valid=False, error="llm_judge_caption_llm_infer_failed", reason=errors[sid]))
                continue
            res = inferences[sid]
            judge_raw = res["judge_raw"]
            try:
                parsed = try_parse_json(judge_raw)
                if parsed is None:
                    raise RuntimeError(f"judge did not return valid JSON: {judge_raw}")
                valid = coerce_bool(_require_key(parsed, "valid", "judge_json"))
                if valid is None:
                    raise RuntimeError("cannot coerce `valid` to bool")
                reason = str(_require_key(parsed, "reason", "judge_json"))
                score = _clamp01(float(parsed["score"])) if "score" in parsed else (1.0 if valid else 0.0)
                rows.append(self.make_result(
                    sample_id=sid, score=score, valid=bool(valid), reason=reason,
                    extra={"caption_a": res["caption_a"], "caption_b": res["caption_b"], "judge_raw": judge_raw},
                ))
            except Exception as exc:
                rows.append(self.make_result(sample_id=sid, score=None, valid=False, error="llm_judge_caption_llm_score_failed", reason=str(exc)))

        return self.finalize(rows)
