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
    return max(0.0, min(1.0, value))


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

        cfg = dict((global_models or {}).get(model_ref, {}))
        provider = str(cfg.get("provider", ""))
        if provider != "gemini":
            raise ValueError(f"expected provider 'gemini', got '{provider}'")

        self.api_key: str = str(cfg.get("api_key", ""))
        self.base_url: str = str(cfg.get("base_url", ""))
        self.model: str = str(cfg.get("model", ""))
        self.headers: dict[str, str] = dict(cfg.get("headers", {}))
        self.model_decode_kwargs: dict[str, Any] = dict(cfg.get("decode_kwargs", {}))

    def _resolve_task(self) -> tuple[str, str, dict[str, Any]]:
        prompts = _require_key(self.task_cfg, "prompts", "tasks.<type>.scorers[]")
        system_prompt = str(_require_key(prompts, "system", "tasks.<type>.scorers[].prompts"))
        user_prompt = str(_require_key(prompts, "user", "tasks.<type>.scorers[].prompts"))
        task_decode_kwargs = dict(self.task_cfg.get("decode_kwargs") or {})
        return system_prompt, user_prompt, task_decode_kwargs

    def _infer_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample["sample_id"])
        system_tmpl, user_tmpl, task_decode_kwargs = self._resolve_task()

        context = dict(sample)
        context["sample_json"] = json.dumps(sample, ensure_ascii=False)

        headers = dict(self.headers)
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.setdefault("Content-Type", "application/json")

        decode = {**self.model_decode_kwargs, **self.decode_kwargs, **task_decode_kwargs}

        messages = []
        if system_tmpl:
            messages.append({"role": "system", "content": [{"type": "text", "text": render_template(system_tmpl, context)}]})

        user_content: list[dict[str, Any]] = [{"type": "text", "text": render_template(user_tmpl, context)}]
        for audio_path in [str(sample["audio_path"]), str(sample["eval_audio_path"])]:
            mime, _ = mimetypes.guess_type(audio_path)
            mime = mime or "audio/wav"
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({"type": "input_audio", "input_audio": {"format": mime, "data": b64}})
        messages.append({"role": "user", "content": user_content})

        payload = {"model": self.model, "messages": messages, "stream": False, **decode}
        resp = requests.post(self.base_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw_response = ""
        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                raw_response = str(msg.get("content", ""))
            elif "content" in data:
                raw_response = str(data["content"])
            elif "text" in data:
                raw_response = str(data["text"])
        return {"sample_id": sample_id, "raw_response": raw_response or json.dumps(data, ensure_ascii=False)}

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        workers = int(self.task_cfg.get("num_workers", self.num_workers))

        inferences: dict[str, str] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._infer_one, s): s for s in samples}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{self.name} [infer]", leave=False):
                sid = str(futures[future]["sample_id"])
                try:
                    inferences[sid] = future.result()["raw_response"]
                except Exception as exc:
                    errors[sid] = str(exc)

        rows: list[dict[str, Any]] = []
        for sample in tqdm(samples, desc=f"{self.name} [score]", leave=False):
            sid = str(sample["sample_id"])
            if sid in errors:
                rows.append(self.make_result(sample_id=sid, score=None, valid=False, error="llm_judge_gemini_infer_failed", reason=errors[sid]))
                continue
            raw = inferences[sid]
            try:
                parsed = try_parse_json(raw)
                if parsed is None:
                    raise RuntimeError(f"judge did not return valid JSON: {raw}")
                valid = coerce_bool(_require_key(parsed, "valid", "judge_json"))
                if valid is None:
                    raise RuntimeError(f"cannot coerce `valid` to bool")
                reason = str(_require_key(parsed, "reason", "judge_json"))
                score = _clamp01(float(parsed["score"])) if "score" in parsed else (1.0 if valid else 0.0)
                rows.append(self.make_result(sample_id=sid, score=score, valid=bool(valid), reason=reason, extra={"judge_raw": raw}))
            except Exception as exc:
                rows.append(self.make_result(sample_id=sid, score=None, valid=False, error="llm_judge_gemini_score_failed", reason=str(exc)))

        return self.finalize(rows)
