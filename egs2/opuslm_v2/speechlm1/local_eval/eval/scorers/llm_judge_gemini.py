from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from .base import coerce_bool, render_template, try_parse_json
from .base import BaseScorer
from .clients import build_client


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
        cfg: dict[str, Any],
        runtime: dict[str, Any],
        global_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, cfg=cfg, runtime=runtime)
        global_config = global_config or {}
        models_cfg = global_config.get("models", {})
        
        model_ref = str(_require_key(self.cfg, "model_ref", f"scorers.{name}"))
        if model_ref not in models_cfg:
            raise KeyError(f"model_ref `{model_ref}` not found in models config")
            
        model_cfg = models_cfg[model_ref]
        self.client = build_client(model_cfg)
        self.decode_kwargs = dict(self.cfg.get("decode_kwargs", {}))

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

        decode = dict(self.decode_kwargs)
        decode.update(task_decode_kwargs)
        
        raw_response = self.client.infer(
            audio_paths=[audio_a_path, audio_b_path],
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            **decode,
        )
        return {"sample_id": sample_id, "raw_response": raw_response}

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if "num_workers" in self.task_cfg:
            workers = int(self.task_cfg["num_workers"])
        elif "num_workers" in self.runtime:
            workers = int(self.runtime["num_workers"])
        else:
            workers = 4
            
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
