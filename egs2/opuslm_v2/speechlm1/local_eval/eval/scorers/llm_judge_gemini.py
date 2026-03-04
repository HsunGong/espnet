from __future__ import annotations

import json
import logging
import os
from typing import Any

import joblib
import jinja2
from tqdm import tqdm
from google import genai
from google.genai import types
import random

random.seed(7)

from .base import BaseScorer, compute_aspect_avg, auto_detect_score_keys


class LLMJudgeGeminiScorer(BaseScorer):
    def __init__(self, *, name: str, model_kwargs: dict, decode_kwargs: dict = {}, batch_size: int = 8, max_samples: int = 10000, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_kwargs["model"]
        self.decode_kwargs = {}
        self.batch_size = batch_size
        self.max_samples =max_samples

    def _infer_one(self, sample: dict, system_prompt: str, user_prompt: jinja2.Template, audio_keys: list[str]) -> dict:
        sid = sample["sample_id"]
        uploaded_files = []
        try:
            uploaded_files = [self.client.files.upload(file=sample[k]) for k in audio_keys]

            # Build contents with labels for AB test
            _audio_labels = ["[Audio A (original)]", "[Audio B (edited)]"]
            contents = []
            for idx, f in enumerate(uploaded_files):
                label = _audio_labels[idx] if idx < len(_audio_labels) else f"[Audio {idx + 1}]"
                contents.append(label)
                contents.append(f)
            contents.append(user_prompt.render(**sample))

            config_kwargs = dict(self.decode_kwargs)
            config_kwargs["response_mime_type"] = "application/json"
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            judge_resp = json.loads(response.text)
            avg_score, aspect_scores = compute_aspect_avg(judge_resp)
            return {
                "sample_id": sid, "valid": True,
                "score": avg_score,
                "reason": judge_resp.get("reason", ""),
                "judge_resp": judge_resp,
                "extra_scores": aspect_scores,
            }
        except Exception as e:
            return {"sample_id": sid, "valid": False, "score": None, "reason": str(e), "judge_resp": None, "extra_scores": {}}
        finally:
            for f in uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                except Exception as e:
                    logging.info(f"Failed to delete {f.name}: {e}")

    def run(self, samples: list[dict]) -> tuple[list[dict], dict]:
        system_prompt = self.task_cfg.get("system_prompt", "")
        user_prompt = jinja2.Template(self.task_cfg["user_prompt"])
        audio_keys = self.task_cfg.get("audio_keys", ["eval_audio_path"])

        if len(samples) > self.max_samples:
            samples = random.choices(samples, k=self.max_samples)

        rows = []
        for res in tqdm(joblib.Parallel(n_jobs=self.batch_size, backend="threading", return_as="generator")(
            joblib.delayed(self._infer_one)(s, system_prompt, user_prompt, audio_keys) for s in samples
        ), total=len(samples), desc=f"{self.name}", leave=False):
            rows.append(self.make_result(
                sample_id=res["sample_id"], score=res["score"], valid=res["valid"],
                reason=res["reason"],
                extra={"judge_resp": res["judge_resp"], **res.get("extra_scores", {})},
                **({} if res["valid"] else {"error": "llm_judge_gemini_infer_failed"}),
            ))
        return self.finalize(rows)

    def finalize(self, rows: list[dict]) -> tuple[list[dict], dict]:
        self.score_keys = auto_detect_score_keys(rows)
        return super().finalize(rows)
