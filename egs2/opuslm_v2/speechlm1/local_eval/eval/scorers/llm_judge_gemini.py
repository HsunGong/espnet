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

from .base import BaseScorer


class LLMJudgeGeminiScorer(BaseScorer):
    def __init__(self, *, name: str, model_kwargs: dict, decode_kwargs: dict = {}, batch_size: int = 8, max_samples: int = 10000, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_kwargs["model"]
        self.decode_kwargs = {}
        self.batch_size = batch_size
        self.max_samples =max_samples

    def _infer_one(self, sample: dict, user_prompt: jinja2.Template, score_key: str, audio_keys: list[str]) -> dict:
        sid = sample["sample_id"]
        uploaded_files = []
        try:
            uploaded_files = [self.client.files.upload(file=sample[k]) for k in audio_keys]
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[*uploaded_files, user_prompt.render(**sample)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    **self.decode_kwargs
                ),
            )
            judge_resp = json.loads(response.text)
            return {"sample_id": sid, "valid": True, "score": judge_resp[score_key], "reason": judge_resp.get("reason", ""), "judge_resp": judge_resp}
        except Exception as e:
            return {"sample_id": sid, "valid": False, "score": None, "reason": str(e), "judge_resp": None}
        finally:
            for f in uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                except Exception as e:
                    logging.info(f"Failed to delete {f.name}: {e}")

    def run(self, samples: list[dict]) -> tuple[list[dict], dict]:
        user_prompt = jinja2.Template(self.task_cfg["user_prompt"])
        score_key = self.task_cfg.get("score_key", "score")
        audio_keys = self.task_cfg["audio_keys"]

        if len(samples) > self.max_samples:
            samples = random.choices(samples, k=self.max_samples)

        rows = []
        for res in tqdm(joblib.Parallel(n_jobs=self.batch_size, backend="threading", return_as="generator")(
            joblib.delayed(self._infer_one)(s, user_prompt, score_key, audio_keys) for s in samples
        ), total=len(samples), desc=f"{self.name}", leave=False):
            rows.append(self.make_result(
                sample_id=res["sample_id"], score=res["score"], valid=res["valid"],
                reason=res["reason"], extra={"judge_resp": res["judge_resp"]},
                **({} if res["valid"] else {"error": "llm_judge_gemini_infer_failed"}),
            ))
        return self.finalize(rows)
