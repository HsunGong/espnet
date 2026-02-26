"""OpenAI-compatible multimodal LLM judge scorer.

Sends audio files + text prompts directly to a vLLM endpoint
(e.g., Qwen3-Omni) for evaluation via the OpenAI-compatible API.

Unlike ``LLMJudgeCaptionLLMScorer``, no separate captioner step is
needed — this scorer sends audio directly to a multimodal model that
can hear the audio and judge it in one shot.

Task-level config keys (set per-task in YAML under the scorer entry):
    system_prompt  : str   – System prompt for the judge.
    user_prompt    : str   – Jinja2 template for the user message.
    score_key      : str   – Key in the JSON response for the main score (default: "score").
    audio_keys     : list  – Sample field names whose values are audio file paths
                             to include in the multimodal request.
    decode_kwargs  : dict  – Extra kwargs forwarded to ``chat_completion``.
    score_keys     : list  – Multi-dimensional score keys to track in the summary.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import jinja2
import joblib
from tqdm import tqdm

from .base import BaseScorer
from .llm_judge_caption_llm import VLLMClient


class LLMJudgeOpenAIScorer(BaseScorer):
    """Multimodal LLM judge using an OpenAI-compatible vLLM endpoint.

    Parameters
    ----------
    model : dict
        Kwargs forwarded to ``VLLMClient`` — must contain at least
        ``base_url`` and ``model``.
    batch_size : int
        Number of concurrent inference threads.
    max_samples : int
        Hard cap on samples to evaluate (randomly sub-sampled if exceeded).
    """

    def __init__(
        self,
        *,
        name: str,
        model: dict,
        batch_size: int = 8,
        max_samples: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.client = VLLMClient(**model)
        self.batch_size = batch_size
        self.max_samples = max_samples

    # ------------------------------------------------------------------
    # Per-sample inference
    # ------------------------------------------------------------------

    def _infer_one(
        self,
        sample: dict[str, Any],
        system_prompt: str,
        user_prompt: jinja2.Template,
        score_key: str,
        audio_keys: list[str],
    ) -> dict[str, Any]:
        sid = sample["sample_id"]
        try:
            # Build multimodal user content: audio file(s) + rendered text
            content: list[dict[str, Any]] = []
            for audio_key in audio_keys:
                audio_path = sample.get(audio_key)
                if audio_path:
                    content.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{audio_path}"},
                        }
                    )

            rendered_user = user_prompt.render(**sample)
            content.append({"type": "text", "text": rendered_user})

            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            judge_resp: dict = self.client.chat_completion(
                messages=messages,
                json_mode=True,
                **self.task_cfg.get("decode_kwargs", {}),
            )

            # Extract numeric sub-scores from the response so that
            # ``summarize_metric_rows`` can compute averages for each.
            extra_scores: dict[str, Any] = {}
            if isinstance(judge_resp, dict):
                for k, v in judge_resp.items():
                    if k not in ("score", "reason") and isinstance(v, (int, float)):
                        extra_scores[k] = float(v)

            return {
                "sample_id": sid,
                "valid": True,
                "score": judge_resp[score_key],
                "reason": judge_resp.get("reason", ""),
                "judge_resp": judge_resp,
                "extra_scores": extra_scores,
            }
        except Exception as e:
            logging.warning(f"[{self.name}] inference failed for {sid}: {e}")
            return {
                "sample_id": sid,
                "valid": False,
                "score": None,
                "reason": f"llm_judge_openai_exception: {e}",
                "judge_resp": None,
                "extra_scores": {},
            }

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system_prompt = self.task_cfg.get("system_prompt", "")
        user_prompt = jinja2.Template(self.task_cfg["user_prompt"])
        score_key = self.task_cfg.get("score_key", "score")
        audio_keys = self.task_cfg.get("audio_keys", ["eval_audio_path"])

        if len(samples) > self.max_samples:
            samples = random.choices(samples, k=self.max_samples)

        rows: list[dict[str, Any]] = []
        for res in tqdm(
            joblib.Parallel(
                n_jobs=self.batch_size,
                backend="threading",
                return_as="generator",
            )(
                joblib.delayed(self._infer_one)(
                    s, system_prompt, user_prompt, score_key, audio_keys,
                )
                for s in samples
            ),
            total=len(samples),
            desc=f"{self.name}",
            leave=False,
        ):
            rows.append(
                self.make_result(
                    sample_id=res["sample_id"],
                    score=res["score"],
                    valid=res["valid"],
                    reason=res["reason"],
                    extra={
                        "judge_resp": res["judge_resp"],
                        **res.get("extra_scores", {}),
                    },
                    **(
                        {}
                        if res["valid"]
                        else {"error": "llm_judge_openai_infer_failed"}
                    ),
                )
            )

        return self.finalize(rows)
