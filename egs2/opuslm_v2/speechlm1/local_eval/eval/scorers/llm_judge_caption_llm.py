from __future__ import annotations

import json
from typing import Any

import joblib
from tqdm import tqdm
import jinja2


import itertools
import json
import requests
import threading
import time
from typing import Any, Dict, List, Optional, Set
import random
import logging

from .base import BaseScorer

def parse_vllm_urls(url_string: str) -> List[str]:
    """Parse colon-separated vLLM URLs.

    Args:
        url_string: Single URL or colon-separated URLs.
            Example: "http://host1:8000/v1,http://host2:8000/v1"

    Returns:
        List of parsed URLs.
    """
    urls = list(filter(lambda x: x.startswith("http"), url_string.split(",")))

    # Clean up URLs
    return [url.rstrip("/") for url in urls if url]


class VLLMClient:
    """Synchronous client for vLLM OpenAI-compatible API with multi-URL support."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        max_concurrent: int = 65536,
        per_endpoint_concurrent: int = 32,
        timeout: int = 360,
        max_retries: int = 3,
        default_kwargs: dict = {},
    ):
        """Initialize the vLLM client.

        Args:
            base_url: Base URL(s) for the vLLM API. Can be a single URL or
                multiple URLs separated by colons.
                Example: "http://host1:8000/v1:http://host2:8000/v1"
            model: Model name to use. Defaults to DEFAULT_MODEL.
            max_concurrent: Hard upper cap on total in-flight requests.
                In practice the effective limit is
                ``len(endpoints) * per_endpoint_concurrent``, whichever is
                smaller.
            per_endpoint_concurrent: Max in-flight requests **per** vLLM
                endpoint.  The actual semaphore value is
                ``min(max_concurrent, n_endpoints * per_endpoint_concurrent)``.
                Default 32 means 4 servers → 128 concurrent requests, which
                is enough to saturate vLLM without triggering queue overflow.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
        """
        self.base_urls = parse_vllm_urls(base_url)
        if not self.base_urls:
            raise ValueError(f"No valid URLs found in: {base_url}")

        print(f"Initialized vLLM client with {len(self.base_urls)} endpoint(s):")
        for url in self.base_urls:
            print(f"  - {url}")

        self.model = model or DEFAULT_MODEL
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        # temperature: float = 0.7,
        # max_tokens: int = 512,
        json_mode: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Send a chat completion request with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: If True, enforce JSON output format.

        Returns:
            Generated text content, or None if all retries failed.
        """
        for k in self.default_kwargs.keys():
            kwargs.setdefault(k, self.default_kwargs[k])

        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        logging.debug(f"Request: {payload}")

        for attempt in range(self.max_retries):
            base_url = random.choice(self.base_urls) # self._get_next_url()
            url = f"{base_url}/chat/completions"

            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    # logging.debug(f"Response: {data}")
                    content = data["choices"][0]["message"]["content"]
                    if json_mode:
                        return json.loads(content)
                    return content
                else:
                    error_text = response.text
                    print(
                        f"API error (attempt {attempt + 1}, "
                        f"{base_url}): {response.status_code} - "
                        f"{error_text[:200]}"
                    )
            except requests.Timeout:
                print(f"Timeout (attempt {attempt + 1}, {base_url})")
            except Exception as e:
                print(f"Error (attempt {attempt + 1}, {base_url}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        return None


class LLMJudgeCaptionLLMScorer(BaseScorer):
    """
    Flow:
      1) captioner: audio_path -> target_caption (audio-only, no prompt text)
      2) judge: uses (source_caption, target_caption) + sample kwargs for prompt templates you write
    """

    def __init__(
        self,
        *,
        name: str,
        captioner: dict,
        judge_model: dict,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.captioner_client = VLLMClient(**captioner)
        self.judge_client = VLLMClient(**judge_model)
        self.batch_size = batch_size

    def _infer_one(self, sample: dict[str, Any], system_prompt: str, user_prompt: jinja2.Template, score_key: str) -> dict[str, Any]:
        sid = sample["sample_id"]
        try:
            target_caption_gen_now = self.captioner_client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": f"file://" + sample["eval_audio_path"]}},
                        ],
                    }
                ],
                **self.task_cfg.get("caption_decode_kwargs", {})
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.render(target_caption_gen_now=target_caption_gen_now, **sample)}
            ]

            judge_resp: dict = self.judge_client.chat_completion(
                messages=messages,
                json_mode=True,
                **self.task_cfg.get("judge_decode_kwargs", {})
            )

            return {
                "sample_id": sid,
                "valid": True,
                "score": judge_resp[score_key],
                "reason": judge_resp.get("reason", ""),
                "judge_resp": judge_resp
            }
        except Exception as e:
            return {
                "sample_id": sid,
                "valid": False,
                "score": None,
                "reason": f"llm_infer_exception: {e}",
                "judge_resp": None,
            }

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        system_prompt = self.task_cfg["system_prompt"]
        user_prompt = jinja2.Template(self.task_cfg["user_prompt"])
        score_key = self.task_cfg.get("score_key", "score")

        for res in tqdm(joblib.Parallel(n_jobs=self.batch_size, backend="threading", return_as="generator")(
            joblib.delayed(self._infer_one)(sample, system_prompt, user_prompt, score_key) for sample in samples
        ), total=len(samples), desc=f"{self.name} [infer & score]", leave=False):

            if not res["valid"]:
                rows.append(
                    self.make_result(
                        sample_id=res["sample_id"],
                        score=None,
                        valid=False,
                        error="llm_judge_caption_llm_infer_failed",
                        reason=res["reason"],
                        extra={"judge_resp": res.get("judge_resp", None)},
                    )
                )
                continue

            score = res["score"]
            rows.append(
                self.make_result(
                    sample_id=res["sample_id"],
                    score=score,
                    valid=bool(res["valid"]),
                    reason=res["reason"],
                    extra={"judge_resp": res.get("judge_resp", None)},
                )
            )

        return self.finalize(rows)
