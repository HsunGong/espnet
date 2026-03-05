from __future__ import annotations

import json
import math
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

from .base import BaseScorer, compute_aspect_avg, auto_detect_score_keys

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
        timeout: int = 3600,
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
                    message = data["choices"][0]["message"]
                    content = message["content"]
                    reasoning_content = message.get("reasoning_content", None)
                    if json_mode:
                        parsed = json.loads(content)
                        if reasoning_content:
                            parsed["_reasoning_content"] = reasoning_content
                        return parsed
                    if reasoning_content:
                        return content, reasoning_content
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

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings via the ``/v1/embeddings`` endpoint.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, or an empty list on failure.
        """
        payload = {"model": self.model, "input": texts}
        for attempt in range(self.max_retries):
            base_url = random.choice(self.base_urls)
            url = f"{base_url}/embeddings"
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    # Sort by index to guarantee order
                    items = sorted(data["data"], key=lambda d: d["index"])
                    return [item["embedding"] for item in items]
                else:
                    print(
                        f"Embeddings API error (attempt {attempt + 1}, "
                        f"{base_url}): {response.status_code} - "
                        f"{response.text[:200]}"
                    )
            except requests.Timeout:
                print(f"Embeddings timeout (attempt {attempt + 1}, {base_url})")
            except Exception as e:
                print(f"Embeddings error (attempt {attempt + 1}, {base_url}): {e}")
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        return []


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        judge_model: dict | None = None,
        embedding_model: dict | None = None,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.captioner_client = VLLMClient(**captioner)
        self.judge_client = VLLMClient(**judge_model) if judge_model else None
        self.embedding_client = VLLMClient(**embedding_model) if embedding_model else None
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Caption generation (shared by both modes)
    # ------------------------------------------------------------------

    def _caption_audio(self, sample: dict[str, Any]) -> str:
        """Generate a caption for ``eval_audio_path`` via the captioner."""
        result = self.captioner_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": "file://" + sample["eval_audio_path"]}},
                    ],
                }
            ],
            **self.task_cfg.get("caption_decode_kwargs", {}),
        )
        if result is None:
            raise RuntimeError("Captioner returned None (all retries exhausted)")
        return str(result).strip() if not isinstance(result, tuple) else str(result[0]).strip()

    # ------------------------------------------------------------------
    # Mode 1: LLM judge
    # ------------------------------------------------------------------

    def _infer_one(self, sample: dict[str, Any], system_prompt: str, user_prompt: jinja2.Template, score_key: str) -> dict[str, Any]:
        sid = sample["sample_id"]
        try:
            target_caption_gen_now = self._caption_audio(sample)

            assert self.judge_client is not None, "judge_model is required for LLM-judge mode"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.render(target_caption_gen_now=target_caption_gen_now, **sample)}
            ]

            judge_resp = self.judge_client.chat_completion(
                messages=messages,
                json_mode=True,
                **self.task_cfg.get("judge_decode_kwargs", {})
            )

            if judge_resp is None:
                raise RuntimeError("Judge LLM returned None (all retries exhausted)")
            if not isinstance(judge_resp, dict):
                raise RuntimeError(f"Judge LLM returned non-dict: {type(judge_resp)}")

            avg_score, aspect_scores = compute_aspect_avg(judge_resp)

            return {
                "sample_id": sid,
                "valid": True,
                "score": avg_score,
                "reason": judge_resp.get("reason", ""),
                "judge_resp": judge_resp,
                "extra_scores": aspect_scores,
                "target_caption_gen_now": target_caption_gen_now,
            }
        except Exception as e:
            return {
                "sample_id": sid,
                "valid": False,
                "score": None,
                "reason": f"llm_infer_exception: {e}",
                "judge_resp": None,
                "extra_scores": {},
                "target_caption_gen_now": None,
            }

    # ------------------------------------------------------------------
    # Mode 2: Embedding similarity  (/v1/embeddings)
    # ------------------------------------------------------------------

    def _infer_one_embedding(
        self,
        sample: dict[str, Any],
        ref_caption_key: str,
    ) -> dict[str, Any]:
        """Caption ``eval_audio_path``, then compute cosine similarity
        between the generated caption and the reference caption via
        ``/v1/embeddings``.
        """
        sid = sample["sample_id"]
        try:
            target_caption_gen_now = self._caption_audio(sample)

            ref_caption = sample.get(ref_caption_key, "")
            if not ref_caption:
                raise RuntimeError(f"Reference caption key '{ref_caption_key}' is empty or missing")

            assert self.embedding_client is not None, "embedding_model is required for embedding mode"
            embeddings = self.embedding_client.get_embeddings([ref_caption, target_caption_gen_now])
            if len(embeddings) < 2:
                raise RuntimeError("Embeddings API returned fewer than 2 vectors")

            sim = _cosine_similarity(embeddings[0], embeddings[1])

            return {
                "sample_id": sid,
                "valid": True,
                "score": round(sim, 6),
                "reason": "",
                "extra_scores": {"caption_similarity": round(sim, 6)},
                "target_caption_gen_now": target_caption_gen_now,
            }
        except Exception as e:
            return {
                "sample_id": sid,
                "valid": False,
                "score": None,
                "reason": f"embedding_exception: {e}",
                "extra_scores": {},
                "target_caption_gen_now": None,
            }

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def run(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mode = self.task_cfg.get("mode", "judge")  # "judge" | "embedding"
        if mode == "embedding":
            return self._run_embedding(samples)
        return self._run_judge(samples)

    # ---------- judge mode ----------

    def _run_judge(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        system_prompt = self.task_cfg["system_prompt"]
        user_prompt = jinja2.Template(self.task_cfg["user_prompt"])
        score_key = self.task_cfg.get("score_key", "score")

        for res in tqdm(joblib.Parallel(n_jobs=self.batch_size, backend="threading", return_as="generator")(
            joblib.delayed(self._infer_one)(sample, system_prompt, user_prompt, score_key) for sample in samples
        ), total=len(samples), desc=f"{self.name} [judge]", leave=False):

            rows.append(
                self.make_result(
                    sample_id=res["sample_id"],
                    score=res["score"],
                    valid=res["valid"],
                    reason=res["reason"],
                    extra={
                        "judge_resp": res.get("judge_resp"),
                        "target_caption_gen_now": res.get("target_caption_gen_now"),
                        **res.get("extra_scores", {}),
                    },
                    **(
                        {}
                        if res["valid"]
                        else {"error": "llm_judge_caption_llm_infer_failed"}
                    ),
                )
            )

        return self.finalize(rows)

    # ---------- embedding mode ----------

    def _run_embedding(self, samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        ref_caption_key = self.task_cfg.get("ref_caption_key", "target_audio_caption")

        for res in tqdm(joblib.Parallel(n_jobs=self.batch_size, backend="threading", return_as="generator")(
            joblib.delayed(self._infer_one_embedding)(sample, ref_caption_key) for sample in samples
        ), total=len(samples), desc=f"{self.name} [embedding]", leave=False):

            rows.append(
                self.make_result(
                    sample_id=res["sample_id"],
                    score=res["score"],
                    valid=res["valid"],
                    reason=res["reason"],
                    extra={
                        "target_caption_gen_now": res.get("target_caption_gen_now"),
                        **res.get("extra_scores", {}),
                    },
                    **(
                        {}
                        if res["valid"]
                        else {"error": "llm_judge_caption_llm_infer_failed"}
                    ),
                )
            )

        return self.finalize(rows)

    def finalize(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.score_keys = auto_detect_score_keys(rows)
        return super().finalize(rows)
