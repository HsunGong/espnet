#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Shared synchronous vLLM client for SFT data simulation."""

import itertools
import json
import requests
import threading
import time
from typing import Any, Dict, List, Optional, Set

import logging

def get_processed_indices(output_file: str, idx_key: str = "idx") -> Set[Any]:
    """Get set of indices/IDs already processed in output file.

    Args:
        output_file: Path to the output JSONL file.
        idx_key: Key name for the index field in output records.

    Returns:
        Set of indices/IDs that have been successfully processed.
        Can contain integers or strings depending on the key type.
    """
    processed: Set[Any] = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if idx_key in record and record[idx_key] is not None:
                        processed.add(record[idx_key])
    except FileNotFoundError:
        pass
    return processed


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


# Default model name
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"


class VLLMClient:
    """Synchronous client for vLLM OpenAI-compatible API with multi-URL support."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        max_concurrent: int = 81920,
        timeout: int = 360,
        max_retries: int = 3,
    ):
        """Initialize the vLLM client.

        Args:
            base_url: Base URL(s) for the vLLM API. Can be a single URL or
                multiple URLs separated by colons.
                Example: "http://host1:8000/v1:http://host2:8000/v1"
            model: Model name to use. Defaults to DEFAULT_MODEL.
            max_concurrent: Maximum number of concurrent requests (total).
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

        # Round-robin URL selector (thread-safe via itertools.cycle)
        self._url_cycle = itertools.cycle(self.base_urls)
        self._url_lock = threading.Lock()

    def _get_next_url(self) -> str:
        """Get next URL in round-robin fashion."""
        with self._url_lock:
            return next(self._url_cycle)

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
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        logging.debug(f"Request: {payload}")

        for attempt in range(self.max_retries):
            base_url = self._get_next_url()
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
