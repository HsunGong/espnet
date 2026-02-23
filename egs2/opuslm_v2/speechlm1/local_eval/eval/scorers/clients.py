from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

def _mime_type_for(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    return "audio/wav"

def _file_to_base64(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class Qwen3Client:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: int,
        headers: dict[str, str] | None = None,
        decode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.decode_kwargs = dict(decode_kwargs or {})
        self.headers = dict(headers or {})
        if self.api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def infer(self, *, messages: list[dict[str, Any]], **decode_overrides: Any) -> Any:
        import requests
        decode = dict(self.decode_kwargs)
        decode.update(decode_overrides)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        payload.update(decode)

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                    if "content" in first["message"]:
                        return first["message"]["content"]
            if "content" in data:
                return data["content"]
            if "text" in data:
                return data["text"]
        return data

class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: int,
        headers: dict[str, str] | None = None,
        decode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_sec = timeout_sec
        self.decode_kwargs = dict(decode_kwargs or {})
        self.headers = dict(headers or {})
        if self.api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def infer(
        self,
        *,
        audio_paths: list[str],
        user_prompt: str,
        system_prompt: str | None = None,
        **decode_overrides: Any,
    ) -> str:
        import requests
        decode = dict(self.decode_kwargs)
        decode.update(decode_overrides)
        stream = bool(decode["stream"]) if "stream" in decode else False

        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for audio_path in audio_paths:
            audio_b64 = _file_to_base64(audio_path)
            audio_mime = _mime_type_for(audio_path)
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"format": audio_mime, "data": audio_b64},
                }
            )
        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        payload.update(decode)

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout_sec,
            stream=stream,
        )
        response.raise_for_status()

        if not stream:
            data = response.json()
            if isinstance(data, dict):
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict) and "message" in first and isinstance(first["message"], dict):
                        if "content" in first["message"]:
                            return str(first["message"]["content"])
                if "content" in data:
                    return str(data["content"])
                if "text" in data:
                    return str(data["text"])
            return json.dumps(data, ensure_ascii=False)

        result = ""
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded:
                continue
            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()
            if decoded in {"[DONE]", "done"}:
                break
            try:
                chunk = json.loads(decoded)
                if "choices" in chunk and isinstance(chunk["choices"], list) and chunk["choices"]:
                    delta = chunk["choices"][0]
                    if isinstance(delta, dict) and "delta" in delta and isinstance(delta["delta"], dict):
                        if "content" in delta["delta"]:
                            result += str(delta["delta"]["content"])
            except json.JSONDecodeError:
                result += decoded
        return result

def build_client(model_cfg: dict[str, Any]) -> Any:
    provider = str(model_cfg.get("provider", ""))
    api_key = str(model_cfg.get("api_key", ""))
    base_url = str(model_cfg.get("base_url", ""))
    model = str(model_cfg.get("model", ""))
    timeout_sec = int(model_cfg.get("timeout_sec", 60))
    headers = dict(model_cfg.get("headers", {}))
    decode_kwargs = dict(model_cfg.get("decode_kwargs", {}))

    if provider == "qwen3":
        return Qwen3Client(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            headers=headers,
            decode_kwargs=decode_kwargs,
        )
    elif provider == "gemini":
        return GeminiClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            headers=headers,
            decode_kwargs=decode_kwargs,
        )
    else:
        raise ValueError(f"unsupported provider: {provider}")
