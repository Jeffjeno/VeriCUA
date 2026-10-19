from __future__ import annotations

import json as _json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agent_runtime.model_client import ModelCapabilities, ModelClient, ModelResponse

logger = logging.getLogger(__name__)

class OpenAIChatCompatible(ModelClient):

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        proxy_url: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.proxy_url = proxy_url

    def _build_opener(self) -> urllib.request.OpenerDirector:
        proxy = self.proxy_url or os.environ.get("AGENT_RUNTIME_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        if proxy:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        else:
            handler = urllib.request.ProxyHandler()
        return urllib.request.build_opener(handler)

    def respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:

        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "top_p": kwargs.pop("top_p", self.top_p),
            **self.extra_body,
            **kwargs,
        }
        if tools:
            body["tools"] = tools

        url = f"{self.base_url}/chat/completions"
        data = _json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)

        t0 = time.monotonic()
        opener = self._build_opener()
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                raw_body = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"Chat completions request failed: HTTP {exc.code}: {err_body}"
            ) from exc

        elapsed = time.monotonic() - t0
        logger.debug("Chat completions took %.2fs", elapsed)

        choice = raw_body["choices"][0]
        message = choice["message"]
        content = message.get("content")

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                }
                for tc in message["tool_calls"]
            ]

        usage = raw_body.get("usage", {})

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=raw_body,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "elapsed_seconds": elapsed,
            },
        )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tool_calling=True,
            multimodal=True,
            max_context_window=None,
            max_output_tokens=self.max_tokens,
        )

    def __repr__(self) -> str:
        return f"OpenAIChatCompatible(base_url={self.base_url!r}, model={self.model!r})"

__all__ = ["OpenAIChatCompatible"]
