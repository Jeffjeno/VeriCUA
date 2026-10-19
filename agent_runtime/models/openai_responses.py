from __future__ import annotations

import json as _json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.model_client import ModelCapabilities, ModelClient, ModelResponse

logger = logging.getLogger(__name__)

@dataclass
class ResponsesResult:

    response_id: str | None
    output: list[dict[str, Any]]
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def computer_calls(self) -> list[dict[str, Any]]:
        return [block for block in self.output if block.get("type") == "computer_call"]

    def messages(self) -> list[dict[str, Any]]:
        return [block for block in self.output if block.get("type") == "message"]

    def reasoning(self) -> list[dict[str, Any]]:
        return [block for block in self.output if block.get("type") == "reasoning"]

class OpenAIResponsesClient(ModelClient):

    def __init__(
        self,
        *,
        model: str = "computer-use-preview",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        extra_body: dict[str, Any] | None = None,
        proxy_url: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("AP_OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("AP_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.proxy_url = proxy_url

    def _build_opener(self) -> urllib.request.OpenerDirector:
        proxy = (
            self.proxy_url
            or os.environ.get("AGENT_RUNTIME_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
        )
        if proxy:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        else:
            handler = urllib.request.ProxyHandler()
        return urllib.request.build_opener(handler)

    def responses_create(
        self,
        *,
        input: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        reasoning: dict[str, Any] | None = None,
        truncation: str | None = None,
        **kwargs: Any,
    ) -> ResponsesResult:

        body: dict[str, Any] = {"model": kwargs.pop("model", self.model)}
        if input is not None:
            body["input"] = input
        if tools:
            body["tools"] = tools
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        if instructions:
            body["instructions"] = instructions
        if reasoning is not None:
            body["reasoning"] = reasoning
        if truncation is not None:
            body["truncation"] = truncation
        body.update(self.extra_body)
        body.update(kwargs)

        url = f"{self.base_url}/responses"
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
                f"Responses API request failed: HTTP {exc.code}: {err_body}"
            ) from exc

        elapsed = time.monotonic() - t0
        logger.debug("Responses API call took %.2fs", elapsed)

        usage_raw = raw_body.get("usage") or {}
        usage = {
            "input_tokens": usage_raw.get("input_tokens", 0),
            "output_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
            "elapsed_seconds": elapsed,
        }
        return ResponsesResult(
            response_id=raw_body.get("id"),
            output=list(raw_body.get("output") or []),
            usage=usage,
            raw=raw_body,
        )

    def respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:

        result = self.responses_create(input=messages, tools=tools, **kwargs)
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in result.output:
            btype = block.get("type")
            if btype == "message":
                for part in block.get("content", []) or []:
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        text_parts.append(part["text"])
            elif btype in ("tool_use", "function_call"):
                tool_calls.append({
                    "id": block.get("call_id") or block.get("id"),
                    "type": btype,
                    "name": block.get("name"),
                    "arguments": block.get("arguments") or block.get("input"),
                })
        return ModelResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            raw=result.raw,
            usage=result.usage,
        )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tool_calling=True,
            multimodal=True,
            prompt_cache=True,
            max_context_window=None,
            max_output_tokens=None,
        )

    def __repr__(self) -> str:
        return f"OpenAIResponsesClient(base_url={self.base_url!r}, model={self.model!r})"

__all__ = ["OpenAIResponsesClient", "ResponsesResult"]
