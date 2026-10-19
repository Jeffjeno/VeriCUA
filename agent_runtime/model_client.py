from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ModelCapabilities:

    tool_calling: bool = False
    multimodal: bool = False
    prompt_cache: bool = False
    max_context_window: int | None = None
    max_output_tokens: int | None = None

class ModelResponse:

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        raw: Any = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.raw = raw
        self.usage = usage or {}

class ModelClient(ABC):

    @abstractmethod
    def respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        pass

    def capabilities(self) -> ModelCapabilities:

        return ModelCapabilities()

__all__ = ["ModelCapabilities", "ModelClient", "ModelResponse"]
