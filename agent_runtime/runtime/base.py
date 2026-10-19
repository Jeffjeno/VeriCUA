from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agent_runtime.types import RunTaskResult

if TYPE_CHECKING:
    from agent_runtime.runner import RunSpec

class Runtime(ABC):

    name: str = ""

    @abstractmethod
    def run(
        self,
        runs: list[RunSpec],
        global_concurrency: int | None = None,
        store: Any = None,
    ) -> list[RunTaskResult]:
        pass

__all__ = ["Runtime"]
