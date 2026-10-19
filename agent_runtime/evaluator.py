from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agent_runtime.types import (
    Action,
    EvaluationEvent,
    EvaluationResult,
    Observation,
    Trace,
)

if TYPE_CHECKING:
    from agent_runtime.env import Env

class Evaluator(ABC):

    supports_native_hook: bool = False

    def on_step(
        self,
        env: Env,
        observation: Observation,
        action: Action,
        next_observation: Observation,
        info: dict[str, Any],
    ) -> list[EvaluationEvent]:

        return []

    @abstractmethod
    def finalize(self, env: Env | None, trace: Trace) -> EvaluationResult:
        pass

    def finalize_from_native(self, native_result: Any, trace: Trace) -> EvaluationResult:

        return self.finalize(env=None, trace=trace)

    def native_hook(self) -> Any:

        return None

__all__ = ["Evaluator"]
