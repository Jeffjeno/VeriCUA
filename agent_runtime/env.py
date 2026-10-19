from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agent_runtime.types import (
    Action,
    EnvCapabilities,
    EvaluationResult,
    Observation,
    RunTaskResult,
    Task,
    Trace,
    TraceEvent,
)

if TYPE_CHECKING:
    from agent_runtime.evaluator import Evaluator
    from agent_runtime.harness import Harness

class Env(ABC):

    def __init__(self, task: Task, session: Any = None, **env_options: Any) -> None:
        self.task = task
        self.session = session
        self.env_options = env_options
        self.trace = Trace(task_id=task.task_id, benchmark=task.benchmark)
        self._capabilities = EnvCapabilities()

    @property
    def capabilities(self) -> EnvCapabilities:
        return self._capabilities

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Observation, dict[str, Any]]:
        pass

    @abstractmethod
    def run(
        self,
        harness: Harness,
        evaluator: Evaluator | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> RunTaskResult:
        pass

    def step(self, action: Action) -> tuple[Observation, float | None, bool, bool, dict[str, Any]]:

        raise NotImplementedError(
            f"{type(self).__name__} does not support step(). "
            "Use env.run(harness, evaluator) instead."
        )

    def run_setup_script(self, script_content: str) -> dict[str, Any]:

        raise NotImplementedError(
            f"{type(self).__name__} does not support run_setup_script()."
        )

    def close(self) -> None:

        self.trace.finalize()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

__all__ = ["Env"]
