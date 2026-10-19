from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_runtime.env import Env
from agent_runtime.evaluator import Evaluator
from agent_runtime.types import Task

class Benchmark(ABC):

    name: str = ""

    @abstractmethod
    def list_tasks(
        self,
        split: str | None = None,
        limit: int | None = None,
        **filters: Any,
    ) -> list[Task]:
        pass

    def get_task(self, task_id: str) -> Task:

        for task in self.list_tasks():
            if task.task_id == task_id:
                return task
        raise KeyError(f"Task '{task_id}' not found in benchmark '{self.name}'")

    @abstractmethod
    def make_env(self, task: Task, session: Any = None, **env_options: Any) -> Env:
        pass

    @abstractmethod
    def make_evaluator(self, task: Task | None = None, mode: str = "native", **kwargs: Any) -> Evaluator:
        pass

    def requirements(self) -> dict[str, Any]:

        return {}

__all__ = ["Benchmark"]
