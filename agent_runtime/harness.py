from __future__ import annotations

from abc import ABC, abstractmethod

from agent_runtime.types import Action, HarnessContext, Observation

class Harness(ABC):

    @abstractmethod
    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        pass

__all__ = ["Harness"]
