"""Common abstractions shared by all OSWorld agents.

This module defines ``BaseAgent`` – an abstract base class that codifies
what every agent implementation inside ``mm_agents`` is expected to do.
It centralizes shared state such as platform/action/observation
configuration, trajectory bookkeeping, environment coupling, and logging
so concrete agents can focus on their model-specific logic.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
from mm_agents.config import AgentConfig
try:
    from pyregister import Registrable
except ImportError:  # pragma: no cover
    class Registrable:  # type: ignore
        pass

AgentObservation = Dict[str, Any]
AgentAction = Any
AgentResponse = Any


class Agent(ABC, Registrable):
    """Abstract base class for all desktop agents.

    Subclasses are expected to implement :meth:`predict` (and optionally
    :meth:`step`) while reusing the shared trajectory/state management
    provided here.  The class deliberately keeps the data structures
    flexible – different agents may store screenshots, accessibility tree
    dumps, or SOM tables – but it enforces consistent member names and
    history management.
    """

    LOGGER_NAME = "baseagent"

    def __init__(self, config_path: str = None, **kwargs):
        super().__init__()
        if config_path is not None:
            self.config = AgentConfig.from_yaml(config_path)
        else:
            self.config = AgentConfig(**kwargs)
        
        self.max_steps = self.config.max_steps
        self.enable_log = self.config.enable_log
        self.log_dir = self.config.log_dir
        self.env = None
        
        self.logger = logging.getLogger(self.LOGGER_NAME)

        if self.enable_log:
            assert self.log_dir is not None, "log_dir must be specified when enable_log=True"
        

    # ------------------------------------------------------------------
    # Public lifecycle helpers
    # ------------------------------------------------------------------
    def bind_env(self, env: Any) -> None:
        """Attach a runtime environment so :meth:`step` can use it."""
        self.env = env

    # def reset(
    #     self,
    #     runtime_logger: Optional[logging.Logger] = None,
    #     **kwargs: Any,
    # ) -> None:
    #     """Reset per-episode state.

    #     Args:
    #         runtime_logger: Optional dedicated logger for the task run.
    #         **kwargs: Forwarded to :meth:`_on_reset` so subclasses can
    #             restore extra state (VM IP, session ids, etc.).
    #     """

    #     self.logger = runtime_logger if runtime_logger else logging.getLogger(self.LOGGER_NAME)
    #     self.reset_history()
    #     self._on_reset(**kwargs)



    # ------------------------------------------------------------------
    # Abstract / overridable behaviour
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self, **kwargs: Any) -> None:
        """Reset per-episode state."""

    @abstractmethod
    def predict(
        self,
        instruction: str,
        obs: AgentObservation,
        **kwargs: Any,
    ) -> Any:
        """Produce actions for the current observation."""
    

        



__all__ = ["Agent", "AgentObservation", "AgentAction", "AgentResponse"]
