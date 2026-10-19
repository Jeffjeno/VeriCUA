from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from eval_runtime.agents.agent_config import AgentConfig
from eval_runtime.agents.registry import register, create_agent

AgentObservation = Dict[str, Any]
AgentAction = Any
AgentResponse = Any

class Agent(ABC):

    LOGGER_NAME = "baseagent"

    def __init__(self, config_path: str = None, **kwargs):
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

    def bind_env(self, env: Any) -> None:
        self.env = env

    @abstractmethod
    def reset(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        instruction: str,
        obs: AgentObservation,
        **kwargs: Any,
    ) -> Any:
        pass

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "Agent":

        params = dict(params)
        agent_type = params.pop("type")
        return create_agent(agent_type, **params)

__all__ = ["Agent", "AgentObservation", "AgentAction", "AgentResponse"]
