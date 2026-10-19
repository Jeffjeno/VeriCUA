from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, Observation

from agent_runtime.harnesses.qwen35vl import (
    _ensure_mm_agents_importable,
    _get_screenshot_bytes,
    _pyautogui_code_to_action,
)

logger = logging.getLogger(__name__)

class Qwen3VLProbeHarness(Harness):

    def __init__(
        self,
        config_path: str | None = None,
        model: str = "qwen3-vl",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 0.9,
        platform: str = "ubuntu",
        history_n: int = 4,
        coordinate_type: str = "relative",
        api_backend: str = "openai",
        enable_thinking: bool = False,
        criteria: str = "",
        logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_mm_agents_importable()
        from mm_agents.qwen3vl_agent_probe_module import Qwen3VLAgent as Qwen3VLProbeAgent

        self.logprobs = logprobs
        self._pending_actions: list[Action] = []

        agent_kwargs: dict[str, Any] = {
            "platform": platform,
            "action_space": "pyautogui",
            "observation_type": "screenshot",
            "history_n": history_n,
            "coordinate_type": coordinate_type,
            "api_backend": api_backend,
            "enable_thinking": enable_thinking,
            "criteria": criteria,
        }
        model_kwargs: dict[str, Any] = {
            "model_name": model,
            "api_key": api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        agent_kwargs["model"] = model_kwargs
        agent_kwargs.update(kwargs)

        if config_path:
            self._agent = Qwen3VLProbeAgent(config_path=config_path, criteria=criteria)
        else:
            self._agent = Qwen3VLProbeAgent(**agent_kwargs)
        self._agent.reset()

    @property
    def criteria(self) -> list[str]:
        return self._agent.criteria

    @property
    def latest_criteria(self) -> str:
        return self._agent.latest_criteria

    @property
    def criteria_images(self) -> list:
        return self._agent.criteria_images

    def reset(self) -> None:
        self._agent.reset()
        self._pending_actions.clear()

    def reset_for_round(self, criteria: str, _logger=None) -> None:
        self._agent.reset_for_round(criteria, _logger)
        self._pending_actions.clear()

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        if self._pending_actions:
            return self._pending_actions.pop(0)

        screenshot_bytes = _get_screenshot_bytes(observation)
        if screenshot_bytes is None:
            return Action.terminate(reason="no_screenshot")

        instruction = observation.instruction or ""
        if not instruction and context and context.task:
            instruction = context.task.instruction or ""

        obs_dict = {"screenshot": screenshot_bytes}

        started_at = time.time()
        response_text, pyautogui_codes = self._agent.predict(instruction, obs_dict)
        elapsed = time.time() - started_at

        actions = []
        for code in pyautogui_codes:
            actions.append(_pyautogui_code_to_action(code, raw=response_text))

        if not actions:
            actions.append(Action.noop(raw=response_text))

        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        step_index = context.step_index if context else 0
        tool_call = getattr(self._agent, "last_tool_call", None)
        metadata: dict[str, Any] = {
            "step_index": step_index,
            "agent": "qwen3vl_probe",
            "model_time": elapsed,
            "response_text": response_text,
            "tool_call": tool_call,
            "latest_criteria": self._agent.latest_criteria,
            "criteria_images": self._agent.criteria_images,
        }
        if self.logprobs:
            metadata["last_messages"] = self._agent.last_messages
        first_action.metadata.setdefault("qwen3vl_probe", {}).update(metadata)

        return first_action
