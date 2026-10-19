from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Any

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, Observation

from agent_runtime.harnesses.qwen35vl import (
    _ensure_mm_agents_importable,
    _get_screenshot_bytes,
)

logger = logging.getLogger(__name__)

def _owl15_code_to_action(code: str, raw: str | None = None) -> Action:

    code = code.strip()

    if code == "DONE":
        return Action.terminate(reason="DONE", raw=raw)
    if code == "FAIL":
        return Action.terminate(reason="FAIL", raw=raw)

    return Action(type=ActionType.CUSTOM, name="pyautogui_script", arguments={"code": code}, raw=raw)

class Owl15Harness(Harness):

    def __init__(
        self,
        config_path: str | None = None,
        model: str = "",
        api_url: str = "",
        api_key: str = "",
        platform: str = "ubuntu",
        max_tokens: int = 1000,
        history_n: int = 1,
        top_p: float = 0.9,
        top_k: int = -1,
        temperature: float = 0.0,
        max_trajectory_length: int = 50,
        engine: str = "openai",
        add_quota: int = 0,
        guide_path: str = "",
        use_history: int = 1,
        keep_full_response: int = 0,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_mm_agents_importable()
        from mm_agents.owl15_agent import Owl15Agent

        self.logprobs = logprobs
        self._pending_actions: list[Action] = []

        owl_kwargs: dict[str, Any] = {
            "model": model,
            "api_url": api_url or os.environ.get("OPENAI_BASE_URL", ""),
            "api_key": api_key or os.environ.get("OPENAI_API_KEY", ""),
            "platform": platform,
            "max_tokens": max_tokens,
            "history_n": history_n,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "max_trajectory_length": max_trajectory_length,
            "engine": engine,
            "add_quota": add_quota,
            "guide_path": guide_path,
            "use_history": use_history,
            "keep_full_response": keep_full_response,
        }
        owl_kwargs.update(kwargs)

        if config_path:
            self._agent = Owl15Agent(config_path=config_path)
        else:
            self._agent = Owl15Agent(**owl_kwargs)
        self._agent.reset()

    def reset(self) -> None:
        self._agent.reset()
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
            actions.append(_owl15_code_to_action(code, raw=response_text))

        if not actions:
            actions.append(Action.noop(raw=response_text))

        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        step_index = context.step_index if context else 0
        metadata: dict[str, Any] = {
            "step_index": step_index,
            "agent": "owl15",
            "model_time": elapsed,
            "response_text": response_text,
        }
        first_action.metadata.setdefault("owl15", {}).update(metadata)

        return first_action
