from __future__ import annotations

import ast
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
    _pyautogui_code_to_action,
)

logger = logging.getLogger(__name__)

_PRESS_RE = re.compile(r"pyautogui\.press\((.+)\)$")

_SCROLL_RE = re.compile(r"pyautogui\.scroll\((-?[\d.]+)\)")

def _evocua_code_to_actions(code: str, raw: str | None = None) -> list[Action]:

    code = code.strip()
    if not code:
        return []

    lines = [ln.strip() for ln in code.split("\n") if ln.strip()]

    if len(lines) > 1:
        return [Action(type=ActionType.PYTHON, arguments={"code": "\n".join(lines)}, raw=raw)]

    line = lines[0]

    m = _PRESS_RE.match(line)
    if m:
        key_expr = m.group(1).strip()
        try:
            key = ast.literal_eval(key_expr)
        except Exception:
            key = key_expr.strip("'\"")
        return [Action(type=ActionType.KEY_PRESS, arguments={"keys": [str(key)]}, raw=raw)]

    m = _SCROLL_RE.match(line)
    if m:
        amount = float(m.group(1))
        return [Action(type=ActionType.SCROLL, arguments={"amount": amount}, raw=raw)]

    return [_pyautogui_code_to_action(line, raw=raw)]

class EvoCUAHarness(Harness):

    def __init__(
        self,
        model: str = "EvoCUA-S2",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_steps: int = 50,
        prompt_style: str = "S2",
        max_history_turns: int = 4,
        screen_size: tuple[int, int] = (1920, 1080),
        coordinate_type: str = "relative",
        password: str = "osworld-public-evaluation",
        resize_factor: int = 32,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_mm_agents_importable()
        from mm_agents.evocua.evocua_agent import EvoCUAAgent

        self.logprobs = logprobs
        self._pending_actions: list[Action] = []

        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)

        self._agent = EvoCUAAgent(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_steps=max_steps,
            prompt_style=prompt_style,
            max_history_turns=max_history_turns,
            screen_size=screen_size,
            coordinate_type=coordinate_type,
            password=password,
            resize_factor=resize_factor,
            **kwargs,
        )
        self._agent.reset()

    def reset(self) -> None:
        self._agent.reset()
        self._pending_actions.clear()

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:

        if self._pending_actions:
            pending = self._pending_actions.pop(0)
            if context:
                pending.metadata.setdefault("evocua", {})["step_index"] = context.step_index
            return pending

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

        actions: list[Action] = []
        for code in pyautogui_codes:
            actions.extend(_evocua_code_to_actions(code, raw=response_text))

        if not actions:
            actions.append(Action.noop(raw=response_text))

        step_index = context.step_index if context else 0

        for i, action in enumerate(actions):
            action.metadata.setdefault("evocua", {}).update({
                "agent": "evocua",
                "response_text": response_text,

                "model_time": elapsed if i == 0 else 0.0,

                "step_index": step_index,
                "pending": i > 0,
            })

        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        return first_action
