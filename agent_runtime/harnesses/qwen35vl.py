from __future__ import annotations

import base64
import logging
import os
import re
import sys
import time
from typing import Any

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, ModalityKind, Observation

logger = logging.getLogger(__name__)

def _ensure_mm_agents_importable() -> None:
    candidate = os.environ.get("PROJECT_ROOT")
    if not candidate:
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

def _get_screenshot_bytes(observation: Observation) -> bytes | None:
    img_part = observation.first(ModalityKind.IMAGE)
    if img_part is None:
        return None
    content = img_part.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        if content.startswith("data:"):
            content = content.split(",", 1)[-1]
        return base64.b64decode(content)
    return None

def _pyautogui_code_to_action(code: str, raw: str | None = None) -> Action:
    code = code.strip()

    if code == "DONE":
        return Action.terminate(reason="DONE", raw=raw)
    if code == "FAIL":
        return Action.terminate(reason="FAIL", raw=raw)
    if code == "WAIT":
        return Action(type=ActionType.CUSTOM, name="wait", arguments={"seconds": 2}, raw=raw)

    match = re.match(r"pyautogui\.click\((\d+),\s*(\d+)\)", code)
    if match:
        return Action(type=ActionType.CLICK, arguments={"x": int(match.group(1)), "y": int(match.group(2))}, raw=raw)

    match = re.match(r"pyautogui\.doubleClick\((\d+),\s*(\d+)\)", code)
    if match:
        return Action(type=ActionType.DOUBLE_CLICK, arguments={"x": int(match.group(1)), "y": int(match.group(2))}, raw=raw)

    match = re.match(r"pyautogui\.rightClick\((\d+),\s*(\d+)\)", code)
    if match:
        return Action(type=ActionType.RIGHT_CLICK, arguments={"x": int(match.group(1)), "y": int(match.group(2))}, raw=raw)

    match = re.match(r"pyautogui\.middleClick\((\d+),\s*(\d+)\)", code)
    if match:
        return Action(type=ActionType.CLICK, arguments={"x": int(match.group(1)), "y": int(match.group(2)), "button": "middle"}, raw=raw)

    match = re.match(r"pyautogui\.moveTo\((\d+),\s*(\d+)\)", code)
    if match:
        return Action(type=ActionType.HOVER, arguments={"x": int(match.group(1)), "y": int(match.group(2))}, raw=raw)

    match = re.match(r"pyautogui\.dragTo\((\d+),\s*(\d+)(?:,\s*duration=([\d.]+))?\)", code)
    if match:
        args: dict[str, Any] = {"x": int(match.group(1)), "y": int(match.group(2))}
        if match.group(3):
            args["duration"] = float(match.group(3))
        return Action(type=ActionType.DRAG, arguments=args, raw=raw)

    match = re.match(r"pyautogui\.typewrite\((.+)\)", code)
    if match:
        try:
            import ast

            tree = ast.parse(f"f({match.group(1)})", mode="eval")
            text = ast.literal_eval(tree.body.args[0])
        except Exception:
            text = match.group(1).strip("'\"")
        return Action(type=ActionType.TYPE, arguments={"text": text}, raw=raw)

    match = re.match(r"pyautogui\.hotkey\((.+)\)", code)
    if match:
        keys = [k.strip().strip("'\"") for k in match.group(1).split(",")]
        return Action(type=ActionType.HOTKEY, arguments={"keys": keys}, raw=raw)

    match = re.match(r"pyautogui\.press\((.+)\)", code)
    if match:
        keys = [k.strip().strip("'\"") for k in match.group(1).split(",")]
        return Action(type=ActionType.KEY_PRESS, arguments={"keys": keys}, raw=raw)

    match = re.match(r"pyautogui\.scroll\((-?\d+)\)", code)
    if match:
        return Action(type=ActionType.SCROLL, arguments={"amount": int(match.group(1))}, raw=raw)

    match = re.match(r"pyautogui\.keyDown\((.+)\)", code)
    if match:
        key = match.group(1).strip("'\"")
        return Action(type=ActionType.KEY_PRESS, arguments={"keys": [key], "action": "down"}, raw=raw)

    match = re.match(r"pyautogui\.keyUp\((.+)\)", code)
    if match:
        key = match.group(1).strip("'\"")
        return Action(type=ActionType.KEY_PRESS, arguments={"keys": [key], "action": "up"}, raw=raw)

    return Action(type=ActionType.CUSTOM, name="pyautogui_raw", arguments={"code": code}, raw=raw)

class Qwen35VLHarness(Harness):

    def __init__(
        self,
        config_path: str | None = None,
        model: str = "qwen35-vl",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 0.9,
        platform: str = "ubuntu",
        history_n: int = 4,
        coordinate_type: str = "absolute",
        image_max: int = 5,
        fold_size: int = 2,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        _ensure_mm_agents_importable()
        from mm_agents.qwen35vl_agent import Qwen35VLAgent

        self.logprobs = logprobs
        self._pending_actions: list[Action] = []

        agent_kwargs: dict[str, Any] = {
            "platform": platform,
            "action_space": "pyautogui",
            "observation_type": "screenshot",
            "history_n": history_n,
            "coordinate_type": coordinate_type,
            "api_backend": "openai",
            "image_max": image_max,
            "fold_size": fold_size,
        }
        model_kwargs: dict[str, Any] = {
            "model_name": model,
            "api_key": api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "base_url": base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if logprobs:
            model_kwargs["logprobs"] = True
        agent_kwargs["model"] = model_kwargs
        agent_kwargs.update(kwargs)

        if config_path:
            self._agent = Qwen35VLAgent(config_path=config_path)
        else:
            self._agent = Qwen35VLAgent(**agent_kwargs)
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
            actions.append(_pyautogui_code_to_action(code, raw=response_text))

        if not actions:
            actions.append(Action.noop(raw=response_text))

        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        step_index = context.step_index if context else 0
        metadata: dict[str, Any] = {
            "step_index": step_index,
            "agent": "qwen35vl",
            "model_time": elapsed,
            "response_text": response_text,
        }
        if self.logprobs:
            metadata["last_messages"] = self._agent.last_messages
            last_response = getattr(self._agent, "last_response", None)
            if last_response is not None:
                from rollout_runtime.logprobs import extract_logprobs_data

                logprobs_data = extract_logprobs_data(last_response.model_dump())
                if logprobs_data is not None:
                    metadata["logprobs_data"] = logprobs_data
        first_action.metadata.setdefault("qwen35vl", {}).update(metadata)

        return first_action
