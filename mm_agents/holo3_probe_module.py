"""Holo3 probe module for GUIRL evaluation.

Adapts the Holo3 structured-output harness into an Agent subclass
with criteria-based evidence probing. Uses structured outputs (not native
function calling) so the model is forced to produce note/thought/tool_call
chain-of-thought before each action — preserving the decision quality
demonstrated by the holo3 harness.
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import os
import re
import struct
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover - hardened for incomplete osworld envs
    def retry(*_a, **_k):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

    def stop_after_attempt(*_a, **_k):  # type: ignore
        return None

    def wait_exponential(*_a, **_k):  # type: ignore
        return None

from eval_runtime.markers import step_or_handle_marker
from mm_agents.base_agent import Agent
from mm_agents.config import Holo31AgentConfig as Holo3AgentConfig
from mm_agents.vlm import VLMWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema (same as holo3 harness + criteria tool)
# ---------------------------------------------------------------------------


class ClickArgs(BaseModel):
    tool_name: Literal["click"]
    element: str = Field(description="Detailed description of the target UI element")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


class DoubleClickArgs(BaseModel):
    tool_name: Literal["double_click"]
    element: str = Field(description="Detailed description of the target UI element")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


class RightClickArgs(BaseModel):
    tool_name: Literal["right_click"]
    element: str = Field(description="Detailed description of the target UI element")
    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


class WriteArgs(BaseModel):
    tool_name: Literal["write"]
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")


class KeypressArgs(BaseModel):
    tool_name: Literal["keypress"]
    keys: list[str] = Field(description='Keys to press, for example ["ctrl", "l"] or ["enter"]')


class ScrollArgs(BaseModel):
    tool_name: Literal["scroll"]
    direction: Literal["up", "down", "left", "right"]
    amount: int = Field(default=3, ge=1, le=20, description="Scroll amount in wheel clicks")
    x: int | None = Field(default=None, ge=0, le=1000)
    y: int | None = Field(default=None, ge=0, le=1000)


class WaitArgs(BaseModel):
    tool_name: Literal["wait"]
    seconds: float = Field(default=1.0, ge=0.1, le=10.0)


class AnswerArgs(BaseModel):
    tool_name: Literal["answer"]
    content: str = Field(description="The answer content, or DONE when the task is complete")


class CriteriaArgs(BaseModel):
    tool_name: Literal["criteria"]
    option: int = Field(description="The criterion number to update, e.g. 1, 2, 3, 4, or 5.")
    state: Literal["success", "failure", "uncertain"] = Field(
        description="The corrected state for this criterion based on current evidence."
    )


class ProbeStep(BaseModel):
    note: str | None = Field(default=None, description="Task-relevant information from the current observation")
    thought: str = Field(description="Reasoning about what to do next")
    tool_call: (
        ClickArgs | DoubleClickArgs | RightClickArgs | WriteArgs
        | KeypressArgs | ScrollArgs | WaitArgs | AnswerArgs | CriteriaArgs
    )


HOLO3_PROBE_SCHEMA = ProbeStep.model_json_schema()


# ---------------------------------------------------------------------------
# System prompt for probe role
# ---------------------------------------------------------------------------

PROBE_SYSTEM_PROMPT_TEMPLATE = """\
You are a GUI evidence probe running inside a desktop GUI agent loop.

A separate GUI agent Jeffery has already attempted the user's task in this environment. \
Your task is not to redo the whole task; your task is to inspect the current GUI state, \
collect just enough evidence, and update the judge criteria using the `criteria` tool.

You operate an {platform} desktop with internet access.
The current date is {current_date}.
The home directory is "{home_dir}".
The computer password is "{client_password}" when sudo is needed.

At each step, choose exactly one tool call from two modes:
1. Exploration: use action tools (click, write, keypress, scroll, wait) to navigate and reveal evidence.
2. Criteria update: use the `criteria` tool to formally update a criterion's state.

IMPORTANT RULES about the `criteria` tool:
- You MUST call the `criteria` tool to update each criterion when you observe evidence. \
Reasoning about evidence in your thought/note is NOT sufficient — the judge only sees \
criteria updates made through the `criteria` tool call.
- Update criteria one at a time. Each `criteria` call updates exactly one criterion number.
- Only call `answer` with "DONE" AFTER you have called `criteria` for every uncertain \
criterion that you have evidence for. If you have not called `criteria` at all, do NOT \
call `answer`.

Follow the screenshot in each <observation>. Choose exactly one tool call per step.
Coordinates must be integers in [0, 1000] relative to the exact screenshot you see, \
with origin at the top-left.

Use write only after the target input is focused. Use click first if focus is \
uncertain. Prefer keypress for shortcuts and Enter. Scroll when content may be \
off-screen.

Do not perform actions that modify the environment (e.g. saving files, deleting, editing). \
Your role is to observe and verify, not to change the state.

<output_format>
```json
{schema}
```
</output_format>"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


@Agent.register("Holo3_Probe")
class Holo3ProbeAgent(Agent):
    """Holo3 structured-output probe agent for GUIRL criteria-based evaluation."""

    def __init__(self, config_path: str = None, **kwargs):
        initial_criteria = kwargs.pop("criteria", "") or ""
        self.criteria = [initial_criteria] if initial_criteria else []
        self.criteria_images: List[Dict[str, Any]] = []
        super().__init__(config_path=config_path, **kwargs)

        if config_path is not None:
            self.config = Holo3AgentConfig.from_yaml(config_path, **kwargs)
        else:
            self.config = Holo3AgentConfig(**kwargs)

        self.platform = self.config.platform
        if self.config.model:
            model_params = self.config.model.model_dump()
            model_params["max_pixels"] = None
            self.model = VLMWrapper(**model_params)
        else:
            self.model = None
        self.client_password = self.config.client_password
        self.enable_thinking = self.config.enable_thinking
        self.reasoning_effort = self.config.reasoning_effort
        self.max_trajectory_length = self.config.max_trajectory_length
        self.max_images = self.config.max_images
        self.action_space = self.config.action_space
        self.observation_type = self.config.observation_type

        assert self.action_space in ["pyautogui"], "Invalid action space"
        assert self.observation_type in ["screenshot"], "Invalid observation type"

        self._turns: List[Dict[str, Any]] = []
        self._pending_tool_output: Optional[str] = None
        self._screen_size: Optional[Tuple[int, int]] = None

        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Any] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.criteria_images: List[Dict[str, Any]] = []
        self.last_messages: Optional[List[Dict]] = None
        self.last_message_step: int = -1
        self.last_tool_call: Optional[Dict[str, Any]] = None
        self.infer_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Criteria management
    # ------------------------------------------------------------------

    def _update_criterion_state(self, option: int, state: str) -> bool:
        current_criteria = self.latest_criteria
        pattern = re.compile(
            rf"(^\s*{option}\.\s*.*?:\s*)(success|failure|uncertain)(\s*$)",
            re.IGNORECASE | re.MULTILINE,
        )
        updated, count = pattern.subn(rf"\g<1>{state}\g<3>", current_criteria, count=1)
        if count != 1:
            return False
        self.criteria.append(updated)
        return True

    @property
    def latest_criteria(self) -> str:
        return self.criteria[-1] if self.criteria else ""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        instruction: str,
        obs: Dict,
        system_prompt_prefix: Optional[str] = None,
        time_list=None,
    ) -> List:
        if time_list is not None:
            time_list.append(("inference_start", time.time()))

        screenshot_bytes = obs["screenshot"]
        raw_screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        screen_size = _png_size_from_bytes(screenshot_bytes)
        if screen_size:
            width, height = screen_size
            self._screen_size = screen_size
        else:
            width, height = self._screen_size or (1920, 1080)

        self.screenshots.append(raw_screenshot_b64)
        current_step = len(self.actions)

        previous_actions = []
        for i in range(current_step):
            if i < len(self.actions):
                previous_actions.append(f"Step {i + 1}: {self.actions[i]}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        current_user_msgs = self._build_current_user_messages(
            raw_screenshot_b64, previous_actions_str,
        )

        messages = self._build_messages(instruction, current_user_msgs, system_prompt_prefix)

        if time_list is not None:
            time_list.append(("prompt_built", time.time()))

        self.last_messages = messages
        self.last_message_step = current_step

        if time_list is not None:
            time_list.append(("messages_saved", time.time()))

        content, low_level_instruction, pyautogui_code = self._infer(
            messages, raw_screenshot_b64, width, height,
        )

        if time_list is not None:
            time_list.append(("agent_inference", time.time()))

        assistant_msg = {"role": "assistant", "content": content}
        self._turns.append({
            "user_msgs": current_user_msgs,
            "assistant_msg": assistant_msg,
        })
        if self.max_trajectory_length > 0 and len(self._turns) > self.max_trajectory_length:
            self._turns = self._pinned_turn_window()

        self.responses.append(content)

        logger.info("Low level instruction: %s", low_level_instruction)
        logger.info("Pyautogui code: %s", pyautogui_code)

        self.actions.append(low_level_instruction)

        if time_list is not None:
            time_list.append(("response_parsed", time.time()))

        return content, pyautogui_code

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_current_user_messages(
        self,
        screenshot_b64: str,
        previous_actions_str: str = "None",
    ) -> List[Dict[str, Any]]:
        user_msgs: List[Dict[str, Any]] = []
        if self._pending_tool_output:
            user_msgs.append({"role": "user", "content": self._pending_tool_output})
            self._pending_tool_output = None

        criteria_text = self.latest_criteria.strip() if self.latest_criteria else "No criteria provided."
        context_text = (
            f"Generate the next step: either explore with an action tool, "
            f"or update exactly one criterion with the `criteria` tool when "
            f"the current screenshot provides evidence.\n\n"
            f"Current Criteria:\n{criteria_text}\n\n"
            f"Previous actions:\n{previous_actions_str}\n"
        )

        if not screenshot_b64:
            user_msgs.append({
                "role": "user",
                "content": f"{context_text}\n<observation>\nNo screenshot available.\n</observation>",
            })
        else:
            user_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{context_text}\n<observation>\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": "\n</observation>"},
                ],
            })
        return user_msgs

    def _build_messages(
        self,
        instruction: str,
        current_user_msgs: List[Dict[str, Any]],
        system_prompt_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        system_prompt = self._system_prompt(instruction)
        if system_prompt_prefix:
            system_prompt = f"{system_prompt_prefix.strip()}\n\n{system_prompt}"

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for turn in self._turn_window():
            messages.extend(turn["user_msgs"])
            messages.append(turn["assistant_msg"])

        messages.extend(current_user_msgs)
        self._trim_images(messages)
        return messages

    def _system_prompt(self, instruction: str) -> str:
        home_dir = "C:\\Users\\user" if self.platform.lower().startswith("win") else "/home/user"
        return PROBE_SYSTEM_PROMPT_TEMPLATE.format(
            platform=self.platform,
            current_date=datetime.datetime.now().strftime("%A, %B %d, %Y"),
            home_dir=home_dir,
            client_password=self.client_password,
            schema=json.dumps(HOLO3_PROBE_SCHEMA, ensure_ascii=False),
        )

    def _turn_window(self) -> List[Dict[str, Any]]:
        if self.max_trajectory_length <= 0:
            return list(self._turns)
        if len(self._turns) <= self.max_trajectory_length:
            return list(self._turns)
        return self._pinned_turn_window()

    def _pinned_turn_window(self) -> List[Dict[str, Any]]:
        if len(self._turns) <= self.max_trajectory_length:
            return list(self._turns)
        pinned = self._turns[:1]
        recent_count = max(0, self.max_trajectory_length - len(pinned))
        recent = self._turns[-recent_count:] if recent_count else []
        return [*pinned, *recent]

    def _trim_images(self, messages: List[Dict[str, Any]]) -> None:
        image_parts: List[Dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_parts.append(part)
        excess = len(image_parts) - max(0, self.max_images)
        for part in image_parts[:max(0, excess)]:
            part.clear()
            part.update({"type": "text", "text": "[older screenshot omitted]"})

    # ------------------------------------------------------------------
    # Inference with retry
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _infer(
        self,
        messages: List[Dict[str, Any]],
        raw_screenshot_b64: str,
        width: int,
        height: int,
    ) -> Tuple[str, str, List[str]]:
        predict_kwargs: Dict[str, Any] = {
            "extra_body": {
                "structured_outputs": {"json": HOLO3_PROBE_SCHEMA},
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }
        if self.reasoning_effort:
            predict_kwargs["extra_body"]["reasoning_effort"] = self.reasoning_effort

        response = self.model.predict(messages, **predict_kwargs)
        content = _choice_content(response)

        step = _parse_step(content)
        self.thoughts.append(step.thought)

        tool_call_info = {
            "name": step.tool_call.tool_name,
            "arguments": step.tool_call.model_dump(),
        }
        self.last_tool_call = tool_call_info

        if step.tool_call.tool_name == "criteria":
            low_level_instruction, pyautogui_code = self._handle_criteria(
                step.tool_call, raw_screenshot_b64,
            )
        else:
            low_level_instruction, pyautogui_code = self._tool_call_to_pyautogui(
                step.tool_call, width, height,
            )

        if not pyautogui_code:
            raise ValueError("No valid actions produced from the response, retrying...")

        self._pending_tool_output = f'<tool_output tool="{step.tool_call.tool_name}">\nok\n</tool_output>'

        self.infer_records.append({
            "messages": copy.deepcopy(messages),
            "response": content,
            "tool_call": tool_call_info,
            "thought": step.thought,
            "note": step.note,
            "low_level_instruction": low_level_instruction,
            "actions": list(pyautogui_code),
        })

        return content, low_level_instruction, pyautogui_code

    # ------------------------------------------------------------------
    # Criteria handling
    # ------------------------------------------------------------------

    def _handle_criteria(
        self, tool_call: CriteriaArgs, raw_screenshot_b64: str,
    ) -> Tuple[str, List[str]]:
        """Handle a criteria tool call.

        Always returns a non-empty ``pyautogui_code`` list (["WAIT"]) so that
        ``_infer``'s "no valid actions" guard never fires and triggers a retry
        loop.  On failure the criteria state is left unchanged and an error is
        logged; evaluation continues normally.
        """
        low_level_instruction = "criteria update"
        try:
            option = tool_call.option
            state = tool_call.state
            if not self._update_criterion_state(option, state):
                logger.error(
                    "criteria tool: option %d not found in criteria text; skipping update", option
                )
                low_level_instruction = f"criteria update failed: option {option} not found"
            else:
                entry = {"option": option, "images": raw_screenshot_b64}
                existing_idx = next(
                    (i for i, e in enumerate(self.criteria_images) if e["option"] == option),
                    None,
                )
                if existing_idx is not None:
                    self.criteria_images[existing_idx] = entry
                else:
                    self.criteria_images.append(entry)
                low_level_instruction = f"Update criterion {option} to {state}"
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Failed to parse criteria tool call: %s", e)
            low_level_instruction = f"criteria update failed: {e}"

        return low_level_instruction, ["WAIT"]

    # ------------------------------------------------------------------
    # Tool call → pyautogui code conversion
    # ------------------------------------------------------------------

    def _tool_call_to_pyautogui(
        self,
        tool_call: BaseModel,
        width: int,
        height: int,
    ) -> Tuple[str, List[str]]:
        func_name = tool_call.tool_name
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        if func_name == "click":
            x, y = self._to_abs_xy(tool_call.x, tool_call.y, width, height)
            element = tool_call.element
            low_level_instruction = f"Click on {element}" if element else "Click"
            pyautogui_code.append(f"pyautogui.click({x}, {y})")

        elif func_name == "double_click":
            x, y = self._to_abs_xy(tool_call.x, tool_call.y, width, height)
            element = tool_call.element
            low_level_instruction = f"Double-click on {element}" if element else "Double-click"
            pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")

        elif func_name == "right_click":
            x, y = self._to_abs_xy(tool_call.x, tool_call.y, width, height)
            element = tool_call.element
            low_level_instruction = f"Right-click on {element}" if element else "Right-click"
            pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")

        elif func_name == "write":
            content = tool_call.content
            press_enter = tool_call.press_enter
            low_level_instruction = f"Type text: {content[:50]}"
            lines = content.split("\n")
            for idx, line in enumerate(lines):
                if line:
                    pyautogui_code.append(f"pyautogui.typewrite({repr(line)}, interval=0.03)")
                if idx < len(lines) - 1:
                    pyautogui_code.append("pyautogui.press('enter')")
            if press_enter:
                pyautogui_code.append("pyautogui.press('enter')")

        elif func_name == "keypress":
            keys = [_map_key(k) for k in tool_call.keys]
            low_level_instruction = f"Press keys: {'+'.join(keys)}"
            keys_str = ", ".join([f"'{key}'" for key in keys])
            if len(keys) > 1:
                pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
            elif len(keys) == 1:
                pyautogui_code.append(f"pyautogui.press({keys_str})")

        elif func_name == "scroll":
            direction = tool_call.direction
            amount = max(1, min(20, tool_call.amount))
            low_level_instruction = f"Scroll {direction} by {amount}"

            if tool_call.x is not None and tool_call.y is not None:
                x, y = self._to_abs_xy(tool_call.x, tool_call.y, width, height)
                pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")

            if direction == "up":
                pyautogui_code.append(f"pyautogui.scroll({amount})")
            elif direction == "down":
                pyautogui_code.append(f"pyautogui.scroll({-amount})")
            elif direction == "left":
                pyautogui_code.append(f"pyautogui.hscroll({-amount})")
            elif direction == "right":
                pyautogui_code.append(f"pyautogui.hscroll({amount})")

        elif func_name == "wait":
            low_level_instruction = "Wait"
            pyautogui_code.append("WAIT")

        elif func_name == "answer":
            content = tool_call.content.strip()
            low_level_instruction = f"Answer: {content}"
            pyautogui_code.append("DONE")

        else:
            low_level_instruction = f"Unknown action: {func_name}"
            logger.warning("Unknown tool call: %s", func_name)

        return low_level_instruction, pyautogui_code

    @staticmethod
    def _to_abs_xy(x_value: Any, y_value: Any, width: int, height: int) -> Tuple[int, int]:
        return _norm1000_to_abs(x_value, width), _norm1000_to_abs(y_value, height)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, _logger=None):
        global logger
        logger = (
            _logger if _logger is not None
            else logging.getLogger("desktopenv.holo3_probe_agent")
        )

        self._turns = []
        self._pending_tool_output = None
        self._screen_size = None
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        self.criteria_images = []
        self.last_messages = None
        self.last_message_step = -1
        self.last_tool_call = None
        self.infer_records = []

    def reset_for_round(self, criteria: str, _logger=None):
        """Reset agent history for a new probe round, keeping model config."""
        self.reset(_logger)
        self.criteria = [criteria] if criteria else []

    def build_messages(self):
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _choice_content(response: Any) -> str:
    """Extract content from chat completion response (object or dict)."""
    if hasattr(response, "choices"):
        choices = response.choices
    elif isinstance(response, dict):
        choices = response.get("choices") or []
    else:
        raise RuntimeError(f"Unexpected response type: {type(response)}")

    if not choices:
        raise RuntimeError("Response has no choices")

    if hasattr(choices[0], "message"):
        message = choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        for field in ("reasoning", "reasoning_content"):
            fallback = getattr(message, field, None)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    else:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        for field in ("reasoning", "reasoning_content"):
            fallback = message.get(field)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()

    raise RuntimeError("Response content is empty")


def _parse_step(content: str) -> ProbeStep:
    try:
        return ProbeStep.model_validate_json(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return ProbeStep.model_validate_json(match.group(0))


def _png_size_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    png_sig = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(png_sig) or len(data) < 24:
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
    except struct.error:
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _norm1000_to_abs(value: Any, size: int) -> int:
    number = float(value if value is not None else 0)
    return _clamp_int(round(number / 1000.0 * size), size)


def _clamp_int(value: int, size: int) -> int:
    return max(0, min(size - 1, value))


def _map_key(key: Any) -> str:
    mapping = {
        "alt": "alt",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "backspace": "backspace",
        "capslock": "capslock",
        "cmd": "command",
        "command": "command",
        "ctrl": "ctrl",
        "delete": "delete",
        "end": "end",
        "enter": "enter",
        "esc": "esc",
        "home": "home",
        "insert": "insert",
        "option": "option",
        "pagedown": "pagedown",
        "pageup": "pageup",
        "shift": "shift",
        "space": "space",
        "super": "super",
        "tab": "tab",
        "win": "win",
    }
    return mapping.get(str(key).lower(), str(key).lower())


# ---------------------------------------------------------------------------
# GUIRL probe round runner (matches holo31_probe_module interface)
# ---------------------------------------------------------------------------


def _write_jsonl(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _omit_large_str(value: Any, limit: int = 512) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return f"<omitted chars≈{len(value)}>"
    return value


def _sanitize_criteria_images_for_disk(entries: Any) -> Any:
    """Drop bulky screenshot base64 from probe traj artifacts."""
    if not isinstance(entries, list):
        return entries
    sanitized = []
    for entry in entries:
        if not isinstance(entry, dict):
            sanitized.append(entry)
            continue
        item = dict(entry)
        images = item.get("images")
        if isinstance(images, str):
            item["images"] = _omit_large_str(images)
        elif isinstance(images, list):
            item["images"] = [_omit_large_str(x) for x in images]
        sanitized.append(item)
    return sanitized


def _sanitize_messages_for_disk(messages: Any) -> Any:
    """Drop multimodal data-urls before writing probe message dumps."""
    if not isinstance(messages, list):
        return messages

    def _sanitize_part(part: Any) -> Any:
        if not isinstance(part, dict):
            return part
        out = dict(part)
        if out.get("type") == "image_url":
            image_url = out.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "")
                out["image_url"] = {
                    **image_url,
                    "url": (
                        f"<omitted data-url bytes≈{len(url)}>"
                        if url.startswith("data:")
                        else _omit_large_str(url)
                    ),
                }
            else:
                out["image_url"] = "<omitted>"
        elif out.get("type") == "image":
            for key in ("source", "data", "image"):
                if key in out:
                    out[key] = "<omitted>"
        return out

    sanitized = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        item = dict(msg)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_sanitize_part(p) for p in content]
        elif isinstance(content, str):
            item["content"] = _omit_large_str(content, limit=200_000)
        sanitized.append(item)
    return sanitized


def run_guirl_probe_round_client(
    probe_agent,
    env_client,
    lease_id: str,
    instruction: str,
    max_steps: int,
    example_result_dir: str,
    sleep_after_execution: float,
    sampler_round: int = 1,
) -> Dict:
    """Run one GUIRL probe round as an inner loop before returning to judge."""
    probe_dir = os.path.join(example_result_dir, "replay_eval", "guirl_probe")
    messages_dir = os.path.join(probe_dir, "messages")
    os.makedirs(messages_dir, exist_ok=True)

    rounds_index_path = os.path.join(probe_dir, "rounds.jsonl")
    message_index_path = os.path.join(probe_dir, "message_traj.jsonl")
    traj_path = os.path.join(probe_dir, "traj.jsonl")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
    _write_jsonl(
        rounds_index_path,
        {
            "event": "round_start",
            "round": sampler_round,
            "timestamp": run_timestamp,
            "instruction": instruction,
            "max_steps": max_steps,
        },
    )

    current_obs = env_client.get_obs(lease_id)
    done = False
    step_idx = 0
    steps_run = 0
    stop_reason = "max_steps"
    total_env_steps = 0

    while not done and step_idx < max_steps:
        step_num = step_idx + 1
        prompt_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
        before_evidence_count = len(getattr(probe_agent, "criteria_images", []))
        response, actions = probe_agent.predict(instruction, current_obs)
        tool_call = getattr(probe_agent, "last_tool_call", None)

        sampler_messages = getattr(probe_agent, "last_messages", None)
        if sampler_messages is not None:
            message_file_name = f"messages_round_{sampler_round}_step_{step_num}_{prompt_timestamp}.json"
            message_file_path = os.path.join(messages_dir, message_file_name)
            try:
                with open(message_file_path, "w", encoding="utf-8") as mf:
                    json.dump(_sanitize_messages_for_disk(sampler_messages), mf, ensure_ascii=False)
                _write_jsonl(
                    message_index_path,
                    {
                        "round": sampler_round,
                        "step_num": step_num,
                        "timestamp": prompt_timestamp,
                        "messages_file": os.path.join("messages", message_file_name),
                    },
                )
            except Exception as msg_e:
                logger.warning("Failed to save GUIRL probe messages at step %d: %s", step_num, msg_e)

        criteria_images = getattr(probe_agent, "criteria_images", [])
        step_record = {
            "round": sampler_round,
            "step": step_num,
            "response": response,
            "tool_call": tool_call,
            "actions": actions,
            "criteria_history_after_parse": getattr(probe_agent, "criteria", []),
            "latest_criteria_after_parse": getattr(probe_agent, "latest_criteria", ""),
            "criteria_images_after_parse": _sanitize_criteria_images_for_disk(criteria_images),
            "new_criteria_images": _sanitize_criteria_images_for_disk(
                criteria_images[before_evidence_count:]
            ),
            "env_steps": [],
        }

        for action_idx, action in enumerate(actions, start=1):
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("GUIRL probe round %d step %d: %s", sampler_round, step_num, action)

            was_marker = action.strip().upper() in ("DONE", "FAIL", "WAIT") if isinstance(action, str) else False
            current_obs, reward, done, info = step_or_handle_marker(
                env_client, lease_id, action, sleep_after_execution
            )
            if not was_marker:
                total_env_steps += 1

            screenshot_rel = os.path.join(
                "replay_eval",
                "guirl_probe",
                f"round_{sampler_round:02d}_step_{step_num:02d}_{action_idx}_{action_timestamp}.png",
            )
            screenshot_path = os.path.join(example_result_dir, screenshot_rel)
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            with open(screenshot_path, "wb") as f:
                f.write(current_obs["screenshot"])
            step_record["env_steps"].append(
                {
                    "action_idx": action_idx,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot": screenshot_rel,
                }
            )
            if done:
                stop_reason = "env_done"
                break

        if step_record["env_steps"]:
            step_record["stop_reason"] = stop_reason if done else None
        try:
            _write_jsonl(traj_path, step_record)
        except OSError as traj_e:
            logger.warning("Failed to append GUIRL probe traj at step %d: %s", step_num, traj_e)
        step_idx += 1
        steps_run += 1

    _write_jsonl(
        rounds_index_path,
        {
            "event": "round_end",
            "round": sampler_round,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f"),
            "steps": steps_run,
            "env_steps": total_env_steps,
            "stop_reason": stop_reason,
            "done": done,
        },
    )
    return {
        "obs": current_obs,
        "done": done,
        "steps": steps_run,
        "env_steps": total_env_steps,
        "stop_reason": stop_reason,
    }
