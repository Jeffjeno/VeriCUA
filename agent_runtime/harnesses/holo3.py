from __future__ import annotations

import base64
import json
import logging
import os
import re
import struct
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, ModalityKind, Observation

logger = logging.getLogger(__name__)

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
    x: int | None = Field(default=None, ge=0, le=1000, description="Optional X coordinate in [0, 1000]")
    y: int | None = Field(default=None, ge=0, le=1000, description="Optional Y coordinate in [0, 1000]")

class WaitArgs(BaseModel):

    tool_name: Literal["wait"]
    seconds: float = Field(default=1.0, ge=0.1, le=10.0)

class AnswerArgs(BaseModel):

    tool_name: Literal["answer"]
    content: str = Field(description="The answer content, or DONE when the task is complete")

class Step(BaseModel):
    note: str | None = Field(default=None, description="Task-relevant information from the previous observation")
    thought: str = Field(description="Reasoning about the next step")
    tool_call: ClickArgs | DoubleClickArgs | RightClickArgs | WriteArgs | KeypressArgs | ScrollArgs | WaitArgs | AnswerArgs

HOLO3_SCHEMA = Step.model_json_schema()

SYSTEM_PROMPT_TEMPLATE = """You are Holo3 running inside a desktop GUI agent loop.

You operate an {platform} desktop with internet access.
The current date is {current_date}.
The home directory is "{home_dir}".
The computer password is "{client_password}" when sudo is needed.

Follow the user task using the screenshot in each <observation>. Choose exactly
one tool call per step. Coordinates must be integers in [0, 1000] relative to
the exact screenshot you see, with origin at the top-left.

Do not ask for confirmation. If the task is complete, use answer with content
"DONE". If the task is impossible because of missing apps, permissions,
contradictory requirements, or other hard blockers, use answer with content
"FAIL: <reason>".

Use write only after the target input is focused. Use click first if focus is
uncertain. Prefer keypress for shortcuts and Enter. Scroll when content may be
off-screen.

<output_format>
```json
{schema}
```
</output_format>"""

class Holo3Harness(Harness):

    def __init__(
        self,
        model: str = "holo3",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        max_trajectory_length: int = 12,
        max_images: int = 3,
        client_password: str = "password",
        proxy_url: str | None = None,
        platform: str = "ubuntu",
        request_timeout: float | None = None,
        enable_thinking: bool = True,
        reasoning_effort: str | None = "medium",
        logprobs: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_trajectory_length = max_trajectory_length
        self.max_images = max_images
        self.client_password = client_password
        self.proxy_url = proxy_url
        self.platform = platform
        self.request_timeout = request_timeout or float(
            os.environ.get("HOLO3_REQUEST_TIMEOUT") or os.environ.get("OPENAI_REQUEST_TIMEOUT") or 600
        )
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.logprobs = logprobs
        self._turns: list[dict[str, Any]] = []
        self._pending_actions: list[Action] = []
        self._pending_tool_output: str | None = None
        self._screen_size: tuple[int, int] | None = None
        self._total_tokens = 0
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._turns.clear()
        self._pending_actions.clear()
        self._pending_tool_output = None
        self._screen_size = None
        self._total_tokens = 0
        self._history.clear()

    def get_token_usage(self) -> dict[str, int]:
        return {"total_tokens": self._total_tokens}

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        self._screen_size = _infer_screen_size(observation) or self._screen_size
        if self._pending_actions:
            return self._pop_pending_action(context)

        instruction = observation.instruction or (context.task.instruction if context and context.task else "")
        pending_tool_output = self._pending_tool_output
        self._pending_tool_output = None
        current_user_msgs = self._build_current_user_messages(
            instruction,
            observation,
            pending_tool_output=pending_tool_output,
        )
        messages = self._build_messages(current_user_msgs)

        step_index = context.step_index if context else len(self._history)
        logger.info(
            "Holo3 step %d: calling %s at %s/chat/completions (turns=%d window=%d images=%d)",
            step_index,
            self.model,
            self.base_url,
            len(self._turns),
            len(self._turn_window()),
            _count_images(messages),
        )
        started_at = time.time()
        response = self._create_chat_completion(messages)
        elapsed = time.time() - started_at

        usage = _get_field(response, "usage", {}) or {}
        self._total_tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

        content = _choice_content(response)
        assistant_msg = {"role": "assistant", "content": content}
        self._turns.append({
            "user_msgs": current_user_msgs,
            "assistant_msg": assistant_msg,
        })
        if self.max_trajectory_length > 0 and len(self._turns) > self.max_trajectory_length:
            self._turns = self._pinned_turn_window()

        step = _parse_step(content)
        actions = self._step_to_actions(step, raw=content)
        first_action = actions[0]
        self._pending_actions.extend(actions[1:])
        if not self._pending_actions and not _is_terminal(first_action):
            self._pending_tool_output = _tool_output(step)

        metadata = {
            "step_index": step_index,
            "agent": "holo3",
            "model": self.model,
            "usage": usage,
            "model_time": elapsed,
            "response_text": content,
            "thought": step.thought,
            "note": step.note,
            "tool_call": step.tool_call.model_dump(),
            "queued_followup_count": len(actions) - 1,
            "screen_size": self._screen_size,
            "context_turns": len(self._turns),
            "context_images": _count_images(messages),
        }
        if self.logprobs:
            logprobs_data = _extract_logprobs(response)
            if logprobs_data is not None:
                metadata["logprobs_data"] = logprobs_data
            metadata["last_messages"] = messages
        first_action.metadata.setdefault("holo3", {}).update(metadata)
        self._history.append({
            "step_index": step_index,
            "thought": step.thought,
            "tool_call": step.tool_call.model_dump(),
            "action_type": first_action.type.value,
            "action_arguments": first_action.arguments,
        })
        if len(self._history) > self.max_trajectory_length:
            self._history = self._history[-self.max_trajectory_length:]

        return first_action

    def _build_current_user_messages(
        self,
        instruction: str,
        observation: Observation,
        *,
        pending_tool_output: str | None,
    ) -> list[dict[str, Any]]:
        user_msgs: list[dict[str, Any]] = []
        if pending_tool_output:
            user_msgs.append({"role": "user", "content": pending_tool_output})
        user_msgs.append(_observation_message(
            instruction,
            observation,
            is_first_observation=len(self._turns) == 0,
        ))
        return user_msgs

    def _build_messages(self, current_user_msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        for turn in self._turn_window():
            messages.extend(turn["user_msgs"])
            messages.append(turn["assistant_msg"])
        messages.extend(current_user_msgs)
        self._trim_images(messages)
        return messages

    def _turn_window(self) -> list[dict[str, Any]]:
        if self.max_trajectory_length <= 0:
            return list(self._turns)
        if len(self._turns) <= self.max_trajectory_length:
            return list(self._turns)
        return self._pinned_turn_window()

    def _pinned_turn_window(self) -> list[dict[str, Any]]:
        if len(self._turns) <= self.max_trajectory_length:
            return list(self._turns)
        pinned = self._turns[:1]
        recent_count = max(0, self.max_trajectory_length - len(pinned))
        recent = self._turns[-recent_count:] if recent_count else []
        return [*pinned, *recent]

    def _create_chat_completion(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is required for Holo3 chat completion requests.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.request_timeout,
        }
        try:
            import httpx
            client_kwargs["http_client"] = httpx.Client(timeout=self.request_timeout, trust_env=False)
        except Exception:
            pass

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,

            "response_format": {"type": "json_schema", "json_schema": {
                "name": "holo3_step", "schema": HOLO3_SCHEMA,
            }},
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }
        if self.reasoning_effort:
            payload["extra_body"]["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.logprobs:
            payload["logprobs"] = True

        client = OpenAI(**client_kwargs)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return _model_dump(client.chat.completions.create(**payload))
            except Exception as exc:
                logger.error("Holo3 API error (attempt %d/5): %s", attempt + 1, exc)
                last_error = exc
                time.sleep(min(5, (attempt + 1) * 2))
        raise RuntimeError(f"Holo3 API failed too many times: {last_error}") from last_error

    def _proxy_url(self) -> str | None:
        return (
            self.proxy_url
            or os.environ.get("AGENT_RUNTIME_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
        )

    def _step_to_actions(self, step: Step, *, raw: str) -> list[Action]:
        tool = step.tool_call
        tool_name = tool.tool_name

        if tool_name == "answer":
            content = tool.content.strip()
            reason = "FAIL" if content.upper().startswith("FAIL") else "DONE"
            return [Action.terminate(reason=reason, raw=raw, metadata={"answer": content})]

        if tool_name == "click":
            x, y = self._to_abs_xy(tool.x, tool.y)
            return [Action(type=ActionType.CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if tool_name == "double_click":
            x, y = self._to_abs_xy(tool.x, tool.y)
            return [Action(type=ActionType.DOUBLE_CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if tool_name == "right_click":
            x, y = self._to_abs_xy(tool.x, tool.y)
            return [Action(type=ActionType.RIGHT_CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if tool_name == "write":
            actions = [Action.type_text(tool.content, raw=raw)]
            if tool.press_enter:
                actions.append(Action(type=ActionType.KEY_PRESS, arguments={"key": "enter"}, raw=raw))
            return actions

        if tool_name == "keypress":
            keys = [_map_key(key) for key in tool.keys]
            if len(keys) == 1:
                return [Action(type=ActionType.KEY_PRESS, arguments={"key": keys[0]}, raw=raw)]
            return [Action(type=ActionType.HOTKEY, arguments={"keys": keys}, raw=raw)]

        if tool_name == "scroll":
            actions: list[Action] = []
            if tool.x is not None and tool.y is not None:
                x, y = self._to_abs_xy(tool.x, tool.y)
                actions.append(Action(type=ActionType.HOVER, arguments={"x": x, "y": y}, raw=raw))
            amount = max(1, int(tool.amount))
            dx, dy = _scroll_delta(tool.direction, amount)
            actions.append(Action(type=ActionType.SCROLL, arguments={"dx": dx, "dy": dy}, raw=raw))
            return actions

        if tool_name == "wait":
            return [Action.noop(raw=raw, metadata={"seconds": tool.seconds})]

        return [Action.noop(raw=raw, metadata={"unsupported_holo3_tool": tool.model_dump()})]

    def _pop_pending_action(self, context: HarnessContext | None) -> Action:
        action = self._pending_actions.pop(0)
        if not self._pending_actions and not _is_terminal(action):
            self._pending_tool_output = "<tool_output>\nok\n</tool_output>"
        action.metadata.setdefault("holo3", {}).update({
            "step_index": context.step_index if context else len(self._history),
            "agent": "holo3",
            "queued_action": True,
            "queued_remaining": len(self._pending_actions),
        })
        return action

    def _to_abs_xy(self, x_value: Any, y_value: Any) -> tuple[int, int]:
        width, height = self._screen_size or (1920, 1080)
        return _norm1000_to_abs(x_value, width), _norm1000_to_abs(y_value, height)

    def _system_prompt(self) -> str:
        home_dir = "C:\\Users\\user" if self.platform.lower().startswith("win") else "/home/user"
        return SYSTEM_PROMPT_TEMPLATE.format(
            client_password=self.client_password,
            current_date=datetime.now().strftime("%A, %B %d, %Y"),
            home_dir=home_dir,
            platform=self.platform,
            schema=json.dumps(HOLO3_SCHEMA, ensure_ascii=False),
        )

    def _trim_images(self, messages: list[dict[str, Any]]) -> None:
        image_parts: list[dict[str, Any]] = []
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

def _observation_message(instruction: str, observation: Observation, *, is_first_observation: bool) -> dict[str, Any]:
    screenshot_b64 = _screenshot_b64(observation)
    task_text = f"<task>\n{instruction}\n</task>\n" if is_first_observation and instruction else ""
    if not screenshot_b64:
        return {
            "role": "user",
            "content": f"{task_text}<observation>\nNo screenshot available.\n</observation>",
        }
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{task_text}<observation>\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            {"type": "text", "text": "\n</observation>"},
        ],
    }

def _count_images(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url" and part.get("image_url"):
                count += 1
    return count

def _parse_step(content: str) -> Step:
    try:
        return Step.model_validate_json(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return Step.model_validate_json(match.group(0))

def _choice_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Holo3 response has no choices: {_redact_images(response)}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    for field in ("reasoning", "reasoning_content"):
        fallback = message.get(field)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    raise RuntimeError(f"Holo3 response content is empty: {_redact_images(response)}")

def _tool_output(step: Step) -> str:
    return f'<tool_output tool="{step.tool_call.tool_name}">\nok\n</tool_output>'

def _scroll_delta(direction: str, amount: int) -> tuple[int, int]:
    if direction == "up":
        return 0, amount
    if direction == "down":
        return 0, -amount
    if direction == "left":
        return -amount, 0
    if direction == "right":
        return amount, 0
    return 0, 0

def _is_terminal(action: Action) -> bool:
    return action.type in (ActionType.TERMINATE, ActionType.SUBMIT)

def _norm1000_to_abs(value: Any, size: int) -> int:
    number = float(value if value is not None else 0)
    return _clamp_int(round(number / 1000.0 * size), size)

def _clamp_int(value: int, size: int) -> int:
    return max(0, min(size - 1, value))

def _screenshot_b64(observation: Observation) -> str | None:
    for part in observation.parts:
        if part.kind == ModalityKind.IMAGE and isinstance(part.content, str) and part.content:
            return part.content
    return None

def _infer_screen_size(observation: Observation) -> tuple[int, int] | None:
    screenshot_b64 = _screenshot_b64(observation)
    if not screenshot_b64:
        return None
    return _png_size_from_base64(screenshot_b64)

def _png_size_from_base64(content: str) -> tuple[int, int] | None:
    try:
        data = base64.b64decode(content[:256])
    except Exception:
        return None
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

def _get_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)

def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _model_dump(item) for key, item in value.items()}
    return value

def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in ("url", "image_url") and isinstance(item, str) and item.startswith("data:image/"):
                redacted[key] = f"<redacted image length={len(item)}>"
            else:
                redacted[key] = _redact_images(item)
        return redacted
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    return value

def _extract_logprobs(response: dict[str, Any]) -> dict[str, Any] | None:

    choices = response.get("choices")
    if not choices:
        return None
    logprobs_obj = choices[0].get("logprobs")
    if not logprobs_obj:
        return None
    content_list = logprobs_obj.get("content")
    if not content_list:
        return None

    tokens: list[dict[str, Any]] = []
    total = 0.0
    minimum = float("inf")
    for token_info in content_list:
        lp = token_info.get("logprob")
        if lp is None:
            continue
        tokens.append({"token": token_info.get("token", ""), "logprob": lp})
        total += lp
        minimum = min(minimum, lp)

    n = len(tokens)
    if n == 0:
        return None
    return {
        "summary": {
            "sequence_logprob": total,
            "mean_token_logprob": total / n,
            "min_token_logprob": minimum,
            "num_generated_tokens": n,
        },
        "tokens": tokens,
    }

__all__ = ["Holo3Harness"]
