from __future__ import annotations

import base64
import json
import logging
import os
import re
import struct
import time
import uuid
from datetime import datetime
from typing import Any

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, ModalityKind, Observation

logger = logging.getLogger(__name__)

HOLO31_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Single left-click at (x, y) normalized coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "Detailed description of the target UI element"},
                    "x": {"type": "integer", "description": "X coordinate as integer in [0, 1000]"},
                    "y": {"type": "integer", "description": "Y coordinate as integer in [0, 1000]"},
                },
                "required": ["element", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double left-click at (x, y) normalized coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "Detailed description of the target UI element"},
                    "x": {"type": "integer", "description": "X coordinate as integer in [0, 1000]"},
                    "y": {"type": "integer", "description": "Y coordinate as integer in [0, 1000]"},
                },
                "required": ["element", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right-click at (x, y) normalized coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "Detailed description of the target UI element"},
                    "x": {"type": "integer", "description": "X coordinate as integer in [0, 1000]"},
                    "y": {"type": "integer", "description": "Y coordinate as integer in [0, 1000]"},
                },
                "required": ["element", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Type text into the currently focused element. Click the target first if focus is uncertain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text to type"},
                    "press_enter": {"type": "boolean", "description": "Whether to press Enter after typing"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keypress",
            "description": "Press one key or a key combination (e.g. ctrl+c, enter, alt+f4).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Keys to press, e.g. ["ctrl", "l"] or ["enter"]',
                    },
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll at the current pointer location or over a given point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "amount": {"type": "integer", "description": "Scroll amount in wheel clicks (1-20, default 3)"},
                    "x": {"type": "integer", "description": "Optional X coordinate in [0, 1000]"},
                    "y": {"type": "integer", "description": "Optional Y coordinate in [0, 1000]"},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait before the next observation (e.g. for loading).",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "Seconds to wait (0.1 - 10.0, default 1.0)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Provide the final answer or signal task completion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The answer content, or DONE when the task is complete"},
                },
                "required": ["content"],
            },
        },
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are Holo3.1 running inside a desktop GUI agent loop.

You operate an {platform} desktop with internet access.
The current date is {current_date}.
The home directory is "{home_dir}".
The computer password is "{client_password}" when sudo is needed.

Follow the user task using the screenshot in each <observation>. Choose exactly
one tool call per step. Coordinates must be integers in [0, 1000] relative to
the exact screenshot you see, with origin at the top-left.

Do not ask for confirmation. If the task is complete, call the answer tool with
content "DONE". If the task is impossible because of missing apps, permissions,
contradictory requirements, or other hard blockers, call the answer tool with
content "FAIL: <reason>".

Use write only after the target input is focused. Use click first if focus is
uncertain. Prefer keypress for shortcuts and Enter. Scroll when content may be
off-screen."""

class Holo31Harness(Harness):

    def __init__(
        self,
        model: str = "holo3.1",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.8,
        max_trajectory_length: int = 20,
        max_images: int = 3,
        client_password: str = "password",
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
        self.platform = platform
        self.request_timeout = request_timeout or float(
            os.environ.get("HOLO31_REQUEST_TIMEOUT") or os.environ.get("OPENAI_REQUEST_TIMEOUT") or 600
        )
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.logprobs = logprobs

        self._turns: list[dict[str, Any]] = []
        self._pending_actions: list[Action] = []
        self._screen_size: tuple[int, int] | None = None
        self._total_tokens = 0
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._turns.clear()
        self._pending_actions.clear()
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
        messages = self._build_messages(instruction, observation)

        step_index = context.step_index if context else len(self._history)
        logger.info(
            "Holo3.1 step %d: calling %s at %s/chat/completions (turns=%d images=%d)",
            step_index,
            self.model,
            self.base_url,
            len(self._turns),
            _count_images(messages),
        )

        started_at = time.time()
        response = self._create_chat_completion(messages)
        elapsed = time.time() - started_at

        usage = _get_field(response, "usage", {}) or {}
        self._total_tokens += int(
            usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )

        msg = (response.get("choices") or [{}])[0].get("message") or {}
        func_name, func_args, func_args_raw, call_id, tool_call_source = _extract_tool_call(msg)
        if tool_call_source != "tool_calls":
            logger.warning(
                "Holo3.1 tool call recovered from %s (vLLM returned empty tool_calls)",
                tool_call_source,
            )

        assistant_content = msg.get("content") or ""
        assistant_msg = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": func_name, "arguments": func_args_raw},
                }
            ],
        }

        actions = self._dispatch_tool_call(func_name, func_args, raw=func_args_raw)
        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        tool_result_content = "ok"
        if _is_terminal(first_action):
            tool_result_content = "task completed"

        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_result_content,
        }

        self._turns.append({
            "observation_msg": self._make_observation_message(instruction, observation),
            "assistant_msg": assistant_msg,
            "tool_msg": tool_msg,
        })
        if self.max_trajectory_length > 0 and len(self._turns) > self.max_trajectory_length:
            self._turns = self._pinned_turn_window()

        metadata = {
            "step_index": step_index,
            "agent": "holo31",
            "model": self.model,
            "usage": usage,
            "model_time": elapsed,
            "tool_call_name": func_name,
            "tool_call_args": func_args,
            "tool_call_id": call_id,
            "tool_call_source": tool_call_source,
            "assistant_content": assistant_content,
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
        first_action.metadata.setdefault("holo31", {}).update(metadata)

        self._history.append({
            "step_index": step_index,
            "tool_call_name": func_name,
            "tool_call_args": func_args,
            "action_type": first_action.type.value,
            "action_arguments": first_action.arguments,
        })
        if len(self._history) > self.max_trajectory_length:
            self._history = self._history[-self.max_trajectory_length:]

        return first_action

    def _build_messages(self, instruction: str, observation: Observation) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]

        for turn in self._turn_window():
            messages.append(turn["observation_msg"])
            messages.append(turn["assistant_msg"])
            messages.append(turn["tool_msg"])

        messages.append(self._make_observation_message(
            instruction, observation, is_first=len(self._turns) == 0,
        ))

        self._trim_images(messages)
        return messages

    def _make_observation_message(
        self,
        instruction: str,
        observation: Observation,
        *,
        is_first: bool = False,
    ) -> dict[str, Any]:
        screenshot_b64 = _screenshot_b64(observation)
        task_text = f"<task>\n{instruction}\n</task>\n" if is_first and instruction else ""

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
        except ImportError as exc:
            raise RuntimeError("openai package is required for Holo3.1 harness.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.request_timeout,
        }
        try:
            import httpx
            client_kwargs["http_client"] = httpx.Client(timeout=self.request_timeout, trust_env=False)
        except ImportError:
            pass

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": HOLO31_TOOLS,
            "tool_choice": "required",
            "temperature": self.temperature,
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
                logger.error("Holo3.1 API error (attempt %d/5): %s", attempt + 1, exc)
                last_error = exc
                time.sleep(min(5, (attempt + 1) * 2))
        raise RuntimeError(f"Holo3.1 API failed too many times: {last_error}") from last_error

    def _dispatch_tool_call(self, name: str, args: dict[str, Any], *, raw: str) -> list[Action]:
        if name == "answer":
            content = args.get("content", "DONE").strip()
            reason = "FAIL" if content.upper().startswith("FAIL") else "DONE"
            return [Action.terminate(reason=reason, raw=raw, metadata={"answer": content})]

        if name == "click":
            x, y = self._to_abs_xy(args.get("x", 0), args.get("y", 0))
            return [Action(type=ActionType.CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if name == "double_click":
            x, y = self._to_abs_xy(args.get("x", 0), args.get("y", 0))
            return [Action(type=ActionType.DOUBLE_CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if name == "right_click":
            x, y = self._to_abs_xy(args.get("x", 0), args.get("y", 0))
            return [Action(type=ActionType.RIGHT_CLICK, arguments={"x": x, "y": y}, raw=raw)]

        if name == "write":
            content = args.get("content", "")
            press_enter = args.get("press_enter", False)
            actions = [Action.type_text(content, raw=raw)]
            if press_enter:
                actions.append(Action(type=ActionType.KEY_PRESS, arguments={"key": "enter"}, raw=raw))
            return actions

        if name == "keypress":
            keys = [_map_key(k) for k in args.get("keys", [])]
            if len(keys) == 1:
                return [Action(type=ActionType.KEY_PRESS, arguments={"key": keys[0]}, raw=raw)]
            return [Action(type=ActionType.HOTKEY, arguments={"keys": keys}, raw=raw)]

        if name == "scroll":
            actions: list[Action] = []
            if args.get("x") is not None and args.get("y") is not None:
                x, y = self._to_abs_xy(args["x"], args["y"])
                actions.append(Action(type=ActionType.HOVER, arguments={"x": x, "y": y}, raw=raw))
            direction = args.get("direction", "down")
            amount = max(1, min(20, int(args.get("amount", 3))))
            dx, dy = _scroll_delta(direction, amount)
            actions.append(Action(type=ActionType.SCROLL, arguments={"dx": dx, "dy": dy}, raw=raw))
            return actions

        if name == "wait":
            seconds = float(args.get("seconds", 1.0))
            return [Action.noop(raw=raw, metadata={"seconds": seconds})]

        return [Action.noop(raw=raw, metadata={"unsupported_holo31_tool": name, "args": args})]

    def _pop_pending_action(self, context: HarnessContext | None) -> Action:
        action = self._pending_actions.pop(0)
        action.metadata.setdefault("holo31", {}).update({
            "step_index": context.step_index if context else len(self._history),
            "agent": "holo31",
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
            part.update({"type": "text", "text": "[screenshot evicted]"})

def _extract_tool_call(msg: dict[str, Any]) -> tuple[str, dict[str, Any], str, str, str]:

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        func = call.get("function") or {}
        func_name = str(func.get("name") or "")
        func_args_raw = func.get("arguments") or "{}"
        if not isinstance(func_args_raw, str):
            func_args_raw = json.dumps(func_args_raw, ensure_ascii=False)
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}")
        try:
            func_args = json.loads(func_args_raw)
        except json.JSONDecodeError:
            func_args = {}
        if func_name:
            return func_name, func_args, func_args_raw, call_id, "tool_calls"

    for field in ("reasoning", "reasoning_content"):
        raw = msg.get(field)
        if not isinstance(raw, str) or not raw.strip():
            continue
        recovered = _parse_tool_call_from_reasoning(raw)
        if recovered is None:
            continue
        func_name, func_args = recovered
        func_args_raw = json.dumps(func_args, ensure_ascii=False)
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        return func_name, func_args, func_args_raw, call_id, field

    raise RuntimeError(f"Holo3.1 response has no tool_calls. message={_redact_images(msg)}")

def _parse_tool_call_from_reasoning(raw: str) -> tuple[str, dict[str, Any]] | None:

    text = raw.strip()
    candidates: list[Any] = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", text, flags=re.S)
        if match:
            try:
                candidates.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass

    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            candidate = candidate[0]
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name") or candidate.get("tool_name") or candidate.get("function")
        if not name:
            continue
        args = candidate.get("arguments")
        if args is None:
            args = candidate.get("parameters")
        if args is None:
            args = candidate.get("args")
        if not isinstance(args, dict):
            args = {}
        return str(name), args
    return None

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

def _norm1000_to_abs(value: Any, size: int) -> int:
    number = float(value if value is not None else 0)
    return _clamp_int(round(number / 1000.0 * size), size)

def _clamp_int(value: int, size: int) -> int:
    return max(0, min(size - 1, value))

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

__all__ = ["Holo31Harness"]
