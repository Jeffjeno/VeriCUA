from __future__ import annotations

import base64
import json
import logging
import os
import struct
import time
from datetime import datetime
from typing import Any

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, ModalityKind, Observation

logger = logging.getLogger(__name__)

OPERATOR_PROMPT = """
Here are some helpful tips:
- You are operating an {PLATFORM} desktop with internet access.
- My computer password is "{CLIENT_PASSWORD}" when sudo is needed.
- The current date is {CURRENT_DATE}.
- The home directory is "{HOME_DIR}".
- Stick to the website or application already opened for the task when possible.
- Prefer Chrome over Firefox/Chromium unless the task says otherwise.
- If you launch GUI applications from bash on Ubuntu, use DISPLAY=:0.
- You can act without asking for confirmation.
- If content may be off-screen, scroll or zoom out before deciding it is unavailable.
- Use the computer tool directly and keep each action grounded in the screenshot.
- If the task is infeasible because of missing apps, permissions, contradictory requirements, or other hard blockers, output exactly "[INFEASIBLE]".
"""

class ClaudeCUAHarness(Harness):

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.5,
        max_trajectory_length: int = 8,
        client_password: str = "password",
        proxy_url: str | None = None,
        platform: str = "ubuntu",
        request_timeout: float | None = None,
        beta: str | None = None,
        computer_tool_type: str | None = None,
        display_number: int = 1,
        gateway_schema_compat: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("CLAUDE_BASE_URL")
            or "https://api.anthropic.com"
        ).rstrip("/")
        self.max_tokens = max_tokens or 1800
        self.temperature = temperature
        self.max_trajectory_length = max_trajectory_length
        self.client_password = client_password
        self.proxy_url = proxy_url
        self.platform = platform
        self.request_timeout = request_timeout or float(
            os.environ.get("CLAUDE_REQUEST_TIMEOUT")
            or os.environ.get("ANTHROPIC_REQUEST_TIMEOUT")
            or os.environ.get("REQUEST_TIMEOUT")
            or 600
        )
        version = _computer_use_version(model)
        self.beta = beta or os.environ.get("CLAUDE_COMPUTER_USE_BETA") or version["beta"]
        self.computer_tool_type = (
            computer_tool_type
            or os.environ.get("CLAUDE_COMPUTER_TOOL_TYPE")
            or version["tool_type"]
        )
        self.display_number = display_number
        if gateway_schema_compat is None:
            gateway_schema_compat = _looks_like_schema_compat_gateway(self.base_url)
        self.gateway_schema_compat = gateway_schema_compat

        self.messages: list[dict[str, Any]] = []
        self.pending_tool_results: list[dict[str, Any]] = []
        self._pending_actions: list[Action] = []
        self._awaiting_tool_result: dict[str, Any] | None = None
        self._screen_size: tuple[int, int] | None = None
        self._total_tokens = 0
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.messages.clear()
        self.pending_tool_results.clear()
        self._pending_actions.clear()
        self._awaiting_tool_result = None
        self._screen_size = None
        self._total_tokens = 0
        self._history.clear()

    def get_token_usage(self) -> dict[str, int]:
        return {"total_tokens": self._total_tokens}

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        self._screen_size = _infer_screen_size(observation) or self._screen_size
        self._append_pending_tool_result(observation)

        if self._pending_actions:
            return self._pop_pending_action(context)

        instruction = observation.instruction or (context.task.instruction if context and context.task else "")
        self._append_user_message(instruction, observation)
        step = context.step_index if context else len(self._history)

        logger.info("Claude-CUA step %d: calling %s", step, self.model)
        started_at = time.time()
        response = self._create_message()
        elapsed = time.time() - started_at

        content = _get_field(response, "content", []) or []
        usage = _get_field(response, "usage", {}) or {}
        self._total_tokens += int(
            usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )

        assistant_content = _model_dump(content)
        self.messages.append({"role": "assistant", "content": assistant_content})

        parsed = self._parse_message_content(assistant_content)
        actions = parsed["actions"] or [Action.noop(raw=json.dumps(response, ensure_ascii=False))]
        first_action = actions[0]
        self._pending_actions.extend(actions[1:])
        self._mark_tool_result_if_needed(first_action)

        metadata = {
            "step_index": step,
            "agent": "claude_cua",
            "model": self.model,
            "usage": usage,
            "model_time": elapsed,
            "beta": self.beta,
            "computer_tool_type": self.computer_tool_type,
            "messages": _redact_images(assistant_content),
            "response_text": parsed["response_text"],
            "infeasible_message": parsed["infeasible_message"],
            "queued_followup_count": len(actions) - 1,
        }
        first_action.metadata.setdefault("claude_cua", {}).update(metadata)

        self._history.append({
            "step_index": step,
            "response_text": parsed["response_text"],
            "action_type": first_action.type.value,
            "action_arguments": first_action.arguments,
        })
        if len(self._history) > self.max_trajectory_length:
            self._history = self._history[-self.max_trajectory_length:]
            self._trim_message_history()

        return first_action

    def _append_user_message(self, instruction: str, observation: Observation) -> None:
        if self.pending_tool_results:
            self.messages.append({"role": "user", "content": list(self.pending_tool_results)})
            self.pending_tool_results.clear()
            return

        screenshot_b64 = _screenshot_b64(observation)
        content: list[dict[str, Any]] = []
        if screenshot_b64:
            content.append(_anthropic_image_block(screenshot_b64))
        content.append({"type": "text", "text": instruction or "Continue."})
        self.messages.append({"role": "user", "content": content})

    def _append_pending_tool_result(self, observation: Observation) -> None:
        if not self._awaiting_tool_result:
            return
        screenshot_b64 = _screenshot_b64(observation)
        content: list[dict[str, Any]] = [{"type": "text", "text": "Success"}]
        if screenshot_b64:
            content.append(_anthropic_image_block(screenshot_b64))
        self.pending_tool_results.append({
            "type": "tool_result",
            "tool_use_id": self._awaiting_tool_result.get("tool_use_id", ""),
            "content": content,
        })
        self._awaiting_tool_result = None

    def _create_message(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": self._operator_prompt(),
            "tools": [self._computer_tool()],
            "messages": self.messages,
            "betas": [self.beta],
        }

        logger.info(
            "Claude-CUA request endpoint=%s model=%s beta=%s tool=%s timeout=%ss api_key=%s",
            self.base_url,
            self.model,
            self.beta,
            self.computer_tool_type,
            self.request_timeout,
            "set" if self.api_key else "missing",
        )
        return self._create_message_with_sdk(payload)

    def _create_message_with_sdk(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from anthropic import Anthropic
        except Exception as exc:
            raise RuntimeError("Anthropic SDK is required for Claude-CUA requests.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.request_timeout,
        }
        proxy_url = self._proxy_url()
        if proxy_url:
            try:
                import httpx

                client_kwargs["http_client"] = httpx.Client(
                    proxy=proxy_url,
                    timeout=self.request_timeout,
                )
            except Exception as exc:
                raise RuntimeError(f"Could not configure Anthropic SDK proxy: {exc}") from exc

        client = Anthropic(**client_kwargs)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = client.beta.messages.create(**payload)
                return _model_dump(response)
            except Exception as exc:
                logger.error("Claude-CUA SDK API error: %s", exc)
                last_error = exc
                time.sleep(min(5, (attempt + 1) * 2))
        raise RuntimeError(f"Claude-CUA SDK API failed too many times: {last_error}") from last_error

    def _computer_tool(self) -> dict[str, Any]:
        width, height = self._screen_size or (1024, 768)
        tool: dict[str, Any] = {
            "type": self.computer_tool_type,
            "name": "computer",
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_number,
        }
        if self.computer_tool_type == "computer_20251124":
            tool["enable_zoom"] = False
        if self.gateway_schema_compat:
            tool.update(_gateway_schema_compat_tool_fields())
        return tool

    def _parse_message_content(self, content: list[Any]) -> dict[str, Any]:
        responses: list[str] = []
        actions: list[Action] = []
        infeasible_message = False
        saw_tool_use = False

        for item in content:
            item_type = _get_field(item, "type")
            if item_type == "text":
                text = str(_get_field(item, "text", "") or "")
                if text:
                    responses.append(text)
                    lower = text.lower()
                    if "[infeasible]" in lower or any(
                        token in lower
                        for token in ("infeasible", "unfeasible", "impossible", "cannot be done", "not feasible")
                    ):
                        infeasible_message = True
            elif item_type == "tool_use":
                saw_tool_use = True
                if _get_field(item, "name") == "computer":
                    actions.extend(self._tool_use_to_actions(item))

        if infeasible_message:
            actions = [Action.terminate(reason="FAIL", raw="\n".join(responses))]
        elif not actions and not saw_tool_use:
            actions = [Action.terminate(reason="DONE", raw="\n".join(responses))]

        return {
            "actions": actions,
            "response_text": "\n".join(part for part in responses if part),
            "infeasible_message": infeasible_message,
        }

    def _tool_use_to_actions(self, item: Any) -> list[Action]:
        tool_use_id = str(_get_field(item, "id", ""))
        action_input = _model_dump(_get_field(item, "input", {}) or {})
        if not isinstance(action_input, dict):
            return []
        raw = json.dumps(action_input, ensure_ascii=False)
        metadata = {
            "tool_use_id": tool_use_id,
            "raw_action": action_input,
        }

        def with_tool(action: Action) -> Action:
            action.metadata.setdefault("claude_cua_tool", {}).update(metadata)
            return action

        def finish(actions: list[Action]) -> list[Action]:
            return self._with_tool_batch_metadata([with_tool(action) for action in actions])

        action_type = _normalize_action_name(action_input.get("action"))
        coordinate = _coordinate(action_input.get("coordinate"))

        if action_type == "screenshot":
            return finish([Action(type=ActionType.SCREENSHOT, raw=raw)])
        if action_type in {"wait"}:
            seconds = float(action_input.get("duration") or 1.0)
            return finish([Action.noop(raw=raw, metadata={"seconds": seconds})])
        if action_type == "mouse_move":
            x, y = self._to_abs_xy(coordinate)
            return finish([Action(type=ActionType.HOVER, arguments={"x": x, "y": y}, raw=raw)])
        if action_type == "left_click":
            return finish([self._click_action(coordinate, raw=raw)])
        if action_type == "right_click":
            x, y = self._to_abs_xy(coordinate)
            return finish([Action(type=ActionType.RIGHT_CLICK, arguments={"x": x, "y": y}, raw=raw)])
        if action_type == "middle_click":
            x, y = self._to_abs_xy(coordinate)
            return finish([Action(type=ActionType.CLICK, arguments={"x": x, "y": y, "button": "middle"}, raw=raw)])
        if action_type == "double_click":
            x, y = self._to_abs_xy(coordinate)
            return finish([Action(type=ActionType.DOUBLE_CLICK, arguments={"x": x, "y": y}, raw=raw)])
        if action_type == "triple_click":
            x, y = self._to_abs_xy(coordinate)
            code = f"pyautogui.tripleClick(x={x}, y={y})"
            return finish([Action(type=ActionType.PYTHON, arguments={"code": code}, raw=raw)])
        if action_type == "left_click_drag":
            return finish(self._drag_actions(action_input, raw=raw))
        if action_type in {"left_mouse_down", "left_mouse_up"}:
            pyautogui_action = "mouseDown" if action_type == "left_mouse_down" else "mouseUp"
            if coordinate:
                x, y = self._to_abs_xy(coordinate)
                code = f"pyautogui.{pyautogui_action}(x={x}, y={y}, button='left')"
            else:
                code = f"pyautogui.{pyautogui_action}(button='left')"
            return finish([Action(type=ActionType.PYTHON, arguments={"code": code}, raw=raw)])
        if action_type == "type":
            return finish([Action.type_text(str(action_input.get("text", "")), raw=raw)])
        if action_type == "key":
            return finish([_key_action(str(action_input.get("text") or action_input.get("key") or ""), raw=raw)])
        if action_type == "hold_key":
            key = _map_key(action_input.get("text") or action_input.get("key") or "")
            duration = float(action_input.get("duration") or 1.0)
            code = (
                f"pyautogui.keyDown({key!r}); "
                f"time.sleep({duration}); "
                f"pyautogui.keyUp({key!r})"
            )
            return finish([Action(type=ActionType.PYTHON, arguments={"code": code}, raw=raw)])
        if action_type == "scroll":
            return finish([self._scroll_action(action_input, raw=raw)])

        logger.warning("Unsupported Claude-CUA computer action: %s", action_type)
        return finish([Action.noop(raw=raw, metadata={"unsupported_action": action_input})])

    def _with_tool_batch_metadata(self, actions: list[Action]) -> list[Action]:
        for index, action in enumerate(actions):
            tool = action.metadata.setdefault("claude_cua_tool", {})
            tool["batch_index"] = index
            tool["batch_size"] = len(actions)
            tool["batch_last"] = index == len(actions) - 1
        return actions

    def _click_action(self, coordinate: tuple[Any, Any] | None, *, raw: str) -> Action:
        if coordinate:
            x, y = self._to_abs_xy(coordinate)
            return Action(type=ActionType.CLICK, arguments={"x": x, "y": y, "button": "left"}, raw=raw)
        return Action(type=ActionType.CLICK, arguments={"button": "left"}, raw=raw)

    def _drag_actions(self, action_input: dict[str, Any], *, raw: str) -> list[Action]:
        start = _coordinate(action_input.get("start_coordinate"))
        end = _coordinate(action_input.get("coordinate"))
        if not start or not end:
            return [Action.noop(raw=raw)]
        start_x, start_y = self._to_abs_xy(start)
        end_x, end_y = self._to_abs_xy(end)
        return [
            Action(type=ActionType.HOVER, arguments={"x": start_x, "y": start_y}, raw=raw),
            Action(type=ActionType.DRAG, arguments={"x": end_x, "y": end_y}, raw=raw),
        ]

    def _scroll_action(self, action_input: dict[str, Any], *, raw: str) -> Action:
        direction = str(action_input.get("scroll_direction") or "down").lower()
        amount = int(action_input.get("scroll_amount") or 3)
        multiplier = max(1, amount)
        dx, dy = 0, 0
        if direction == "up":
            dy = 3 * multiplier
        elif direction == "down":
            dy = -3 * multiplier
        elif direction == "left":
            dx = -3 * multiplier
        elif direction == "right":
            dx = 3 * multiplier
        return Action(type=ActionType.SCROLL, arguments={"dx": dx, "dy": dy, "direction": direction}, raw=raw)

    def _mark_tool_result_if_needed(self, action: Action) -> None:
        tool = action.metadata.get("claude_cua_tool")
        if not tool:
            return
        if tool.get("batch_last") is False:
            return
        self._awaiting_tool_result = {
            "tool_use_id": tool.get("tool_use_id", ""),
        }

    def _pop_pending_action(self, context: HarnessContext | None) -> Action:
        action = self._pending_actions.pop(0)
        self._mark_tool_result_if_needed(action)
        action.metadata.setdefault("claude_cua", {}).update({
            "step_index": context.step_index if context else len(self._history),
            "agent": "claude_cua",
            "queued_action": True,
            "queued_remaining": len(self._pending_actions),
        })
        return action

    def _to_abs_xy(self, coordinate: tuple[Any, Any] | None) -> tuple[int, int]:
        width, height = self._screen_size or (1024, 768)
        if not coordinate:
            return 0, 0
        return _coord_to_abs(coordinate[0], width), _coord_to_abs(coordinate[1], height)

    def _operator_prompt(self) -> str:
        home_dir = "C:\\Users\\user" if self.platform.lower().startswith("win") else "/home/user"
        return OPERATOR_PROMPT.format(
            CLIENT_PASSWORD=self.client_password,
            CURRENT_DATE=datetime.now().strftime("%A, %B %d, %Y"),
            HOME_DIR=home_dir,
            PLATFORM=self.platform,
        )

    def _proxy_url(self) -> str | None:
        return (
            self.proxy_url
            or os.environ.get("AGENT_RUNTIME_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
        )

    def _trim_message_history(self) -> None:
        if self.max_trajectory_length <= 0:
            return
        keep_pairs = self.max_trajectory_length * 2 + 2
        if len(self.messages) > keep_pairs:
            self.messages = self.messages[-keep_pairs:]

def _anthropic_image_block(screenshot_b64: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": screenshot_b64,
        },
    }

def _computer_use_version(model: str) -> dict[str, str]:
    normalized = model.lower().replace("_", "-").replace(".", "-")
    new_markers = (
        "opus-4-7",
        "opus-4-6",
        "sonnet-4-6",
        "opus-4-5",
    )
    if any(marker in normalized for marker in new_markers):
        return {"beta": "computer-use-2025-11-24", "tool_type": "computer_20251124"}
    return {"beta": "computer-use-2025-01-24", "tool_type": "computer_20250124"}

def _looks_like_schema_compat_gateway(base_url: str) -> bool:
    env_value = os.environ.get("CLAUDE_GATEWAY_SCHEMA_COMPAT")
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes"}
    return False

def _gateway_schema_compat_tool_fields() -> dict[str, Any]:
    return {
        "description": (
            "Control a desktop computer by moving/clicking the mouse, typing keys/text, "
            "scrolling, dragging, waiting, and taking screenshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "mouse_move",
                        "left_click",
                        "left_mouse_down",
                        "left_mouse_up",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "key",
                        "type",
                        "hold_key",
                        "wait",
                    ],
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "start_coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "text": {"type": "string"},
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
                "scroll_amount": {"type": "integer", "minimum": 1},
                "duration": {"type": "number", "minimum": 0},
            },
            "required": ["action"],
            "additionalProperties": True,
        },
    }

def _coordinate(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return None

def _normalize_action_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

def _key_action(text: str, *, raw: str) -> Action:
    keys = [_map_key(part) for part in text.replace("+", " ").split() if part.strip()]
    if not keys and text:
        keys = [_map_key(text)]
    if len(keys) <= 1:
        return Action(type=ActionType.KEY_PRESS, arguments={"key": keys[0] if keys else ""}, raw=raw)
    return Action(type=ActionType.HOTKEY, arguments={"keys": keys}, raw=raw)

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
            if key == "data" and isinstance(item, str) and len(item) > 100:
                redacted[key] = f"<redacted image length={len(item)}>"
            else:
                redacted[key] = _redact_images(item)
        return redacted
    if isinstance(value, list):
        return [_redact_images(item) for item in value]
    return value

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

def _coord_to_abs(value: Any, size: int) -> int:
    number = float(value if value is not None else 0)
    return _clamp_int(round(number), size)

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
        "control": "ctrl",
        "delete": "delete",
        "end": "end",
        "enter": "enter",
        "esc": "esc",
        "escape": "esc",
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

__all__ = ["ClaudeCUAHarness"]
