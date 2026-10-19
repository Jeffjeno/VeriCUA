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
- You can act without asking for confirmation.
- If content may be off-screen, scroll or zoom out before deciding it is unavailable.
- When possible, bundle multiple GUI actions into one computer-use turn.
- If the task is infeasible because of missing apps, permissions, contradictory requirements, or other hard blockers, output exactly "[INFEASIBLE]".
"""

class GPT54CUAHarness(Harness):

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.5,
        max_trajectory_length: int = 100,
        client_password: str = "password",
        proxy_url: str | None = None,
        platform: str = "ubuntu",
        reasoning_effort: str = "medium",
        request_timeout: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_trajectory_length = max_trajectory_length
        self.client_password = client_password
        self.proxy_url = proxy_url
        self.platform = platform
        self.reasoning_effort = reasoning_effort
        self.request_timeout = request_timeout or float(
            os.environ.get("GPT54_REQUEST_TIMEOUT") or os.environ.get("OPENAI_REQUEST_TIMEOUT") or 600
        )
        self._last_response_id: str | None = None
        self.pending_input_items: list[dict[str, Any]] = []
        self._pending_actions: list[Action] = []
        self._awaiting_call_output: dict[str, Any] | None = None
        self._screen_size: tuple[int, int] | None = None
        self._total_tokens = 0
        self._history: list[dict[str, Any]] = []
        self._input_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._last_response_id = None
        self.pending_input_items.clear()
        self._pending_actions.clear()
        self._awaiting_call_output = None
        self._screen_size = None
        self._total_tokens = 0
        self._history.clear()
        self._input_history.clear()

    def get_token_usage(self) -> dict[str, int]:
        return {"total_tokens": self._total_tokens}

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        self._screen_size = _infer_screen_size(observation) or self._screen_size
        self._append_pending_call_output(observation)

        if self._pending_actions:
            return self._pop_pending_action(context)

        instruction = observation.instruction or (context.task.instruction if context and context.task else "")
        new_input_items = self._build_request_input(instruction, observation)
        request_input = [*self._input_history, *new_input_items]
        step = context.step_index if context else len(self._history)

        logger.info("GPT54-CUA step %d: calling %s", step, self.model)
        started_at = time.time()
        response = self._create_response(request_input)
        elapsed = time.time() - started_at

        response_id = _get_field(response, "id")
        if response_id:
            self._last_response_id = str(response_id)
        output = _get_field(response, "output", []) or []
        self._append_response_context(new_input_items, output)
        usage = _get_field(response, "usage", {}) or {}
        self._total_tokens += int(usage.get("total_tokens") or usage.get("input_tokens", 0) + usage.get("output_tokens", 0))

        parsed = self._parse_response_output(output)
        actions = parsed["actions"] or [Action.noop(raw=json.dumps(response, ensure_ascii=False))]
        first_action = actions[0]
        self._pending_actions.extend(actions[1:])
        self._mark_call_output_if_needed(first_action)

        metadata = {
            "step_index": step,
            "agent": "gpt54_cua",
            "model": self.model,
            "response_id": self._last_response_id,
            "usage": usage,
            "model_time": elapsed,
            "messages": _redact_response_output(output),
            "response_text": parsed["response_text"],
            "infeasible_message": parsed["infeasible_message"],
            "queued_followup_count": len(actions) - 1,
        }
        first_action.metadata.setdefault("gpt54_cua", {}).update(metadata)

        self._history.append({
            "step_index": step,
            "response_id": self._last_response_id,
            "response_text": parsed["response_text"],
            "action_type": first_action.type.value,
            "action_arguments": first_action.arguments,
        })
        if len(self._history) > self.max_trajectory_length:
            self._history = self._history[-self.max_trajectory_length:]
        self._trim_input_history()

        return first_action

    def _create_response(self, request_input: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self._operator_prompt(),
            "input": request_input,
            "tools": [{"type": "computer"}],
            "parallel_tool_calls": False,
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "concise",
            },
            "truncation": "auto",
        }
        if self.max_tokens is not None:
            payload["max_output_tokens"] = self.max_tokens
        logger.info(
            "GPT54-CUA request endpoint=%s model=%s timeout=%ss api_key=%s history_items=%d",
            f"{self.base_url}/responses",
            self.model,
            self.request_timeout,
            "set" if self.api_key else "missing",
            len(request_input),
        )

        return self._create_response_with_sdk(payload)

    def _create_response_with_sdk(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is required for GPT54-CUA Responses API requests.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.request_timeout,
        }
        proxy_url = self._proxy_url()
        if proxy_url:
            try:
                import httpx

                try:
                    client_kwargs["http_client"] = httpx.Client(
                        proxy=proxy_url,
                        timeout=self.request_timeout,
                    )
                except TypeError:
                    client_kwargs["http_client"] = httpx.Client(
                        proxies=proxy_url,
                        timeout=self.request_timeout,
                    )
            except Exception as exc:
                raise RuntimeError(f"Could not configure OpenAI SDK proxy: {exc}") from exc

        client = OpenAI(**client_kwargs)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = client.responses.create(**payload)
                body = _model_dump(response)
                response_error = _get_field(_get_field(body, "error", {}), "message")
                if response_error:
                    raise RuntimeError(str(response_error))
                if _get_field(body, "status") == "failed":
                    raise RuntimeError("Responses API request failed.")
                return body
            except Exception as exc:
                logger.error("GPT54-CUA SDK API error: %s", exc)
                last_error = exc
                time.sleep(min(5, (attempt + 1) * 2))
        raise RuntimeError(f"GPT54-CUA SDK API failed too many times: {last_error}") from last_error

    def _proxy_url(self) -> str | None:
        return (
            self.proxy_url
            or os.environ.get("AGENT_RUNTIME_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
        )

    def _build_request_input(self, instruction: str, observation: Observation) -> list[dict[str, Any]]:
        if self.pending_input_items:
            pending = list(self.pending_input_items)
            self.pending_input_items.clear()
            return pending

        screenshot_b64 = _screenshot_b64(observation)
        if not screenshot_b64:
            return [{"role": "user", "content": [{"type": "input_text", "text": instruction or "Continue."}]}]

        text = instruction if not self._input_history else "Continue from the latest screenshot."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "high",
                    },
                ],
            }
        ]

    def _append_response_context(self, new_input_items: list[dict[str, Any]], output: list[Any]) -> None:
        dumped_input = _model_dump(new_input_items)
        if isinstance(dumped_input, list):
            self._input_history.extend(dumped_input)
        elif dumped_input:
            self._input_history.append(dumped_input)

        self._input_history.extend(_response_output_to_input_items(output))

    def _trim_input_history(self) -> None:
        if self.max_trajectory_length <= 0:
            return
        if self._awaiting_call_output or self._pending_actions:
            return
        max_items = max(2, self.max_trajectory_length * 2)
        if len(self._input_history) > max_items:
            pinned = self._input_history[:1]
            recent = self._input_history[-(max_items - len(pinned)):]
            self._input_history = [*pinned, *recent]

    def _append_pending_call_output(self, observation: Observation) -> None:
        if not self._awaiting_call_output:
            return
        screenshot_b64 = _screenshot_b64(observation)
        if not screenshot_b64:
            return
        output_item: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": self._awaiting_call_output.get("call_id", ""),
            "output": {
                "type": "computer_screenshot",
                "image_url": f"data:image/png;base64,{screenshot_b64}",
                "detail": "high",
            },
        }
        pending_checks = self._awaiting_call_output.get("pending_checks") or []
        if pending_checks:
            output_item["acknowledged_safety_checks"] = pending_checks
        self.pending_input_items.append(output_item)
        self._awaiting_call_output = None

    def _pop_pending_action(self, context: HarnessContext | None) -> Action:
        action = self._pending_actions.pop(0)
        self._mark_call_output_if_needed(action)
        action.metadata.setdefault("gpt54_cua", {}).update({
            "step_index": context.step_index if context else len(self._history),
            "agent": "gpt54_cua",
            "queued_action": True,
            "queued_remaining": len(self._pending_actions),
        })
        return action

    def _mark_call_output_if_needed(self, action: Action) -> None:
        batch = action.metadata.get("gpt54_cua_batch")
        if not batch or not batch.get("batch_last"):
            return
        self._awaiting_call_output = {
            "call_id": batch.get("call_id", ""),
            "pending_checks": batch.get("pending_checks", []),
        }

    def _parse_response_output(self, output: list[Any]) -> dict[str, Any]:
        responses: list[str] = []
        actions: list[Action] = []
        infeasible_message = False
        saw_message = False
        saw_computer_call = False

        for item in output:
            item_type = _get_field(item, "type")
            if item_type == "message":
                saw_message = True
                text = _message_text(item)
                if text:
                    responses.append(text)
                    lower = text.lower()
                    if "[infeasible]" in lower or any(
                        token in lower
                        for token in ("infeasible", "unfeasible", "impossible", "cannot be done", "not feasible")
                    ):
                        infeasible_message = True
            elif item_type == "reasoning":
                text = _reasoning_text(item)
                if text:
                    responses.append(text)
            elif item_type == "computer_call":
                saw_computer_call = True
                actions.extend(self._computer_call_to_actions(item))

        if infeasible_message:
            actions = [Action.terminate(reason="FAIL", raw="\n".join(responses))]
        elif not actions and saw_message and not saw_computer_call:
            actions = [Action.terminate(reason="DONE", raw="\n".join(responses))]

        return {
            "actions": actions,
            "response_text": "\n".join(part for part in responses if part),
            "infeasible_message": infeasible_message,
        }

    def _computer_call_to_actions(self, item: Any) -> list[Action]:
        raw_actions = _get_field(item, "actions")
        if raw_actions is None:
            single_action = _get_field(item, "action")
            raw_actions = [single_action] if single_action is not None else []

        call_id = _get_field(item, "call_id", "")
        pending_checks = _model_dump(_get_field(item, "pending_safety_checks", []))
        raw_actions = list(raw_actions)
        converted: list[Action] = []
        batch_size = len(raw_actions)

        for index, raw_action in enumerate(raw_actions):
            action_info = _action_to_dict(raw_action)
            batch_metadata = {
                "call_id": call_id,
                "pending_checks": pending_checks,
                "batch_index": index,
                "batch_size": batch_size,
                "batch_last": index == batch_size - 1,
                "raw_action": action_info,
            }
            action_list = self._computer_action_to_agent_runtime(action_info, batch_metadata)
            converted.extend(action_list)

        for physical_index, action in enumerate(converted):
            action.metadata["gpt54_cua_batch"] = {
                "call_id": call_id,
                "pending_checks": pending_checks,
                "batch_index": physical_index,
                "batch_size": len(converted),
                "batch_last": physical_index == len(converted) - 1,
            }
        return converted

    def _computer_action_to_agent_runtime(
        self,
        action_info: dict[str, Any],
        batch_metadata: dict[str, Any],
    ) -> list[Action]:
        action_type = str(action_info.get("type") or "").lower()
        args = action_info.get("args") or {}
        raw = json.dumps(action_info, ensure_ascii=False)

        def with_batch(action: Action) -> Action:
            action.metadata.setdefault("gpt54_cua_batch", {}).update(batch_metadata)
            return action

        if action_type == "click":
            x, y = self._to_abs_xy(args.get("x"), args.get("y"))
            button = str(args.get("button", "left"))
            if button == "right":
                return [with_batch(Action(type=ActionType.RIGHT_CLICK, arguments={"x": x, "y": y}, raw=raw))]
            return [with_batch(Action(type=ActionType.CLICK, arguments={"x": x, "y": y, "button": button}, raw=raw))]

        if action_type == "double_click":
            x, y = self._to_abs_xy(args.get("x"), args.get("y"))
            return [with_batch(Action(type=ActionType.DOUBLE_CLICK, arguments={"x": x, "y": y}, raw=raw))]

        if action_type == "move":
            x, y = self._to_abs_xy(args.get("x"), args.get("y"))
            return [with_batch(Action(type=ActionType.HOVER, arguments={"x": x, "y": y}, raw=raw))]

        if action_type == "drag":
            return [with_batch(action) for action in self._drag_actions(args, raw=raw)]

        if action_type == "type":
            return [with_batch(Action.type_text(str(args.get("text", "")), raw=raw))]

        if action_type == "keypress":
            keys = args.get("keys")
            if not keys and args.get("key"):
                keys = [args.get("key")]
            if isinstance(keys, str):
                keys = [keys]
            mapped = [_map_key(key) for key in (keys or [])]
            if len(mapped) == 1:
                return [with_batch(Action(type=ActionType.KEY_PRESS, arguments={"key": mapped[0]}, raw=raw))]
            return [with_batch(Action(type=ActionType.HOTKEY, arguments={"keys": mapped}, raw=raw))]

        if action_type == "scroll":
            actions: list[Action] = []
            if args.get("x") is not None and args.get("y") is not None:
                x, y = self._to_abs_xy(args.get("x"), args.get("y"))
                actions.append(Action(type=ActionType.HOVER, arguments={"x": x, "y": y}, raw=raw))
            scroll_x = int(args.get("scroll_x") or args.get("scrollX") or args.get("delta_x") or args.get("deltaX") or 0)
            scroll_y = int(args.get("scroll_y") or args.get("scrollY") or args.get("delta_y") or args.get("deltaY") or 0)
            if not scroll_x and not scroll_y:
                return [with_batch(Action.noop(raw=raw))]
            actions.append(Action(type=ActionType.SCROLL, arguments={"dx": -scroll_x, "dy": -scroll_y}, raw=raw))
            return [with_batch(action) for action in actions]

        if action_type in ("wait", "screenshot"):
            seconds = max(0.1, float(args.get("ms", 1000)) / 1000.0) if action_type == "wait" else 0.1
            return [with_batch(Action.noop(raw=raw, metadata={"seconds": seconds}))]

        logger.warning("Unsupported GPT54-CUA computer action: %s", action_type)
        return [with_batch(Action.noop(raw=raw, metadata={"unsupported_action": action_info}))]

    def _drag_actions(self, args: dict[str, Any], *, raw: str) -> list[Action]:
        path = args.get("path")
        if not path and args.get("from") and args.get("to"):
            path = [args["from"], args["to"]]
        if not path or len(path) < 2:
            return [Action.noop(raw=raw)]

        points = [_point_xy(point) for point in path]
        if any(x is None or y is None for x, y in points):
            return [Action.noop(raw=raw)]

        start_x, start_y = self._to_abs_xy(points[0][0], points[0][1])
        actions = [Action(type=ActionType.HOVER, arguments={"x": start_x, "y": start_y}, raw=raw)]
        for point_x, point_y in points[1:]:
            x, y = self._to_abs_xy(point_x, point_y)
            actions.append(Action(type=ActionType.DRAG, arguments={"x": x, "y": y}, raw=raw))
        return actions

    def _to_abs_xy(self, x_value: Any, y_value: Any) -> tuple[int, int]:
        width, height = self._screen_size or (1920, 1080)
        return _coord_to_abs(x_value, width), _coord_to_abs(y_value, height)

    def _operator_prompt(self) -> str:
        home_dir = "C:\\Users\\user" if self.platform.lower().startswith("win") else "/home/user"
        return OPERATOR_PROMPT.format(
            CLIENT_PASSWORD=self.client_password,
            CURRENT_DATE=datetime.now().strftime("%A, %B %d, %Y"),
            HOME_DIR=home_dir,
            PLATFORM=self.platform,
        )

def _action_to_dict(action: Any) -> dict[str, Any]:
    raw = _model_dump(action)
    if isinstance(raw, dict):
        action_type = raw.get("type")
        return {"type": action_type, "args": {key: value for key, value in raw.items() if key != "type"}}
    action_type = getattr(action, "type", None)
    args: dict[str, Any] = {}
    for attr in dir(action):
        if attr.startswith("_") or attr == "type":
            continue
        try:
            args[attr] = _model_dump(getattr(action, attr))
        except Exception:
            continue
    return {"type": action_type, "args": args}

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

def _message_text(item: Any) -> str:
    content = _get_field(item, "content", [])
    if isinstance(content, list):
        parts = []
        for part in content:
            if _get_field(part, "type") == "output_text":
                parts.append(str(_get_field(part, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")

def _reasoning_text(item: Any) -> str:
    summary = _get_field(item, "summary", [])
    if isinstance(summary, list):
        return "\n".join(str(_get_field(part, "text", "")) for part in summary if _get_field(part, "text", ""))
    return str(summary or "")

def _redact_response_output(output: list[Any]) -> list[Any]:
    return _redact_images(_model_dump(output))

def _response_output_to_input_items(output: list[Any]) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for item in output:
        dumped = _model_dump(item)
        if not isinstance(dumped, dict):
            continue
        item_type = dumped.get("type")

        if item_type == "message":
            content = _message_content_for_input(dumped.get("content", []))
            if content:
                input_items.append({
                    "type": "message",
                    "role": dumped.get("role") or "assistant",
                    "content": content,
                })
            continue

        if item_type == "computer_call":
            computer_call: dict[str, Any] = {"type": "computer_call"}
            call_id = dumped.get("call_id")
            if call_id:
                computer_call["call_id"] = call_id
            if dumped.get("action") is not None:
                computer_call["action"] = dumped["action"]
            elif dumped.get("actions") is not None:
                computer_call["actions"] = dumped["actions"]
            if "action" in computer_call or "actions" in computer_call:
                input_items.append(computer_call)
            continue

        if item_type == "function_call":
            function_call = {
                key: dumped[key]
                for key in ("type", "call_id", "name", "arguments")
                if dumped.get(key) is not None
            }
            if "call_id" in function_call and "name" in function_call:
                input_items.append(function_call)

    return input_items

def _message_content_for_input(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    converted: list[dict[str, Any]] = []
    for part in content:
        dumped = _model_dump(part)
        if not isinstance(dumped, dict):
            continue
        part_type = dumped.get("type")
        text = dumped.get("text")
        if part_type in {"output_text", "refusal"} and text is not None:
            converted.append({"type": part_type, "text": str(text)})
    return converted

def _redact_images(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in ("image_url", "url") and isinstance(item, str) and item.startswith("data:image/"):
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

def _point_xy(point: Any) -> tuple[Any, Any]:
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return point[0], point[1]
    if isinstance(point, dict):
        return point.get("x"), point.get("y")
    return getattr(point, "x", None), getattr(point, "y", None)

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

__all__ = ["GPT54CUAHarness"]
