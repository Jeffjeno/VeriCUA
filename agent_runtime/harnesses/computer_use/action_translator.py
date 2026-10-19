from __future__ import annotations

from typing import Any

from agent_runtime.types import Action, ActionType

_BUTTON_MAP = {
    "left": "left",
    "right": "right",
    "middle": "middle",
    "wheel": "middle",
    "back": "back",
    "forward": "forward",
}

def openai_to_agent_runtime(call: dict[str, Any]) -> Action:

    actions = call.get("actions")
    if actions is None:

        single = call.get("action")
        actions = [single] if single else []

    call_id = call.get("call_id") or call.get("id")
    pending = call.get("pending_safety_checks") or []
    metadata: dict[str, Any] = {
        "source": "openai_computer_use",
        "call_id": call_id,
    }
    if pending:
        metadata["pending_safety_checks"] = pending
    if len(actions) > 1:
        metadata["extra_actions"] = actions[1:]

    primary = actions[0] if actions else {"type": "wait"}
    base = _translate_single(primary)
    merged = {**base.metadata, **metadata}
    return base.model_copy(update={"metadata": merged, "raw": _safe_dumps(call)})

def _translate_single(action_obj: dict[str, Any]) -> Action:
    atype = action_obj.get("type")

    if atype == "click":
        button = _BUTTON_MAP.get(action_obj.get("button") or "left", "left")
        x = int(action_obj.get("x", 0))
        y = int(action_obj.get("y", 0))
        if button == "right":
            return Action(type=ActionType.RIGHT_CLICK, arguments={"x": x, "y": y})
        return Action(type=ActionType.CLICK, arguments={"x": x, "y": y, "button": button})

    if atype == "double_click":
        return Action(
            type=ActionType.DOUBLE_CLICK,
            arguments={"x": int(action_obj.get("x", 0)), "y": int(action_obj.get("y", 0))},
        )

    if atype == "scroll":
        return Action(
            type=ActionType.SCROLL,
            arguments={
                "x": int(action_obj.get("x", 0)),
                "y": int(action_obj.get("y", 0)),
                "dx": int(action_obj.get("scroll_x", 0)),
                "dy": int(action_obj.get("scroll_y", 0)),
            },
        )

    if atype == "type":
        return Action(type=ActionType.TYPE, arguments={"text": action_obj.get("text", "")})

    if atype == "keypress":
        keys = action_obj.get("keys") or []
        if not isinstance(keys, list):
            keys = [keys]
        keys = [str(k) for k in keys]
        if len(keys) <= 1:
            return Action(
                type=ActionType.KEY_PRESS,
                arguments={"key": keys[0] if keys else ""},
            )
        return Action(type=ActionType.HOTKEY, arguments={"keys": keys})

    if atype == "move":
        return Action(
            type=ActionType.HOVER,
            arguments={"x": int(action_obj.get("x", 0)), "y": int(action_obj.get("y", 0))},
        )

    if atype == "drag":
        path = action_obj.get("path") or []
        if path:
            end = path[-1]
            start = path[0]
            return Action(
                type=ActionType.DRAG,
                arguments={
                    "x": int(end.get("x", 0)),
                    "y": int(end.get("y", 0)),
                    "start_x": int(start.get("x", 0)),
                    "start_y": int(start.get("y", 0)),
                    "path": path,
                },
            )
        return Action(type=ActionType.NOOP)

    if atype == "wait":
        ms = int(action_obj.get("ms") or action_obj.get("duration_ms") or 1000)
        return Action(type=ActionType.NOOP, arguments={"wait_ms": ms})

    if atype == "screenshot":
        return Action(type=ActionType.SCREENSHOT)

    return Action(
        type=ActionType.CUSTOM,
        name=f"openai_computer_{atype or 'unknown'}",
        arguments=dict(action_obj),
    )

def acknowledged_safety_checks(action: Action) -> list[dict[str, Any]]:

    return list(action.metadata.get("pending_safety_checks") or [])

def call_id_of(action: Action) -> str | None:
    return action.metadata.get("call_id")

def _safe_dumps(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)

__all__ = ["openai_to_agent_runtime", "acknowledged_safety_checks", "call_id_of"]
