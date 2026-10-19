from __future__ import annotations

from pathlib import Path
from typing import Any

from eval_runtime.vagen_active.qwen_eval import (
    QwenVagenEvalAgent,
)
from eval_runtime.web_arena.holo_probe import _safe_probe_navigation_url

BROWSER_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "browser_use",
        "description": (
            "Inspect the live WebArena browser with one low-risk action. "
            "Coordinates are pixels in the current screenshot. Never submit "
            "forms or mutate task data during verification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "click", "double_click", "hover", "scroll", "type",
                        "key", "go_back", "go_forward", "goto", "wait",
                    ],
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "text": {"type": "string"},
                "content": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
                "amount": {"type": "integer", "minimum": 1},
                "pixels": {"type": "integer", "minimum": 1},
                "keys": {"type": "array", "items": {"type": "string"}},
                "key": {"type": "string"},
                "url": {"type": "string"},
                "press_enter": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}

CHECK_SCREENSHOT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_screenshot",
        "description": "Inspect saved screenshots from the recorded WebArena trajectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
            },
            "required": ["image_names"],
            "additionalProperties": False,
        },
    },
}

def _coordinate(args: dict[str, Any], width: int, height: int) -> tuple[int, int] | None:
    value = args.get("coordinate")
    if value is None and args.get("x") is not None and args.get("y") is not None:
        value = (args["x"], args["y"])
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (
            min(width - 1, max(0, int(float(value[0])))),
            min(height - 1, max(0, int(float(value[1])))),
        )
    except (TypeError, ValueError):
        return None

def browser_action_from_qwen(
    args: dict[str, Any], *, width: int, height: int, current_url: str
) -> tuple[str | None, str | None]:

    action = str(args.get("action") or args.get("command") or "").strip().lower()
    point = _coordinate(args, width, height)
    if action in {"click", "left_click"}:
        if point is None:
            return None, "click requires coordinate [x, y]"
        return f"mouse_click({point[0]}, {point[1]}, button='left')", None
    if action in {"double_click", "doubleclick"}:
        if point is None:
            return None, "double_click requires coordinate [x, y]"
        return f"mouse_dblclick({point[0]}, {point[1]}, button='left')", None
    if action in {"hover", "mouse_move"}:
        if point is None:
            return None, "hover requires coordinate [x, y]"
        return f"mouse_move({point[0]}, {point[1]})", None
    if action in {"scroll", "hscroll"}:
        direction = str(args.get("direction") or args.get("scroll_direction") or "down").lower()
        try:
            amount = max(1, min(max(width, height), int(float(args.get("amount", args.get("pixels", 500))))))
        except (TypeError, ValueError):
            amount = 500
        if direction in {"up", "down"}:
            dx, dy = 0, -amount if direction == "up" else amount
        elif direction in {"left", "right"}:
            dx, dy = (-amount if direction == "left" else amount), 0
        else:
            return None, f"unsupported scroll direction: {direction}"
        return f"scroll({dx}, {dy})", None
    if action in {"type", "write", "keyboard_type"}:
        text = str(args.get("text") or args.get("content") or "")
        if not text:
            return None, "type requires non-empty text"

        if args.get("enter") or args.get("press_enter"):
            return None, "form submission is forbidden during VAGEN verification"
        return f"keyboard_type({text!r})", None
    if action in {"key", "keypress", "keyboard_press"}:
        keys = args.get("keys") or args.get("key")
        if isinstance(keys, list):
            key_text = "+".join(str(value) for value in keys)
        else:
            key_text = str(keys or "")
        if not key_text:
            return None, "key requires a key name"
        if any(value in key_text.lower().split("+") for value in ("enter", "return")):
            return None, "form submission is forbidden during VAGEN verification"
        return f"keyboard_press({key_text!r})", None
    if action in {"wait", "screenshot", "noop"}:
        try:
            milliseconds = max(0, min(10000, int(float(args.get("milliseconds", args.get("duration", 500))))))
        except (TypeError, ValueError):
            milliseconds = 500
        return f"noop({milliseconds})", None
    if action in {"go_back", "back"}:
        return "go_back()", None
    if action in {"go_forward", "forward"}:
        return "go_forward()", None
    if action in {"goto", "navigate"}:
        requested_url = str(args.get("url") or args.get("text") or "")
        routed, reason = _safe_probe_navigation_url(requested_url, current_url)
        if routed is None:
            return None, reason or "navigation target is outside active WebArena origins"
        return f"goto({routed!r})", None
    if action in {"terminate", "done", "finish", "answer"}:
        return None, "logical termination is handled by the final judgment"
    return None, f"unsupported browser action: {action or '<empty>'}"

class WebArenaQwenVagenEvalAgent(QwenVagenEvalAgent):

    DEFAULT_TOOLS = ("browser_use", "check_screenshot")

    def __init__(self, *, platform: str = "WebArena", **kwargs: Any) -> None:
        self.platform = platform
        super().__init__(**kwargs)

    def _request_tools(self) -> list[dict[str, Any]]:

        return [BROWSER_USE_TOOL, CHECK_SCREENSHOT_TOOL]

    @staticmethod
    def _system_prompt(max_round: int) -> str:
        return f"""You are a strict WebArena task verifier following the VAGEN evaluation procedure.
You receive the recorded trajectory and its screenshots first. Judge from that evidence before probing.
If evidence is inconclusive, use exactly one browser_use call per turn for low-risk, read-only inspection,
or use check_screenshot for a recorded screenshot. Never redo a failed task. Never create, edit, delete,
send, submit, purchase, press Enter in a form, or otherwise mutate user data. Stay on the active WebArena
site origins; never navigate to localhost, 127.0.0.1, another node, Google, OpenStreetMap, or a public link.
browser_use actions: click, double_click, hover, scroll, type (without submit), key (without Enter),
go_back, go_forward, goto (same-origin only), and wait. Coordinates are screenshot pixels.
Use the native browser_use/check_screenshot functions when available. If the
server returns textual tool calls, use XML containing JSON:
<tool_call>{{\"name\":\"browser_use\",\"arguments\":{{...}}}}</tool_call>.
Use check_screenshot with {{\"image_names\":[\"step_N\"]}}. A FAILURE with LOW confidence means
the evidence is inconclusive: use browser_use for read-only inspection before making the final decision.
You have at most {max_round} rounds. When evidence is sufficient, output exactly:
EVALUATION RESULT:
Status: SUCCESS or FAILURE
Confidence: HIGH, MEDIUM, or LOW
Reasoning: detailed evidence-based reasoning"""

    def _accept_final_result(self, result: dict[str, Any]) -> bool:

        return not (
            not bool(result.get("success"))
            and str(result.get("confidence") or "").upper() == "LOW"
        )

    def _build_trajectory_prompt(
        self, instruction: str, trajectory: dict[str, Any], count: int
    ) -> str:
        actions = "\n".join(
            f"Step {idx + 1}: {str(action.get('raw_response', action))}"
            for idx, action in enumerate(trajectory.get("actions", []))
        ) or "None"
        return f"""You are evaluating a WebArena browser automation task.
Task instruction:
{instruction}

Execution trajectory ({count} screenshots):
{actions}

The attached screenshots are the latest recorded browser observations, with the last one being the final
recorded state. First make a static assessment. Only if inconclusive use low-risk browser inspection.
Do not alter data or continue an incomplete workflow."""

    def _execute_probe_tool(
        self,
        *,
        name: str,
        args: dict[str, Any],
        env: Any,
        save_dir: str,
        eval_steps: int,
        image_map: dict[str, bytes],
    ) -> tuple[str, bytes | None, bool]:
        if name == "check_screenshot":
            return super()._execute_probe_tool(
                name=name, args=args, env=env, save_dir=save_dir,
                eval_steps=eval_steps, image_map=image_map,
            )
        if name not in {"browser_use", "browser"}:
            return f"Unknown WebArena verifier tool: {name}", None, False

        current = env.get_obs("")
        screenshot = current.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)):
            return "Browser observation has no screenshot", None, False
        from PIL import Image
        import io
        with Image.open(io.BytesIO(bytes(screenshot))) as image:
            width, height = image.size
        command, error = browser_action_from_qwen(
            args, width=width, height=height,
            current_url=str(current.get("url") or ""),
        )
        if command is None:
            return f"Browser action rejected: {error}", bytes(screenshot), False
        next_obs = env.step_browser(command)
        next_screenshot = next_obs.get("screenshot")
        if not isinstance(next_screenshot, (bytes, bytearray)):
            return "Browser action returned no screenshot", None, False
        next_screenshot = bytes(next_screenshot)
        probe_dir = Path(save_dir) / "eval_result"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / f"round_{eval_steps}.png").write_bytes(next_screenshot)
        return "Browser inspection action executed.", next_screenshot, False

__all__ = [
    "BROWSER_USE_TOOL",
    "CHECK_SCREENSHOT_TOOL",
    "WebArenaQwenVagenEvalAgent",
    "browser_action_from_qwen",
]
