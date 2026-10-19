from __future__ import annotations

import base64
import ast
import io
import json
import logging
import os
import re
import time
from typing import Any

from openai import OpenAI
from PIL import Image

logger = logging.getLogger("eval_runtime.vagen_active.qwen_eval")

_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_STATUS_RE = re.compile(r"\bStatus\s*:\s*(SUCCESS|FAILURE)\b", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"\bConfidence\s*:\s*(HIGH|MEDIUM|LOW)\b", re.IGNORECASE)
_REASON_RE = re.compile(
    r"\bReasoning\s*:\s*(.*?)(?=\n\s*(?:Status|Confidence)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)

def _resize_png(data: bytes, size: tuple[int, int] = (1280, 720)) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()

def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

def _json_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:

            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

def _extract_tool_calls(message: Any, text: str) -> list[dict[str, Any]]:

    calls: list[dict[str, Any]] = []

    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", None)
        if name:
            calls.append({
                "id": str(getattr(call, "id", "")),
                "name": str(name),
                "arguments": _json_args(getattr(fn, "arguments", "{}")),
                "native": True,
            })
    if calls:
        return calls
    for raw in _TOOL_RE.findall(text or ""):
        try:
            obj = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            calls.append({
                "id": "",
                "name": str(obj["name"]),
                "arguments": _json_args(obj.get("arguments")),
                "native": False,
            })

    if not calls:
        for candidate in re.findall(r"\{\s*[\"']name[\"']\s*:.*?\}\s*$", text or "", re.DOTALL):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(obj, dict) and obj.get("name"):
                calls.append({"name": str(obj["name"]), "arguments": _json_args(obj.get("arguments"))})
                break
    return calls

def _message_text(message: Any) -> str:

    content = getattr(message, "content", None)
    if content:
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    value = block.get("text")
                    if value:
                        parts.append(str(value))
                else:
                    value = getattr(block, "text", None)
                    if value:
                        parts.append(str(value))
            if parts:
                return "\n".join(parts)
        return str(content)
    return str(getattr(message, "reasoning_content", None) or "")

def _computer_command(args: dict[str, Any], screen_size: tuple[int, int]) -> str:
    action = str(args.get("action") or args.get("command") or "").strip().lower().replace(" ", "_")
    width, height = screen_size
    def scale(value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (int(float(value[0]) * width / 1280), int(float(value[1]) * height / 720))
        return None

    point = scale(args.get("coordinate"))
    if action in {"left_click", "click"}:
        return f"pyautogui.click({point[0]}, {point[1]})" if point else "pyautogui.click()"
    if action == "right_click":
        return f"pyautogui.rightClick({point[0]}, {point[1]})" if point else "pyautogui.rightClick()"
    if action == "double_click":
        return f"pyautogui.doubleClick({point[0]}, {point[1]})" if point else "pyautogui.doubleClick()"
    if action == "middle_click":
        return f"pyautogui.middleClick({point[0]}, {point[1]})" if point else "pyautogui.middleClick()"
    if action == "triple_click":
        return f"pyautogui.tripleClick({point[0]}, {point[1]})" if point else "pyautogui.tripleClick()"
    if action == "mouse_move":
        duration = float(args.get("duration") or 0.5)
        return f"pyautogui.moveTo({point[0]}, {point[1]}, duration={duration:g})" if point else "pyautogui.moveTo(0, 0)"
    if action == "left_mouse_down":
        return f"pyautogui.mouseDown({point[0]}, {point[1]})" if point else "pyautogui.mouseDown()"
    if action == "left_mouse_up":
        return f"pyautogui.mouseUp({point[0]}, {point[1]})" if point else "pyautogui.mouseUp()"
    if action == "left_press":
        if point:
            return f"pyautogui.mouseDown({point[0]}, {point[1]}); pyautogui.sleep(1); pyautogui.mouseUp({point[0]}, {point[1]})"
        return "pyautogui.mouseDown(); pyautogui.sleep(1); pyautogui.mouseUp()"
    if action == "left_click_drag":
        end = point or (0, 0)
        start = scale(args.get("start_coordinate"))
        duration = float(args.get("duration") or 0.5)
        prefix = f"pyautogui.moveTo({start[0]}, {start[1]}, duration={duration})" if start else ""
        drag = f"pyautogui.dragTo({end[0]}, {end[1]}, duration={duration})"
        return f"{prefix}; {drag}" if prefix else drag
    if action in {"scroll", "hscroll"}:
        amount = args.get("pixels", args.get("scroll_amount", 0))
        try:
            amount = float(amount or 0)
        except (TypeError, ValueError):
            amount = 0.0
        direction = str(args.get("scroll_direction") or "").lower()
        if direction in {"down", "left"}:
            amount = -abs(float(amount))
        elif direction in {"up", "right"}:
            amount = abs(float(amount))
        fn = "hscroll" if action == "hscroll" or direction in {"left", "right"} else "scroll"
        location = f", {point[0]}, {point[1]}" if point else ""
        return f"pyautogui.{fn}({amount:g}{location})"
    if action == "type":
        text = str(args.get("text") or "")

        if text.isascii():
            return f"pyautogui.write({text!r}, interval=0.03)"
        return f"import pyperclip; pyperclip.copy({text!r}); pyautogui.hotkey('ctrl', 'v')"
    if action == "key":
        keys = args.get("keys") or args.get("key") or args.get("text") or []
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+") if k.strip()]
        keys = [
            {"page_down": "pagedown", "page_up": "pageup", "escape": "esc", "super_l": "win"}.get(str(k).lower(), str(k).lower())
            for k in keys
        ]
        if not keys:
            logger.warning("Qwen computer key action had no key value; using no-op")
            return "pyautogui.sleep(0.1)"
        return f"pyautogui.hotkey({', '.join(repr(k) for k in keys)})" if len(keys) > 1 else f"pyautogui.press({keys[0]!r})"
    if action == "wait":

        duration = float(args.get("duration") or 1.0)
        return f"pyautogui.sleep({duration:g})"
    if action == "screenshot":
        return "pyautogui.sleep(0.1)"
    if action in {"terminate", "done", "finish"}:

        return "pyautogui.sleep(0.1)"
    return "pyautogui.sleep(0.1)"

def _trajectory_prompt(instruction: str, trajectory: dict[str, Any], count: int) -> str:

    actions = "\n".join(
        f"Step {idx + 1}: {str(action.get('raw_response', action))}"
        for idx, action in enumerate(trajectory.get("actions", []))
    ) or "None"
    return f"""You are evaluating a GUI automation task.
Task instruction:
{instruction}

Execution trajectory ({count} screenshots):
{actions}

The attached screenshots are the most recent observations (the last one is
the final screenshot). First perform a static assessment.
If it is inconclusive, use read-only screenshot, shell, or Python probes. Do
not alter user data while verifying the task."""

class QwenVagenEvalAgent:

    DEFAULT_TOOLS = ("execute_python", "execute_shell", "check_screenshot")

    def __init__(
        self,
        *,
        model: str,
        api_url: str,
        api_key: str = "EMPTY",
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
        top_p: float | None = 0.95,
        screen_size: tuple[int, int] = (1920, 1080),
        only_n_most_recent_images_eval: int = 10,
        enable_thinking: bool = False,
        max_retries: int = 3,
        format_retry_rounds: int = 2,
        request_timeout: float = 120.0,
    ):
        self.model = model
        self.request_timeout = float(request_timeout)
        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_url,
            timeout=self.request_timeout,
            max_retries=0,
        )
        self.max_tokens = int(max_tokens)
        self.temperature = temperature
        self.top_p = top_p
        self.screen_size = screen_size
        self.only_n_most_recent_images_eval = int(only_n_most_recent_images_eval)
        self.enable_thinking = bool(enable_thinking)
        self.max_retries = int(max_retries)
        self.format_retry_rounds = max(0, int(format_retry_rounds))

    @staticmethod
    def _system_prompt(max_round: int) -> str:
        return f"""You are a strict GUI task verifier. Follow the VAGEN evaluation procedure and inspect evidence before judging.
You may use exactly these textual tools. Emit one call per turn as XML containing JSON:
<tool_call>{{\"name\":\"computer_use\",\"arguments\":{{...}}}}</tool_call>
Tools:
- computer_use: action in key, type, mouse_move, left_click, left_click_drag, right_click, middle_click, double_click, scroll, hscroll, wait, terminate; coordinate is [x,y] in a 1280x720 screenshot.
- execute_python: arguments {{\"code\": \"...\"}} (read-only verification only).
- execute_shell: arguments {{\"command\": \"...\"}} (read-only verification only).
- check_screenshot: arguments {{\"image_names\": [\"step_N\"]}}.
Use no state-changing verification actions. You have at most {max_round} rounds.
When done, output exactly:
EVALUATION RESULT:
Status: SUCCESS or FAILURE
        Confidence: HIGH, MEDIUM, or LOW
        Reasoning: detailed evidence-based reasoning"""

    def _build_trajectory_prompt(
        self, instruction: str, trajectory: dict[str, Any], count: int
    ) -> str:

        return _trajectory_prompt(instruction, trajectory, count)

    def _accept_final_result(self, result: dict[str, Any]) -> bool:

        del result
        return True

    def _request_tools(self) -> list[dict[str, Any]]:

        return []

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

        screenshot: bytes | None = None
        screenshot_is_resized = False
        if name in {"computer_use", "computer"}:
            command = _computer_command(args, self.screen_size)
            obs, _, _, info = env.step({
                "name": "computer", "input": args,
                "action_type": "tool_use", "command": command,
            })
            screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
            result_text = str(info or "")
            if screenshot:
                probe_dir = os.path.join(save_dir, "eval_result")
                os.makedirs(probe_dir, exist_ok=True)
                with open(
                    os.path.join(probe_dir, f"round_{eval_steps}.png"), "wb"
                ) as image_file:
                    image_file.write(screenshot)
        elif name == "execute_python":
            obs, _, _, info = env.step({
                "action_type": "EXECUTE_PYTHON",
                "command": str(args.get("code", "")),
            })
            del obs
            result_text = str(info or "")
        elif name == "execute_shell":
            obs, _, _, info = env.step({
                "action_type": "EXECUTE_SHELL",
                "command": str(args.get("command", "")),
            })
            del obs
            result_text = str(info or "")
        elif name == "check_screenshot":
            names = args.get("image_names") or []
            if isinstance(names, str):
                names = [names]
            selected = image_map.get(str(names[0])) if names else None
            if selected is None:
                requested = ", ".join(str(value) for value in names) or "none"
                result_text = f"Screenshot(s) not found in trajectory: {requested}."
            else:
                result_text = f"Loaded screenshot {names[0]} from trajectory."
            screenshot = selected
            screenshot_is_resized = True
        else:
            result_text = f"Unknown tool: {name}"
        return result_text, screenshot, screenshot_is_resized

    def _request(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if tools:
            kwargs["tools"] = tools
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Qwen evaluator request failed after {self.max_retries} attempts: {last}")

    @staticmethod
    def _parse_final(text: str) -> dict[str, Any] | None:
        status = _STATUS_RE.search(text or "")
        if not status:
            return None
        confidence = _CONFIDENCE_RE.search(text or "")
        reason = _REASON_RE.search(text or "")
        return {
            "success": status.group(1).upper() == "SUCCESS",
            "score": 1.0 if status.group(1).upper() == "SUCCESS" else 0.0,
            "confidence": confidence.group(1).upper() if confidence else "LOW",
            "reason": reason.group(1).strip() if reason else text.strip(),
            "raw_output": text,
            "parse_method": "format",
        }

    def eval_task(self, env: Any, task_config: dict[str, Any], trajectory: dict[str, Any], save_dir: str, max_round: int = 12) -> dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        screenshots = trajectory.get("screenshots") or []
        if not screenshots:
            raise ValueError("Qwen VAGEN evaluation requires at least one trajectory screenshot")
        kept = screenshots[-self.only_n_most_recent_images_eval:] if self.only_n_most_recent_images_eval > 0 else screenshots
        actions = trajectory.get("actions") or []
        prompt = self._build_trajectory_prompt(
            task_config.get("instruction", ""), {"actions": actions}, len(screenshots)
        )
        initial_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        resized_screenshots = [_resize_png(data) for data in screenshots]
        for index, data in enumerate(kept, start=len(screenshots) - len(kept) + 1):
            initial_content.append({"type": "text", "text": f"Screenshot step_{index}:"})
            initial_content.append({"type": "image_url", "image_url": {"url": _data_url(resized_screenshots[index - 1])}})
        messages: list[dict[str, Any]] = [

            {"role": "system", "content": self._system_prompt(max_round)},
            {"role": "user", "content": initial_content},
        ]
        image_map = {f"step_{idx + 1}": data for idx, data in enumerate(resized_screenshots)}
        request_tools = self._request_tools()
        eval_steps = 0
        last_text = ""
        for _ in range(max_round):
            eval_steps += 1
            response = self._request(messages, tools=request_tools)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("Qwen evaluator response contained no choices")
            message = choices[0].message
            text = _message_text(message)
            last_text = text
            assistant_message: dict[str, Any] = {"role": "assistant", "content": text or None}
            native_tool_calls = getattr(message, "tool_calls", None) or []
            if native_tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": str(getattr(call, "id", "")),
                        "type": "function",
                        "function": {
                            "name": str(getattr(getattr(call, "function", None), "name", "")),
                            "arguments": str(getattr(getattr(call, "function", None), "arguments", "{}")),
                        },
                    }
                    for call in native_tool_calls
                ]
            messages.append(assistant_message)
            calls = _extract_tool_calls(message, text)
            final = self._parse_final(text)
            if final and not calls and self._accept_final_result(final):
                final["eval_steps"] = eval_steps
                return final
            if not calls:
                messages.append({"role": "user", "content": [{"type": "text", "text": "The evidence is insufficient for a confident judgment. Use exactly one low-risk read-only probe tool now, or provide the required final EVALUATION RESULT format only if the evidence is conclusive."}]})
                continue
            result_parts: list[dict[str, Any]] = []
            for call in calls[:1]:
                name, args = call["name"], call["arguments"]
                result_text, screenshot, screenshot_is_resized = self._execute_probe_tool(
                    name=name,
                    args=args,
                    env=env,
                    save_dir=save_dir,
                    eval_steps=eval_steps,
                    image_map=image_map,
                )
                image_part = None
                if screenshot:
                    image_data = screenshot if screenshot_is_resized else _resize_png(screenshot)
                    image_part = {"type": "image_url", "image_url": {"url": _data_url(image_data)}}
                if call.get("native"):

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result_text[:12000],
                    })
                    if image_part:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Current browser screenshot after the tool call."},
                                image_part,
                            ],
                        })
                else:
                    result_parts.append({"type": "text", "text": result_text[:12000]})
                    if image_part:
                        result_parts.append(image_part)
            if result_parts:
                messages.append({"role": "user", "content": result_parts})
        forced_text = last_text
        for retry_idx in range(self.format_retry_rounds + 1):
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "FORMAT ERROR. Do not call any tool and do not add analysis. Return exactly these four lines, replacing the placeholders: \nEVALUATION RESULT:\nStatus: SUCCESS or FAILURE\nConfidence: HIGH, MEDIUM, or LOW\nReasoning: one concise evidence-based explanation",
                }],
            })
            forced = self._request(messages)
            forced_choices = getattr(forced, "choices", None) or []
            if not forced_choices:
                raise RuntimeError("Qwen evaluator forced-final response contained no choices")
            forced_text = _message_text(forced_choices[0].message) or forced_text
            final = self._parse_final(forced_text)
            if final:

                final["eval_steps"] = eval_steps
                final["format_retry_attempts"] = retry_idx
                return final
            messages.append({
                "role": "assistant",
                "content": forced_text,
            })
        return {
            "success": False,
            "score": 0.0,
            "confidence": "LOW",
            "reason": forced_text,
            "raw_output": forced_text,
            "parse_method": "failed",
            "eval_steps": eval_steps,
            "format_retry_attempts": self.format_retry_rounds,
        }
