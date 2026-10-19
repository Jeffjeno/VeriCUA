from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai
from PIL import Image
from requests.exceptions import SSLError

from agent_runtime.harness import Harness
from agent_runtime.types import Action, HarnessContext, Observation

from agent_runtime.harnesses.qwen35vl import (
    _ensure_mm_agents_importable,
    _get_screenshot_bytes,
    _pyautogui_code_to_action,
)

logger = logging.getLogger(__name__)

def _smart_resize(height: int, width: int, factor: int = 32, max_pixels: int = 16 * 16 * 4 * 12800) -> Tuple[int, int]:
    _ensure_mm_agents_importable()
    from mm_agents.utils.qwen_vl_utils import smart_resize
    return smart_resize(height=height, width=width, factor=factor, max_pixels=max_pixels)

def _process_image(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes))
    w, h = image.size
    rh, rw = _smart_resize(height=h, width=w, factor=32, max_pixels=16 * 16 * 4 * 12800)
    image = image.resize((rw, rh))
    buf = BytesIO()
    image.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    return Image.open(BytesIO(image_bytes)).size

def _image_size_from_base64(b64: str) -> Tuple[int, int]:
    import base64 as _b64
    return Image.open(BytesIO(_b64.b64decode(b64))).size

def _adjust_coordinates(
    x: float, y: float,
    *, coordinate_type: str,
    original_width: int = None, original_height: int = None,
    processed_width: int = None, processed_height: int = None,
) -> Tuple[int, int]:
    if not (original_width and original_height):
        return int(x), int(y)
    if coordinate_type == "absolute":
        if processed_width and processed_height:
            return int(x * original_width / processed_width), int(y * original_height / processed_height)
        return int(x), int(y)

    return int(x * original_width / 999), int(y * original_height / 999)

_MAX_RETRY = int(os.getenv("OSWORLD_MAX_RETRY_TIMES", "5"))

def _extract_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else (getattr(p, "text", None) or "")
            for p in content
        )
    return str(content)

def _merge_reasoning_content(content, reasoning_content) -> str:
    text = _extract_content_text(content)
    think = _extract_content_text(reasoning_content).strip()
    if not think:
        return text
    return f"<think>\n{think}\n</think>\n\n{text.lstrip()}"

def _call_openai_compatible(
    payload: Dict,
    model: str,
    *,
    base_url: Optional[str],
    api_key: Optional[str],
    default_max_tokens: int,
    default_temperature: float,
    default_top_p: float,
) -> str:
    resolved_base = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    timeout = float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", "130"))

    try:
        client = openai.OpenAI(base_url=resolved_base, api_key=resolved_key, timeout=timeout)
    except TypeError:
        client = openai.OpenAI(base_url=resolved_base, api_key=resolved_key)

    retryable = tuple(
        e for e in [
            SSLError,
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "BadRequestError", None),
            getattr(openai, "InternalServerError", None),
        ] if isinstance(e, type)
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRY + 1):
        try:
            kw: Dict[str, Any] = dict(
                model=model,
                messages=payload["messages"],
                max_tokens=payload.get("max_tokens", default_max_tokens),
                temperature=payload.get("temperature", default_temperature),
                top_p=payload.get("top_p", default_top_p),
            )
            extra_body = payload.get("extra_body")
            if extra_body:
                kw["extra_body"] = extra_body
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            reasoning = getattr(msg, "reasoning_content", None)
            return _merge_reasoning_content(content, reasoning)
        except retryable as exc:
            last_err = exc
            logger.warning("[QwenRefAgent] attempt %d/%d failed: %s", attempt, _MAX_RETRY, exc)
            time.sleep(min(5.0 * attempt, 30.0))

    if last_err is not None:
        raise last_err
    return ""

def _parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
    params: Dict = {}
    m = re.search(r"<function=([^>]+)>", xml_content)
    if not m or m.group(1) != "computer_use":
        return None
    for pm in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL):
        name, value = pm.group(1), pm.group(2).strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return params

def _iter_tool_call_params(response: str):
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        p = _parse_xml_tool_call(m.group(1))
        if p:
            yield p

def _parse_coordinate(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, list) and len(raw) >= 2:
        return raw[0], raw[1]
    return None

def _parse_keys(raw_keys, *, lowercase: bool = False) -> List[str]:
    if isinstance(raw_keys, str):
        try:
            raw_keys = json.loads(raw_keys)
        except Exception:
            try:
                raw_keys = ast.literal_eval(raw_keys)
            except Exception:
                pass

    def clean(k):
        return str(k).strip(" \t\r\n[](){}\"'")

    def flatten(obj) -> List[str]:
        if obj is None:
            return []
        if isinstance(obj, list):
            out = []
            for i in obj:
                out.extend(flatten(i))
            return out
        parts = []
        for part in re.split(r"\s*\+\s*", str(obj).strip()):
            c = clean(part)
            if c:
                parts.append(c)
        return parts

    keys = flatten(raw_keys)
    return [k.lower() for k in keys] if lowercase else keys

def _parse_number(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)

def _extract_action_line(response: str) -> str:
    for line in response.split("\n"):
        s = line.strip()
        if s.lower().startswith("action:"):
            return s.split(":", 1)[-1].strip()
    return ""

def _looks_infeasible(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in [
        "[infeasible]", "infeasible", "not possible", "impossible", "not feasible",
        "cannot be completed", "can't be completed", "cannot be done", "cannot complete",
        "can't complete", "unable to complete",
    ]):
        return True
    return False

def _py_string(text: str) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)

def _termination_code(status) -> str:
    n = str(status or "success").strip().lower()
    return "FAIL" if n in {"fail", "failed", "failure", "error", "infeasible"} else "DONE"

def _parse_internal_response(
    response: str,
    *,
    coordinate_type: str,
    original_width: int = None,
    original_height: int = None,
    processed_width: int = None,
    processed_height: int = None,
) -> Tuple[str, List[str]]:

    low_level = ""
    codes: List[str] = []

    if not response or not response.strip():
        return low_level, codes

    infeasible = _looks_infeasible(response)

    def adjust(x, y):
        return _adjust_coordinates(
            x, y,
            coordinate_type=coordinate_type,
            original_width=original_width, original_height=original_height,
            processed_width=processed_width, processed_height=processed_height,
        )

    def handle(params: Dict) -> None:
        action = params.get("action")
        if not action:
            return
        coord = _parse_coordinate(params.get("coordinate"))

        if action == "left_click":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.click({x}, {y})")
            else:
                codes.append("pyautogui.click()")
        elif action == "right_click":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.rightClick({x}, {y})")
            else:
                codes.append("pyautogui.rightClick()")
        elif action == "middle_click":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.middleClick({x}, {y})")
            else:
                codes.append("pyautogui.middleClick()")
        elif action == "double_click":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                codes.append("pyautogui.doubleClick()")
        elif action == "triple_click":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.click({x}, {y}, clicks=3)")
            else:
                codes.append("pyautogui.click(clicks=3)")
        elif action == "type":
            text = "" if params.get("text") is None else str(params.get("text", ""))
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            if "\n" not in normalized:
                codes.append(f"pyautogui.typewrite({_py_string(normalized)})")
            else:
                chunks = normalized.split("\n")
                for idx, chunk in enumerate(chunks):
                    if chunk:
                        codes.append(f"pyautogui.typewrite({_py_string(chunk)})")
                    if idx < len(chunks) - 1:
                        codes.append(f"pyautogui.press({_py_string('enter')})")
        elif action == "key":
            keys = _parse_keys(params.get("keys", []), lowercase=True)
            if not keys:
                return
            ks = ", ".join(_py_string(k) for k in keys)
            codes.append(f"pyautogui.hotkey({ks})" if len(keys) > 1 else f"pyautogui.press({ks})")
        elif action == "key_down":
            for k in _parse_keys(params.get("keys", []), lowercase=True):
                codes.append(f"pyautogui.keyDown({_py_string(k)})")
        elif action == "key_up":
            for k in _parse_keys(params.get("keys", []), lowercase=True):
                codes.append(f"pyautogui.keyUp({_py_string(k)})")
        elif action == "scroll":
            px = int(_parse_number(params.get("pixels", 0), default=0))
            codes.append(f"pyautogui.scroll({px})")
        elif action == "hscroll":
            px = int(_parse_number(params.get("pixels", 0), default=0))
            codes.append(f"pyautogui.scroll({px})")
        elif action == "mouse_move":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.moveTo({x}, {y})")
            else:
                codes.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coord:
                x, y = adjust(*coord)
                dur = _parse_number(params.get("duration", 0.5), default=0.5)
                codes.append(f"pyautogui.dragTo({x}, {y}, duration={dur})")
            else:
                codes.append("pyautogui.dragTo(0, 0)")
        elif action == "left_mouse_down":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.moveTo({x}, {y})")
            codes.append("pyautogui.mouseDown(button='left')")
        elif action == "left_mouse_up":
            if coord:
                x, y = adjust(*coord)
                codes.append(f"pyautogui.moveTo({x}, {y})")
            codes.append("pyautogui.mouseUp(button='left')")
        elif action == "wait":
            codes.append("WAIT")
        elif action == "screenshot":
            codes.append("WAIT")
        elif action == "terminate":
            codes.append(_termination_code(params.get("status", "success")))
        elif action == "call_user":
            codes.append("FAIL" if infeasible else "DONE")
        elif action == "answer":
            codes.append("DONE")

    low_level = _extract_action_line(response)
    for p in _iter_tool_call_params(response):
        handle(p)

    if not codes:
        codes.append("FAIL" if infeasible else "DONE")

    return low_level, codes

_INTERNAL_ACTION_DESC = """\
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation."""

def _build_description_prompt(processed_width: int, processed_height: int, coordinate_type: str) -> str:
    resolution = (
        f"* The screen's resolution is {processed_width}x{processed_height}."
        if coordinate_type == "absolute"
        else "* The screen's resolution is 1000x1000."
    )
    return "\n".join([
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
        resolution,
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ])

def _build_internal_tools_def(processed_width: int, processed_height: int, coordinate_type: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": _build_description_prompt(processed_width, processed_height, coordinate_type),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": _INTERNAL_ACTION_DESC,
                        "enum": [
                            "key", "key_down", "key_up", "left_mouse_down", "left_mouse_up",
                            "type", "mouse_move", "left_click", "left_click_drag", "right_click",
                            "middle_click", "double_click", "triple_click", "scroll", "hscroll",
                            "screenshot", "wait", "terminate", "call_user",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`."},
                    "text": {"type": "string", "description": "Required only by `action=type` and `action=call_user`."},
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {"type": "string", "description": "Task status for terminate.", "enum": ["success", "failure"]},
                },
            },
        },
    }

def _build_internal_system_prompt(tools_def: Dict, collapse_text: str) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def, ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "For normal UI interaction steps:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "For terminal steps, you may either:\n"
        "- output a final natural-language response with no tool call, or\n"
        "- use a terminal tool call such as call_user or terminate.\n\n"
        "Rules:\n"
        "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything after a tool call.\n"
        "- Use call_user when you need user information or confirmation.\n"
        "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
        "- If the task is infeasible, say so explicitly in the response."
    )

def _build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return (
        "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n"
        f"{previous_actions_str}"
    )

_COLLAPSE_TEXT_DEFAULT = "This screenshot has been collapsed."

def _ensure_empty_think_prefix(response: str) -> str:
    text = response or ""
    if re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL):
        return text
    return "<think>\n\n</think>\n\n" + text.lstrip("\n")

def _wrap_tool_response(parts: List[Dict]) -> List[Dict]:
    return (
        [{"type": "text", "text": "<tool_response>\n"}]
        + parts
        + [{"type": "text", "text": "\n</tool_response>"}]
    )

def _update_folding_state(total: int, folded: int, image_max: int, fold_size: int) -> int:
    while (total - folded) > image_max:
        folded += fold_size
    return min(folded, total)

def _build_messages(
    *,
    system_prompt: str,
    instruction_prompt: str,
    screenshots: List[str],
    responses: List[str],
    start_step: int,
    total_steps: int,
    folded_prefix_k: int,
    collapse_text: str,
    enable_thinking: bool = False,
) -> List[Dict]:
    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first = step_num == start_step
        is_current = step_num == total_steps
        is_collapsed = step_num <= folded_prefix_k

        if is_collapsed:
            user_content = (
                [{"type": "text", "text": instruction_prompt}]
                if is_first
                else _wrap_tool_response([{"type": "text", "text": collapse_text}])
            )
        else:
            img_url = f"data:image/png;base64,{screenshots[step_num - 1]}"
            if is_first:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = _wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )

        messages.append({"role": "user", "content": user_content})

        if not is_current:
            resp_idx = step_num - 1
            if resp_idx < len(responses):
                resp_text = responses[resp_idx]
                if enable_thinking:
                    resp_text = _ensure_empty_think_prefix(resp_text)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": resp_text}],
                })

    return messages

class OfficialQwenAgent:

    COLLAPSE_TEXT = _COLLAPSE_TEXT_DEFAULT

    def __init__(
        self,
        model: str = "Qwen3-VL-8B-Thinking",
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        coordinate_type: str = "relative",
        history_n: int = 100,
        image_max: int = 20,
        fold_size: int = 10,
        enable_thinking: bool = True,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.coordinate_type = coordinate_type
        self.history_n = history_n
        self.image_max = image_max
        self.fold_size = fold_size
        self.enable_thinking = enable_thinking
        self.base_url = base_url
        self.api_key = api_key

        self.screenshots: List[str] = []
        self.responses: List[str] = []
        self.actions: List[str] = []
        self.folded_prefix_k: int = 0
        self.last_messages: Optional[List[Dict]] = None

    def reset(self, _logger=None, *args, **kwargs):
        global logger
        if _logger is not None:
            logger = _logger
        self.screenshots = []
        self.responses = []
        self.actions = []
        self.folded_prefix_k = 0
        self.last_messages = None

    def _build_payload(self, messages: List[Dict]) -> Dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def call_llm(self, payload: Dict) -> str:
        return _call_openai_compatible(
            payload,
            payload["model"],
            base_url=self.base_url,
            api_key=self.api_key,
            default_max_tokens=self.max_tokens,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
        )

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        screenshot_bytes = obs["screenshot"]
        original_width, original_height = _image_size_from_bytes(screenshot_bytes)
        processed_b64 = _process_image(screenshot_bytes)
        processed_width, processed_height = _image_size_from_base64(processed_b64)

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)

        self.folded_prefix_k = _update_folding_state(
            total_steps, self.folded_prefix_k, self.image_max, self.fold_size
        )

        start_step = max(1, total_steps - self.history_n)
        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}"
            for i in range(start_step - 1)
            if i < len(self.actions)
        ]
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        tools_def = _build_internal_tools_def(processed_width, processed_height, self.coordinate_type)
        system_prompt = _build_internal_system_prompt(tools_def, self.COLLAPSE_TEXT)
        instruction_prompt = _build_instruction_prompt(instruction, previous_actions_str)

        messages = _build_messages(
            system_prompt=system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=self.screenshots,
            responses=self.responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.COLLAPSE_TEXT,
            enable_thinking=self.enable_thinking,
        )

        self.last_messages = messages
        payload = self._build_payload(messages)
        response = self.call_llm(payload)
        logger.info("[QwenRefAgent] response: %s", response[:200])
        self.responses.append(response or "")

        low_level, pyautogui_codes = _parse_internal_response(
            response or "",
            coordinate_type=self.coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        self.actions.append(low_level)
        return response or "", pyautogui_codes

class Qwen3VLRefHarness(Harness):

    def __init__(
        self,
        model: str = "Qwen3-VL-8B-Thinking",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 0.9,
        history_n: int = 100,
        image_max: int = 20,
        fold_size: int = 10,
        coordinate_type: str = "relative",
        enable_thinking: bool = True,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> None:
        self.logprobs = logprobs
        self._pending_actions: list[Action] = []

        self._agent = OfficialQwenAgent(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            coordinate_type=coordinate_type,
            history_n=history_n,
            image_max=image_max,
            fold_size=fold_size,
            enable_thinking=enable_thinking,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
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

        started_at = time.time()
        response_text, pyautogui_codes = self._agent.predict(instruction, {"screenshot": screenshot_bytes})
        elapsed = time.time() - started_at

        actions = [_pyautogui_code_to_action(code, raw=response_text) for code in pyautogui_codes]
        if not actions:
            actions.append(Action.noop(raw=response_text))

        first_action = actions[0]
        self._pending_actions.extend(actions[1:])

        step_index = context.step_index if context else 0
        metadata: dict[str, Any] = {
            "step_index": step_index,
            "agent": "qwen3vl-ref",
            "model_time": elapsed,
            "response_text": response_text,
        }
        if self.logprobs:
            metadata["last_messages"] = self._agent.last_messages
        first_action.metadata.setdefault("qwen3vl_ref", {}).update(metadata)
        return first_action
