import base64
import json
import logging
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image

from mm_agents.base_agent import Agent
from mm_agents.config import Qwen35VLAgentConfig
from mm_agents.utils.qwen_vl_utils import smart_resize
from mm_agents.vlm import VLMWrapper


logger = None


def process_image(image_bytes: bytes) -> str:
    """Resize + re-encode screenshot and return base64 PNG."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize((resized_width, resized_height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@Agent.register("Qwen35VL")
class Qwen35VLAgent(Agent):
    """
    Lightweight Qwen3.5-VL agent.

    Characteristics:
    - OpenAI-compatible API only.
    - XML tool-call output format.
    - History truncation by `history_n`.
    - Old screenshot folding by `image_max` / `fold_size`.
    """

    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path=config_path, **kwargs)
        if config_path is not None:
            self.config = Qwen35VLAgentConfig.from_yaml(config_path)
        else:
            self.config = self._build_config_from_legacy_kwargs(kwargs)

        model_payload = self.config.model.model_dump() if self.config.model else {}
        model_payload.setdefault("top_p", 0.9)

        self.platform = self.config.platform
        self.model = VLMWrapper(**model_payload) if model_payload else None
        self.action_space = self.config.action_space
        self.observation_type = self.config.observation_type
        self.history_n = self.config.history_n
        self.add_thought_prefix = self.config.add_thought_prefix
        self.coordinate_type = self.config.coordinate_type
        self.api_backend = self.config.api_backend
        self.image_max = int(self.config.image_max)
        self.fold_size = int(self.config.fold_size)
        self.collapse_text = (
            self.config.collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        )

        if self.action_space != "pyautogui":
            raise ValueError("qwen35vl_agent only supports pyautogui action space")
        if self.observation_type != "screenshot":
            raise ValueError("qwen35vl_agent only supports screenshot observations")
        if self.api_backend != "openai":
            raise ValueError("qwen35vl_agent only supports OpenAI-compatible APIs")
        if self.image_max < 1:
            raise ValueError("image_max must be >= 1")
        if self.fold_size < 1:
            raise ValueError("fold_size must be >= 1")

        self.last_messages = None
        self.last_message_step = -1
        self.last_response = None
        self._reset_state()

    @staticmethod
    def _build_config_from_legacy_kwargs(
        kwargs: Dict,
    ) -> Qwen35VLAgentConfig:
        config_kwargs = dict(kwargs)
        raw_model = config_kwargs.pop("model", None)
        model_payload: Dict = {}

        if raw_model is not None:
            if hasattr(raw_model, "model_dump"):
                model_payload.update(raw_model.model_dump())
            elif isinstance(raw_model, dict):
                model_payload.update(raw_model)
            else:
                model_payload["model_name"] = str(raw_model)

        legacy_model_keys = [
            "model_name",
            "api_key",
            "base_url",
            "max_retry",
            "retry_waiting_seconds",
            "max_tokens",
            "temperature",
            "max_pixels",
            "policy",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "enable_thinking",
            "frequency_penalty",
        ]
        for key in legacy_model_keys:
            if key in config_kwargs:
                model_payload[key] = config_kwargs.pop(key)

        model_payload.setdefault("model_name", "qwen35-vl")
        model_payload.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "dummy"))
        model_payload.setdefault(
            "base_url",
            os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        )
        model_payload.setdefault("max_tokens", 32768)
        model_payload.setdefault("temperature", 0.0)
        model_payload.setdefault("top_p", 0.9)

        config_kwargs["model"] = model_payload
        return Qwen35VLAgentConfig(**config_kwargs)

    def _reset_state(self) -> None:
        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[Dict] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []
        self.folded_prefix_k = 0
        self.last_messages = None
        self.last_message_step = -1
        self.last_response = None

    @staticmethod
    def _py_string(text: str) -> str:
        return json.dumps("" if text is None else str(text), ensure_ascii=False)

    def _update_folding_state(self, total_screenshots: int) -> None:
        while (total_screenshots - self.folded_prefix_k) > self.image_max:
            self.folded_prefix_k += self.fold_size
        if self.folded_prefix_k > total_screenshots:
            self.folded_prefix_k = total_screenshots

    def _should_collapse_step(self, step_num_1based: int) -> bool:
        return step_num_1based <= self.folded_prefix_k

    def _wrap_tool_response(self, parts: List[Dict]) -> List[Dict]:
        return (
            [{"type": "text", "text": "<tool_response>\n"}]
            + parts
            + [{"type": "text", "text": "\n</tool_response>"}]
        )

    @staticmethod
    def _sanitize_messages_for_dump(messages: List[Dict]) -> List[Dict]:
        sanitized: List[Dict] = []
        for message in messages:
            cloned = {"role": message.get("role"), "content": []}
            for part in message.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ((part.get("image_url") or {}).get("url")) or ""
                    if url.startswith("data:image/"):
                        cloned["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url[:40] + "...<omitted>"},
                            }
                        )
                    else:
                        cloned["content"].append(part)
                else:
                    cloned["content"].append(part)
            sanitized.append(cloned)
        return sanitized

    def predict(
        self,
        instruction: str,
        obs: Dict,
        system_prompt_prefix: Optional[str] = None,
        time_list=None,
    ) -> Tuple[str, List[str]]:
        if time_list is not None:
            time_list.append(("inference_start", time.time()))
        if self.model is None:
            raise ValueError("Qwen35VLAgent requires a valid model config")

        screenshot_bytes = obs["screenshot"]
        current_step = len(self.actions)

        original_img = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = original_img.size

        processed_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
        processed_width, processed_height = processed_img.size

        self.screenshots.append(processed_b64)
        total_steps = len(self.screenshots)
        self._update_folding_state(total_steps)

        start_step = max(1, total_steps - self.history_n)

        previous_actions = [
            f"Step {i + 1}: {self.actions[i]}"
            for i in range(0, min(start_step - 1, len(self.actions)))
        ]
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if self.coordinate_type == "absolute"
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "scroll",
                                "hscroll",
                                "wait",
                                "terminate",
                                "answer",
                            ],
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`."},
                        "text": {
                            "type": "string",
                            "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                        },
                        "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                        "pixels": {"type": "number", "description": "Scroll amount."},
                        "time": {"type": "number", "description": "Seconds to wait."},
                        "status": {
                            "type": "string",
                            "description": "Task status for terminate.",
                            "enum": ["success", "failure"],
                        },
                    },
                },
            },
        }

        system_prompt = (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def)
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
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
            f"- Collapsed screenshots appear as text: {self.collapse_text}\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )
        if system_prompt_prefix:
            system_prompt = f"{system_prompt_prefix.strip()}\n\n{system_prompt}"

        instruction_prompt = (
            f"\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n"
            f"{previous_actions_str}"
        )

        messages: List[Dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        for step_num in range(start_step, total_steps + 1):
            is_first_turn = step_num == start_step
            is_collapsed = self._should_collapse_step(step_num)

            if is_collapsed:
                parts = [{"type": "text", "text": self.collapse_text}]
                if is_first_turn:
                    user_content = [{"type": "text", "text": instruction_prompt}]
                else:
                    user_content = self._wrap_tool_response(parts)
                messages.append({"role": "user", "content": user_content})
            else:
                img_url = f"data:image/png;base64,{self.screenshots[step_num - 1]}"
                if is_first_turn:
                    user_content = [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": instruction_prompt},
                    ]
                else:
                    user_content = self._wrap_tool_response(
                        [{"type": "image_url", "image_url": {"url": img_url}}]
                    )
                messages.append({"role": "user", "content": user_content})

            if step_num <= total_steps - 1 and (step_num - 1) < len(self.responses):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": self.responses[step_num - 1]}],
                    }
                )

        if time_list is not None:
            time_list.append(("prompt_built", time.time()))
        self.last_messages = messages
        self.last_message_step = current_step

        try:
            draft_dir = "./draft/message_cache"
            os.makedirs(draft_dir, exist_ok=True)
            message_file_path = os.path.join(
                draft_dir, f"qwen35vl_messages_step_{current_step}.json"
            )
            with open(message_file_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    self._sanitize_messages_for_dump(messages),
                    file_obj,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            if logger:
                logger.warning("[Qwen35VLAgent] failed to dump debug messages: %s", exc)

        if time_list is not None:
            time_list.append(("messages_saved", time.time()))

        response_obj = self.model.predict(messages)
        self.last_response = response_obj
        response = self._extract_content_text(response_obj.choices[0].message.content)

        if time_list is not None:
            time_list.append(("agent_inference", time.time()))
        if logger:
            logger.info("Qwen35VL Output: %s", response)
        self.responses.append(response or "")

        low_level_instruction, pyautogui_code = self.parse_response(
            response or "",
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )

        if logger:
            logger.info("Low level instruction: %s", low_level_instruction)
            logger.info("Pyautogui code: %s", pyautogui_code)

        self.actions.append(low_level_instruction)
        if time_list is not None:
            time_list.append(("response_parsed", time.time()))
        return response or "", pyautogui_code

    def parse_response(
        self,
        response: str,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ) -> Tuple[str, List[str]]:
        low_level_instruction = ""
        pyautogui_code: List[str] = []

        if not response or not response.strip():
            return low_level_instruction, pyautogui_code

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
            if not (original_width and original_height):
                return int(x), int(y)
            if self.coordinate_type == "absolute":
                if processed_width and processed_height:
                    x_scale = original_width / processed_width
                    y_scale = original_height / processed_height
                    return int(x * x_scale), int(y * y_scale)
                return int(x), int(y)
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
            params: Dict = {}
            func_match = re.search(r"<function=([^>]+)>", xml_content)
            if not func_match or func_match.group(1) != "computer_use":
                return None

            for match in re.finditer(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                xml_content,
                re.DOTALL,
            ):
                name = match.group(1)
                value = match.group(2).strip()
                if value.startswith("[") or value.startswith("{"):
                    try:
                        params[name] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                params[name] = value
            return params

        def parse_keys(raw_keys):
            if isinstance(raw_keys, str):
                try:
                    raw_keys = json.loads(raw_keys)
                except Exception:
                    raw_keys = [raw_keys]
            if isinstance(raw_keys, list):
                return [str(key).strip() for key in raw_keys]
            return [str(raw_keys).strip()]

        def parse_coordinate(raw_coord):
            if isinstance(raw_coord, str):
                try:
                    raw_coord = json.loads(raw_coord)
                except Exception:
                    return None
            if isinstance(raw_coord, list) and len(raw_coord) >= 2:
                return raw_coord[0], raw_coord[1]
            return None

        def process_tool_call_params(params: Dict) -> None:
            action = params.get("action")
            if not action:
                return

            coordinate = parse_coordinate(params.get("coordinate"))
            text = params.get("text")

            def press_modifier_keys() -> None:
                if text:
                    for key in str(text).split("+"):
                        key = key.strip().lower()
                        if key:
                            pyautogui_code.append(
                                f"pyautogui.keyDown({self._py_string(key)})"
                            )

            def release_modifier_keys() -> None:
                if text:
                    keys = [
                        key.strip().lower()
                        for key in str(text).split("+")
                        if key.strip()
                    ]
                    for key in reversed(keys):
                        pyautogui_code.append(
                            f"pyautogui.keyUp({self._py_string(key)})"
                        )

            if action == "left_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.click({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.click()")
                release_modifier_keys()
            elif action == "right_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.rightClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.rightClick()")
                release_modifier_keys()
            elif action == "middle_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.middleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.middleClick()")
                release_modifier_keys()
            elif action == "double_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()
            elif action == "triple_click":
                press_modifier_keys()
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.doubleClick({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
                release_modifier_keys()
            elif action == "type":
                text = params.get("text", "")
                pyautogui_code.append(
                    f"pyautogui.typewrite({self._py_string(text)})"
                )
            elif action == "key":
                keys = parse_keys(params.get("keys", []))
                keys_str = ", ".join(self._py_string(key) for key in keys)
                if len(keys) > 1:
                    pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                else:
                    pyautogui_code.append(f"pyautogui.press({keys_str})")
            elif action in {"scroll", "hscroll"}:
                press_modifier_keys()
                pixels = params.get("pixels", 0)
                try:
                    pixels = int(float(pixels))
                except Exception:
                    pixels = 0
                pyautogui_code.append(f"pyautogui.scroll({pixels})")
                release_modifier_keys()
            elif action == "wait":
                pyautogui_code.append("WAIT")
            elif action in {"terminate", "answer"}:
                pyautogui_code.append("DONE")
            elif action == "mouse_move":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    pyautogui_code.append(f"pyautogui.moveTo({x}, {y})")
                else:
                    pyautogui_code.append("pyautogui.moveTo(0, 0)")
            elif action == "left_click_drag":
                if coordinate:
                    x, y = adjust_coordinates(*coordinate)
                    duration = 0.5
                    if "duration" in params:
                        try:
                            duration = float(params["duration"])
                        except Exception:
                            duration = 0.5
                    pyautogui_code.append(
                        f"pyautogui.dragTo({x}, {y}, duration={duration})"
                    )
                else:
                    pyautogui_code.append("pyautogui.dragTo(0, 0)")

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                low_level_instruction = stripped.split("Action:", 1)[-1].strip()
                break

        for tool_call_match in re.finditer(
            r"<tool_call>(.*?)</tool_call>", response, re.DOTALL
        ):
            params = parse_xml_tool_call(tool_call_match.group(1))
            if params:
                process_tool_call_params(params)

        if not low_level_instruction and pyautogui_code:
            first_code = pyautogui_code[0]
            if first_code == "DONE":
                low_level_instruction = "Task completed"
            elif first_code == "WAIT":
                low_level_instruction = "Waiting"
            elif "." in first_code:
                low_level_instruction = (
                    f"Performing {first_code.split('.', 1)[1].split('(', 1)[0]} action"
                )
            else:
                low_level_instruction = "Performing action"

        return low_level_instruction, pyautogui_code

    @staticmethod
    def _extract_content_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(part.get("text", ""))
                else:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def reset(self, _logger=None, **kwargs):
        global logger
        logger = (
            _logger
            if _logger is not None
            else logging.getLogger("desktopenv.qwen35vl_agent")
        )
        self._reset_state()
