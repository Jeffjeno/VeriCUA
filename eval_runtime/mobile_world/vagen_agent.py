from __future__ import annotations

import base64
import datetime
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from eval_runtime.vagen_active.qwen_eval import _TOOL_RE, _json_args

from .env_client import MobileWorldEvalEnvClient, validate_mobile_action

logger = logging.getLogger("eval_runtime.mobile_world.vagen_agent")

def _xml_tool_use_blocks(text: str) -> list[dict[str, Any]]:

    blocks: list[dict[str, Any]] = []
    for index, raw in enumerate(_TOOL_RE.findall(text or "")):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not payload.get("name"):
            continue
        blocks.append(
            {
                "type": "tool_use",
                "id": f"xml_tool_{index}",
                "name": str(payload["name"]),
                "input": _json_args(payload.get("arguments")),
            }
        )
    return blocks

MODEL_VIEW_WIDTH = 648
MODEL_VIEW_HEIGHT = 1440

MOBILE_USE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": (
            "Inspect the live Android environment with native MobileWorld actions. "
            "Every screenshot shown to you is exactly 648x1440. Supply click and "
            "drag coordinates in that 648x1440 screenshot coordinate system; the "
            "tool scales them to the native Android device. Use low-risk navigation "
            "to verify state; do not redo the user's task. For scroll or swipe, "
            "pass a direction only. For a coordinate gesture, use drag with "
            "start/end coordinates instead of swipe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": (
                        "Android action. click/double_tap/long_press require x,y; "
                        "scroll/swipe require direction; drag requires start_x, "
                        "start_y, end_x, end_y; open_app requires app_name."
                    ),
                    "enum": [
                        "click",
                        "double_tap",
                        "long_press",
                        "input_text",
                        "navigate_back",
                        "navigate_home",
                        "keyboard_enter",
                        "open_app",
                        "scroll",
                        "swipe",
                        "drag",
                        "wait",
                    ],
                },
                "x": {
                    "type": "integer",
                    "description": "X coordinate in the displayed 648x1440 screenshot.",
                    "minimum": 0,
                    "maximum": 647,
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate in the displayed 648x1440 screenshot.",
                    "minimum": 0,
                    "maximum": 1439,
                },
                "text": {"type": "string"},
                "app_name": {"type": "string"},
                "direction": {
                    "type": "string",
                    "description": "Required for scroll and swipe.",
                    "enum": ["left", "right", "up", "down"],
                },
                "start_x": {
                    "type": "integer",
                    "description": (
                        "Drag start x in the 648x1440 screenshot; never use with swipe."
                    ),
                    "minimum": 0,
                    "maximum": 647,
                },
                "start_y": {
                    "type": "integer",
                    "description": (
                        "Drag start y in the 648x1440 screenshot; never use with swipe."
                    ),
                    "minimum": 0,
                    "maximum": 1439,
                },
                "end_x": {
                    "type": "integer",
                    "description": (
                        "Drag end x in the 648x1440 screenshot; never use with swipe."
                    ),
                    "minimum": 0,
                    "maximum": 647,
                },
                "end_y": {
                    "type": "integer",
                    "description": (
                        "Drag end y in the 648x1440 screenshot; never use with swipe."
                    ),
                    "minimum": 0,
                    "maximum": 1439,
                },
            },
            "required": ["action_type"],
            "additionalProperties": False,
        },
    },
}

CHECK_SCREENSHOT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_screenshot",
        "description": (
            "Load one or more trajectory screenshots by step name, for example "
            "step_1 or step_7."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                }
            },
            "required": ["image_names"],
            "additionalProperties": False,
        },
    },
}

def _build_mobile_eval_prompt(
    instruction: str,
    trajectory: dict[str, Any],
    screenshot_count: int,
) -> str:
    actions_summary = "\n".join(
        f"Step {index + 1}: {action.get('raw_response', action)}"
        for index, action in enumerate(trajectory.get("actions", []))
    )
    return f"""You are an expert evaluator for Android GUI automation tasks. Determine whether the task was successfully completed from the execution trajectory and the live MobileWorld environment.

Available tools:
1. `check_screenshot`: inspect up to three trajectory screenshots such as step_1 or step_7.
2. `mobile_use`: inspect the live Android device using native Android actions. Every image you see is exactly 648x1440; provide x/y and drag coordinates in that displayed-image coordinate system. The tool automatically scales them to the 1080x2400 device. Use this only for low-risk verification and never redo the task.

Progressive verification protocol:
1. First assess only the trajectory, action history, and final screenshot. If evidence is sufficient, immediately provide the final judgment.
2. If inconclusive, inspect only the most relevant intermediate screenshots with `check_screenshot`.
3. Only if still inconclusive, use low-risk `mobile_use` actions to reveal current state. Do not create, delete, send, submit, purchase, or otherwise mutate user data while verifying.
4. Re-read the latest screenshot before every coordinate action. Never repeat an unchanged click more than once; switch strategy or conclude if an action has no effect.
5. If the trajectory already proves that a required step was omitted or failed, immediately return FAILURE. Do not investigate how to complete the task, retrieve information the policy failed to retrieve, or reconstruct the missing workflow.

Judge strictly whether the full objective was achieved, whether any required part is incomplete, and whether the final state matches the instruction.

Final judgment format (required):
EVALUATION RESULT:
Reasoning: Your detailed reasoning explaining why the task succeeded or failed
Status: SUCCESS or FAILURE
Confidence: HIGH or MEDIUM or LOW

Use actual API tool calls, not textual imitations.

Task Instruction:
{instruction}

Execution Trajectory:
Total steps: {screenshot_count}
Actions taken:
{actions_summary}

Last Screenshot (step {screenshot_count}):"""

class MobileWorldClaudeEvalAgent:

    DEFAULT_TOOLS: list[str] = []

    def __init__(
        self,
        *,
        platform: str,
        model: str,
        max_tokens: int,
        api_url: str,
        api_key: str,
        system_prompt_suffix: str = "",
        only_n_most_recent_images_eval: int = 10,
        temperature: float | None = None,
        top_p: float | None = None,
        screen_size: tuple[int, int] = (1080, 2400),
        request_timeout: float = 120.0,
        **_: Any,
    ) -> None:
        self.platform = platform
        self.model_name = model
        self.max_tokens = int(max_tokens)
        self.api_url = api_url
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.only_n_most_recent_images_eval = int(only_n_most_recent_images_eval)
        self.temperature = temperature
        self.top_p = top_p
        self.native_screen_width = int(screen_size[0])
        self.native_screen_height = int(screen_size[1])
        self.request_timeout = float(request_timeout)
        self.eval_messages: list[dict[str, Any]] = []
        self.eval_image_names: list[str] = []

    def _get_sampling_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        return params

    @staticmethod
    def _encode_model_view_image(image: Image.Image) -> str:
        prepared = image.convert("RGB").resize(
            (MODEL_VIEW_WIDTH, MODEL_VIEW_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        prepared.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _model_action_to_native(
        self,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        action = dict(tool_input)
        x_scale = self.native_screen_width / MODEL_VIEW_WIDTH
        y_scale = self.native_screen_height / MODEL_VIEW_HEIGHT
        for key in ("x", "start_x", "end_x"):
            if key in action and action[key] is not None:
                action[key] = min(
                    self.native_screen_width - 1,
                    max(0, round(float(action[key]) * x_scale)),
                )
        for key in ("y", "start_y", "end_y"):
            if key in action and action[key] is not None:
                action[key] = min(
                    self.native_screen_height - 1,
                    max(0, round(float(action[key]) * y_scale)),
                )
        return action

    @staticmethod
    def _convert_messages_to_openai_format(
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if system_prompt:
            output.append({"role": "system", "content": system_prompt})
        for message in messages:
            role = message["role"]
            content = message.get("content") or []
            if role == "assistant":
                text = "".join(
                    str(item.get("text") or "")
                    for item in content
                    if item.get("type") == "text"
                )
                tool_calls = [
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(
                                item.get("input") or {},
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for item in content
                    if item.get("type") == "tool_use"
                ]
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": text or None,
                }
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                output.append(assistant)
                continue

            user_parts: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for item in content:
                item_type = item.get("type")
                if item_type == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": item.get("tool_use_id", ""),
                            "content": str(item.get("content") or ""),
                        }
                    )
                elif item_type == "text":
                    user_parts.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    source = item.get("source") or {}
                    user_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{source.get('media_type', 'image/png')};"
                                    f"base64,{source.get('data', '')}"
                                ),
                                "detail": "high",
                            },
                        }
                    )
            output.extend(tool_results)
            if user_parts:
                output.append({"role": "user", "content": user_parts})
        return output

    @staticmethod
    def _response_blocks(response_json: dict[str, Any]) -> list[dict[str, Any]]:
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError(f"VAGEN response has no choices: {response_json!r}")
        message = choices[0].get("message") or {}
        blocks: list[dict[str, Any]] = []
        native_tool_calls = message.get("tool_calls") or []
        if message.get("content"):
            text = str(message["content"])
            blocks.append({"type": "text", "text": text})

            if not native_tool_calls:
                blocks.extend(_xml_tool_use_blocks(text))
        for tool_call in native_tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "input": arguments,
                }
            )
        return blocks

    def _trim_images(self, keep: int) -> None:
        image_locations: list[tuple[int, int]] = []
        for message_index, message in enumerate(self.eval_messages):
            for item_index, item in enumerate(message.get("content") or []):
                if item.get("type") == "image":
                    image_locations.append((message_index, item_index))
        remove_count = max(0, len(image_locations) - max(0, keep))
        to_remove = set(image_locations[:remove_count])
        for message_index, message in enumerate(self.eval_messages):
            message["content"] = [
                item
                for item_index, item in enumerate(message.get("content") or [])
                if (message_index, item_index) not in to_remove
            ]

    @staticmethod
    def _read_key_images(
        tool_input: dict[str, Any],
        image_map: dict[str, str],
    ) -> dict[str, Any]:
        requested = tool_input.get("image_names") or []
        images = []
        loaded = []
        for name in requested[:3]:
            path = image_map.get(str(name))
            if path and Path(path).is_file():
                images.append(Image.open(path))
                loaded.append(str(name))
        if not images:
            return {
                "success": False,
                "text": (
                    "No requested screenshots were found. Available names: "
                    f"{sorted(image_map)}"
                ),
                "images": [],
            }
        return {
            "success": True,
            "text": f"Loaded trajectory screenshots: {', '.join(loaded)}",
            "images": images,
        }

    def _add_tool_result_to_eval_messages(
        self,
        *,
        tool_id: str,
        tool_result: dict[str, Any],
    ) -> None:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(tool_result.get("text") or "Success"),
            }
        ]
        for image in tool_result.get("images") or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_model_view_image(image),
                    },
                }
            )
        if tool_result.get("screenshot"):
            with Image.open(io.BytesIO(tool_result["screenshot"])) as screenshot_image:
                screenshot_data = self._encode_model_view_image(screenshot_image)
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_data,
                    },
                }
            )
        new_images = sum(item.get("type") == "image" for item in content)
        self._trim_images(max(self.only_n_most_recent_images_eval - new_images, 0))
        self.eval_messages.append({"role": "user", "content": content})

    @staticmethod
    def _parse_eval_result(reasoning_text: str, eval_steps: int) -> dict[str, Any]:
        if not reasoning_text:
            return {
                "success": False,
                "score": 0.0,
                "reason": "No reasoning provided by evaluator",
                "eval_steps": eval_steps,
                "confidence": "UNKNOWN",
                "parse_method": "empty",
            }
        success: bool | None = None
        confidence = "UNKNOWN"
        reasoning = reasoning_text
        parse_method = "failed"
        full_match = re.search(
            r"EVALUATION RESULT:\s*Reasoning:\s*(.+?)\s*"
            r"Status:\s*(SUCCESS|FAILURE)\s*"
            r"Confidence:\s*(HIGH|MEDIUM|LOW)",
            reasoning_text,
            re.IGNORECASE | re.DOTALL,
        )
        if full_match:
            reasoning = full_match.group(1).strip()
            success = full_match.group(2).upper() == "SUCCESS"
            confidence = full_match.group(3).upper()
            parse_method = "full_format"
        else:
            status_match = re.search(
                r"Status:\s*(SUCCESS|FAILURE)",
                reasoning_text,
                re.IGNORECASE,
            )
            if status_match:
                success = status_match.group(1).upper() == "SUCCESS"
                parse_method = "partial_format"
                confidence_match = re.search(
                    r"Confidence:\s*(HIGH|MEDIUM|LOW)",
                    reasoning_text,
                    re.IGNORECASE,
                )
                confidence = (
                    confidence_match.group(1).upper()
                    if confidence_match
                    else "MEDIUM"
                )
                reasoning_match = re.search(
                    r"Reasoning:\s*(.+?)(?=\n\n|\Z)",
                    reasoning_text,
                    re.IGNORECASE | re.DOTALL,
                )
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
        if success is None:
            return {
                "success": False,
                "score": 0.0,
                "reason": (
                    "Failed to parse evaluation result. The evaluator did not "
                    "provide output in the required format."
                ),
                "eval_steps": eval_steps,
                "confidence": "UNKNOWN",
                "parse_method": "failed",
                "raw_output": reasoning_text,
            }
        return {
            "success": success,
            "score": 1.0 if success else 0.0,
            "reason": reasoning,
            "eval_steps": eval_steps,
            "confidence": confidence,
            "parse_method": parse_method,
            "raw_output": reasoning_text,
        }

    def _save_eval_log(
        self,
        save_dir: str,
        messages: list[dict[str, Any]],
        result: dict[str, Any],
        task_config: dict[str, Any],
    ) -> None:
        simplified: list[dict[str, Any]] = []
        for message in messages:
            content = []
            for item in message.get("content") or []:
                if item.get("type") == "image":
                    content.append({"type": "image", "note": "[image_data_removed]"})
                else:
                    content.append(item)
            simplified.append({"role": message["role"], "content": content})
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cua_eval_log.json").write_text(
            json.dumps(
                {
                    "task_instruction": task_config.get("instruction", "N/A"),
                    "task_id": task_config.get("id", "N/A"),
                    "evaluation_result": result,
                    "conversation_history": simplified,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model_name,
                    "active_tool": "mobile_use",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "cua_eval_summary.txt").write_text(
            (
                f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}\n"
                f"Score: {result['score']}\n"
                f"Confidence: {result.get('confidence', 'UNKNOWN')}\n"
                f"Parse Method: {result.get('parse_method', 'unknown')}\n"
                f"Evaluation Steps: {result['eval_steps']}\n"
                f"Reasoning:\n{result['reason']}\n"
            ),
            encoding="utf-8",
        )

    def _call_eval_model(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        task_id: str,
        round_number: int,
        max_retries: int = 5,
    ) -> Any:
        openai_messages = self._convert_messages_to_openai_format(
            self.eval_messages,
            system_prompt,
        )
        payload: dict[str, Any] = {
            "model": self.model_name.replace("-openai", ""),
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }
        if hasattr(self, "enable_thinking"):
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(getattr(self, "enable_thinking"))
            }
        if tools:
            payload["tools"] = tools
        payload.update(self._get_sampling_params())
        trace_id = (
            f"{task_id}_mobile_eval_{round_number}_{int(time.time() * 1000)}"
        )
        tool_names = [
            str((tool.get("function") or {}).get("name") or tool.get("name") or "")
            for tool in tools
        ]
        image_count = sum(
            part.get("type") == "image_url"
            for message in openai_messages
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
        )
        logger.info(
            "VAGEN model request task=%s round=%d model=%s url=%s "
            "messages=%d images=%d tools=%s temperature=%s top_p=%s trace_id=%s",
            task_id,
            round_number,
            payload["model"],
            self.api_url,
            len(openai_messages),
            image_count,
            tool_names,
            payload.get("temperature"),
            payload.get("top_p"),
            trace_id,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "M-TraceId": trace_id,
        }
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code == 200:
                    response_json = response.json()
                    blocks = self._response_blocks(response_json)
                    choice = (response_json.get("choices") or [{}])[0]
                    logger.info(
                        "VAGEN model response task=%s round=%d model=%s "
                        "finish_reason=%s block_types=%s tool_calls=%s",
                        task_id,
                        round_number,
                        response_json.get("model"),
                        choice.get("finish_reason"),
                        [block.get("type") for block in blocks],
                        [
                            block.get("name")
                            for block in blocks
                            if block.get("type") == "tool_use"
                        ],
                    )
                    return blocks
                last_error = RuntimeError(
                    f"VAGEN API HTTP {response.status_code}: {response.text[:2000]}"
                )
            except requests.RequestException as exc:
                last_error = exc
            if attempt < max_retries:
                time.sleep(min(8, attempt * 2))
        raise RuntimeError(
            f"VAGEN MobileWorld evaluator request failed after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error

    def _execute_mobile_action(
        self,
        tool_input: dict[str, Any],
        env: MobileWorldEvalEnvClient,
    ) -> dict[str, Any]:
        try:
            native_input = self._model_action_to_native(tool_input)
            action = validate_mobile_action(native_input, allow_terminal=False)
            result = env.step(action)
            logger.info(
                "VAGEN mobile action model_view=%s native=%s executed=%s",
                json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
                json.dumps(action, ensure_ascii=False, sort_keys=True),
                result.executed,
            )
            return {
                "success": True,
                "text": (
                    "Android action executed. "
                    f"Model-view input={json.dumps(tool_input, ensure_ascii=False)}; "
                    f"native action={json.dumps(action, ensure_ascii=False)}"
                ),
                "screenshot": result.screenshot,
                "images": [],
            }
        except Exception as exc:
            logger.warning(
                "VAGEN mobile action failed model_view=%s error=%s",
                json.dumps(tool_input, ensure_ascii=False, sort_keys=True),
                exc,
            )
            return {
                "success": False,
                "text": f"Native Android action failed: {exc}",
                "screenshot": env.screenshot(wait=False),
                "images": [],
            }

    def eval_task(
        self,
        env: MobileWorldEvalEnvClient,
        task_config: dict[str, Any],
        trajectory: dict[str, Any],
        save_dir: str,
        max_round: int = 35,
    ) -> dict[str, Any]:
        self.eval_messages = []
        self.eval_image_names = []
        screenshots = list(trajectory.get("screenshots", []))
        image_map: dict[str, str] = {}
        image_dir = Path(save_dir) / "tmp_imgs"
        image_dir.mkdir(parents=True, exist_ok=True)
        for index, screenshot in enumerate(screenshots, start=1):
            image_path = image_dir / f"eval_step_{index}.png"
            image_path.write_bytes(screenshot)
            image_map[f"step_{index}"] = str(image_path)

        prompt = _build_mobile_eval_prompt(
            str(task_config.get("instruction") or ""),
            trajectory,
            len(screenshots),
        )
        initial_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if screenshots:
            with Image.open(io.BytesIO(screenshots[-1])) as final_image:
                final_image_data = self._encode_model_view_image(final_image)
            initial_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": final_image_data,
                    },
                }
            )
            self.eval_image_names.append("final_screenshot")
        self.eval_messages.append({"role": "user", "content": initial_content})

        system_prompt = (
            "You are a careful Android GUI evaluation agent. You may inspect "
            "trajectory images and use the native mobile_use tool for low-risk "
            "verification. Never perform actions that change user data and never "
            "continue or redo an incomplete task. If the recorded trajectory "
            "already proves failure, return FAILURE without probing. "
            f"You have at most {max_round} rounds. Return the required "
            "EVALUATION RESULT format as soon as evidence is sufficient."
        )
        if self.system_prompt_suffix:
            system_prompt += " " + self.system_prompt_suffix
        tools = [CHECK_SCREENSHOT_TOOL, MOBILE_USE_TOOL]
        final_reasoning: list[str] = []
        eval_steps = 0
        evaluation_complete = False
        task_id = str(task_config.get("id") or "mobile_eval_task")
        interaction_dir = Path(save_dir) / "eval_result"
        interaction_dir.mkdir(parents=True, exist_ok=True)

        for round_number in range(1, max_round + 1):
            eval_steps = round_number
            response = self._call_eval_model(
                system_prompt=system_prompt,
                tools=tools,
                task_id=task_id,
                round_number=round_number,
            )
            response_params = response
            self.eval_messages.append(
                {"role": "assistant", "content": response_params}
            )
            self.eval_image_names.append("no_screenshot")
            for block in response_params:
                if block.get("type") == "text":
                    final_reasoning.append(str(block.get("text") or ""))

            tool_calls = [
                block for block in response_params
                if block.get("type") == "tool_use"
            ]
            if not tool_calls:
                evaluation_complete = True
                break
            for tool_call in tool_calls:
                name = str(tool_call.get("name") or "")
                tool_input = dict(tool_call.get("input") or {})
                if name == "check_screenshot":
                    tool_result = self._read_key_images(tool_input, image_map)
                elif name == "mobile_use":
                    tool_result = self._execute_mobile_action(tool_input, env)
                    screenshot = tool_result.get("screenshot")
                    if screenshot:
                        timestamp = datetime.datetime.now().strftime(
                            "%Y%m%d@%H%M%S%f"
                        )
                        (interaction_dir / f"round_{round_number}_{timestamp}.png").write_bytes(
                            screenshot
                        )
                else:
                    tool_result = {
                        "success": False,
                        "text": f"Unsupported MobileWorld evaluator tool: {name}",
                        "images": [],
                    }
                self._add_tool_result_to_eval_messages(
                    tool_id=str(tool_call.get("id") or ""),
                    tool_result=tool_result,
                )

        if not evaluation_complete:
            self._trim_images(self.only_n_most_recent_images_eval)
            for final_attempt in range(1, 4):
                if final_attempt == 1:
                    final_instruction = (
                        "Stop probing now. Based on the evidence already collected, "
                        "provide the final judgment in the required EVALUATION RESULT "
                        "format. Do not call tools, describe a next action, or continue "
                        "investigating."
                    )
                    self.eval_messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": final_instruction}],
                        }
                    )
                    response_params = self._call_eval_model(
                        system_prompt=system_prompt,
                        tools=[],
                        task_id=task_id,
                        round_number=max_round + final_attempt,
                    )
                else:
                    evidence = "\n".join(final_reasoning)[-24000:]
                    final_instruction = f"""You are now a final-decision formatter, not an interactive agent. The prior evaluator failed to stop probing.

Task:
{task_config.get("instruction", "")}

Evidence and observations already gathered:
{evidence}

Do not propose or describe another action. Output exactly one judgment beginning with these literal words and fields:
EVALUATION RESULT:
Reasoning: <concise evidence-based reason>
Status: SUCCESS or FAILURE
Confidence: HIGH or MEDIUM or LOW"""
                    conversation = self.eval_messages
                    self.eval_messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": final_instruction}],
                        }
                    ]
                    try:
                        response_params = self._call_eval_model(
                            system_prompt=(
                                "Return only the requested final evaluation record. "
                                "You cannot use tools or take further actions."
                            ),
                            tools=[],
                            task_id=task_id,
                            round_number=max_round + final_attempt,
                        )
                    finally:
                        self.eval_messages = conversation
                    self.eval_messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": final_instruction}],
                        }
                    )

                self.eval_messages.append(
                    {"role": "assistant", "content": response_params}
                )
                response_text = "\n".join(
                    str(block.get("text") or "")
                    for block in response_params
                    if block.get("type") == "text"
                )
                if response_text:
                    final_reasoning.append(response_text)
                final_parse = self._parse_eval_result(response_text, eval_steps)
                if final_parse.get("parse_method") not in {"empty", "failed"}:
                    break
                logger.warning(
                    "VAGEN final judgment format invalid task=%s attempt=%d; retrying",
                    task_id,
                    final_attempt,
                )

        combined_reasoning = "\n".join(final_reasoning)
        result = self._parse_eval_result(combined_reasoning, eval_steps)
        self._save_eval_log(save_dir, self.eval_messages, result, task_config)
        return result

class MobileWorldQwenVagenEvalAgent(MobileWorldClaudeEvalAgent):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.enable_thinking = False

        self.api_url = self._resolve_chat_completions_url(self.api_url)

    @staticmethod
    def _resolve_chat_completions_url(api_url: str) -> str:
        url = str(api_url).rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"

__all__ = [
    "CHECK_SCREENSHOT_TOOL",
    "MOBILE_USE_TOOL",
    "MobileWorldClaudeEvalAgent",
    "MobileWorldQwenVagenEvalAgent",
]
