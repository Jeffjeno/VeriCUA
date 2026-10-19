from __future__ import annotations

import base64
import copy
import datetime
import io
import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urljoin, urlsplit

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("eval_runtime.web_arena.holo_probe")

class _Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Click(_Tool):
    tool_name: Literal["click"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    button: Literal["left", "middle", "right"] = "left"

class DoubleClick(_Tool):
    tool_name: Literal["double_click"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    button: Literal["left", "middle", "right"] = "left"

class Hover(_Tool):
    tool_name: Literal["hover"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)

class Drag(_Tool):
    tool_name: Literal["drag"]
    start_element: str
    end_element: str
    start_x: int = Field(ge=0, le=1000)
    start_y: int = Field(ge=0, le=1000)
    end_x: int = Field(ge=0, le=1000)
    end_y: int = Field(ge=0, le=1000)

class Write(_Tool):
    tool_name: Literal["write"]
    content: str
    press_enter: bool = False

class Keypress(_Tool):
    tool_name: Literal["keypress"]
    keys: str

class Scroll(_Tool):
    tool_name: Literal["scroll"]
    direction: Literal["up", "down", "left", "right"]
    amount: int = Field(default=500, ge=1, le=1000)

class Goto(_Tool):
    tool_name: Literal["goto"]
    url: str

class GoBack(_Tool):
    tool_name: Literal["go_back"]

class GoForward(_Tool):
    tool_name: Literal["go_forward"]

class NewTab(_Tool):
    tool_name: Literal["new_tab"]
    url: str | None = None

class SwitchTab(_Tool):
    tool_name: Literal["switch_tab"]
    index: int = Field(ge=0)

class CloseTab(_Tool):
    tool_name: Literal["close_tab"]

class Wait(_Tool):
    tool_name: Literal["wait"]
    milliseconds: int = Field(default=1000, ge=0, le=10000)

class Criteria(_Tool):
    tool_name: Literal["criteria"]
    option: int = Field(ge=1)
    state: Literal["success", "failure", "uncertain"]

class Answer(_Tool):
    tool_name: Literal["answer"]
    content: str = "DONE"

ProbeTool = Annotated[
    Click | DoubleClick | Hover | Drag | Write | Keypress | Scroll | Goto
    | GoBack | GoForward | NewTab | SwitchTab | CloseTab | Wait | Criteria | Answer,
    Field(discriminator="tool_name"),
]

class ProbeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = None
    thought: str = ""
    tool_call: ProbeTool

PROBE_SCHEMA = ProbeStep.model_json_schema()

SYSTEM_PROMPT = """You are a visual evidence probe inspecting a live web browser after
another agent attempted a task. Do not redo the task. Reveal only enough state to
judge the criteria, using low-risk browser navigation. Never create, edit, delete,
send, submit, purchase, or otherwise mutate data.
Stay inside the currently configured WebArena sites and the current site origin.
Never follow or enter a public-web URL such as Google, OpenStreetMap, or another
external domain. If useful evidence is not visible, use scroll, go_back, or
criteria/answer instead of leaving WebArena.

Choose exactly one structured tool call per turn. Coordinates are integers in
[0, 1000] relative to the screenshot. Use `criteria` to update exactly one numbered
criterion when visible evidence supports it. The judge only receives evidence saved
through `criteria`. Use `answer` with DONE once useful inspection is complete.

Original task:
{instruction}

Current criteria:
{criteria}

Current date: {current_date}

Output must match this JSON schema:
{schema}
"""

def _png_data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

def _image_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size

def _decode_json_object(content: str) -> dict[str, Any]:

    decoder = json.JSONDecoder()
    first_object: dict[str, Any] | None = None
    last_error: Exception | None = None
    for match in re.finditer(r"\{", content):
        try:
            value, _end = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            continue
        if first_object is None:
            first_object = value
        if "tool_name" in value or "tool_call" in value:
            return value
    if first_object is not None:
        return first_object
    if last_error is not None:
        raise last_error
    raise ValueError("Holo response did not contain a JSON object")

def _normalize_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:

    if "tool_name" in payload:
        return {"note": None, "thought": "", "tool_call": payload}

    normalized = dict(payload)
    tool_call = normalized.get("tool_call")
    if tool_call is None:
        for key in ("tool", "action"):
            candidate = normalized.get(key)
            if isinstance(candidate, dict) and "tool_name" in candidate:
                tool_call = candidate
                break
    if isinstance(tool_call, str):
        tool_call = _decode_json_object(tool_call)
    if not isinstance(tool_call, dict):
        raise ValueError("Holo response did not contain a tool_call object")
    normalized["tool_call"] = tool_call
    normalized.setdefault("thought", "")
    if normalized.get("thought") is None:
        normalized["thought"] = ""
    return normalized

def _parse_probe_step(content: str) -> ProbeStep:

    try:
        return ProbeStep.model_validate_json(content)
    except Exception as original_error:
        try:
            payload = _normalize_probe_payload(_decode_json_object(content))
        except Exception:
            raise original_error
        tool_call = payload.get("tool_call")
        if (
            isinstance(tool_call, dict)
            and tool_call.get("tool_name") == "scroll"
            and isinstance(tool_call.get("amount"), int)
        ):
            amount = int(tool_call["amount"])
            clamped = min(1000, max(1, amount))
            if amount != clamped:
                logger.warning(
                    "Clamping Holo scroll amount from %d to %d", amount, clamped
                )
                tool_call["amount"] = clamped
        return ProbeStep.model_validate(payload)

def _active_webarena_origins() -> set[tuple[str, str]]:

    try:
        from scripts.webarena_data_replay.common import wa_site_env_defaults
    except ModuleNotFoundError:
        from common import wa_site_env_defaults

    origins: set[tuple[str, str]] = set()
    for key, value in wa_site_env_defaults().items():
        if key == "WA_FULL_RESET":
            continue
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            origins.add((parsed.scheme.lower(), parsed.netloc.lower()))
    return origins

def _safe_probe_navigation_url(url: str, current_url: str) -> tuple[str | None, str | None]:

    try:
        from scripts.webarena_data_replay.common import rewrite_webarena_action_urls
    except ModuleNotFoundError:
        from common import rewrite_webarena_action_urls

    routed = rewrite_webarena_action_urls(str(url))
    parsed = urlsplit(routed)
    if not parsed.scheme and not parsed.netloc:
        routed = urljoin(current_url, routed)
        parsed = urlsplit(routed)
    origins = _active_webarena_origins()
    origin = (parsed.scheme.lower(), parsed.netloc.lower())
    if origin not in origins:
        return None, f"navigation target {url!r} is outside the active WebArena origins"
    return routed, None

def _point(value: int, extent: int) -> int:
    return min(extent - 1, max(0, int(value / 1000 * extent)))

def browser_action_from_tool(
    tool: Any, width: int, height: int, *, navigation_url: str | None = None
) -> str:
    if isinstance(tool, Click):
        return f"mouse_click({_point(tool.x, width)}, {_point(tool.y, height)}, button={tool.button!r})"
    if isinstance(tool, DoubleClick):
        return f"mouse_dblclick({_point(tool.x, width)}, {_point(tool.y, height)}, button={tool.button!r})"
    if isinstance(tool, Hover):
        return f"mouse_move({_point(tool.x, width)}, {_point(tool.y, height)})"
    if isinstance(tool, Drag):
        return (
            f"mouse_drag_and_drop({_point(tool.start_x, width)}, {_point(tool.start_y, height)}, "
            f"{_point(tool.end_x, width)}, {_point(tool.end_y, height)})"
        )
    if isinstance(tool, Write):
        action = f"keyboard_type({tool.content!r})"
        return action + ("\nkeyboard_press('ENTER')" if tool.press_enter else "")
    if isinstance(tool, Keypress):
        return f"keyboard_press({tool.keys!r})"
    if isinstance(tool, Scroll):
        extent = height if tool.direction in {"up", "down"} else width
        distance = max(1, int(tool.amount / 1000 * extent))
        dx = distance if tool.direction == "right" else -distance if tool.direction == "left" else 0
        dy = distance if tool.direction == "down" else -distance if tool.direction == "up" else 0
        return f"scroll({dx}, {dy})"
    if isinstance(tool, Goto):
        return f"goto({(navigation_url if navigation_url is not None else tool.url)!r})"
    if isinstance(tool, GoBack):
        return "go_back()"
    if isinstance(tool, GoForward):
        return "go_forward()"
    if isinstance(tool, NewTab):
        url = navigation_url if navigation_url is not None else tool.url
        return "new_tab()" + (f"\ngoto({url!r})" if url else "")
    if isinstance(tool, SwitchTab):
        return f"tab_focus({tool.index})"
    if isinstance(tool, CloseTab):
        return "tab_close()"
    if isinstance(tool, Wait):
        return f"noop({tool.milliseconds})"
    raise TypeError(f"Tool does not map to a BrowserGym action: {type(tool).__name__}")

class WebArenaHoloProbeAgent:

    def __init__(self, *, model: dict[str, Any], criteria: str = "", max_images: int = 3,
                 max_trajectory_length: int = 15, reasoning_effort: str = "medium", **_: Any):
        self.model = dict(model)
        self.max_images = int(max_images)
        self.max_trajectory_length = int(max_trajectory_length)
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(
            api_key=str(self.model.get("api_key") or "EMPTY"),
            base_url=str(self.model["base_url"]), timeout=180, max_retries=0,
        )
        self.criteria = [criteria] if criteria else []
        self.criteria_images: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tool_call: dict[str, Any] | None = None
        self.last_raw_response: str = ""
        self.last_parse_error: str | None = None
        self.last_blocked_navigation: dict[str, str] | None = None
        self._pending_result: str | None = None

    @property
    def latest_criteria(self) -> str:
        return self.criteria[-1] if self.criteria else ""

    def reset(self, _logger: Any = None) -> None:
        del _logger
        self.criteria_images = []
        self.messages = []
        self.last_messages = None
        self.last_tool_call = None
        self.last_raw_response = ""
        self.last_parse_error = None
        self.last_blocked_navigation = None
        self._pending_result = None

    def reset_for_round(self, criteria: str, _logger: Any = None) -> None:
        self.reset(_logger)
        self.criteria = [criteria] if criteria else []

    def _update_criteria(self, option: int, state: str, screenshot: bytes) -> bool:
        pattern = re.compile(
            rf"(^\s*{option}\.\s*.*?:\s*)(success|failure|uncertain)(\s*$)",
            re.IGNORECASE | re.MULTILINE,
        )
        updated, count = pattern.subn(rf"\g<1>{state}\g<3>", self.latest_criteria, count=1)
        if count != 1:
            logger.warning("Criterion %d was not found in current WebArena criteria", option)
            return False
        self.criteria.append(updated)
        encoded = base64.b64encode(screenshot).decode("ascii")
        entry = {"option": option, "images": encoded}
        self.criteria_images = [item for item in self.criteria_images if item.get("option") != option]
        self.criteria_images.append(entry)
        return True

    def predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        screenshot = obs["screenshot"]
        if not isinstance(screenshot, (bytes, bytearray)):
            raise TypeError("WebArena Holo probe expects PNG screenshot bytes")
        screenshot = bytes(screenshot)
        self.last_parse_error = None
        self.last_blocked_navigation = None
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    instruction=instruction,
                    criteria=self.latest_criteria or "No criteria provided.",
                    current_date=datetime.date.today().isoformat(),
                    schema=json.dumps(PROBE_SCHEMA, ensure_ascii=False, separators=(",", ":")),
                ),
            })
        if self._pending_result:
            self.messages.append({"role": "user", "content": self._pending_result})
            self._pending_result = None
        tabs = obs.get("open_pages_urls") or []
        context = f"Current URL: {obs.get('url', '')}\nOpen tabs: {tabs}"
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"<observation>\n{context}\n"},
                {"type": "image_url", "image_url": {"url": _png_data_url(screenshot)}},
                {"type": "text", "text": "\n</observation>"},
            ],
        })
        image_parts = [part for message in self.messages if isinstance(message.get("content"), list)
                       for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url"]
        excess = len(image_parts) if self.max_images == 0 else max(0, len(image_parts) - self.max_images)
        for part in image_parts[:excess]:
            part.clear()
            part.update({"type": "text", "text": "[older screenshot omitted]"})
        last_error: Exception | None = None
        last_content = ""
        parse_failure_seen = False
        parse_recovered = False
        step: ProbeStep | None = None
        for attempt in range(3):
            try:
                request_messages = self.messages
                if attempt:
                    request_messages = self.messages + [{
                        "role": "user",
                        "content": (
                            "Return exactly one compact valid JSON object. Do not use "
                            "markdown fences and do not put unescaped double quotes in "
                            "thought. A bare tool object is also acceptable."
                        ),
                    }]
                response = self.client.chat.completions.create(
                    model=str(self.model["model_name"]), messages=request_messages,
                    temperature=float(self.model.get("temperature", 0.0)),
                    max_completion_tokens=int(self.model.get("max_tokens", 8192)),
                    reasoning_effort=self.reasoning_effort,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "structured_outputs": {"json": PROBE_SCHEMA},
                    },
                )
                content = response.choices[0].message.content or ""
                last_content = content
                self.last_raw_response = content
                step = _parse_probe_step(content)
                break
            except Exception as exc:
                last_error = exc
                if last_content:
                    parse_failure_seen = True
                if attempt == 2:
                    break
                logger.warning("WebArena Holo probe request %d/3 failed: %s", attempt + 1, exc)
        if step is None:
            if not parse_failure_seen:
                raise last_error or RuntimeError("WebArena Holo probe request failed")
            self.last_parse_error = repr(last_error)
            logger.error(
                "Holo returned unparsable JSON after 3 attempts; using a no-op WAIT: %s",
                last_error,
            )
            step = ProbeStep(
                thought="The probe response was malformed; keep the browser unchanged.",
                tool_call=Wait(tool_name="wait", milliseconds=0),
            )
            parse_recovered = True
        normalized = step.model_dump_json()
        self.messages.append({"role": "assistant", "content": normalized})
        if self.max_trajectory_length > 0:
            max_messages = 1 + self.max_trajectory_length * 3
            if len(self.messages) > max_messages:
                self.messages = self.messages[:1] + self.messages[-(max_messages - 1):]
        self.last_messages = copy.deepcopy(self.messages)
        self.last_tool_call = {
            "name": step.tool_call.tool_name,
            "arguments": step.tool_call.model_dump(),
        }
        if parse_recovered:
            action = "WAIT"
            self._pending_result = (
                '<tool_output tool="wait">malformed probe JSON recovered with a '
                "no-op; return a valid JSON tool on the next turn</tool_output>"
            )
        elif isinstance(step.tool_call, Criteria):
            updated = self._update_criteria(
                step.tool_call.option, step.tool_call.state, screenshot
            )
            action = "WAIT"
            if not updated:
                self._pending_result = (
                    f'<tool_output tool="criteria">error: criterion '
                    f'{step.tool_call.option} not found</tool_output>'
                )
        elif isinstance(step.tool_call, Answer):
            action = "DONE"
        else:
            width, height = _image_size(screenshot)
            navigation_url: str | None = None
            requested_url = None
            if isinstance(step.tool_call, Goto):
                requested_url = step.tool_call.url
            elif isinstance(step.tool_call, NewTab) and step.tool_call.url:
                requested_url = step.tool_call.url
            if requested_url is not None:
                navigation_url, reason = _safe_probe_navigation_url(
                    requested_url, str(obs.get("url") or "")
                )
                if navigation_url is None:
                    self.last_blocked_navigation = {
                        "requested_url": requested_url,
                        "reason": reason or "outside active WebArena origins",
                    }
                    action = "WAIT"
                    self._pending_result = (
                        '<tool_output tool="navigation">blocked: external navigation '
                        "is not allowed; inspect the current WebArena page instead"
                        "</tool_output>"
                    )
                else:
                    action = browser_action_from_tool(
                        step.tool_call, width, height, navigation_url=navigation_url
                    )
            else:
                action = browser_action_from_tool(step.tool_call, width, height)
        if self._pending_result is None:
            self._pending_result = (
                f'<tool_output tool="{step.tool_call.tool_name}">ok</tool_output>'
            )
        return normalized, [action]

def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def run_webarena_guirl_probe_round(
    *, probe_agent: WebArenaHoloProbeAgent, env_client: Any, lease_id: str,
    instruction: str, max_steps: int, example_result_dir: str,
    sleep_after_execution: float, sampler_round: int = 1,
) -> dict[str, Any]:
    del lease_id, sleep_after_execution
    probe_dir = Path(example_result_dir) / "replay_eval" / "guirl_probe"
    current_obs = env_client.get_obs("")
    done = False
    env_steps = 0
    stop_reason = "max_steps"
    steps_run = 0
    blocked_navigation_count = 0
    _append_jsonl(probe_dir / "rounds.jsonl", {
        "event": "round_start", "round": sampler_round,
        "timestamp": datetime.datetime.now().isoformat(), "max_steps": max_steps,
        "platform": "webarena", "agent_type": "holo",
    })
    for step_num in range(1, max_steps + 1):
        before_count = len(probe_agent.criteria_images)
        response, actions = probe_agent.predict(instruction, current_obs)
        action = actions[0]
        if action == "DONE":
            done = True
            stop_reason = "agent_done"
        elif action == "WAIT":
            if probe_agent.last_blocked_navigation:
                blocked_navigation_count += 1
                stop_reason = "navigation_blocked"
            elif probe_agent.last_parse_error:
                stop_reason = "parse_recovery"
            else:
                stop_reason = "criteria_update"
        else:
            current_obs = env_client.step_browser(action)
            env_steps += 1
        image_path = probe_dir / f"round_{sampler_round:02d}_step_{step_num:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(current_obs["screenshot"])
        _append_jsonl(probe_dir / "traj.jsonl", {
            "round": sampler_round, "step": step_num, "response": response,
            "tool_call": probe_agent.last_tool_call, "actions": actions,
            "latest_criteria_after_parse": probe_agent.latest_criteria,
            "new_criteria_images": len(probe_agent.criteria_images) - before_count,
            "blocked_navigation": probe_agent.last_blocked_navigation,
            "parse_error": probe_agent.last_parse_error,
            "done": done,
        })
        steps_run += 1
        if done:
            break
    _append_jsonl(probe_dir / "rounds.jsonl", {
        "event": "round_end", "round": sampler_round,
        "timestamp": datetime.datetime.now().isoformat(), "steps": steps_run,
        "env_steps": env_steps, "done": done, "stop_reason": stop_reason,
        "blocked_navigation_count": blocked_navigation_count,
    })
    return {
        "obs": current_obs, "done": done, "steps": steps_run,
        "env_steps": env_steps, "stop_reason": stop_reason,
        "blocked_navigation_count": blocked_navigation_count,
        "platform": "webarena", "agent_type": "holo",
    }

__all__ = [
    "PROBE_SCHEMA", "WebArenaHoloProbeAgent", "browser_action_from_tool",
    "run_webarena_guirl_probe_round",
]
