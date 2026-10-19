from __future__ import annotations

import base64
import datetime
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .env_client import MobileWorldEvalEnvClient, TERMINAL_WITHOUT_EXECUTION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBILE_WORLD_SRC = PROJECT_ROOT / "third_party" / "MobileWorld" / "src"
if str(MOBILE_WORLD_SRC) not in sys.path:
    sys.path.insert(0, str(MOBILE_WORLD_SRC))

def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _json_action(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        return dict(action.model_dump(exclude_none=True))
    if isinstance(action, dict):
        return dict(action)
    raise TypeError(f"Native probe produced an invalid action: {action!r}")

class MobileWorldNativeProbeAgent:

    native_agent_type = "qwen3vl"

    def __init__(
        self,
        *,
        model: dict[str, Any],
        criteria: str = "",
        history_n: int = 4,
        max_evidence_images: int = 4,
        **_: Any,
    ) -> None:
        self.model = dict(model)
        self.history_n = int(history_n)
        self.max_evidence_images = int(max_evidence_images)
        self.criteria = [criteria] if criteria else []
        self.criteria_images: list[dict[str, Any]] = []
        self.responses: list[str] = []
        self.actions: list[dict[str, Any]] = []
        self.last_tool_call: dict[str, Any] | None = None
        self.last_messages: Any = None
        self._native_agent: Any = None
        self._instruction = ""

    @property
    def latest_criteria(self) -> str:
        return self.criteria[-1] if self.criteria else ""

    def _build_probe_instruction(self, instruction: str) -> str:
        criteria = self.latest_criteria.strip() or "No explicit criteria were provided."
        return (
            "You are an evidence probe inspecting the current state of an Android "
            "device after another agent attempted a task. Do not repeat the whole "
            "task and do not make destructive or irreversible changes. Use the "
            "native mobile actions only to reveal enough on-screen evidence for the "
            "judge. Prefer navigation, opening relevant views, scrolling, and wait. "
            "Do not ask the user, call MCP tools, submit answers, or modify user "
            "data. When sufficient evidence is visible, terminate.\n\n"
            f"Original task:\n{instruction}\n\n"
            f"Current judge criteria:\n{criteria}"
        )

    def _make_native_agent(self) -> Any:
        if self.native_agent_type == "qwen3vl":
            from mobile_world.agents.implementations.qwen3vl import Qwen3VLAgentMCP

            agent_cls = Qwen3VLAgentMCP
            runtime_conf: dict[str, Any] = {
                "temperature": self.model.get("temperature", 0.0),
                "max_tokens": self.model.get("max_tokens", 32768),
            }
        elif self.native_agent_type == "gui_owl_1_5":
            from mobile_world.agents.implementations.gui_owl_1_5 import GUIOWL15AgentMCP

            agent_cls = GUIOWL15AgentMCP
            runtime_conf = {
                "history_n": self.history_n,
                "temperature": self.model.get("temperature", 0.0),
                "top_p": self.model.get("top_p", 1.0),
                "max_tokens": self.model.get("max_tokens", 32768),
            }
        else:
            raise ValueError(f"Unsupported native MobileWorld probe: {self.native_agent_type}")
        return agent_cls(
            model_name=str(self.model["model_name"]),
            llm_base_url=str(self.model["base_url"]),
            api_key=str(self.model.get("api_key") or "EMPTY"),
            tools=[],
            runtime_conf=runtime_conf,
        )

    def reset(self, _logger: Any = None) -> None:
        del _logger
        self.criteria_images = []
        self.responses = []
        self.actions = []
        self.last_tool_call = None
        self.last_messages = None
        if self._native_agent is not None:
            self._native_agent.reset()

    def reset_for_round(self, criteria: str, _logger: Any = None) -> None:
        self.reset(_logger)
        self.criteria = [criteria] if criteria else []

        self._native_agent = None

    def record_evidence(self, screenshot: bytes, option: int | None = None) -> None:
        entry = {
            "option": option if option is not None else len(self.criteria_images) + 1,
            "images": base64.b64encode(screenshot).decode("ascii"),
        }
        self.criteria_images.append(entry)
        if self.max_evidence_images > 0:
            self.criteria_images = self.criteria_images[-self.max_evidence_images :]

    def predict(
        self,
        instruction: str,
        obs: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        self._instruction = instruction
        if self._native_agent is None:
            self._native_agent = self._make_native_agent()
            self._native_agent.initialize(self._build_probe_instruction(instruction))

        screenshot = obs["screenshot"]
        if not isinstance(screenshot, (bytes, bytearray)):
            raise TypeError("MobileWorld GUIRL probe expects PNG screenshot bytes")
        pil_image = Image.open(BytesIO(bytes(screenshot)))
        prediction, action = self._native_agent.predict(
            {
                "screenshot": pil_image,
                "tool_call": obs.get("tool_call"),
                "ask_user_response": obs.get("ask_user_response"),
            }
        )
        native_action = _json_action(action)
        self.responses.append(str(prediction))
        self.actions.append(native_action)
        self.last_tool_call = {
            "name": "mobile_use",
            "arguments": native_action,
        }
        return str(prediction), [native_action]

class MobileWorldQwen3VLProbeAgent(MobileWorldNativeProbeAgent):
    native_agent_type = "qwen3vl"

class MobileWorldGUIOwlProbeAgent(MobileWorldNativeProbeAgent):
    native_agent_type = "gui_owl_1_5"

def run_mobile_world_guirl_probe_round(
    *,
    probe_agent: MobileWorldNativeProbeAgent,
    env_client: MobileWorldEvalEnvClient,
    lease_id: str,
    instruction: str,
    max_steps: int,
    example_result_dir: str,
    sleep_after_execution: float,
    sampler_round: int = 1,
) -> dict[str, Any]:

    del lease_id, sleep_after_execution
    probe_dir = (
        Path(example_result_dir)
        / "replay_eval"
        / "guirl_probe"
    )
    probe_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = probe_dir / "rounds.jsonl"
    traj_path = probe_dir / "traj.jsonl"
    _append_jsonl(
        rounds_path,
        {
            "event": "round_start",
            "round": sampler_round,
            "timestamp": datetime.datetime.now().isoformat(),
            "instruction": instruction,
            "max_steps": max_steps,
            "platform": "mobile_world",
            "agent_type": probe_agent.native_agent_type,
        },
    )

    current_obs = env_client.get_obs()
    probe_agent.record_evidence(current_obs["screenshot"], option=0)
    done = False
    stop_reason = "max_steps"
    total_env_steps = 0
    steps_run = 0
    for step_num in range(1, max_steps + 1):
        before_count = len(probe_agent.criteria_images)
        response, actions = probe_agent.predict(instruction, current_obs)
        action = actions[0]
        action_type = action.get("action_type")
        before_name = f"round_{sampler_round:02d}_step_{step_num:02d}_before.png"
        (probe_dir / before_name).write_bytes(current_obs["screenshot"])

        if action_type in TERMINAL_WITHOUT_EXECUTION or action_type in {
            "answer",
            "error_env",
            "ask_user",
            "mcp",
        }:
            done = True
            stop_reason = str(action_type)
        else:
            current_obs = env_client.execute_action(action)
            total_env_steps += 1
            probe_agent.record_evidence(current_obs["screenshot"], option=step_num)
            after_name = f"round_{sampler_round:02d}_step_{step_num:02d}_after.png"
            (probe_dir / after_name).write_bytes(current_obs["screenshot"])

        steps_run += 1
        _append_jsonl(
            traj_path,
            {
                "round": sampler_round,
                "step": step_num,
                "response": response,
                "tool_call": probe_agent.last_tool_call,
                "actions": actions,
                "native_action": action,
                "criteria_history_after_parse": probe_agent.criteria,
                "latest_criteria_after_parse": probe_agent.latest_criteria,
                "new_criteria_images": len(probe_agent.criteria_images) - before_count,
                "done": done,
            },
        )
        if done:
            break

    _append_jsonl(
        rounds_path,
        {
            "event": "round_end",
            "round": sampler_round,
            "timestamp": datetime.datetime.now().isoformat(),
            "steps": steps_run,
            "env_steps": total_env_steps,
            "done": done,
            "stop_reason": stop_reason,
        },
    )
    return {
        "obs": current_obs,
        "done": done,
        "steps": steps_run,
        "env_steps": total_env_steps,
        "stop_reason": stop_reason,
        "round": sampler_round,
        "platform": "mobile_world",
        "agent_type": probe_agent.native_agent_type,
    }

__all__ = [
    "MobileWorldGUIOwlProbeAgent",
    "MobileWorldNativeProbeAgent",
    "MobileWorldQwen3VLProbeAgent",
    "run_mobile_world_guirl_probe_round",
]
