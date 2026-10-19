from __future__ import annotations

import base64
import contextlib
import json as _json
import logging
import queue
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from agent_runtime.backend import BackendProvider, BackendSession
from agent_runtime.benchmark import Benchmark
from agent_runtime.env import Env
from agent_runtime.evaluator import Evaluator
from agent_runtime.harness import Harness
from agent_runtime.registry import register_benchmark
from agent_runtime.types import (
    Action,
    ActionType,
    EnvCapabilities,
    EvaluationResult,
    HarnessContext,
    Modality,
    ModalityKind,
    Observation,
    RunTaskResult,
    Task,
    Trace,
    TraceEvent,
)

logger = logging.getLogger(__name__)

_MW_CLICK = "click"
_MW_DOUBLE_TAP = "double_tap"
_MW_LONG_PRESS = "long_press"
_MW_SWIPE = "swipe"
_MW_SCROLL = "scroll"
_MW_INPUT_TEXT = "input_text"
_MW_NAVIGATE_HOME = "navigate_home"
_MW_NAVIGATE_BACK = "navigate_back"
_MW_KEYBOARD_ENTER = "keyboard_enter"
_MW_OPEN_APP = "open_app"
_MW_WAIT = "wait"
_MW_DRAG = "drag"
_MW_ASK_USER = "ask_user"
_MW_MCP = "mcp"
_MW_ANSWER = "answer"
_MW_STATUS = "status"
_MW_FINISHED = "finished"
_MW_UNKNOWN = "unknown"
_MW_ENV_FAIL = "error_env"

_MW_TERMINAL_ACTIONS = frozenset({_MW_FINISHED, _MW_STATUS, _MW_UNKNOWN, _MW_ENV_FAIL})

class MobileWorldClient:

    def __init__(
        self,
        base_url: str = "http://localhost:6800",
        device: str = "emulator-5554",
        timeout: float = 300.0,
        step_wait_time: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device = device
        self.timeout = timeout
        self.step_wait_time = step_wait_time
        self._device_initialized = False

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        timeout: float | None = None,
    ) -> Any:

        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        data: bytes | None = None
        if json_body is not None:
            data = _json.dumps(json_body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            method=method.upper(),
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                body = resp.read().decode("utf-8")
                if body:
                    return _json.loads(body)
                return None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(
                f"HTTP {exc.code} for {method} {path}: {body}"
            ) from exc

    def _get(self, path: str, **kw: Any) -> Any:
        return self._request("GET", path, **kw)

    def _post(self, path: str, **kw: Any) -> Any:
        return self._request("POST", path, **kw)

    def _ensure_device(self) -> None:
        if not self._device_initialized:
            self.init_device()

    def health_check(self) -> bool:
        try:
            result = self._get("/health", timeout=10)
            return bool(result and result.get("ok", False))
        except Exception:
            return False

    def init_device(self) -> dict[str, Any]:
        result = self._post("/init", json_body={"device": self.device})
        self._device_initialized = True
        return result

    def list_tasks(
        self,
        enable_mcp: bool = False,
        enable_user_interaction: bool = False,
    ) -> list[dict[str, Any]]:

        self._ensure_device()
        all_tasks: list[dict[str, Any]] = self._get("/task/list")

        filtered = []
        for t in all_tasks:
            tags = t.get("tags", [])
            if not enable_mcp and "agent-mcp" in tags:
                continue
            if not enable_user_interaction and "agent-user-interaction" in tags:
                continue
            filtered.append(t)
        return filtered

    def get_task_goal(self, task_name: str) -> str:
        self._ensure_device()
        return self._get("/task/goal", params={"task_name": task_name})

    def get_task_metadata(self, task_name: str) -> dict[str, Any]:
        self._ensure_device()
        return self._get("/task/metadata", params={"task_name": task_name})

    def get_task_complexity(self, task_name: str) -> float:
        self._ensure_device()
        return float(self._get("/task/complexity", params={"task_name": task_name}))

    def initialize_task(self, task_name: str) -> None:

        self._ensure_device()
        self._post("/task/init", json_body={"task_name": task_name, "req_device": self.device})

    def get_screenshot_b64(self) -> str:

        self._ensure_device()
        result = self._get(
            "/screenshot",
            params={"device": self.device, "return_b64": "true"},
        )
        return result["b64_png"]

    def step(self, action_payload: dict[str, Any]) -> dict[str, Any]:

        self._ensure_device()
        try:
            return self._post(
                "/step",
                json_body={"device": self.device, "action": action_payload},
            )
        except RuntimeError as exc:
            if action_payload.get("action_type") == _MW_ASK_USER:
                raise
            return {"result": str(exc)}

    def evaluate(self, task_name: str) -> tuple[float, str]:

        self._ensure_device()
        result = self._get(
            "/task/eval",
            json_body={"task_name": task_name, "req_device": self.device},
        )
        score = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        return score, reason

    def tear_down(self, task_name: str) -> None:

        self._ensure_device()
        try:
            self._post(
                "/task/tear_down",
                json_body={"task_name": task_name, "req_device": self.device},
            )
        except Exception as exc:
            logger.warning("tear_down failed for %s: %s", task_name, exc)

    def close(self) -> None:
        pass

_PLATFORM_TO_MW: dict[ActionType, str] = {
    ActionType.TAP: _MW_CLICK,
    ActionType.CLICK: _MW_CLICK,
    ActionType.DOUBLE_CLICK: _MW_DOUBLE_TAP,
    ActionType.LONG_PRESS: _MW_LONG_PRESS,
    ActionType.SWIPE: _MW_SWIPE,
    ActionType.SCROLL: _MW_SCROLL,
    ActionType.TYPE: _MW_INPUT_TEXT,
    ActionType.DRAG: _MW_DRAG,
    ActionType.ASK_USER: _MW_ASK_USER,
    ActionType.NOOP: _MW_WAIT,
    ActionType.TERMINATE: _MW_FINISHED,
    ActionType.SUBMIT: _MW_ANSWER,
}

_MW_TO_PLATFORM: dict[str, ActionType] = {
    _MW_CLICK: ActionType.TAP,
    _MW_DOUBLE_TAP: ActionType.DOUBLE_CLICK,
    _MW_LONG_PRESS: ActionType.LONG_PRESS,
    _MW_SWIPE: ActionType.SWIPE,
    _MW_SCROLL: ActionType.SCROLL,
    _MW_INPUT_TEXT: ActionType.TYPE,
    _MW_NAVIGATE_HOME: ActionType.KEY_PRESS,
    _MW_NAVIGATE_BACK: ActionType.KEY_PRESS,
    _MW_KEYBOARD_ENTER: ActionType.KEY_PRESS,
    _MW_OPEN_APP: ActionType.CUSTOM,
    _MW_WAIT: ActionType.NOOP,
    _MW_DRAG: ActionType.DRAG,
    _MW_ASK_USER: ActionType.ASK_USER,
    _MW_MCP: ActionType.TOOL_CALL,
    _MW_ANSWER: ActionType.SUBMIT,
    _MW_STATUS: ActionType.TERMINATE,
    _MW_FINISHED: ActionType.TERMINATE,
    _MW_UNKNOWN: ActionType.NOOP,
    _MW_ENV_FAIL: ActionType.TERMINATE,
}

def action_to_native(action: Action) -> dict[str, Any]:

    args = action.arguments
    native: dict[str, Any] = {}

    if action.type in (ActionType.TAP, ActionType.CLICK):
        native = {"action_type": _MW_CLICK, "x": args.get("x"), "y": args.get("y")}
    elif action.type == ActionType.DOUBLE_CLICK:
        native = {"action_type": _MW_DOUBLE_TAP, "x": args.get("x"), "y": args.get("y")}
    elif action.type == ActionType.LONG_PRESS:
        native = {"action_type": _MW_LONG_PRESS, "x": args.get("x"), "y": args.get("y")}
    elif action.type == ActionType.SWIPE:
        native = {
            "action_type": _MW_SWIPE,
            "x": args.get("x"),
            "y": args.get("y"),
            "direction": args.get("direction", "up"),
        }
    elif action.type == ActionType.SCROLL:
        native = {
            "action_type": _MW_SCROLL,
            "direction": args.get("direction", "down"),
        }
    elif action.type == ActionType.TYPE:
        native = {
            "action_type": _MW_INPUT_TEXT,
            "text": args.get("text", ""),
            "clear_text": args.get("clear_text"),
        }
    elif action.type == ActionType.KEY_PRESS:
        key = args.get("key", action.name or "")
        key_lower = key.lower()
        if key_lower in ("home", "navigate_home"):
            native = {"action_type": _MW_NAVIGATE_HOME}
        elif key_lower in ("back", "navigate_back"):
            native = {"action_type": _MW_NAVIGATE_BACK}
        elif key_lower in ("enter", "keyboard_enter"):
            native = {"action_type": _MW_KEYBOARD_ENTER}
        else:
            native = {"action_type": _MW_KEYBOARD_ENTER}
    elif action.type == ActionType.DRAG:
        native = {
            "action_type": _MW_DRAG,
            "start_x": args.get("start_x"),
            "start_y": args.get("start_y"),
            "end_x": args.get("end_x"),
            "end_y": args.get("end_y"),
        }
    elif action.type == ActionType.TOOL_CALL:
        native = {
            "action_type": _MW_MCP,
            "action_name": action.name,
            "action_json": args,
        }
    elif action.type == ActionType.ASK_USER:
        native = {"action_type": _MW_ASK_USER, "text": args.get("text", "")}
    elif action.type == ActionType.SUBMIT:
        text = args.get("answer") or args.get("text") or ""
        native = {"action_type": _MW_ANSWER, "text": str(text)}
    elif action.type == ActionType.TERMINATE:
        reason = args.get("reason", "")
        goal_status = args.get("goal_status", reason)
        native = {"action_type": _MW_FINISHED, "goal_status": goal_status}
    elif action.type == ActionType.NOOP:
        native = {"action_type": _MW_WAIT}
    elif action.type == ActionType.CUSTOM:
        custom_name = (action.name or "").lower()
        if custom_name == "open_app":
            native = {"action_type": _MW_OPEN_APP, "app_name": args.get("app_name", "")}
        elif custom_name == "status":
            native = {"action_type": _MW_STATUS, "goal_status": args.get("goal_status", "")}
        else:
            native = {"action_type": custom_name, **args}
    else:
        native = {"action_type": _MW_UNKNOWN}

    native = {k: v for k, v in native.items() if v is not None}
    return native

def native_to_action(native: dict[str, Any], raw: str | None = None) -> Action:

    action_type_str = native.get("action_type", _MW_UNKNOWN)
    platform_type = _MW_TO_PLATFORM.get(action_type_str, ActionType.CUSTOM)

    args: dict[str, Any] = {}
    name: str | None = None

    if action_type_str == _MW_CLICK:
        args = {"x": native.get("x"), "y": native.get("y")}
    elif action_type_str == _MW_DOUBLE_TAP:
        args = {"x": native.get("x"), "y": native.get("y")}
    elif action_type_str == _MW_LONG_PRESS:
        args = {"x": native.get("x"), "y": native.get("y")}
    elif action_type_str == _MW_SWIPE:
        args = {"x": native.get("x"), "y": native.get("y"), "direction": native.get("direction")}
    elif action_type_str == _MW_SCROLL:
        args = {"direction": native.get("direction")}
    elif action_type_str == _MW_INPUT_TEXT:
        args = {"text": native.get("text", ""), "clear_text": native.get("clear_text")}
    elif action_type_str in (_MW_NAVIGATE_HOME, _MW_NAVIGATE_BACK, _MW_KEYBOARD_ENTER):
        key_map = {
            _MW_NAVIGATE_HOME: "home",
            _MW_NAVIGATE_BACK: "back",
            _MW_KEYBOARD_ENTER: "enter",
        }
        args = {"key": key_map[action_type_str]}
    elif action_type_str == _MW_OPEN_APP:
        name = "open_app"
        args = {"app_name": native.get("app_name", "")}
    elif action_type_str == _MW_DRAG:
        args = {
            "start_x": native.get("start_x"),
            "start_y": native.get("start_y"),
            "end_x": native.get("end_x"),
            "end_y": native.get("end_y"),
        }
    elif action_type_str == _MW_ASK_USER:
        args = {"text": native.get("text", "")}
    elif action_type_str == _MW_MCP:
        name = native.get("action_name", "mcp_tool")
        args = native.get("action_json", {})
    elif action_type_str == _MW_ANSWER:
        args = {"answer": native.get("text", "")}
    elif action_type_str in (_MW_STATUS, _MW_FINISHED):
        args = {"reason": native.get("goal_status", action_type_str)}
    elif action_type_str == _MW_ENV_FAIL:
        args = {"reason": "env_fail"}

    args = {k: v for k, v in args.items() if v is not None}
    return Action(
        type=platform_type,
        name=name,
        arguments=args,
        raw=raw,
        metadata={"native": native},
    )

class MobileWorldEnv(Env):

    def __init__(self, task: Task, session: Any = None, **env_options: Any) -> None:
        super().__init__(task, session, **env_options)
        self._capabilities = EnvCapabilities(step=True, screenshot=True)
        self._task_name: str = (task.native_ref or {}).get("task_name", task.task_id)
        self._task_goal: str | None = None
        self._step_count = 0
        self._terminated = False
        self._last_obs: Observation | None = None

        if session is not None and hasattr(session, "client"):
            self.client: MobileWorldClient = session.client
        else:
            raise ValueError(
                "MobileWorldEnv requires a MobileWorldSession with a .client attribute. "
                "Use MobileWorldBackendProvider.create_session() to obtain one."
            )

    def _get_screenshot_observation(
        self,
        step_index: int,
        ask_user_response: str | None = None,
        tool_call: Any | None = None,
        terminal: bool = False,
    ) -> Observation:
        screenshot_b64 = self.client.get_screenshot_b64()
        parts: list[Modality] = [
            Modality(kind=ModalityKind.IMAGE, content=screenshot_b64, mime_type="image/png"),
        ]
        if ask_user_response is not None:
            parts.append(Modality(
                kind=ModalityKind.TEXT,
                content=ask_user_response,
                name="ask_user_response",
            ))
        if tool_call is not None:
            parts.append(Modality(
                kind=ModalityKind.TOOL_RESULT,
                content=tool_call,
                name="tool_call",
            ))
        return Observation(
            parts=parts,
            instruction=self._task_goal,
            step_index=step_index,
            terminal=terminal,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self.client.init_device()
        self.client.initialize_task(self._task_name)
        self._task_goal = self.client.get_task_goal(self._task_name)
        self._step_count = 0
        self._terminated = False

        self._run_task_hook("pre_task")

        time.sleep(self.client.step_wait_time)
        obs = self._get_screenshot_observation(step_index=0)
        obs.instruction = self._task_goal
        self._last_obs = obs
        self.trace.record_step(observation=obs)

        return obs, {"task_name": self._task_name, "task_goal": self._task_goal}

    def step(
        self, action: Action
    ) -> tuple[Observation, float | None, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset() first.")

        self._step_count += 1
        native_action = action_to_native(action)
        action_type = native_action.get("action_type", _MW_UNKNOWN)

        action.metadata["native"] = native_action

        terminated = False
        truncated = False
        ask_user_response: str | None = None
        tool_call: Any | None = None
        info: dict[str, Any] = {"native_action": native_action}

        if action_type in _MW_TERMINAL_ACTIONS:
            terminated = True
        elif action_type == _MW_ANSWER:
            step_result = self.client.step(native_action)
            info["step_result"] = step_result
            terminated = True
        else:
            step_result = self.client.step(native_action)
            info["step_result"] = step_result

            if action_type == _MW_ASK_USER:
                result_text = step_result.get("result", "")
                if result_text:
                    ask_user_response = str(result_text)

        time.sleep(self.client.step_wait_time)

        obs = self._get_screenshot_observation(
            step_index=self._step_count,
            ask_user_response=ask_user_response,
            tool_call=tool_call,
            terminal=terminated,
        )
        self._last_obs = obs

        self.trace.record_step(
            observation=obs,
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated:
            self._terminated = True

        return obs, None, terminated, truncated, info

    def run(
        self,
        harness: Harness,
        evaluator: Evaluator | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> RunTaskResult:
        if self._last_obs is None:
            self.reset()

        effective_max_steps = max_steps or self.task.max_steps or 15
        start_time = time.time()

        obs = self._last_obs
        assert obs is not None

        for step_idx in range(effective_max_steps):
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Timeout reached after %d steps", step_idx)
                break

            context = HarnessContext(task=self.task, step_index=step_idx)
            try:
                action = harness.act(obs, context)
            except Exception:
                logger.exception("Harness.act() failed at step %d", step_idx)
                break

            obs, _, terminated, truncated, info = self.step(action)

            if terminated or truncated:
                break

        self._run_task_hook("post_task")

        evaluation = None
        if evaluator is not None:
            evaluation = evaluator.finalize(env=self, trace=self.trace)
            self.trace.record_evaluation(evaluation)

        self.trace.finalize()
        return RunTaskResult(trace=self.trace, evaluation=evaluation)

    def run_setup_script(self, script_content: str) -> dict[str, Any]:

        session = self.session
        container_id = None
        if hasattr(session, "_container") and session._container:
            container_id = session._container.id

        if container_id is None:
            return {"exit_code": -1, "output": "No container reference available"}

        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "/bin/bash", "-c", script_content],
                capture_output=True, text=True, timeout=300,
            )
            return {"exit_code": result.returncode, "output": result.stdout + result.stderr}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "output": "Script timed out after 300s"}

    def _run_task_hook(self, hook_name: str) -> None:

        env_setup = self.task.env_setup
        if not env_setup:
            return
        script_path_str = env_setup.get(hook_name)
        if not script_path_str:
            return

        from pathlib import Path as _Path
        script_path = _Path(script_path_str)
        if not script_path.exists():
            logger.warning("Task hook %s script not found: %s", hook_name, script_path)
            return

        session = self.session
        container_id = None
        if hasattr(session, "_container") and session._container:
            container_id = session._container.id

        if container_id is None:
            logger.warning(
                "Cannot run task hook %s for %s: no container reference available",
                hook_name, self.task.task_id,
            )
            return

        content = script_path.read_text()
        logger.info("[%s/%s] running hook %s", self.task.task_id, hook_name, script_path)
        t0 = time.time()
        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "/bin/bash", "-c", content],
                capture_output=True, text=True, timeout=300,
            )
            elapsed = time.time() - t0
            logger.info(
                "[%s/%s] hook %s exit=%d (%.1fs)",
                self.task.task_id, hook_name, script_path, result.returncode, elapsed,
            )
            if result.stdout:
                for line in result.stdout.splitlines()[-5:]:
                    logger.debug("[%s/%s] %s", self.task.task_id, hook_name, line)
        except subprocess.TimeoutExpired:
            logger.warning("[%s/%s] hook %s timed out after 300s", self.task.task_id, hook_name, script_path)

    def close(self) -> None:
        try:
            self.client.tear_down(self._task_name)
        except Exception:
            logger.warning("tear_down failed during close for %s", self._task_name)
        super().close()

class MobileWorldEvaluator(Evaluator):

    def __init__(self, task: Task | None = None) -> None:
        self.task = task

    def finalize(self, env: Env | None, trace: Trace) -> EvaluationResult:
        if env is None or not isinstance(env, MobileWorldEnv):
            return EvaluationResult(
                success=False,
                score=0.0,
                reason="MobileWorldEvaluator requires a MobileWorldEnv",
                error_type="evaluator.no_env",
            )

        if not (self.task and self.task.native_ref and "task_name" in self.task.native_ref):
            return EvaluationResult(
                success=None,
                score=None,
                reason=(
                    "custom task has no native MobileWorld evaluator config — "
                    "native scoring skipped (pass a custom Evaluator to score it)"
                ),
                native_result={"evaluated": False, "task_id": env.task.task_id},
            )

        task_name = env._task_name
        try:
            score, reason = env.client.evaluate(task_name)
        except Exception as exc:
            logger.exception("evaluate() failed for %s", task_name)
            return EvaluationResult(
                success=False,
                score=0.0,
                reason=f"Evaluation call failed: {exc}",
                error_type="evaluator.server_error",
            )

        return EvaluationResult(
            success=score > 0.0,
            score=score,
            reason=reason,
            native_result={"score": score, "reason": reason},
        )

@dataclass
class ContainerInfo:

    id: str
    port: int
    name: str | None = None

def _docker_run(
    image: str,
    *,
    host_port: int | None = None,
    name: str | None = None,
    extra_args: list[str] | None = None,
) -> str:

    cmd = ["docker", "run", "-d", "--privileged", "--device", "/dev/kvm"]
    if host_port is not None:
        cmd += ["-p", f"{host_port}:6800"]
    else:
        cmd += ["-p", "0:6800"]
    if name:
        cmd += ["--name", name]
    if extra_args:
        cmd += extra_args
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()[:12]

def _docker_port(container_id: str, container_port: int = 6800) -> int:

    result = subprocess.run(
        ["docker", "port", container_id, str(container_port)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip().rsplit(":", 1)[-1])

def _docker_rm(container_id: str) -> None:

    subprocess.run(
        ["docker", "rm", "-f", container_id],
        capture_output=True, check=False,
    )

def _wait_for_health(
    url: str, *, timeout: float = 300, interval: float = 5,
) -> None:

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
                if data.get("ok") or data.get("status") == "success":
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(interval)
    raise TimeoutError(
        f"Container at {url} not healthy within {timeout}s"
        + (f": {last_err}" if last_err else "")
    )

class MobileWorldSession(BackendSession):

    def __init__(
        self,
        task: Task,
        client: MobileWorldClient,
        *,
        container: ContainerInfo | None = None,
        provider: MobileWorldBackendProvider | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(task, **kwargs)
        self.client = client
        self._container = container
        self._provider = provider
        self._mode = mode

    def close(self) -> None:
        if self._mode == "reuse" and self._container and self._provider:

            self._provider._release_container(self._container)
            self._container = None
        elif self._mode == "ephemeral" and self._container:
            _docker_rm(self._container.id)
            self._container = None
        super().close()

class MobileWorldBackendProvider(BackendProvider):

    def __init__(
        self,
        server_url: str | None = None,
        image: str | None = None,
        num_instances: int = 1,
        reuse: bool = True,
        device: str = "emulator-5554",
        timeout: float = 300.0,
        step_wait_time: float = 1.0,
        docker_args: list[str] | None = None,
        boot_timeout: float = 300.0,
        env_hooks: dict[str, Any] | None = None,
    ) -> None:
        self._managed = image is not None
        self._reuse = reuse
        self._image = image
        self._num_instances = num_instances
        self.server_url = server_url or "http://localhost:6800"
        self.device = device
        self.timeout = timeout
        self.step_wait_time = step_wait_time
        self._docker_args = docker_args or []
        self._boot_timeout = boot_timeout
        self.env_hooks = env_hooks or {}

        self._client: MobileWorldClient | None = None
        self._container_queue: queue.Queue[ContainerInfo] | None = None
        self._all_containers: list[ContainerInfo] = []
        self._setup_done = False

    def _get_client(self) -> MobileWorldClient:
        if self._client is None:
            self._client = MobileWorldClient(
                base_url=self.server_url,
                device=self.device,
                timeout=self.timeout,
                step_wait_time=self.step_wait_time,
            )
        return self._client

    def _release_container(self, container: ContainerInfo) -> None:

        if self._container_queue is not None:
            self._container_queue.put(container)

    def setup(self) -> None:
        if not self._managed or self._setup_done:
            return
        if self._reuse:
            self._container_queue = queue.Queue()
            for i in range(self._num_instances):
                name = f"agent_runtime_mw_{i}"
                cid = _docker_run(
                    self._image, name=name, extra_args=self._docker_args,
                )
                port = _docker_port(cid)
                container = ContainerInfo(id=cid, port=port, name=name)
                self._all_containers.append(container)
                logger.info("Started container %s (port %d)", name, port)
            for c in self._all_containers:
                _wait_for_health(
                    f"http://localhost:{c.port}",
                    timeout=self._boot_timeout,
                )
                logger.info("Container %s healthy", c.name)
                self._run_env_hooks(c)
                self._container_queue.put(c)
        self._setup_done = True

    def _run_env_hooks(self, container: ContainerInfo) -> None:

        if not self.env_hooks:
            return

        post_start = self.env_hooks.get("post_start")
        if not post_start:
            return

        from pathlib import Path as _Path
        script_path = _Path(post_start)
        if not script_path.exists():
            logger.warning("env_hooks post_start script not found: %s", post_start)
            return

        script_content = script_path.read_text()
        logger.info("Running env_hook:post_start on container %s", container.name or container.id)
        t0 = time.time()
        try:
            result = subprocess.run(
                ["docker", "exec", container.id, "/bin/bash", "-c", script_content],
                capture_output=True, text=True, timeout=600,
            )
            elapsed = time.time() - t0
            if result.returncode != 0:
                logger.warning(
                    "env_hook:post_start on %s failed (exit=%d, %.1fs): %s",
                    container.name, result.returncode, elapsed,
                    result.stderr[-500:] if result.stderr else "",
                )
            else:
                logger.info("env_hook:post_start on %s completed in %.1fs", container.name, elapsed)
        except subprocess.TimeoutExpired:
            logger.warning("env_hook:post_start on %s timed out after 600s", container.name)

    def teardown(self) -> None:
        for c in self._all_containers:
            logger.info("Removing container %s", c.name or c.id)
            _docker_rm(c.id)
        self._all_containers.clear()
        if self._container_queue is not None:
            while not self._container_queue.empty():
                with contextlib.suppress(Exception):
                    self._container_queue.get_nowait()
            self._container_queue = None
        self._setup_done = False

    def create_session(self, task: Task, **kwargs: Any) -> MobileWorldSession:
        if not self._managed:
            return MobileWorldSession(task=task, client=self._get_client())

        if self._reuse:
            if not self._setup_done:
                self.setup()
            assert self._container_queue is not None
            container = self._container_queue.get()
            client = MobileWorldClient(
                base_url=f"http://localhost:{container.port}",
                device=self.device,
                timeout=self.timeout,
                step_wait_time=self.step_wait_time,
            )
            return MobileWorldSession(
                task=task, client=client,
                container=container, provider=self, mode="reuse",
            )

        cid = _docker_run(self._image, extra_args=self._docker_args)
        port = _docker_port(cid)
        url = f"http://localhost:{port}"
        _wait_for_health(url, timeout=self._boot_timeout)
        container = ContainerInfo(id=cid, port=port)
        client = MobileWorldClient(
            base_url=url, device=self.device,
            timeout=self.timeout, step_wait_time=self.step_wait_time,
        )
        return MobileWorldSession(
            task=task, client=client,
            container=container, provider=self, mode="ephemeral",
        )

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "type": "mobile_world",
            "protocol": "http",
            "session_type": "android_emulator",
        }
        if self._managed:
            caps["managed"] = True
            caps["reuse"] = self._reuse
            caps["num_instances"] = self._num_instances
        else:
            caps["managed"] = False
            caps["server_url"] = self.server_url
        return caps

    def task_resources(self) -> dict[str, float]:
        return {"docker_slot": 1}

@register_benchmark(
    "mobile_world",
    description="MobileWorld Android GUI benchmark (HTTP server adapter).",
    overwrite=True,
)
class MobileWorldBenchmark(Benchmark):

    name = "mobile_world"

    def __init__(
        self,
        server_url: str = "http://localhost:6800",
        device: str = "emulator-5554",
        timeout: float = 300.0,
        enable_mcp: bool = False,
        enable_user_interaction: bool = False,
    ) -> None:
        self.server_url = server_url
        self.device = device
        self.timeout = timeout
        self.enable_mcp = enable_mcp
        self.enable_user_interaction = enable_user_interaction
        self._client: MobileWorldClient | None = None

    def _get_client(self) -> MobileWorldClient:
        if self._client is None:
            self._client = MobileWorldClient(
                base_url=self.server_url,
                device=self.device,
                timeout=self.timeout,
            )
        return self._client

    def list_tasks(
        self,
        split: str | None = None,
        limit: int | None = None,
        **filters: Any,
    ) -> list[Task]:
        client = self._get_client()
        enable_mcp = filters.pop("enable_mcp", self.enable_mcp)
        enable_user_interaction = filters.pop(
            "enable_user_interaction", self.enable_user_interaction
        )

        raw_tasks = client.list_tasks(
            enable_mcp=enable_mcp,
            enable_user_interaction=enable_user_interaction,
        )

        tasks: list[Task] = []
        for raw in raw_tasks:
            task_name = raw["name"]
            tags_list = raw.get("tags", [])
            apps = raw.get("apps", [])

            suite = None
            if apps:
                suite = apps[0] if len(apps) == 1 else ",".join(sorted(apps))

            tasks.append(Task(
                task_id=task_name,
                benchmark="mobile_world",
                suite=suite,
                instruction=None,
                native_ref={
                    "task_name": task_name,
                    "tags": tags_list,
                    "apps": apps,
                },
                resource_requirements={
                    "android_emulator": 1,
                    "kvm": True,
                },
                metadata={
                    "tags": tags_list,
                    "apps": apps,
                },
                tags=tags_list,
            ))

        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def get_task(self, task_id: str) -> Task:
        client = self._get_client()
        goal = client.get_task_goal(task_id)
        metadata = client.get_task_metadata(task_id)
        tags_list = metadata.get("tags", [])
        apps = metadata.get("apps", [])

        try:
            complexity = client.get_task_complexity(task_id)
        except Exception:
            complexity = None

        suite = None
        if apps:
            suite = apps[0] if len(apps) == 1 else ",".join(sorted(apps))

        return Task(
            task_id=task_id,
            benchmark="mobile_world",
            suite=suite,
            instruction=goal,
            native_ref={
                "task_name": task_id,
                "tags": tags_list,
                "apps": apps,
            },
            resource_requirements={
                "android_emulator": 1,
                "kvm": True,
            },
            metadata={
                "tags": tags_list,
                "apps": apps,
                "complexity": complexity,
            },
            tags=tags_list,
        )

    def make_env(self, task: Task, session: Any = None, **env_options: Any) -> MobileWorldEnv:
        return MobileWorldEnv(task=task, session=session, **env_options)

    def make_evaluator(
        self, task: Task | None = None, mode: str = "native", **kwargs: Any
    ) -> MobileWorldEvaluator:
        return MobileWorldEvaluator(task=task)

__all__ = [
    "ContainerInfo",
    "MobileWorldBackendProvider",
    "MobileWorldBenchmark",
    "MobileWorldClient",
    "MobileWorldEnv",
    "MobileWorldEvaluator",
    "MobileWorldSession",
    "action_to_native",
    "native_to_action",
]
