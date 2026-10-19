from __future__ import annotations

import base64
import io
import json as _json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
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

_AW_CLICK = "click"
_AW_LONG_PRESS = "long_press"
_AW_SWIPE = "swipe"
_AW_SCROLL = "scroll"
_AW_INPUT_TEXT = "input_text"
_AW_NAVIGATE_HOME = "navigate_home"
_AW_NAVIGATE_BACK = "navigate_back"
_AW_KEYBOARD_ENTER = "keyboard_enter"
_AW_OPEN_APP = "open_app"
_AW_WAIT = "wait"
_AW_STATUS = "status"

_AW_TERMINAL_ACTIONS = frozenset({_AW_STATUS})

class AndroidWorldClient:

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: float = 300.0,
        step_wait_time: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.step_wait_time = step_wait_time

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

    def health_check(self) -> bool:
        try:
            result = self._get("/health", timeout=10)
            return bool(result and result.get("status") == "success")
        except Exception:
            return False

    def reset(self, go_home: bool = True) -> dict[str, Any]:
        return self._post("/reset", params={"go_home": str(go_home).lower()})

    def list_task_types(self, max_index: int = -1) -> list[str]:
        result = self._get("/suite/task_list", params={"max_index": str(max_index)})
        return result.get("task_list", [])

    def get_task_length(self, task_type: str) -> int:
        result = self._get("/suite/task_length", params={"task_type": task_type})
        return result.get("length", 0)

    def get_task_goal(self, task_type: str, task_idx: int = 0) -> str:
        result = self._get(
            "/task/goal",
            params={"task_type": task_type, "task_idx": str(task_idx)},
        )
        return result.get("goal", "")

    def get_task_template(self, task_type: str, task_idx: int = 0) -> str:
        result = self._get(
            "/task/template",
            params={"task_type": task_type, "task_idx": str(task_idx)},
        )
        return result.get("template", "")

    def initialize_task(self, task_type: str, task_idx: int = 0) -> dict[str, Any]:
        return self._post(
            "/task/initialize",
            params={"task_type": task_type, "task_idx": str(task_idx)},
        )

    def get_screenshot_pixels(self) -> list[int]:
        result = self._get(
            "/screenshot",
            params={"wait_to_stabilize": "false"},
        )
        return result.get("pixels", [])

    def get_screenshot_png_bytes(self) -> bytes:

        pixels = self.get_screenshot_pixels()

        from PIL import Image

        if not pixels:
            img = Image.new("RGB", (1080, 2400), (0, 0, 0))
        elif isinstance(pixels[0], list):

            height = len(pixels)
            width = len(pixels[0]) if height > 0 else 0
            raw = bytearray()
            for row in pixels:
                for px in row:
                    raw.append(px[0])
                    raw.append(px[1])
                    raw.append(px[2])
            img = Image.frombytes("RGB", (width, height), bytes(raw))
        else:

            n = len(pixels)
            channels = 3
            width = 1080
            height = (n // channels) // width
            raw_bytes = bytes(pixels[: height * width * channels])
            img = Image.frombytes("RGB", (width, height), raw_bytes)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_screenshot_b64(self) -> str:

        return base64.b64encode(self.get_screenshot_png_bytes()).decode("ascii")

    def execute_action(self, action_dict: dict[str, Any]) -> dict[str, Any]:
        return self._post("/execute_action", json_body=action_dict)

    def get_task_score(self, task_type: str, task_idx: int = 0) -> float:
        result = self._get(
            "/task/score",
            params={"task_type": task_type, "task_idx": str(task_idx)},
        )
        return float(result.get("score", 0.0))

    def tear_down(self, task_type: str, task_idx: int = 0) -> dict[str, Any]:
        try:
            return self._post(
                "/task/tear_down",
                params={"task_type": task_type, "task_idx": str(task_idx)},
            )
        except Exception as exc:
            logger.warning("tear_down failed for %s[%d]: %s", task_type, task_idx, exc)
            return {}

    def close(self) -> None:
        pass

_PLATFORM_TO_AW: dict[ActionType, str] = {
    ActionType.TAP: _AW_CLICK,
    ActionType.CLICK: _AW_CLICK,
    ActionType.LONG_PRESS: _AW_LONG_PRESS,
    ActionType.SWIPE: _AW_SWIPE,
    ActionType.SCROLL: _AW_SCROLL,
    ActionType.TYPE: _AW_INPUT_TEXT,
    ActionType.NOOP: _AW_WAIT,
    ActionType.TERMINATE: _AW_STATUS,
}

_AW_TO_PLATFORM: dict[str, ActionType] = {
    _AW_CLICK: ActionType.TAP,
    _AW_LONG_PRESS: ActionType.LONG_PRESS,
    _AW_SWIPE: ActionType.SWIPE,
    _AW_SCROLL: ActionType.SCROLL,
    _AW_INPUT_TEXT: ActionType.TYPE,
    _AW_NAVIGATE_HOME: ActionType.KEY_PRESS,
    _AW_NAVIGATE_BACK: ActionType.KEY_PRESS,
    _AW_KEYBOARD_ENTER: ActionType.KEY_PRESS,
    _AW_OPEN_APP: ActionType.CUSTOM,
    _AW_WAIT: ActionType.NOOP,
    _AW_STATUS: ActionType.TERMINATE,
}

def action_to_native(action: Action) -> dict[str, Any]:

    args = action.arguments
    native: dict[str, Any] = {}

    if action.type in (ActionType.TAP, ActionType.CLICK):
        native = {"action_type": _AW_CLICK, "x": args.get("x"), "y": args.get("y")}
    elif action.type == ActionType.LONG_PRESS:
        native = {"action_type": _AW_LONG_PRESS, "x": args.get("x"), "y": args.get("y")}
    elif action.type == ActionType.SWIPE:
        native = {
            "action_type": _AW_SWIPE,
            "x": args.get("x"),
            "y": args.get("y"),
            "direction": args.get("direction", "up"),
        }
    elif action.type == ActionType.SCROLL:
        native = {
            "action_type": _AW_SCROLL,
            "x": args.get("x", 540),
            "y": args.get("y", 1200),
            "direction": args.get("direction", "down"),
        }
    elif action.type == ActionType.TYPE:
        native = {"action_type": _AW_INPUT_TEXT, "text": args.get("text", "")}
    elif action.type == ActionType.KEY_PRESS:
        key = args.get("key", action.name or "")
        key_lower = key.lower()
        if key_lower in ("home", "navigate_home"):
            native = {"action_type": _AW_NAVIGATE_HOME}
        elif key_lower in ("back", "navigate_back"):
            native = {"action_type": _AW_NAVIGATE_BACK}
        elif key_lower in ("enter", "keyboard_enter"):
            native = {"action_type": _AW_KEYBOARD_ENTER}
        else:
            native = {"action_type": _AW_KEYBOARD_ENTER}
    elif action.type == ActionType.TERMINATE:
        reason = args.get("reason", "")
        native = {"action_type": _AW_STATUS, "goal_status": reason}
    elif action.type == ActionType.NOOP:
        native = {"action_type": _AW_WAIT}
    elif action.type == ActionType.CUSTOM:
        custom_name = (action.name or "").lower()
        if custom_name == "open_app":
            native = {"action_type": _AW_OPEN_APP, "app_name": args.get("app_name", "")}
        else:
            native = {"action_type": custom_name, **args}
    else:
        native = {"action_type": _AW_WAIT}

    native = {k: v for k, v in native.items() if v is not None}
    return native

def native_to_action(native: dict[str, Any], raw: str | None = None) -> Action:

    action_type_str = native.get("action_type", _AW_WAIT)
    platform_type = _AW_TO_PLATFORM.get(action_type_str, ActionType.CUSTOM)

    args: dict[str, Any] = {}
    name: str | None = None

    if action_type_str == _AW_CLICK:
        args = {"x": native.get("x"), "y": native.get("y")}
    elif action_type_str == _AW_LONG_PRESS:
        args = {"x": native.get("x"), "y": native.get("y")}
    elif action_type_str == _AW_SWIPE:
        args = {"x": native.get("x"), "y": native.get("y"), "direction": native.get("direction")}
    elif action_type_str == _AW_SCROLL:
        args = {"x": native.get("x"), "y": native.get("y"), "direction": native.get("direction")}
    elif action_type_str == _AW_INPUT_TEXT:
        args = {"text": native.get("text", "")}
    elif action_type_str in (_AW_NAVIGATE_HOME, _AW_NAVIGATE_BACK, _AW_KEYBOARD_ENTER):
        key_map = {
            _AW_NAVIGATE_HOME: "home",
            _AW_NAVIGATE_BACK: "back",
            _AW_KEYBOARD_ENTER: "enter",
        }
        args = {"key": key_map[action_type_str]}
    elif action_type_str == _AW_OPEN_APP:
        name = "open_app"
        args = {"app_name": native.get("app_name", "")}
    elif action_type_str == _AW_STATUS:
        args = {"reason": native.get("goal_status", "")}

    args = {k: v for k, v in args.items() if v is not None}
    return Action(
        type=platform_type,
        name=name,
        arguments=args,
        raw=raw,
        metadata={"native": native},
    )

class AndroidWorldEnv(Env):

    def __init__(self, task: Task, session: Any = None, **env_options: Any) -> None:
        super().__init__(task, session, **env_options)
        self._capabilities = EnvCapabilities(step=True, screenshot=True)
        ref = task.native_ref or {}
        self._task_type: str = ref.get("task_type", task.task_id)
        self._task_idx: int = ref.get("task_idx", 0)
        self._task_goal: str | None = None
        self._step_count = 0
        self._terminated = False
        self._last_obs: Observation | None = None
        self._screenshot_dir: str | None = env_options.get("screenshot_dir")

        if session is not None and hasattr(session, "client"):
            self.client: AndroidWorldClient = session.client
        else:
            raise ValueError(
                "AndroidWorldEnv requires an AndroidWorldSession with a .client attribute."
            )

    def _save_screenshot(self, png_bytes: bytes, step_index: int) -> str | None:
        if not self._screenshot_dir:
            return None
        import os
        os.makedirs(self._screenshot_dir, exist_ok=True)
        filename = f"step_{step_index:03d}.png"
        filepath = os.path.join(self._screenshot_dir, filename)
        with open(filepath, "wb") as f:
            f.write(png_bytes)
        return filepath

    def _get_screenshot_observation(
        self,
        step_index: int,
        terminal: bool = False,
    ) -> Observation:
        png_bytes = self.client.get_screenshot_png_bytes()
        screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")

        rel_uri = self._save_screenshot(png_bytes, step_index)

        parts: list[Modality] = [
            Modality(
                kind=ModalityKind.IMAGE,
                content=screenshot_b64 if not rel_uri else None,
                uri=rel_uri,
                mime_type="image/png",
            ),
        ]
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
        self.client.reset(go_home=True)
        self.client.initialize_task(self._task_type, self._task_idx)
        self._task_goal = self.client.get_task_goal(self._task_type, self._task_idx)
        self._step_count = 0
        self._terminated = False

        time.sleep(self.client.step_wait_time)
        obs = self._get_screenshot_observation(step_index=0)
        obs.instruction = self._task_goal
        self._last_obs = obs
        self.trace.record_step(observation=obs)

        return obs, {"task_type": self._task_type, "task_idx": self._task_idx, "task_goal": self._task_goal}

    def step(
        self, action: Action
    ) -> tuple[Observation, float | None, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset() first.")

        self._step_count += 1
        native_action = action_to_native(action)
        action_type = native_action.get("action_type", _AW_WAIT)

        action.metadata["native"] = native_action

        terminated = False
        truncated = False
        info: dict[str, Any] = {"native_action": native_action}

        if action_type in _AW_TERMINAL_ACTIONS:
            terminated = True
        else:
            try:
                result = self.client.execute_action(native_action)
                info["step_result"] = result
            except Exception as exc:
                logger.warning("execute_action failed: %s", exc)
                info["step_error"] = str(exc)

        time.sleep(self.client.step_wait_time)

        obs = self._get_screenshot_observation(
            step_index=self._step_count,
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

        evaluation = None
        if evaluator is not None:
            evaluation = evaluator.finalize(env=self, trace=self.trace)
            self.trace.record_evaluation(evaluation)

        self.trace.finalize()
        return RunTaskResult(trace=self.trace, evaluation=evaluation)

    def close(self) -> None:
        try:
            self.client.tear_down(self._task_type, self._task_idx)
        except Exception:
            logger.warning("tear_down failed during close for %s", self._task_type)
        super().close()

class AndroidWorldEvaluator(Evaluator):

    def __init__(self, task: Task | None = None) -> None:
        self.task = task

    def finalize(self, env: Env | None, trace: Trace) -> EvaluationResult:
        if env is None or not isinstance(env, AndroidWorldEnv):
            return EvaluationResult(
                success=False,
                score=0.0,
                reason="AndroidWorldEvaluator requires an AndroidWorldEnv",
                error_type="evaluator.no_env",
            )

        if not (self.task and self.task.native_ref and "task_type" in self.task.native_ref):
            return EvaluationResult(
                success=None,
                score=None,
                reason=(
                    "custom task has no native AndroidWorld evaluator config — "
                    "native scoring skipped (pass a custom Evaluator to score it)"
                ),
                native_result={"evaluated": False, "task_id": env.task.task_id},
            )

        task_type = env._task_type
        task_idx = env._task_idx
        try:
            score = env.client.get_task_score(task_type, task_idx)
        except Exception as exc:
            logger.exception("get_task_score() failed for %s[%d]", task_type, task_idx)
            return EvaluationResult(
                success=False,
                score=0.0,
                reason=f"Evaluation call failed: {exc}",
                error_type="evaluator.server_error",
            )

        return EvaluationResult(
            success=score > 0.0,
            score=score,
            reason=None if score > 0.0 else "task not successful",
            native_result={"score": score, "task_type": task_type, "task_idx": task_idx},
        )

class AndroidWorldSession(BackendSession):

    def __init__(
        self,
        task: Task,
        client: AndroidWorldClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(task, **kwargs)
        self.client = client

    def close(self) -> None:
        super().close()

class AndroidWorldBackendProvider(BackendProvider):

    def __init__(
        self,
        server_url: str = "http://localhost:5000",
        timeout: float = 300.0,
        step_wait_time: float = 1.0,
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self.step_wait_time = step_wait_time
        self._client: AndroidWorldClient | None = None

    def _get_client(self) -> AndroidWorldClient:
        if self._client is None:
            self._client = AndroidWorldClient(
                base_url=self.server_url,
                timeout=self.timeout,
                step_wait_time=self.step_wait_time,
            )
        return self._client

    def create_session(self, task: Task, **kwargs: Any) -> AndroidWorldSession:
        return AndroidWorldSession(task=task, client=self._get_client(), **kwargs)

    def capabilities(self) -> dict[str, Any]:
        return {
            "type": "android_world",
            "protocol": "http",
            "session_type": "android_emulator_docker",
            "server_url": self.server_url,
        }

    def task_resources(self) -> dict[str, float]:
        return {"docker_slot": 1}

@register_benchmark(
    "android_world",
    description="AndroidWorld Android GUI benchmark (Docker server adapter).",
    overwrite=True,
)
class AndroidWorldBenchmark(Benchmark):

    name = "android_world"

    def __init__(
        self,
        server_url: str = "http://localhost:5000",
        timeout: float = 300.0,
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self._client: AndroidWorldClient | None = None

    def _get_client(self) -> AndroidWorldClient:
        if self._client is None:
            self._client = AndroidWorldClient(
                base_url=self.server_url,
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
        task_types = client.list_task_types(max_index=-1)

        tasks: list[Task] = []
        for task_type in task_types:
            n_instances = client.get_task_length(task_type)
            for idx in range(n_instances):
                template = ""
                try:
                    template = client.get_task_template(task_type, idx)
                except Exception:
                    pass

                task_id = f"{task_type}_{idx}" if n_instances > 1 else task_type
                tasks.append(Task(
                    task_id=task_id,
                    benchmark="android_world",
                    suite=task_type,
                    instruction=None,
                    native_ref={
                        "task_type": task_type,
                        "task_idx": idx,
                    },
                    resource_requirements={
                        "android_emulator": 1,
                        "kvm": True,
                        "docker": True,
                    },
                    metadata={
                        "template": template,
                    },
                ))

        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def get_task(self, task_id: str) -> Task:
        client = self._get_client()
        task_type = task_id
        task_idx = 0

        if "_" in task_id:
            parts = task_id.rsplit("_", 1)
            if parts[-1].isdigit():
                task_type, task_idx = parts[0], int(parts[1])

        goal = client.get_task_goal(task_type, task_idx)
        template = client.get_task_template(task_type, task_idx)

        return Task(
            task_id=task_id,
            benchmark="android_world",
            suite=task_type,
            instruction=goal,
            native_ref={
                "task_type": task_type,
                "task_idx": task_idx,
            },
            resource_requirements={
                "android_emulator": 1,
                "kvm": True,
                "docker": True,
            },
            metadata={
                "template": template,
            },
        )

    def make_env(self, task: Task, session: Any = None, **env_options: Any) -> AndroidWorldEnv:
        return AndroidWorldEnv(task=task, session=session, **env_options)

    def make_evaluator(
        self, task: Task | None = None, mode: str = "native", **kwargs: Any
    ) -> AndroidWorldEvaluator:
        return AndroidWorldEvaluator(task=task)

__all__ = [
    "AndroidWorldBackendProvider",
    "AndroidWorldBenchmark",
    "AndroidWorldClient",
    "AndroidWorldEnv",
    "AndroidWorldEvaluator",
    "AndroidWorldSession",
    "action_to_native",
    "native_to_action",
]
