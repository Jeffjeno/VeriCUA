from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from .dataset import MobileWorldEpisode

logger = logging.getLogger("eval_runtime.mobile_world.env")

TERMINAL_WITHOUT_EXECUTION = frozenset({"finished", "unknown", "error_env"})
TERMINAL_AFTER_EXECUTION = frozenset({"answer"})
UNSUPPORTED_ACTIVE_ACTIONS = frozenset({"ask_user", "mcp"})
SUPPORTED_ACTIONS = frozenset(
    {
        "answer",
        "click",
        "double_tap",
        "drag",
        "finished",
        "input_text",
        "keyboard_enter",
        "long_press",
        "navigate_back",
        "navigate_home",
        "open_app",
        "scroll",
        "status",
        "swipe",
        "unknown",
        "wait",
        "error_env",
    }
)

class MobileWorldClientError(RuntimeError):
    pass

class MobileWorldEnvironmentUnhealthy(MobileWorldClientError):
    pass

@dataclass(frozen=True)
class MobileWorldStepResult:
    action: dict[str, Any]
    executed: bool
    terminal: bool
    screenshot: bytes
    response: dict[str, Any] | None

@dataclass(frozen=True)
class MobileWorldReplayResult:
    task_name: str
    initial_screenshot: bytes
    steps: tuple[MobileWorldStepResult, ...]
    score: float
    reason: str | None

    @property
    def final_screenshot(self) -> bytes:
        return self.steps[-1].screenshot if self.steps else self.initial_screenshot

def validate_mobile_action(
    action: dict[str, Any],
    *,
    allow_terminal: bool = True,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise TypeError(f"MobileWorld action must be an object, got {type(action).__name__}")
    normalized = {key: value for key, value in action.items() if value is not None}
    action_type = normalized.get("action_type")
    if action_type is None:
        return {"action_type": None}
    action_type = str(action_type)
    normalized["action_type"] = action_type
    if action_type not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported MobileWorld action_type: {action_type!r}")
    if action_type in UNSUPPORTED_ACTIVE_ACTIONS:
        raise ValueError(
            f"Action {action_type!r} requires interaction unavailable to an evaluator"
        )
    if not allow_terminal and action_type in TERMINAL_WITHOUT_EXECUTION:
        raise ValueError(f"Terminal action is not allowed here: {action_type!r}")

    required: dict[str, tuple[str, ...]] = {
        "click": ("x", "y"),
        "double_tap": ("x", "y"),
        "long_press": ("x", "y"),
        "input_text": ("text",),
        "open_app": ("app_name",),
        "drag": ("start_x", "start_y", "end_x", "end_y"),
    }
    missing = [key for key in required.get(action_type, ()) if key not in normalized]
    if missing:
        raise ValueError(f"Action {action_type!r} is missing fields: {missing}")
    if action_type in {"scroll", "swipe"}:
        direction = normalized.get("direction")
        if direction not in {"left", "right", "up", "down"}:
            raise ValueError(
                f"Action {action_type!r} requires direction left/right/up/down"
            )
    coordinate_fields = (
        "x",
        "y",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    )
    for key in coordinate_fields:
        if key in normalized:
            value = normalized[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Action coordinate {key!r} must be numeric")
            normalized[key] = round(value)
    if "index" in normalized:
        raise ValueError("Index-based actions are not supported by MobileWorld HTTP")
    return normalized

class MobileWorldEvalEnvClient:

    def __init__(
        self,
        base_url: str,
        *,
        device: str = "emulator-5554",
        step_wait_time: float = 1.0,
        request_timeout: float = 60.0,
        recovery_timeout: float = 180.0,
        health_poll_interval: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device = device
        self.step_wait_time = float(step_wait_time)
        self.request_timeout = float(request_timeout)
        self.recovery_timeout = float(recovery_timeout)
        self.health_poll_interval = float(health_poll_interval)
        self.session = session or requests.Session()
        self.lease_id = f"mobile-world:{self.base_url}:{self.device}"
        self.active_task: str | None = None
        self._initialized = False

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout if timeout is not None else self.request_timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise MobileWorldClientError(
                f"{method} {self.base_url}{path} failed: {exc}"
            ) from exc
        if not response.ok:
            body = response.text[:2000]
            raise MobileWorldClientError(
                f"{method} {self.base_url}{path} returned "
                f"HTTP {response.status_code}: {body}"
            )
        return response

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._request("POST", "/init", json={"device": self.device})
        self._initialized = True

    def health_details(self) -> dict[str, Any]:
        try:
            response = self._request("GET", "/health", timeout=60.0)
            payload = response.json()
            return payload if isinstance(payload, dict) else {"ok": False, "raw": payload}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def health(self) -> bool:
        return bool(self.health_details().get("ok", False))

    def wait_until_healthy(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + (
            self.recovery_timeout if timeout is None else float(timeout)
        )
        last_status: dict[str, Any] = {"ok": False}
        while time.monotonic() < deadline:
            last_status = self.health_details()
            if last_status.get("ok"):
                return last_status
            time.sleep(self.health_poll_interval)
        raise MobileWorldEnvironmentUnhealthy(
            f"Backend did not become healthy within "
            f"{self.recovery_timeout if timeout is None else timeout}s: "
            f"{self.base_url} ({last_status})"
        )

    def task_list(self) -> list[dict[str, Any]]:
        self.ensure_initialized()
        payload = self._request("GET", "/task/list").json()
        if not isinstance(payload, list):
            raise MobileWorldClientError(f"Invalid /task/list response: {payload!r}")
        return [dict(value) for value in payload if isinstance(value, dict)]

    def task_goal(self, task_name: str) -> str:
        self.ensure_initialized()
        return str(
            self._request(
                "GET",
                "/task/goal",
                params={"task_name": task_name},
            ).json()
        )

    def initialize_task(self, task_name: str, *, attempts: int = 2) -> bytes:
        self.ensure_initialized()
        last_error: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                self.wait_until_healthy()
                self._request(
                    "POST",
                    "/task/init",
                    timeout=300.0,
                    json={"task_name": task_name, "req_device": self.device},
                )
                self.active_task = task_name
                return self.screenshot(wait=True)
            except Exception as exc:
                last_error = exc
                self.active_task = None
                logger.warning(
                    "MobileWorld task init failed on %s (attempt %d/%d): %s",
                    self.base_url,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(self.health_poll_interval)
                    self._initialized = False
                    self.ensure_initialized()
        raise MobileWorldClientError(
            f"Failed to initialize task {task_name!r} on {self.base_url}: {last_error}"
        ) from last_error

    def screenshot(self, *, wait: bool = False) -> bytes:
        self.ensure_initialized()
        if wait and self.step_wait_time > 0:
            time.sleep(self.step_wait_time)
        response = self._request(
            "GET",
            "/screenshot",
            params={"device": self.device, "return_b64": True},
        )
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("b64_png"):
            raise MobileWorldClientError(f"Invalid screenshot response: {payload!r}")
        try:
            return base64.b64decode(str(payload["b64_png"]), validate=True)
        except Exception as exc:
            raise MobileWorldClientError("Invalid base64 screenshot payload") from exc

    def step(
        self,
        action: dict[str, Any],
        *,
        wait: bool = True,
    ) -> MobileWorldStepResult:
        normalized = validate_mobile_action(action)
        action_type = normalized.get("action_type")
        if action_type in TERMINAL_WITHOUT_EXECUTION or action_type is None:
            if wait and self.step_wait_time > 0:
                time.sleep(self.step_wait_time)
            return MobileWorldStepResult(
                action=normalized,
                executed=False,
                terminal=action_type in TERMINAL_WITHOUT_EXECUTION,
                screenshot=self.screenshot(wait=False),
                response=None,
            )
        response = self._request(
            "POST",
            "/step",
            json={"device": self.device, "action": normalized},
        )
        response_payload = response.json()
        return MobileWorldStepResult(
            action=normalized,
            executed=True,
            terminal=action_type in TERMINAL_AFTER_EXECUTION,
            screenshot=self.screenshot(wait=wait),
            response=(
                dict(response_payload)
                if isinstance(response_payload, dict)
                else {"raw": response_payload}
            ),
        )

    def evaluate_task(self, task_name: str | None = None) -> tuple[float, str | None]:
        resolved_task = task_name or self.active_task
        if not resolved_task:
            raise MobileWorldClientError("No active task to evaluate")
        payload = self._request(
            "GET",
            "/task/eval",
            json={"task_name": resolved_task, "req_device": self.device},
        ).json()
        if not isinstance(payload, dict) or "score" not in payload:
            raise MobileWorldClientError(f"Invalid task evaluation response: {payload!r}")
        return float(payload["score"]), (
            str(payload["reason"]) if payload.get("reason") is not None else None
        )

    def tear_down_task(self, task_name: str | None = None) -> None:
        resolved_task = task_name or self.active_task
        if not resolved_task:
            return
        try:
            self._request(
                "POST",
                "/task/tear_down",
                json={"task_name": resolved_task, "req_device": self.device},
            )
        finally:
            self.active_task = None

    def replay_episode(self, episode: MobileWorldEpisode) -> MobileWorldReplayResult:
        initial = self.initialize_task(episode.record.task_name)
        step_results: list[MobileWorldStepResult] = []
        for source_step in episode.steps:
            result = self.step(source_step.action)
            step_results.append(result)
            if result.terminal:
                break
        score, reason = self.evaluate_task(episode.record.task_name)
        return MobileWorldReplayResult(
            task_name=episode.record.task_name,
            initial_screenshot=initial,
            steps=tuple(step_results),
            score=score,
            reason=reason,
        )

    def get_obs(self, lease_id: str | None = None) -> dict[str, Any]:
        del lease_id
        return {"screenshot": self.screenshot(wait=False), "accessibility_tree": None}

    def execute_action(
        self,
        lease_id_or_action: str | dict[str, Any] | None,
        action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action is None:
            if not isinstance(lease_id_or_action, dict):
                raise TypeError("execute_action requires a MobileWorld action object")
            action = lease_id_or_action
        result = self.step(action)
        return {
            "screenshot": result.screenshot,
            "accessibility_tree": None,
            "done": result.terminal,
            "response": result.response,
        }

    def evaluate(self, lease_id: str | None = None) -> float:
        del lease_id
        return self.evaluate_task()[0]

    def prepare_for_probe(self, lease_id: str | None = None) -> None:
        del lease_id

    def close(self) -> None:
        self.session.close()

__all__ = [
    "MobileWorldClientError",
    "MobileWorldEnvironmentUnhealthy",
    "MobileWorldEvalEnvClient",
    "MobileWorldReplayResult",
    "MobileWorldStepResult",
    "SUPPORTED_ACTIONS",
    "TERMINAL_AFTER_EXECUTION",
    "TERMINAL_WITHOUT_EXECUTION",
    "validate_mobile_action",
]
