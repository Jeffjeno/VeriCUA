from __future__ import annotations

import atexit
import logging
import math
import os
import uuid
from typing import Any

import eval_runtime

from agent_runtime.benchmarks.osworld import (
    OSWorldBackendProvider,
    OSWorldEnv,
    OSWorldEvaluator,
)
from agent_runtime.types import Action, ActionType, Task

logger = logging.getLogger(__name__)

def _obs_to_dict(env: OSWorldEnv) -> dict[str, Any]:

    desktop_env = env.desktop_env
    controller = desktop_env.controller
    require_terminal = getattr(desktop_env, "require_terminal", False)
    return {
        "screenshot": controller.get_screenshot(),
        "terminal": controller.get_terminal_output() if require_terminal else None,
    }

def _task_config_to_ap_task(task_config: dict[str, Any]) -> Task:

    domain = task_config.get("_domain", "unknown")
    related_apps = task_config.get("related_apps", [])
    suite = domain if domain != "unknown" else (
        related_apps[0] if len(related_apps) == 1 else ",".join(sorted(related_apps))
    )
    return Task(
        task_id=task_config.get("id", str(uuid.uuid4())),
        benchmark="osworld",
        suite=suite,
        instruction=task_config.get("instruction"),
        native_ref={
            "task_config": task_config,
            "domain": domain,
            "snapshot": task_config.get("snapshot", "init_state"),
        },
    )

def cleanup_docker_containers(tag: str = "") -> int:

    try:
        import docker as docker_lib
        client = docker_lib.from_env()
        if tag:
            label_filters = [f"osworld.tag={tag}"]
        else:
            label_filters = ["osworld.managed=true"]
        containers = client.containers.list(all=True, filters={"label": label_filters})
        removed = 0
        for c in containers:
            try:
                c.remove(force=True)
                removed += 1
            except Exception:
                pass
        if removed:
            logger.info("Cleaned up %d orphan Docker container(s) (tag=%s)", removed, tag or "*")
        return removed
    except Exception as exc:
        logger.debug("Docker orphan cleanup skipped: %s", exc)
        return 0

def cleanup_docker_by_group(run_group: str) -> int:

    if not run_group:
        return 0
    try:
        import docker as docker_lib
        client = docker_lib.from_env()
        containers = client.containers.list(
            all=True, filters={"label": [f"osworld.run_group={run_group}"]}
        )
        removed = 0
        for c in containers:
            try:
                c.remove(force=True)
                removed += 1
            except Exception:
                pass
        if removed:
            logger.info(
                "Pre-run cleanup: removed %d container(s) with run_group=%s on this node",
                removed, run_group,
            )
        return removed
    except Exception as exc:
        logger.debug("Docker run_group cleanup skipped: %s", exc)
        return 0

def _ensure_docker_tag(docker_tag: str) -> str:

    if docker_tag:
        os.environ["OSWORLD_DOCKER_TAG"] = docker_tag
        return docker_tag
    existing = os.environ.get("OSWORLD_DOCKER_TAG", "").strip()
    if existing:
        return existing
    auto_tag = f"eval-ap-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    os.environ["OSWORLD_DOCKER_TAG"] = auto_tag
    logger.info("Auto-generated OSWORLD_DOCKER_TAG=%s for container isolation", auto_tag)
    return auto_tag

class AgentRuntimeEnvClient:

    def __init__(
        self,
        osworld_path: str,
        provider_name: str = "docker",
        path_to_vm: str = "",
        headless: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 2.0,
        docker_tag: str = "",
        **kwargs: Any,
    ):
        self._osworld_path = osworld_path
        self._provider_name = provider_name
        self._path_to_vm = path_to_vm
        self._headless = headless
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._sleep_after_execution = sleep_after_execution
        self._docker_tag = _ensure_docker_tag(docker_tag)
        self._extra_kwargs = kwargs
        self._closed = False

        self._sessions: dict[str, dict[str, Any]] = {}

        self._backend = self._create_backend()
        self._backend.setup()

        atexit.register(self._atexit_cleanup)

    def _create_backend(self) -> OSWorldBackendProvider:

        backend_kwargs: dict[str, Any] = {
            "osworld_path": self._osworld_path,
            "provider_name": self._provider_name,
            "headless": self._headless,
            "screen_size": (self._screen_width, self._screen_height),
            "reuse": True,
            "num_instances": 1,
            "require_a11y_tree": False,
        }
        if self._path_to_vm:
            backend_kwargs["path_to_vm"] = self._path_to_vm

        backend_kwargs.update(self._extra_kwargs)
        return OSWorldBackendProvider(**backend_kwargs)

    def _atexit_cleanup(self) -> None:

        if not self._closed:
            try:
                self.close_all()
            except Exception:
                pass

    def allocate(self, episode_id: str) -> dict[str, Any]:

        if self._closed:
            raise RuntimeError("Client has been closed")
        lease_id = episode_id
        try:
            placeholder_task = Task(
                task_id=lease_id,
                benchmark="osworld",
                suite="eval",
                instruction="placeholder",
            )
            session = self._backend.create_session(placeholder_task)
            self._sessions[lease_id] = {
                "session": session,
                "env": None,
                "task": None,
            }
            logger.info("Allocated env for episode %s", episode_id)
            return {"ok": True, "lease_id": lease_id}
        except Exception as exc:
            logger.error("Failed to allocate env for %s: %s", episode_id, exc)
            raise

    def reset(self, lease_id: str, task_config: dict[str, Any] | None) -> dict[str, Any]:

        entry = self._sessions.get(lease_id)
        if entry is None:
            raise RuntimeError(f"No session for lease_id={lease_id}")

        old_env = entry.get("env")
        if old_env is not None:
            try:
                old_env.close()
            except Exception as exc:
                logger.warning("Error closing previous env on re-reset for lease %s: %s", lease_id, exc)

        task = _task_config_to_ap_task(task_config or {})
        env = OSWorldEnv(
            task=task,
            session=entry["session"],
            action_space="pyautogui",
            sleep_after_execution=self._sleep_after_execution,
        )
        entry["env"] = env
        entry["task"] = task

        obs, info = env.reset()
        logger.info("Reset env for lease %s, task=%s", lease_id, task.task_id)
        return _obs_to_dict(env)

    def prepare_for_probe(self, lease_id: str | None = None) -> None:

        if lease_id is None:
            raise ValueError("lease_id is required for prepare_for_probe()")
        entry = self._sessions.get(lease_id)
        if entry is None or entry.get("env") is None:
            return
        env: OSWorldEnv = entry["env"]
        if getattr(env, "_terminated", False):
            logger.info(
                "Clearing OSWorldEnv._terminated for GUIRL probe (lease=%s, VM state preserved)",
                lease_id,
            )
            env._terminated = False

    def step(
        self,
        lease_id: str,
        action: str,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:

        from eval_runtime.markers import is_logical_marker
        assert not is_logical_marker(action), (
            f"Logical marker '{action}' must not reach env_client.step(). "
            "Use eval_runtime.markers.step_or_handle_marker() in the runner."
        )

        entry = self._sessions.get(lease_id)
        if entry is None or entry.get("env") is None:
            raise RuntimeError(f"No active env for lease_id={lease_id}")

        env: OSWorldEnv = entry["env"]

        ap_action = Action(type=ActionType.PYTHON, arguments={"code": action})
        obs, reward, terminated, truncated, info = env.step(ap_action)
        obs_dict = _obs_to_dict(env)
        done = bool(terminated or truncated)
        return obs_dict, float(reward or 0.0), done, (info or {})

    def execute(self, lease_id: str, action: str) -> dict[str, Any]:

        obs, _, _, _ = self.step(lease_id, action)
        return obs

    def get_obs(self, lease_id: str) -> dict[str, Any]:

        entry = self._sessions.get(lease_id)
        if entry is None or entry.get("env") is None:
            raise RuntimeError(f"No active env for lease_id={lease_id}")
        return _obs_to_dict(entry["env"])

    def evaluate(self, lease_id: str, config: dict[str, Any] | None = None) -> float:

        entry = self._sessions.get(lease_id)
        if entry is None or entry.get("env") is None:
            raise RuntimeError(f"No active env for lease_id={lease_id}")

        env: OSWorldEnv = entry["env"]
        evaluator = OSWorldEvaluator(task=entry.get("task"))
        result = evaluator.finalize(env=env, trace=env.trace)
        if result is None:
            raise RuntimeError(
                f"OSWorld native evaluation returned no result for lease_id={lease_id}"
            )
        if result.error_type:

            if (
                result.error_type == "evaluator.eval_error"
                and result.score is not None
            ):
                logger.warning(
                    "Native eval soft-fail for lease_id=%s (%s); using score=%s",
                    lease_id,
                    result.reason or result.error_type,
                    result.score,
                )
                return float(result.score)
            raise RuntimeError(
                "OSWorld native evaluation failed for "
                f"lease_id={lease_id}: {result.error_type}: "
                f"{result.reason or 'no reason provided'}"
            )
        if result.score is None:
            raise RuntimeError(
                "OSWorld native evaluation produced no score for "
                f"lease_id={lease_id}: {result.reason or 'not evaluated'}"
            )
        score = float(result.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(
                "OSWorld native evaluation produced invalid score "
                f"{score!r} for lease_id={lease_id}"
            )
        return score

    def close(self, lease_id: str) -> None:

        entry = self._sessions.pop(lease_id, None)
        if entry is None:
            return

        env = entry.get("env")
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logger.warning("Error closing env for lease %s: %s", lease_id, exc)

        session = entry.get("session")
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                logger.error("session.close() failed for lease %s: %s — "
                             "force-releasing VM back to pool", lease_id, exc)

                desktop_env = getattr(session, "desktop_env", None)
                if desktop_env is not None and self._backend._env_queue is not None:
                    try:
                        self._backend._env_queue.put(desktop_env)
                    except Exception:
                        pass

        logger.info("Closed env for lease %s", lease_id)

    def heartbeat(self, lease_id: str) -> None:

        pass

    def close_all(self) -> None:

        if self._closed:
            return
        self._closed = True

        for lease_id in list(self._sessions.keys()):
            self.close(lease_id)

        try:
            self._backend.teardown()
            logger.info("Backend teardown complete")
        except Exception as exc:
            logger.warning("Error during backend teardown: %s", exc)
