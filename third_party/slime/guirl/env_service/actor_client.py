from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable
from typing import Any

from .master_client import TaskRecord

logger = logging.getLogger("guirl.actor_client")

class ActorClient:

    def __init__(
        self,
        *,
        env_client,
        harness=None,
        harness_factory: Callable[[], Any] | None = None,
        max_steps: int = 15,
    ):
        if (harness is None) == (harness_factory is None):
            raise ValueError("provide exactly one of harness or harness_factory")
        self.env_client = env_client
        self.harness = harness
        self.harness_factory = harness_factory
        self.max_steps = max_steps
        self._shared_harness_lock = asyncio.Lock()

    async def run_task(
        self,
        task: TaskRecord,
        lease_id: str,
        initial_obs: dict[str, Any],
        *,
        fence_token: str | None = None,
        operation_prefix: str | None = None,
    ) -> dict[str, Any]:

        if self.harness_factory is not None:
            harness = self.harness_factory()
            return await self._run_task_with_harness(
                harness,
                task,
                lease_id,
                initial_obs,
                fence_token=fence_token,
                operation_prefix=operation_prefix,
            )

        async with self._shared_harness_lock:
            return await self._run_task_with_harness(
                self.harness,
                task,
                lease_id,
                initial_obs,
                fence_token=fence_token,
                operation_prefix=operation_prefix,
            )

    async def _run_task_with_harness(
        self,
        harness,
        task: TaskRecord,
        lease_id: str,
        initial_obs: dict[str, Any],
        *,
        fence_token: str | None,
        operation_prefix: str | None,
    ) -> dict[str, Any]:

        from agent_runtime.types import ActionType, HarnessContext, Task
        from agent_runtime.benchmarks.osworld import action_to_pyautogui

        try:
            instruction = _extract_instruction(task)

            ap_task = Task(
                task_id=task.task_id,
                benchmark="osworld",
                suite="guirl",
                instruction=instruction,
            )

            trajectory: list[dict[str, Any]] = []
            screenshots: list[str] = []
            raw_obs = initial_obs
            obs = _raw_obs_to_observation(initial_obs, instruction=instruction, step_index=0)

            for step_idx in range(self.max_steps):
                screenshot_b64 = _extract_screenshot_b64(raw_obs)
                if screenshot_b64:
                    screenshots.append(screenshot_b64)

                context = HarnessContext(task=ap_task, step_index=step_idx)

                t_call = time.monotonic()
                action = await asyncio.to_thread(harness.act, obs, context)
                t_llm = time.monotonic() - t_call

                if action.type in (ActionType.TERMINATE, ActionType.SUBMIT):
                    reason = action.arguments.get("reason", "DONE")
                    trajectory.append({
                        "step": step_idx,
                        "action": reason,
                        "is_terminal": True,
                        "t_llm": t_llm,
                    })
                    break

                if action.type == ActionType.NOOP:
                    trajectory.append({
                        "step": step_idx,
                        "action": "WAIT",
                        "is_terminal": False,
                        "t_llm": t_llm,
                    })
                    raw_obs = await _call_env_operation(
                        self.env_client.get_obs,
                        lease_id,
                        operation_id=_operation_id(
                            operation_prefix, "get-obs", step_idx
                        ),
                        fence_token=fence_token,
                    )
                    obs = _raw_obs_to_observation(raw_obs, instruction=instruction, step_index=step_idx + 1)
                    continue

                if action.type == ActionType.PYTHON:
                    action_code = action.arguments.get("code", action.arguments.get("command", "pass"))
                else:
                    action_code = action_to_pyautogui(action)

                if not action_code:
                    trajectory.append({
                        "step": step_idx,
                        "action": "pass",
                        "is_terminal": False,
                        "t_llm": t_llm,
                    })
                    raw_obs = await _call_env_operation(
                        self.env_client.get_obs,
                        lease_id,
                        operation_id=_operation_id(
                            operation_prefix, "get-obs", step_idx
                        ),
                        fence_token=fence_token,
                    )
                    obs = _raw_obs_to_observation(raw_obs, instruction=instruction, step_index=step_idx + 1)
                    continue

                raw_obs, reward, done, info = await _call_env_operation(
                    self.env_client.step,
                    lease_id,
                    action_code,
                    operation_id=_operation_id(
                        operation_prefix, "step", step_idx
                    ),
                    fence_token=fence_token,
                )

                obs = _raw_obs_to_observation(raw_obs, instruction=instruction, step_index=step_idx + 1)

                trajectory.append({
                    "step": step_idx,
                    "action": action_code,
                    "reward": reward,
                    "done": done,
                    "t_llm": t_llm,
                })

                if done:
                    break

            final_obs = await _call_env_operation(
                self.env_client.get_obs,
                lease_id,
                operation_id=_operation_id(
                    operation_prefix, "get-obs-final"
                ),
                fence_token=fence_token,
            )
            final_b64 = _extract_screenshot_b64(final_obs)
            if final_b64:
                screenshots.append(final_b64)

            return {
                "trajectory": trajectory,
                "screenshots": screenshots,
                "num_steps": len(trajectory),
            }
        finally:
            reset = getattr(harness, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:
                    logger.warning(
                        "Harness reset failed for task %s: %s", task.task_id, exc
                    )

    async def healthcheck(self) -> bool:

        return self.harness is not None or self.harness_factory is not None

def _extract_instruction(task: TaskRecord) -> str:
    if isinstance(task.prompt, dict):
        return task.prompt.get("instruction", "")
    if isinstance(task.prompt, str):
        return task.prompt
    return task.task_config.get("instruction", "")

def _operation_id(
    prefix: str | None,
    operation: str,
    index: int | None = None,
) -> str | None:
    if prefix is None:
        return None
    suffix = f":{index}" if index is not None else ""
    return f"{prefix}:{operation}{suffix}"

async def _call_env_operation(
    operation,
    *args,
    operation_id: str | None,
    fence_token: str | None,
):

    if operation_id is None and fence_token is None:
        return await operation(*args)
    return await operation(
        *args,
        operation_id=operation_id,
        fence_token=fence_token,
    )

def _raw_obs_to_observation(
    raw_obs: dict[str, Any],
    *,
    instruction: str = "",
    step_index: int = 0,
) -> Any:

    from agent_runtime.types import Modality, ModalityKind, Observation

    parts: list[Modality] = []

    screenshot = raw_obs.get("screenshot", b"")
    if isinstance(screenshot, bytes) and screenshot:
        screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
        parts.append(Modality(
            kind=ModalityKind.IMAGE,
            content=screenshot_b64,
            mime_type="image/png",
        ))

    return Observation(
        parts=parts,
        instruction=instruction,
        step_index=step_index,
    )

def _extract_screenshot_b64(raw_obs: dict[str, Any]) -> str:

    screenshot = raw_obs.get("screenshot", b"")
    if isinstance(screenshot, bytes) and screenshot:
        return base64.b64encode(screenshot).decode("utf-8")
    return ""
