from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger("guirl.master_client")

@dataclass
class TaskRecord:
    task_id: str
    batch_id: str
    sample_index: int
    prompt: dict | list | str
    task_config: dict[str, Any]
    step_id: int = 0
    status: str = "pending"
    lease_id: str | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    S: float | None = None
    error: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchSet:
    step_id: int
    batch_id: str
    tasks: list[TaskRecord]
    status: str = "init"
    metadata: dict[str, Any] = field(default_factory=dict)

class TaskSamplerProtocol(Protocol):
    async def sample_next(self, count: int) -> BatchSet: ...

class ActorClientProtocol(Protocol):
    async def run_task(
        self, task: TaskRecord, lease_id: str, initial_obs: dict[str, Any]
    ) -> dict[str, Any]: ...

class MasterClient:

    def __init__(
        self,
        *,
        env_client,
        task_sampler: TaskSamplerProtocol,
        actor_client: ActorClientProtocol | None = None,
        actor_clients: dict[str, ActorClientProtocol] | None = None,
        batch_size: int = 8,
        env_capacity: int = 8,
        agent_timeout_seconds: float = 600.0,
        pull_first_item_timeout_seconds: float | None = None,
        env_restart_fn: Callable[[], Awaitable[None]] | None = None,
        env_restart_threshold_seconds: float = 300.0,
        env_restart_cooldown_seconds: float = 300.0,
        env_health_probe_fn: Callable[[], Awaitable[bool]] | None = None,
        env_health_probe_interval_seconds: float = 60.0,
    ):
        self.env_client = env_client
        self.task_sampler = task_sampler
        if actor_clients:
            self.actor_clients: dict[str, ActorClientProtocol] = dict(actor_clients)
        elif actor_client is not None:
            self.actor_clients = {"default": actor_client}
        else:
            raise ValueError("MasterClient requires actor_client or actor_clients")

        self.actor_client = next(iter(self.actor_clients.values()))
        self.batch_size = batch_size

        self.env_capacity = max(0, env_capacity)
        self.agent_timeout_seconds = agent_timeout_seconds

        if pull_first_item_timeout_seconds is None:
            pull_first_item_timeout_seconds = max(1500.0, agent_timeout_seconds + 300.0)
        self.pull_first_item_timeout_seconds = pull_first_item_timeout_seconds

        self._finished: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=max(1, self.env_capacity)
        )
        self._inflight: set[asyncio.Task] = set()

        self._pending_records: deque[TaskRecord] = deque()
        self._running = False
        self._prefill_task: asyncio.Task | None = None

        self._env_restart_fn = env_restart_fn
        self._env_restart_threshold_seconds = env_restart_threshold_seconds
        self._env_restart_cooldown_seconds = env_restart_cooldown_seconds
        self._first_alloc_failure_time: float | None = None
        self._last_restart_time: float = 0.0

        self._env_health_probe_fn = env_health_probe_fn
        self._env_health_probe_interval_seconds = env_health_probe_interval_seconds
        self._last_health_probe_time: float = 0.0
        self._last_health_probe_ok: bool | None = None

        self._first_probe_failure_time: float | None = None

        self._probe_ever_ok: bool = False

    async def start(self) -> None:

        if self._running:
            return
        self._running = True
        self._prefill_task = asyncio.create_task(self._prefill_loop())

    async def stop(self) -> None:

        self._running = False
        if self._prefill_task:
            self._prefill_task.cancel()
            try:
                await self._prefill_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._prefill_task = None

        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        for task in inflight:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._inflight.clear()

        queued_closes = []
        while True:
            try:
                entry = self._finished.get_nowait()
            except asyncio.QueueEmpty:
                break
            queued_closes.append(self._safe_close(entry.get("lease_id")))
        if queued_closes:
            await asyncio.gather(*queued_closes)

    async def pull_finished(self, num_needed: int) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []

        first = await asyncio.wait_for(
            self._finished.get(), timeout=self.pull_first_item_timeout_seconds
        )
        results.append(first)

        while len(results) < num_needed:
            try:
                item = self._finished.get_nowait()
                results.append(item)
            except asyncio.QueueEmpty:
                break

        return results

    async def _maybe_restart_env(self) -> bool:

        if self._env_restart_fn is None:
            return False
        now = time.monotonic()

        if (
            self._env_health_probe_fn is not None
            and now - self._last_health_probe_time >= self._env_health_probe_interval_seconds
        ):
            self._last_health_probe_time = now
            try:
                self._last_health_probe_ok = await self._env_health_probe_fn()
            except Exception as exc:
                logger.warning("Env health probe errored (%s); treating as unhealthy", exc)
                self._last_health_probe_ok = False
            if self._last_health_probe_ok:
                self._probe_ever_ok = True
                self._first_probe_failure_time = None
            elif self._first_probe_failure_time is None:
                self._first_probe_failure_time = now

        if self._env_health_probe_fn is not None and not self._probe_ever_ok:
            return False

        alloc_streak = (
            now - self._first_alloc_failure_time
            if self._first_alloc_failure_time is not None else 0.0
        )
        probe_streak = (
            now - self._first_probe_failure_time
            if self._first_probe_failure_time is not None else 0.0
        )
        alloc_trigger = alloc_streak >= self._env_restart_threshold_seconds
        probe_trigger = probe_streak >= self._env_restart_threshold_seconds
        if not alloc_trigger and not probe_trigger:
            return False
        if (
            alloc_trigger
            and not probe_trigger
            and self._env_health_probe_fn is not None
            and self._last_health_probe_ok
        ):
            logger.warning(
                "Allocate failing for %.0fs but env health probe is OK — "
                "not restarting (stack may be saturated, not dead)",
                alloc_streak,
            )
            return False
        if now - self._last_restart_time < self._env_restart_cooldown_seconds:
            return False
        self._last_restart_time = now

        logger.error(
            "Env stack looks dead (allocate failing %.0fs, probe failing %.0fs) — "
            "restarting (inflight=%d, pending=%d, queued=%d)",
            alloc_streak, probe_streak,
            len(self._inflight), len(self._pending_records), self._finished.qsize(),
        )

        inflight = list(self._inflight)
        for task in inflight:
            task.cancel()
        for task in inflight:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._inflight.clear()

        self._pending_records.clear()

        queued_closes = []
        while True:
            try:
                entry = self._finished.get_nowait()
            except asyncio.QueueEmpty:
                break
            queued_closes.append(self._safe_close(entry.get("lease_id")))
        if queued_closes:
            await asyncio.gather(*queued_closes)

        try:
            await self._env_restart_fn()
            logger.info("Env restart completed")
        except Exception:
            logger.exception("Env restart failed; will retry after cooldown")

        self._first_alloc_failure_time = None
        self._first_probe_failure_time = None
        self._last_health_probe_time = 0.0
        self._last_health_probe_ok = None
        return True

    async def _prefill_loop(self) -> None:

        num_actors = len(self.actor_clients)
        while self._running:
            try:
                if await self._maybe_restart_env():
                    continue
                outstanding = self._finished.qsize() + len(self._inflight)
                budget = self.env_capacity - outstanding
                if budget <= 0:

                    if self._inflight:
                        await asyncio.wait(
                            self._inflight, timeout=2.0, return_when=asyncio.FIRST_COMPLETED
                        )
                    else:
                        await asyncio.sleep(1.0)
                    continue

                if len(self._pending_records) < budget:
                    tasks_needed = math.ceil(
                        (budget - len(self._pending_records)) / num_actors
                    )
                    n = min(tasks_needed, self.batch_size)

                    batch = await self.task_sampler.sample_next(n)
                    self._pending_records.extend(self._expand_records(batch.tasks))

                while budget > 0 and self._pending_records:
                    record = self._pending_records.popleft()
                    task = asyncio.create_task(self._run_and_enqueue(record))
                    self._inflight.add(task)
                    task.add_done_callback(self._inflight.discard)
                    budget -= 1

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Prefill loop error: %s", exc)
                await asyncio.sleep(5.0)

    def _expand_records(self, tasks: list[TaskRecord]) -> list[TaskRecord]:

        if len(self.actor_clients) == 1:
            name = next(iter(self.actor_clients))
            for record in tasks:
                record.metadata.setdefault("actor", name)
            return list(tasks)

        expanded: list[TaskRecord] = []
        for record in tasks:
            for actor_name in self.actor_clients:
                expanded.append(
                    TaskRecord(
                        task_id=record.task_id,
                        batch_id=record.batch_id,
                        sample_index=record.sample_index,
                        prompt=record.prompt,
                        task_config=dict(record.task_config),
                        step_id=record.step_id,
                        metadata={**record.metadata, "actor": actor_name},
                    )
                )
        return expanded

    async def _run_and_enqueue(self, record: TaskRecord) -> None:

        try:
            await self._run_single_task(record)
            if record.status == "finished":
                entry = self._record_to_entry(record)
                await self._finished.put(entry)
        except asyncio.CancelledError:

            await self._safe_close(record.lease_id)
            raise

    async def _run_single_task(self, record: TaskRecord) -> None:

        t_start = time.monotonic()
        try:
            try:
                alloc_result = await self.env_client.allocate(
                    f"guirl-{record.task_id}-{uuid.uuid4().hex[:8]}"
                )
            except Exception:

                if self._first_alloc_failure_time is None:
                    self._first_alloc_failure_time = time.monotonic()
                raise

            self._first_alloc_failure_time = None
            lease_id = alloc_result["lease_id"]
            record.lease_id = lease_id

            task_config = dict(record.task_config)
            task_config["_domain"] = record.metadata.get("app", "unknown")
            initial_obs = await self.env_client.reset(lease_id, task_config)
            record.timing["t_reset"] = time.monotonic() - t_start

            actor_name = record.metadata.get("actor")
            actor_client = self.actor_clients.get(actor_name, self.actor_client)

            t_actor = time.monotonic()
            if self.agent_timeout_seconds and self.agent_timeout_seconds > 0:
                result = await asyncio.wait_for(
                    actor_client.run_task(record, lease_id, initial_obs),
                    timeout=self.agent_timeout_seconds,
                )
            else:
                result = await actor_client.run_task(record, lease_id, initial_obs)
            record.trajectory = result.get("trajectory", [])
            record.screenshots = result.get("screenshots", [])
            record.timing["t_actor"] = time.monotonic() - t_actor

            t_eval = time.monotonic()
            record.S = await self.env_client.evaluate(lease_id)
            record.timing["t_evaluate"] = time.monotonic() - t_eval

            record.status = "finished"
            record.timing["t_total"] = time.monotonic() - t_start
            logger.info(
                "Task %s (actor=%s) finished: S=%.1f, steps=%d, time=%.1fs",
                record.task_id, actor_name or "default", record.S or 0,
                len(record.trajectory), record.timing["t_total"],
            )

        except asyncio.TimeoutError:
            record.status = "timeout"
            record.error = "actor_timeout"
            record.timing["t_total"] = time.monotonic() - t_start
            logger.error("Task %s actor timed out after %.0fs", record.task_id, self.agent_timeout_seconds)
            await self._safe_close(record.lease_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.timing["t_total"] = time.monotonic() - t_start
            logger.error("Task %s failed: %s", record.task_id, exc)
            await self._safe_close(record.lease_id)

    async def _safe_close(self, lease_id: str | None) -> None:
        if not lease_id:
            return
        try:
            await asyncio.wait_for(
                self.env_client.close(lease_id),
                timeout=float(os.environ.get("GUIRL_CLEANUP_TIMEOUT_SECONDS", "15")),
            )
        except Exception as exc:
            logger.warning("Best-effort cleanup failed for lease %s: %s", lease_id, exc)

    @staticmethod
    def _record_to_entry(record: TaskRecord) -> dict[str, Any]:

        return {
            "task_id": record.task_id,
            "batch_id": record.batch_id,
            "sample_index": record.sample_index,
            "prompt": record.prompt,
            "task_config": record.task_config,
            "lease_id": record.lease_id,
            "trajectory": record.trajectory,
            "screenshots": record.screenshots,
            "S": record.S,
            "timing": record.timing,
            "metadata": record.metadata,
        }
