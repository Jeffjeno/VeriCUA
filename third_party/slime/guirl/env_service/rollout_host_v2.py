from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .artifact_store import ArtifactStore
from .batch_models import (
    ArtifactRef,
    BatchId,
    BatchRecord,
    BatchState,
    CloseIntent,
    ErrorCode,
    LeaseRecord,
    LeaseStatus,
    ManifestRevision,
    ManifestSlot,
    OutboxStatus,
    ProtocolError,
    ReplacementIntent,
    ReplacementStatus,
    SlotKey,
    SlotRecord,
    SlotState,
)
from .master_client import TaskRecord
from .state_store import SQLiteStateStore

logger = logging.getLogger("guirl.rollout_host_v2")

_INFRA_ERROR_CODES = frozenset({
    "lease_lost",
    "lease_not_active",
    "lease_recovering",
    "fenced",
    "fence_required",
    "worker_stale",
    "allocation_cancelled",
})
_INFRA_ERROR_MARKERS = (
    "is not active",
    "is lost and cannot be routed",
    "connection refused",
    "connect call failed",
    "cannot connect to host",
    "no such container",
    "docker_port_allocations.lck",
    "worker incarnation changed",
)

def _is_infra_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in _INFRA_ERROR_CODES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _INFRA_ERROR_MARKERS)

@dataclass(frozen=True)
class RolloutHostV2Config:
    env_capacity: int
    actor_concurrency: int | Mapping[str, int] | None = None
    max_attempts: int = 2
    max_task_replacements: int = 2

    infra_retry_budget: int = 20
    failure_policy: str = "replace_task_group"
    claim_ttl_seconds: float = 600.0
    close_retry_seconds: float = 1.0
    supervisor_retry_seconds: float = 0.1
    actor_timeout_seconds: float = 1200.0
    batch_deadline_seconds: float = 6 * 60 * 60
    one_active_batch: bool = True

    def __post_init__(self) -> None:
        if self.env_capacity <= 0:
            raise ValueError("env_capacity must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.infra_retry_budget < 0:
            raise ValueError("infra_retry_budget must be non-negative")
        if self.max_task_replacements < 0:
            raise ValueError("max_task_replacements must be non-negative")
        if self.failure_policy not in {"replace_task_group", "fail_batch"}:
            raise ValueError("failure_policy must be replace_task_group or fail_batch")
        if self.actor_timeout_seconds <= 0:
            raise ValueError("actor_timeout_seconds must be positive")
        if self.batch_deadline_seconds <= 0:
            raise ValueError("batch_deadline_seconds must be positive")
        if isinstance(self.actor_concurrency, Mapping):
            for actor_id, value in self.actor_concurrency.items():
                if not str(actor_id) or int(value) <= 0:
                    raise ValueError(
                        "actor_concurrency mapping requires non-empty actors "
                        "and positive limits"
                    )
        elif self.actor_concurrency is not None and int(self.actor_concurrency) <= 0:
            raise ValueError("actor_concurrency must be positive")

    def concurrency_for(self, actor_id: str, actor_count: int = 1) -> int:
        fair_share = max(1, (self.env_capacity + actor_count - 1) // actor_count)
        value = (
            self.actor_concurrency.get(actor_id, fair_share)
            if isinstance(self.actor_concurrency, Mapping)
            else (
                fair_share
                if self.actor_concurrency is None
                else self.actor_concurrency
            )
        )
        return int(value)

@dataclass(frozen=True)
class RunOutcome:
    lease_id: str | None = None
    retain_capacity: bool = False

class LoopSupervisor:

    def __init__(self, retry_seconds: float = 0.1) -> None:
        self.retry_seconds = retry_seconds
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._stopping = False
        self.restart_counts: dict[str, int] = {}

    def start(
        self,
        name: str,
        factory: Callable[[], Awaitable[None]],
    ) -> None:
        if name in self._tasks:
            return
        self._tasks[name] = asyncio.create_task(
            self._run(name, factory),
            name=f"guirl-v2-supervisor-{name}",
        )

    async def _run(
        self,
        name: str,
        factory: Callable[[], Awaitable[None]],
    ) -> None:
        while not self._stopping:
            try:
                await factory()
                if not self._stopping:
                    raise RuntimeError(f"supervised loop {name} exited unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
                logger.exception("v2 loop %s failed; restarting", name)
                await asyncio.sleep(self.retry_seconds)

    async def stop(self) -> None:
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

class FairScheduler:

    def __init__(
        self,
        *,
        actor_ids: Sequence[str],
        env_capacity: int,
        actor_concurrency: int | Mapping[str, int] | None,
        runner: Callable[[SlotRecord], Awaitable[RunOutcome]],
        initial_leases: Sequence[LeaseRecord] = (),
        initial_capacity_holds: Sequence[str] = (),
        supervisor: LoopSupervisor | None = None,
    ) -> None:
        self.actor_ids = tuple(actor_ids)
        self._queues = {actor_id: deque() for actor_id in self.actor_ids}
        self._queued: set[tuple[SlotKey, int]] = set()
        self._running = {actor_id: 0 for actor_id in self.actor_ids}
        fair_share = max(
            1,
            (env_capacity + len(self.actor_ids) - 1) // len(self.actor_ids),
        )
        self._limits = {
            actor_id: max(
                1,
                int(
                    actor_concurrency.get(actor_id, fair_share)
                    if isinstance(actor_concurrency, Mapping)
                    else (
                        fair_share
                        if actor_concurrency is None
                        else actor_concurrency
                    )
                ),
            )
            for actor_id in self.actor_ids
        }
        self._runner = runner
        self._condition = asyncio.Condition()
        self._round_robin_index = 0
        self._capacity_limit = env_capacity
        live = [
            lease
            for lease in initial_leases
            if lease.status in {LeaseStatus.ACTIVE, LeaseStatus.CLOSING}
        ]

        retained = {
            *(lease.lease_id for lease in live),
            *(str(value) for value in initial_capacity_holds),
        }
        self._capacity_in_use = len(retained)
        self._retained_leases = retained
        self._released_before_register: dict[str, SlotKey] = {}
        self._worker_tasks: set[asyncio.Task[Any]] = set()
        self._task_slots: dict[asyncio.Task[Any], SlotRecord] = {}
        self._supervisor = supervisor or LoopSupervisor()
        self._owns_supervisor = supervisor is None
        self._started = False
        self._stopping = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._supervisor.start("scheduler", self._dispatch_loop)

    async def stop(self) -> None:
        self._stopping = True
        async with self._condition:
            self._condition.notify_all()
        tasks = list(self._worker_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_supervisor:
            await self._supervisor.stop()

    async def enqueue(self, slot: SlotRecord, *, retry: bool = False) -> None:
        if slot.key.actor_id not in self._queues:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "slot actor is not configured",
                slot.key.to_dict(),
            )
        identity = (slot.key, slot.task_group_revision)
        async with self._condition:
            if identity in self._queued:
                return
            queue = self._queues[slot.key.actor_id]
            if retry:
                queue.appendleft(slot)
            else:
                queue.append(slot)
            self._queued.add(identity)
            self._condition.notify_all()

    async def enqueue_batch(self, slots: Sequence[SlotRecord]) -> None:
        for slot in slots:
            if slot.state in {SlotState.PENDING, SlotState.RETRY_PENDING}:
                await self.enqueue(slot, retry=slot.state == SlotState.RETRY_PENDING)

    async def _dispatch_loop(self) -> None:
        while not self._stopping:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._stopping
                    or (
                        self._capacity_in_use < self._capacity_limit
                        and self._has_dispatchable()
                    )
                )
                if self._stopping:
                    return
                actor_id = self._next_actor()
                if actor_id is None:
                    continue
                slot = self._queues[actor_id].popleft()
                self._queued.discard((slot.key, slot.task_group_revision))
                self._running[actor_id] += 1
                self._capacity_in_use += 1
            task = asyncio.create_task(
                self._execute(actor_id, slot),
                name=(
                    f"guirl-v2-slot-{slot.key.batch_id.run_id}-"
                    f"{slot.key.task_index}-{slot.key.actor_id}"
                ),
            )
            self._worker_tasks.add(task)
            self._task_slots[task] = slot
            task.add_done_callback(self._worker_finished)

    def _worker_finished(self, task: asyncio.Task[Any]) -> None:
        self._worker_tasks.discard(task)
        slot = self._task_slots.pop(task, None)
        if slot is not None and not any(item.key == slot.key for item in self._task_slots.values()):
            for lease_id, key in list(self._released_before_register.items()):
                if key == slot.key:
                    self._released_before_register.pop(lease_id, None)

    def _has_dispatchable(self) -> bool:
        return any(
            self._queues[actor_id]
            and self._running[actor_id] < self._limits[actor_id]
            for actor_id in self.actor_ids
        )

    def _next_actor(self) -> str | None:
        for offset in range(len(self.actor_ids)):
            index = (self._round_robin_index + offset) % len(self.actor_ids)
            actor_id = self.actor_ids[index]
            if (
                self._queues[actor_id]
                and self._running[actor_id] < self._limits[actor_id]
            ):
                self._round_robin_index = (index + 1) % len(self.actor_ids)
                return actor_id
        return None

    async def _execute(self, actor_id: str, slot: SlotRecord) -> None:
        outcome = RunOutcome()
        try:
            outcome = await self._runner(slot)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("unhandled slot runner error for %s", slot.key)
        finally:
            if outcome.retain_capacity and outcome.lease_id:
                if outcome.lease_id in self._released_before_register:
                    self._released_before_register.pop(outcome.lease_id)
                    self._capacity_in_use = max(
                        0, self._capacity_in_use - 1
                    )
                else:
                    self._retained_leases.add(outcome.lease_id)
            else:
                self._capacity_in_use = max(0, self._capacity_in_use - 1)
            async with self._condition:
                self._running[actor_id] -= 1
                self._condition.notify_all()

    async def release_lease(self, lease_id: str, key: SlotKey | None = None) -> None:
        if lease_id in self._retained_leases:
            self._retained_leases.remove(lease_id)
            self._capacity_in_use = max(0, self._capacity_in_use - 1)
        elif key is not None and any(slot.key == key for slot in self._task_slots.values()):

            self._released_before_register[lease_id] = key
        async with self._condition:
            self._condition.notify_all()

    async def reserve_lease(self, lease_id: str) -> None:
        if (
            lease_id not in self._retained_leases
            and lease_id not in self._released_before_register
        ):
            self._retained_leases.add(lease_id)
            self._capacity_in_use += 1
        async with self._condition:
            self._condition.notify_all()

    async def cancel_batch(self, batch_id: BatchId) -> None:
        async with self._condition:
            for actor_id, queue in self._queues.items():
                kept = deque(
                    slot
                    for slot in queue
                    if slot.key.batch_id != batch_id
                )
                removed = [
                    slot for slot in queue if slot.key.batch_id == batch_id
                ]
                self._queues[actor_id] = kept
                for slot in removed:
                    self._queued.discard(
                        (slot.key, slot.task_group_revision)
                    )
            for task, slot in list(self._task_slots.items()):
                if (
                    slot.key.batch_id == batch_id
                    and task is not asyncio.current_task()
                ):
                    task.cancel()
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        return {
            "pending": {
                actor_id: len(self._queues[actor_id])
                for actor_id in self.actor_ids
            },
            "running": dict(self._running),
            "actor_limits": dict(self._limits),
            "live_leases": len(self._retained_leases),
            "env_capacity": self._capacity_limit,
            "capacity_in_use": self._capacity_in_use,
            "available_capacity": max(
                0, self._capacity_limit - self._capacity_in_use
            ),
        }

class ResultBroker:
    def __init__(
        self,
        store: SQLiteStateStore,
        artifacts: ArtifactStore,
        *,
        on_claim_deadline: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self._condition = asyncio.Condition()
        self._on_claim_deadline = on_claim_deadline

    async def notify_ready(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def pull_ready(
        self,
        batch_id: BatchId,
        *,
        consumer: str,
        limit: int,
        request_id: str,
        ttl_seconds: float,
        wait_seconds: float = 0.0,
        include_payload: bool = True,
    ) -> dict[str, Any]:
        batch = self.store.get_batch(batch_id)
        if batch.state in {
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
            BatchState.COMMITTED,
        }:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "batch does not accept result pulls",
                {"state": batch.state.value, **batch_id.to_dict()},
            )
        request_payload = {
            "batch_id": batch_id.to_dict(),
            "consumer": consumer,
            "limit": limit,
            "ttl_seconds": ttl_seconds,
            "include_payload": include_payload,
        }
        replay = self.store.get_idempotent_response(
            "pull_ready_response",
            request_id,
            request_payload,
        )
        if replay is not None:
            return await asyncio.to_thread(
                self.artifacts.read_json, ArtifactRef.from_dict(replay["response_artifact"]),
            )
        deadline = asyncio.get_running_loop().time() + max(0.0, wait_seconds)
        while True:
            claims = self.store.claim_ready_slots(
                batch_id,
                consumer_id=consumer,
                limit=limit,
                request_id=request_id,
                ttl_seconds=ttl_seconds,
                store_empty_replay=False,
            )
            if claims.claims or wait_seconds <= 0:
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            async with self._condition:
                try:
                    await asyncio.wait_for(self._condition.wait(), remaining)
                except asyncio.TimeoutError:
                    break
        current_batch = self.store.get_batch(batch_id)
        if current_batch.state in {
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
            BatchState.COMMITTED,
        }:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "batch stopped accepting results while pull was pending",
                {
                    "state": current_batch.state.value,
                    **batch_id.to_dict(),
                },
            )
        entries = [
            self._claim_entry(claim, include_payload=False)
            for claim in claims.claims
        ]
        if include_payload:
            for entry in entries:
                entry["actor_payload"] = await asyncio.to_thread(
                    self.artifacts.read_json, ArtifactRef.from_dict(entry["actor_artifact"]),
                )
        response = {
            "batch_id": batch_id.to_dict(),
            "request_id": request_id,
            "entries": entries,
        }
        response_artifact = await asyncio.to_thread(self.artifacts.write_json, response)
        stored_marker = self.store.put_idempotent_response(
            "pull_ready_response",
            request_id,
            request_payload,
            {"response_artifact": response_artifact.to_dict()},
        )
        if claims.claims and self._on_claim_deadline is not None:
            self._on_claim_deadline()
        current_batch = self.store.get_batch(batch_id)
        if current_batch.state in {
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
            BatchState.COMMITTED,
        }:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "batch stopped accepting results before pull completed",
                {
                    "state": current_batch.state.value,
                    **batch_id.to_dict(),
                },
            )
        return await asyncio.to_thread(
            self.artifacts.read_json, ArtifactRef.from_dict(stored_marker["response_artifact"]),
        )

    def _claim_entry(self, claim, *, include_payload: bool) -> dict[str, Any]:
        slot = self.store.get_slot(claim.key)
        lease = self.store.get_claim_lease(claim.claim_id)
        artifact = self.store.get_slot_artifact(
            claim.key,
            role="actor",
            attempt=lease.attempt,
        )
        entry = {
            "claim_token": claim.claim_id,
            "generation": claim.generation,
            "expires_at": claim.expires_at,
            "slot": claim.key.to_dict(),
            "task_id": slot.task_id,
            "task_group_id": slot.task_group_id,
            "task_group_revision": slot.task_group_revision,
            "manifest_revision": slot.manifest_revision,
            "actor_id": claim.key.actor_id,
            "attempt": lease.attempt,
            "slot_version": slot.version,
            "actor_artifact": artifact.to_dict(),
            "lease": _lease_to_dict(lease),
        }
        if include_payload:
            entry["actor_payload"] = self.artifacts.read_json(artifact)
        return entry

class LeaseManager:

    def __init__(
        self,
        *,
        store: SQLiteStateStore,
        env_client: Any,
        scheduler: FairScheduler,
        retry_seconds: float,
        max_attempts: int,
        on_retry: Callable[[SlotRecord], Awaitable[None]],
        on_exhausted: Callable[[SlotKey, int], Awaitable[None]],
        clock: Callable[[], float] = time.time,
        supervisor: LoopSupervisor | None = None,
    ) -> None:
        self.store = store
        self.env_client = env_client
        self.scheduler = scheduler
        self.retry_seconds = retry_seconds
        self.max_attempts = max_attempts
        self.on_retry = on_retry
        self.on_exhausted = on_exhausted
        self.clock = clock
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued: set[str] = set()
        self._allocation_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued_allocations: set[str] = set()
        self._allocation_completion: dict[str, asyncio.Event] = {}
        self._completion: dict[str, asyncio.Event] = {}
        self._callbacks: set[asyncio.Task[Any]] = set()
        self._deadline_changed = asyncio.Event()
        self._supervisor = supervisor or LoopSupervisor()
        self._owns_supervisor = supervisor is None
        self._started = False
        self._stopping = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for intent in self.store.list_close_outbox(limit=100_000):
            self.enqueue(intent)
        for allocation in self.store.list_allocation_cleanups(
            statuses=["PENDING"]
        ):
            self.enqueue_allocation(allocation)

        workers = min(4, max(1, self.scheduler._capacity_limit))
        for index in range(workers):
            self._supervisor.start(f"lease-close-{index}", self._close_loop)
            self._supervisor.start(f"allocation-close-{index}", self._allocation_close_loop)
        self._supervisor.start("claim-expiry", self._expiry_loop)

    async def stop(self) -> None:
        self._stopping = True
        self._deadline_changed.set()
        callbacks = list(self._callbacks)
        for callback in callbacks:
            callback.cancel()
        if callbacks:
            await asyncio.gather(*callbacks, return_exceptions=True)
        if self._owns_supervisor:
            await self._supervisor.stop()

    def claim_deadline_changed(self) -> None:
        self._deadline_changed.set()

    @staticmethod
    def _finish_waiters(events, key):
        event = events.pop(key, None)
        if event is not None:
            event.set()

    def enqueue(self, intent: CloseIntent) -> None:
        intent = self.store.get_close_intent(intent.intent_id)
        if intent.status != OutboxStatus.PENDING or intent.intent_id in self._queued:
            return
        self._queued.add(intent.intent_id)
        self._completion.setdefault(intent.intent_id, asyncio.Event())
        self._queue.put_nowait(intent.intent_id)

    @staticmethod
    def allocation_hold_id(allocation_key: str) -> str:
        return f"allocation:{allocation_key}"

    def enqueue_allocation(self, allocation: Mapping[str, Any]) -> None:
        allocation_key = str(allocation["allocation_key"])
        if (
            str(allocation.get("status")) != "PENDING"
            or allocation_key in self._queued_allocations
        ):
            return
        self._queued_allocations.add(allocation_key)
        self._allocation_completion.setdefault(allocation_key, asyncio.Event())
        self._allocation_queue.put_nowait(allocation_key)

    async def wait_closed(self, intent: CloseIntent) -> None:
        intent = self.store.get_close_intent(intent.intent_id)
        if intent.status in {OutboxStatus.CLOSED, OutboxStatus.LOST}:
            return
        self.enqueue(intent)
        await self._completion[intent.intent_id].wait()

    async def recover_expired_claims(self, now: float | None = None) -> None:
        self.store.recover_expired_claims(
            now=self.clock() if now is None else now
        )
        for intent in self.store.list_close_outbox(limit=100_000):
            self.enqueue(intent)

    async def close_slots(
        self,
        slots: Sequence[SlotRecord],
        *,
        wait: bool = True,
        reason: str = "task-group-superseded",
    ) -> None:
        identities = {slot.key for slot in slots}
        intents: list[CloseIntent] = []
        close_by_lease = {
            intent.lease_id: intent for intent in self.store.list_close_outbox(limit=100_000)
        }
        for lease in self.store.list_leases(
            statuses=[LeaseStatus.ACTIVE, LeaseStatus.CLOSING]
        ):
            if lease.key not in identities:
                continue
            existing = close_by_lease.get(lease.lease_id)
            intent = (
                existing
                if existing
                else self.store.enqueue_attempt_close(
                    lease.lease_id,
                    action="NONE",
                    reason=reason,
                )
            )
            intents.append(intent)
        allocations = [
            allocation
            for allocation in self.store.list_allocation_cleanups(
                statuses=["ALLOCATING", "PENDING"]
            )
            if allocation["key"] in identities
        ]
        for allocation in allocations:
            pending = self.store.quarantine_allocation(
                str(allocation["allocation_key"]),
                action="NONE",
                reason=reason,
            )
            self.enqueue_allocation(pending)
        if wait:
            waits = [self.wait_closed(intent) for intent in intents]
            for allocation in allocations:
                allocation_key = str(allocation["allocation_key"])
                event = self._allocation_completion.setdefault(
                    allocation_key,
                    asyncio.Event(),
                )
                waits.append(event.wait())
            if waits:
                await asyncio.gather(*waits)
        else:
            for intent in intents:
                self.enqueue(intent)

    async def _close_loop(self) -> None:
        while not self._stopping:
            intent_id = await self._queue.get()
            try:
                intent = self.store.get_close_intent(intent_id)
                if intent.status != OutboxStatus.PENDING:
                    self._queued.discard(intent_id)
                    self._finish_waiters(self._completion, intent_id)
                    continue
                lost = False
                try:
                    result = await _call_optional_keywords(
                        self.env_client.close_strict,
                        intent.lease_id,
                        operation_id=intent.operation_id,
                        fence_token=intent.fence_token,
                    )
                    lost = (
                        isinstance(result, Mapping)
                        and str(result.get("status") or "").upper() == "LOST"
                    )
                except Exception as exc:
                    if str(getattr(exc, "code", "")).lower() != "lease_lost":
                        raise
                    lost = True
                lease = self.store.get_lease(intent.lease_id)
                slot = self.store.mark_close_intent_terminal(
                    intent_id,
                    lost=lost,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("close intent %s failed: %s", intent_id, exc)
                try:
                    self.store.record_close_failure(intent_id, str(exc))
                except Exception:
                    logger.exception("failed to persist close failure %s", intent_id)
                await asyncio.sleep(self.retry_seconds)
                self._queued.discard(intent_id)
                try:
                    self.enqueue(self.store.get_close_intent(intent_id))
                except Exception:
                    logger.exception("failed to reload close intent %s", intent_id)
                continue
            self._queued.discard(intent_id)
            await self.scheduler.release_lease(intent.lease_id, intent.key)
            self._finish_waiters(self._completion, intent_id)
            if slot.state == SlotState.RETRY_PENDING:
                await self.on_retry(slot)
            if intent.reason.startswith("attempt-exhausted"):
                triggering_revision = int(
                    lease.owner.get(
                        "task_group_revision",
                        slot.task_group_revision,
                    )
                )
                callback = asyncio.create_task(
                    self.on_exhausted(intent.key, triggering_revision),
                    name="guirl-v2-task-group-replacement",
                )
                self._callbacks.add(callback)
                callback.add_done_callback(self._replacement_finished)

    async def _allocation_close_loop(self) -> None:

        max_cleanup_attempts = int(
            os.environ.get("GUI_ENV_CLEANUP_MAX_ATTEMPTS", "6")
        )
        while not self._stopping:
            allocation_key = await self._allocation_queue.get()
            abandoned = False
            try:
                allocation = self.store.get_allocation_cleanup(allocation_key)
                if str(allocation["status"]) != "PENDING":
                    self._queued_allocations.discard(allocation_key)
                    self._finish_waiters(self._allocation_completion, allocation_key)
                    continue
                cancel = getattr(self.env_client, "cancel_allocation", None)
                if not callable(cancel):
                    raise RuntimeError(
                        "Env Master does not expose allocation cancellation"
                    )
                await _call_optional_keywords(
                    cancel,
                    allocation_key,
                    operation_id=str(allocation["operation_id"]),
                    fence_token=str(allocation["fence_token"]),
                    generation=int(allocation["attempt"]),
                )
                terminal, slot = self.store.mark_allocation_cleanup_terminal(
                    allocation_key
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "allocation cleanup %s failed: %s",
                    allocation_key,
                    exc,
                )
                try:
                    pending = self.store.record_allocation_cleanup_failure(
                        allocation_key,
                        str(exc),
                    )
                except Exception:
                    logger.exception(
                        "failed to persist allocation cleanup failure %s",
                        allocation_key,
                    )
                    self._queued_allocations.discard(allocation_key)
                    continue
                attempts = int(pending.get("cleanup_attempts", 0))
                if attempts < max_cleanup_attempts:
                    await asyncio.sleep(self.retry_seconds)
                    self._queued_allocations.discard(allocation_key)
                    self.enqueue_allocation(pending)
                    continue
                logger.error(
                    "allocation cleanup %s abandoned after %d attempts; "
                    "releasing capacity hold (worker fencing owns the stale "
                    "allocation)",
                    allocation_key,
                    attempts,
                )
                try:
                    terminal, slot = (
                        self.store.mark_allocation_cleanup_terminal(
                            allocation_key
                        )
                    )
                except Exception:
                    logger.exception(
                        "failed to force-terminal abandoned cleanup %s",
                        allocation_key,
                    )
                    self._queued_allocations.discard(allocation_key)
                    continue
                abandoned = True
            self._queued_allocations.discard(allocation_key)
            await self.scheduler.release_lease(
                self.allocation_hold_id(allocation_key), slot.key,
            )
            self._finish_waiters(self._allocation_completion, allocation_key)
            if (
                str(terminal["action"]) == "RETRY"
                and slot.state == SlotState.RETRY_PENDING
            ):
                if abandoned:
                    logger.warning(
                        "retrying slot %s on a fresh allocation after "
                        "abandoned cleanup %s",
                        slot.key,
                        allocation_key,
                    )
                await self.on_retry(slot)

    def _replacement_finished(self, task: asyncio.Task[Any]) -> None:
        self._callbacks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "task-group replacement failed",
                exc_info=task.exception(),
            )

    async def _expiry_loop(self) -> None:
        while not self._stopping:
            expiry = self.store.next_claim_expiry()
            self._deadline_changed.clear()
            if expiry is None:
                await self._deadline_changed.wait()
                continue
            delay = max(0.0, expiry - self.clock())
            if delay > 0:
                try:
                    await asyncio.wait_for(self._deadline_changed.wait(), delay)
                    continue
                except asyncio.TimeoutError:
                    pass
            await self.recover_expired_claims()

    async def reconcile_startup(self) -> None:

        remote: list[dict[str, Any]] | None = None
        last_error: BaseException | None = None
        for attempt in range(1, 6):
            try:
                remote = list(await self.env_client.list_leases())
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "could not list env leases during v2 reconciliation "
                    "(attempt %d/5): %s",
                    attempt,
                    exc,
                )
                if attempt < 5:
                    await asyncio.sleep(self.retry_seconds)
        if remote is None:
            raise RuntimeError(
                "Env lease inventory unavailable during v2 reconciliation"
            ) from last_error

        for allocation in self.store.list_allocation_cleanups(
            statuses=["ALLOCATING"]
        ):
            slot = self.store.get_slot(allocation["key"])
            if slot.state == SlotState.RUNNING:
                continue
            batch = self.store.get_batch(slot.key.batch_id)
            retry = (
                slot.state == SlotState.RETRY_PENDING
                and batch.state
                in {
                    BatchState.PLANNED,
                    BatchState.RUNNING,
                    BatchState.RECOVERING,
                }
            )
            pending = self.store.quarantine_allocation(
                str(allocation["allocation_key"]),
                action="RETRY" if retry else "NONE",
                reason="host-restart-during-allocation-quarantine",
            )
            self.enqueue_allocation(pending)
        remote_by_id = {
            str(item.get("lease_id")): item
            for item in remote
            if item.get("lease_id")
        }
        persisted = self.store.list_leases(
            statuses=[LeaseStatus.ACTIVE, LeaseStatus.CLOSING]
        )
        persisted_ids = {lease.lease_id for lease in persisted}
        for lease in persisted:
            remote_item = remote_by_id.get(lease.lease_id)
            remote_status = (
                str(remote_item.get("status") or "ACTIVE").upper()
                if remote_item is not None
                else "MISSING"
            )
            slot = self.store.get_slot(lease.key)
            if (
                remote_status == "ACTIVE"
                and lease.status == LeaseStatus.ACTIVE
                and slot.state in {SlotState.READY, SlotState.CLAIMED}
            ):

                continue
            intent = self._startup_close_intent(
                lease,
                reason=f"startup-remote-{remote_status.lower()}",
            )
            if remote_status in {"ACTIVE", "CLOSING"}:
                if intent is not None:
                    self.enqueue(intent)
                continue
            if remote_status not in {"CLOSED", "LOST", "MISSING"}:
                logger.warning(
                    "treating unknown remote lease status %s as lost for %s",
                    remote_status,
                    lease.lease_id,
                )
                remote_status = "LOST"
            if intent is None:
                intent = self.store.enqueue_attempt_close(
                    lease.lease_id,
                    action="NONE",
                    reason=f"startup-remote-{remote_status.lower()}",
                )
            local_lease = self.store.get_lease(lease.lease_id)
            slot = self.store.mark_close_intent_terminal(
                intent.intent_id,
                lost=remote_status != "CLOSED",
            )
            await self.scheduler.release_lease(lease.lease_id, lease.key)
            if slot.state == SlotState.RETRY_PENDING:
                await self.on_retry(slot)
            if intent.reason.startswith("attempt-exhausted"):
                await self.on_exhausted(
                    intent.key,
                    int(
                        local_lease.owner.get(
                            "task_group_revision",
                            slot.task_group_revision,
                        )
                    ),
                )

        for batch in self.store.list_batches(
            states=[
                BatchState.PLANNED,
                BatchState.RUNNING,
                BatchState.DRAINING,
                BatchState.RECOVERING,
            ]
        ):
            for slot in self.store.list_slots(
                batch.batch_id,
                states=[SlotState.RUNNING],
            ):
                attempt_base = int(slot.metadata.get("attempt_base", 0))
                attempt_ordinal = slot.current_attempt - attempt_base
                retryable = attempt_ordinal < self.max_attempts
                allocation_key = (
                    f"{slot.key.batch_id.run_id}:"
                    f"{slot.key.batch_id.train_step_id}:"
                    f"{slot.key.task_index}:{slot.key.actor_id}:"
                    f"{slot.current_attempt}"
                )
                fence_token = f"fence:{allocation_key}"
                owner = {
                    "component": "rollout-host-v2",
                    **slot.key.to_dict(),
                    "attempt": slot.current_attempt,
                    "task_group_revision": slot.task_group_revision,
                }
                if not self.store.has_allocation_hold(
                    slot.key,
                    slot.current_attempt,
                ):
                    self.store.begin_allocation(
                        slot.key,
                        slot.current_attempt,
                        allocation_key=allocation_key,
                        fence_token=fence_token,
                        owner=owner,
                    )
                    await self.scheduler.reserve_lease(
                        self.allocation_hold_id(allocation_key)
                    )
                try:
                    self.store.finish_attempt(
                        slot.key,
                        slot.current_attempt,
                        succeeded=False,
                        retryable=retryable,
                        error={
                            "type": "HostRestart",
                            "message": (
                                "attempt interrupted before lease "
                                "registration"
                            ),
                        },
                    )
                except ProtocolError as exc:
                    if exc.code != ErrorCode.FENCED:
                        raise
                pending = self.store.quarantine_allocation(
                    allocation_key,
                    action="RETRY" if retryable else "NONE",
                    reason="host-restart-before-lease-registration",
                )
                self.enqueue_allocation(pending)
                current = self.store.get_slot(slot.key)
                if retryable:

                    pass
                else:
                    await self.on_exhausted(
                        slot.key,
                        slot.task_group_revision,
                    )
        for item in remote:
            lease_id = str(item.get("lease_id"))
            owner = item.get("owner") or {}
            remote_status = str(item.get("status") or "ACTIVE").upper()
            if (
                lease_id not in persisted_ids
                and owner.get("component") == "rollout-host-v2"
                and remote_status in {"ACTIVE", "CLOSING"}
            ):
                allocation_key = str(item.get("allocation_key") or "")
                fence_token = str(item.get("fence_token") or "")
                generation = int(item.get("generation") or owner.get("attempt") or 1)
                last_close_error: BaseException | None = None
                for close_attempt in range(1, 6):
                    try:
                        cancel = getattr(
                            self.env_client,
                            "cancel_allocation",
                            None,
                        )
                        if callable(cancel) and allocation_key:
                            await _call_optional_keywords(
                                cancel,
                                allocation_key,
                                operation_id=(
                                    f"reconcile-cancel:{allocation_key}"
                                ),
                                fence_token=fence_token,
                                generation=generation,
                            )
                        else:
                            await _call_optional_keywords(
                                self.env_client.close_strict,
                                lease_id,
                                operation_id=f"reconcile-close:{lease_id}",
                                fence_token=fence_token,
                            )
                        last_close_error = None
                        break
                    except Exception as exc:
                        last_close_error = exc
                        if close_attempt < 5:
                            await asyncio.sleep(self.retry_seconds)
                if last_close_error is not None:
                    raise RuntimeError(
                        f"could not quarantine unknown owned lease {lease_id}"
                    ) from last_close_error

    def _startup_close_intent(
        self,
        lease: LeaseRecord,
        *,
        reason: str,
    ) -> CloseIntent | None:
        existing = [
            item
            for status in (
                OutboxStatus.PENDING,
                OutboxStatus.CLOSED,
                OutboxStatus.LOST,
            )
            for item in self.store.list_close_outbox(
                status=status,
                limit=100_000,
            )
            if item.lease_id == lease.lease_id
        ]
        if existing:
            return existing[0]
        slot = self.store.get_slot(lease.key)
        if lease.claim_id is not None:
            claim = self.store.get_claim(lease.claim_id)
            if claim.status.value == "ACTIVE":
                return self.store.nack_claim(
                    claim.claim_id,
                    claim.generation,
                    reason=reason,
                    request_id=(
                        f"startup-nack:{claim.claim_id}:{claim.generation}"
                    ),
                )
        if slot.state == SlotState.RUNNING:
            attempt_base = int(slot.metadata.get("attempt_base", 0))
            attempt_ordinal = lease.attempt - attempt_base
            retryable = attempt_ordinal < self.max_attempts
            try:
                self.store.finish_attempt(
                    lease.key,
                    lease.attempt,
                    succeeded=False,
                    retryable=retryable,
                    error={
                        "type": "HostRestart",
                        "message": (
                            "actor attempt interrupted by host restart"
                        ),
                    },
                )
            except ProtocolError as exc:
                if exc.code != ErrorCode.FENCED:
                    raise
            return self.store.enqueue_attempt_close(
                lease.lease_id,
                action="RETRY" if retryable else "NONE",
                reason=(
                    reason if retryable else "attempt-exhausted:startup"
                ),
            )
        if slot.state == SlotState.READY:
            slot = self.store.update_slot_state(
                slot.key,
                SlotState.RETRY_PENDING,
                expected_version=slot.version,
                expected_attempt=lease.attempt,
                reason=reason,
            )
        return self.store.enqueue_attempt_close(
            lease.lease_id,
            action=(
                "RETRY"
                if slot.state == SlotState.RETRY_PENDING
                else "NONE"
            ),
            reason=reason,
        )

class BatchCoordinator:
    def __init__(
        self,
        *,
        store: SQLiteStateStore,
        task_sampler: Any,
        actor_clients: Mapping[str, Any],
        config: RolloutHostV2Config,
    ) -> None:
        self.store = store
        self.task_sampler = task_sampler
        self.actor_clients = dict(actor_clients)
        self.actor_ids = tuple(self.actor_clients)
        self.config = config
        self.scheduler: FairScheduler | None = None
        self.lease_manager: LeaseManager | None = None
        self._on_batch_failed: Callable[[], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()
        self._replacement_locks: dict[
            tuple[BatchId, str, int], asyncio.Lock
        ] = weakref.WeakValueDictionary()

    def bind(
        self,
        scheduler: FairScheduler,
        lease_manager: LeaseManager,
        on_batch_failed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.lease_manager = lease_manager
        self._on_batch_failed = on_batch_failed

    async def acquire_batch(
        self,
        *,
        run_id: str,
        train_step_id: int,
        tasks_per_step: int,
        request_id: str,
        config_fingerprint: str,
    ) -> ManifestRevision:
        if tasks_per_step <= 0 or not request_id or not config_fingerprint:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "tasks_per_step, request_id and config_fingerprint are required",
            )
        batch_id = BatchId(run_id, train_step_id)
        async with self._lock:
            try:
                existing = self.store.get_batch(batch_id)
            except ProtocolError as exc:
                if exc.code != ErrorCode.NOT_FOUND:
                    raise
            else:
                metadata = existing.metadata
                if (
                    metadata.get("acquire_request_id") != request_id
                    or metadata.get("config_fingerprint") != config_fingerprint
                    or int(metadata.get("tasks_per_step", -1)) != tasks_per_step
                    or list(metadata.get("actor_order") or []) != list(self.actor_ids)
                ):
                    raise ProtocolError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "batch identity or request was reused with different input",
                        batch_id.to_dict(),
                    )
                return self.store.get_manifest(batch_id)
            if self.config.one_active_batch:
                active_states = [
                    BatchState.PLANNED,
                    BatchState.RUNNING,
                    BatchState.DRAINING,
                    BatchState.COMPLETE,
                    BatchState.SEALED,
                    BatchState.RECOVERING,
                    BatchState.IN_DOUBT,
                ]
                active = self.store.list_batches(states=active_states)
                if active:
                    raise ProtocolError(
                        ErrorCode.ALREADY_EXISTS,
                        "another rollout batch is still active",
                        {"active_batch": active[0].batch_id.to_dict()},
                    )
            sampled = await self.task_sampler.sample_batch(
                train_step_id,
                tasks_per_step,
            )
            tasks = list(sampled.tasks)
            if len(tasks) != tasks_per_step:
                raise ProtocolError(
                    ErrorCode.VALIDATION_ERROR,
                    "task sampler returned an unexpected task count",
                    {"expected": tasks_per_step, "actual": len(tasks)},
                )
            manifest = self._manifest_for_tasks(batch_id, tasks)
            self.store.create_batch(
                batch_id,
                manifest,
                request_id=request_id,
                metadata={
                    "acquire_request_id": request_id,
                    "config_fingerprint": config_fingerprint,
                    "tasks_per_step": tasks_per_step,
                    "actor_order": list(self.actor_ids),
                    "sampler_metadata": dict(getattr(sampled, "metadata", {}) or {}),
                },
                initial_state=BatchState.RUNNING,
            )
            slots = self.store.list_slots(batch_id)
            if self.scheduler is not None:
                await self.scheduler.enqueue_batch(slots)
            return self.store.get_manifest(batch_id)

    def _manifest_for_tasks(
        self,
        batch_id: BatchId,
        tasks: Sequence[TaskRecord],
    ) -> tuple[ManifestSlot, ...]:
        slots: list[ManifestSlot] = []
        for index, task in enumerate(tasks):
            group_id = (
                f"{batch_id.run_id}:{batch_id.train_step_id}:task:{index}"
            )
            metadata = {
                "prompt": task.prompt,
                "task_config": dict(task.task_config),
                "task_metadata": dict(task.metadata),
                "sample_index": task.sample_index,
                "attempt_base": 0,
            }
            for actor_id in self.actor_ids:
                slots.append(
                    ManifestSlot(
                        task_index=index,
                        actor_id=actor_id,
                        task_id=task.task_id,
                        task_group_id=group_id,
                        metadata=metadata,
                    )
                )
        return tuple(slots)

    async def handle_exhausted(
        self,
        key: SlotKey,
        triggering_group_revision: int,
    ) -> None:
        batch = self.store.get_batch(key.batch_id)
        if batch.state not in {
            BatchState.RUNNING,
            BatchState.DRAINING,
            BatchState.RECOVERING,
        }:
            return
        failed = self.store.get_slot(key)
        if (
            failed.task_group_revision != triggering_group_revision
            or failed.state != SlotState.FAILED
        ):
            return
        if self.config.failure_policy == "fail_batch":
            await self.fail_and_drain(
                key.batch_id,
                "slot-attempts-exhausted",
            )
            return
        if (
            triggering_group_revision - 1
            >= self.config.max_task_replacements
        ):
            await self.fail_and_drain(
                key.batch_id,
                "task-replacement-cap-exhausted",
            )
            return
        intent = self.store.begin_replacement_intent(
            key.batch_id,
            failed.task_group_id,
            triggering_group_revision,
        )
        if intent is not None:
            await self.resume_replacement(intent)

    async def resume_replacement(
        self,
        intent: ReplacementIntent,
    ) -> None:
        identity = (
            intent.batch_id,
            intent.task_group_id,
            intent.triggering_group_revision,
        )
        lock = self._replacement_locks.setdefault(identity, asyncio.Lock())
        async with lock:
            current = next(
                (
                    item
                    for item in self.store.list_replacement_intents()
                    if (
                        item.batch_id,
                        item.task_group_id,
                        item.triggering_group_revision,
                    )
                    == identity
                ),
                None,
            )
            if current is None or current.status in {
                ReplacementStatus.COMPLETED,
                ReplacementStatus.FAILED,
            }:
                return
            batch = self.store.get_batch(current.batch_id)
            if batch.state not in {
                BatchState.RUNNING,
                BatchState.DRAINING,
                BatchState.RECOVERING,
            }:
                return
            group_slots = [
                slot
                for slot in self.store.list_slots(current.batch_id)
                if (
                    slot.task_group_id == current.task_group_id
                    and slot.task_group_revision
                    == current.triggering_group_revision
                )
            ]
            if any(slot.state in {SlotState.ACKING, SlotState.DONE} for slot in group_slots):
                await self.fail_and_drain(
                    current.batch_id, "replacement-has-acknowledged-slots",
                )
                self.store.update_replacement_status(
                    current, ReplacementStatus.FAILED, error="replacement-has-acknowledged-slots",
                )
                return
            if current.status == ReplacementStatus.PENDING:
                expected_identities = {
                    (slot.key.task_index, slot.key.actor_id)
                    for slot in group_slots
                }
                task_indexes = {
                    slot.key.task_index for slot in group_slots
                }
                required_identities = {
                    (task_index, actor_id)
                    for task_index in task_indexes
                    for actor_id in self.actor_ids
                }
                if (
                    len(task_indexes) != 1
                    or expected_identities != required_identities
                ):
                    await self.fail_and_drain(
                        current.batch_id,
                        "replacement-membership-invalid",
                    )
                    return
                sample_next = getattr(
                    self.task_sampler, "sample_next", None
                )
                if callable(sample_next):
                    sampled = await sample_next(1)
                else:
                    sampled = await self.task_sampler.sample_batch(
                        current.batch_id.train_step_id * 1_000_000
                        + current.triggering_group_revision,
                        1,
                    )
                if len(sampled.tasks) != 1:
                    raise ProtocolError(
                        ErrorCode.VALIDATION_ERROR,
                        "replacement sampler must return exactly one task",
                    )
                task = sampled.tasks[0]
                task_index = next(iter(task_indexes))
                attempt_bases = {
                    slot.key.actor_id: slot.current_attempt
                    for slot in group_slots
                }
                replacements = tuple(
                    ManifestSlot(
                        task_index=task_index,
                        actor_id=actor_id,
                        task_id=task.task_id,
                        task_group_id=current.task_group_id,
                        metadata={
                            "prompt": task.prompt,
                            "task_config": dict(task.task_config),
                            "task_metadata": dict(task.metadata),
                            "sample_index": task.sample_index,
                            "attempt_base": attempt_bases[actor_id],
                        },
                    )
                    for actor_id in self.actor_ids
                )
                current = self.store.store_replacement_payload(
                    current,
                    replacement_task=_task_to_dict(task),
                    replacement_slots=replacements,
                )
            if current.status == ReplacementStatus.CLOSING:
                if self.lease_manager is not None:
                    await self.lease_manager.close_slots(
                        group_slots,
                        wait=False,
                        reason="task-group-superseded",
                    )
                if not self.store.replacement_group_leases_terminal(
                    current
                ):
                    return
                current = self.store.update_replacement_status(
                    current,
                    ReplacementStatus.READY_TO_COMMIT,
                )
            if current.status != ReplacementStatus.READY_TO_COMMIT:
                return
            latest_group_slots = [
                slot
                for slot in self.store.list_slots(current.batch_id)
                if slot.task_group_id == current.task_group_id
            ]
            latest_revision = max(
                (
                    slot.task_group_revision
                    for slot in latest_group_slots
                ),
                default=0,
            )
            if latest_revision == current.triggering_group_revision:
                batch = self.store.get_batch(current.batch_id)
                try:
                    replacement = self.store.replace_task_group(
                        current.batch_id,
                        current.task_group_id,
                        current.replacement_slots,
                        request_id=(
                            f"replace:{current.batch_id.run_id}:"
                            f"{current.batch_id.train_step_id}:"
                            f"{current.task_group_id}:"
                            f"{current.triggering_group_revision + 1}"
                        ),
                        expected_batch_version=batch.version,
                    )
                except ProtocolError as exc:
                    if exc.code != ErrorCode.INVALID_TRANSITION:
                        raise

                    await self.fail_and_drain(
                        current.batch_id, "replacement-invalid-transition",
                    )
                    self.store.update_replacement_status(
                        current, ReplacementStatus.FAILED, error=str(exc),
                    )
                    return
            elif latest_revision == current.triggering_group_revision + 1:
                replacement_slots = tuple(
                    slot
                    for slot in latest_group_slots
                    if slot.task_group_revision == latest_revision
                )
                replacement = type("_Replacement", (), {
                    "active_slots": replacement_slots
                })()
            else:
                return
            self.store.update_replacement_status(
                current,
                ReplacementStatus.COMPLETED,
            )
            if self.scheduler is not None:
                await self.scheduler.enqueue_batch(
                    replacement.active_slots
                )

    async def resume_pending_replacements(self) -> None:
        for failed in self.store.list_exhausted_slots():
            await self.handle_exhausted(
                failed.key,
                failed.task_group_revision,
            )
        for intent in self.store.list_replacement_intents(
            statuses=[
                ReplacementStatus.PENDING,
                ReplacementStatus.CLOSING,
                ReplacementStatus.READY_TO_COMMIT,
            ]
        ):
            await self.resume_replacement(intent)

    async def fail_and_drain(
        self,
        batch_id: BatchId,
        reason: str,
    ) -> BatchRecord:
        batch, intents = self.store.fail_batch_and_drain(
            batch_id,
            reason=reason,
        )
        if self.scheduler is not None:
            await self.scheduler.cancel_batch(batch_id)
        if self.lease_manager is not None:
            for intent in intents:
                self.lease_manager.enqueue(intent)
            for allocation in self.store.list_allocation_cleanups(
                statuses=["PENDING"],
                batch_id=batch_id,
            ):
                self.lease_manager.enqueue_allocation(allocation)
        if self._on_batch_failed is not None:
            await self._on_batch_failed()
        return batch

    def seal_batch(self, batch_id: BatchId) -> BatchRecord:
        batch = self.store.get_batch(batch_id)
        if batch.state == BatchState.SEALED:
            return batch
        manifest = self.store.get_manifest(batch_id)
        slots = {
            (slot.key.task_index, slot.key.actor_id): slot
            for slot in self.store.list_slots(batch_id)
            if slot.manifest_revision == manifest.revision
        }
        conflicts = []
        for spec in manifest.slots:
            slot = slots.get((spec.task_index, spec.actor_id))
            if slot is not None and (
                slot.task_id != spec.task_id
                or slot.task_group_id != spec.task_group_id
            ):
                conflicts.append(spec.key(batch_id).to_dict())
        if conflicts:
            if batch.state != BatchState.INVALID:
                self.store.update_batch_state(
                    batch_id,
                    BatchState.INVALID,
                    expected_version=batch.version,
                    reason="manifest-slot-conflict",
                )
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "manifest conflicts with current slot records",
                {"conflicts": conflicts},
            )
        completeness = self.store.validate_expected_slot_completeness(batch_id)
        open_leases = [
            lease
            for lease in self.store.list_leases(batch_id=batch_id)
            if lease.status not in {LeaseStatus.CLOSED, LeaseStatus.LOST}
        ]
        if not completeness.complete or open_leases:
            missing = [
                item.to_dict()
                for item in (*completeness.missing, *completeness.not_done)
            ]
            raise ProtocolError(
                ErrorCode.BATCH_INCOMPLETE,
                "batch is not complete",
                {
                    "missing_slots": missing,
                    "open_leases": [
                        _lease_to_dict(lease) for lease in open_leases
                    ],
                },
            )
        if batch.state == BatchState.RUNNING:
            batch = self.store.update_batch_state(
                batch_id,
                BatchState.COMPLETE,
                expected_version=batch.version,
                reason="all-current-slots-done",
            )
        if batch.state != BatchState.COMPLETE:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "batch can only be sealed from COMPLETE",
                {"state": batch.state.value},
            )
        return self.store.update_batch_state(
            batch_id,
            BatchState.SEALED,
            expected_version=batch.version,
            reason="validated-and-sealed",
        )

    def commit_batch(self, batch_id: BatchId) -> BatchRecord:
        batch = self.store.get_batch(batch_id)
        if batch.state == BatchState.COMMITTED:
            return batch
        if batch.state != BatchState.SEALED:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "batch can only be committed from SEALED",
                {"state": batch.state.value},
            )
        return self.store.update_batch_state(
            batch_id,
            BatchState.COMMITTED,
            expected_version=batch.version,
            reason="trainer-commit",
        )

    async def fail_batch(
        self,
        batch_id: BatchId,
        reason: str,
    ) -> BatchRecord:
        return await self.fail_and_drain(batch_id, reason)

    async def fail_expired_batch(self, batch_id: BatchId) -> BatchRecord:

        batch = self.store.get_batch(batch_id)
        if batch.state not in {
            BatchState.PLANNED,
            BatchState.RUNNING,
            BatchState.DRAINING,
            BatchState.RECOVERING,
        }:
            return batch
        return await self.fail_and_drain(
            batch_id,
            "batch-deadline-exceeded",
        )

    def mark_in_doubt(self, batch_id: BatchId, error: str) -> BatchRecord:

        batch = self.store.get_batch(batch_id)
        if batch.state == BatchState.IN_DOUBT:
            return batch
        if batch.state != BatchState.SEALED:
            raise ProtocolError(
                ErrorCode.INVALID_TRANSITION,
                "only a SEALED batch may be marked IN_DOUBT",
                {"state": batch.state.value},
            )
        return self.store.update_batch_state(
            batch_id,
            BatchState.IN_DOUBT,
            expected_version=batch.version,
            reason=f"trainer-optimizer-error:{error}",
        )

    async def recover_batch(
        self,
        batch_id: BatchId,
        *,
        request_id: str,
    ) -> BatchRecord:
        batch = self.store.recover_failed_batch(
            batch_id,
            request_id=request_id,
        )
        if self.scheduler is not None:
            await self.scheduler.enqueue_batch(
                self.store.list_slots(batch_id)
            )
        return batch

class RolloutHostV2:
    def __init__(
        self,
        *,
        store: SQLiteStateStore,
        artifacts: ArtifactStore,
        task_sampler: Any,
        env_client: Any,
        actor_clients: Mapping[str, Any],
        config: RolloutHostV2Config,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not actor_clients:
            raise ValueError("RolloutHostV2 requires at least one actor")
        self.store = store
        self.artifacts = artifacts
        self.task_sampler = task_sampler
        self.env_client = env_client
        self.actor_clients = dict(actor_clients)
        if any(not actor_id for actor_id in self.actor_clients):
            raise ValueError("RolloutHostV2 actor names must not be empty")
        if isinstance(config.actor_concurrency, Mapping):
            unknown = set(config.actor_concurrency) - set(self.actor_clients)
            if unknown:
                raise ValueError(
                    "actor_concurrency contains unknown actors: "
                    + ", ".join(sorted(map(str, unknown)))
                )
        self.config = config
        self.clock = clock
        self.supervisor = LoopSupervisor(config.supervisor_retry_seconds)
        self.coordinator = BatchCoordinator(
            store=store,
            task_sampler=task_sampler,
            actor_clients=actor_clients,
            config=config,
        )
        live = store.list_leases(
            statuses=[LeaseStatus.ACTIVE, LeaseStatus.CLOSING]
        )
        allocation_holds = [
            LeaseManager.allocation_hold_id(str(item["allocation_key"]))
            for item in store.list_allocation_cleanups(
                statuses=["ALLOCATING", "PENDING"]
            )
        ]
        self.scheduler = FairScheduler(
            actor_ids=tuple(actor_clients),
            env_capacity=config.env_capacity,
            actor_concurrency={
                actor_id: config.concurrency_for(actor_id, len(actor_clients))
                for actor_id in actor_clients
            },
            runner=self._run_slot,
            initial_leases=live,
            initial_capacity_holds=allocation_holds,
            supervisor=self.supervisor,
        )
        self.lease_manager = LeaseManager(
            store=store,
            env_client=env_client,
            scheduler=self.scheduler,
            retry_seconds=config.close_retry_seconds,
            max_attempts=config.max_attempts,
            on_retry=self._schedule_retry,
            on_exhausted=self.coordinator.handle_exhausted,
            clock=clock,
            supervisor=self.supervisor,
        )
        self.broker = ResultBroker(
            store,
            artifacts,
            on_claim_deadline=self.lease_manager.claim_deadline_changed,
        )
        self.coordinator.bind(
            self.scheduler,
            self.lease_manager,
            self.broker.notify_ready,
        )
        self._started = False
        self._stopping = False
        self._started_at: float | None = None

        self._infra_retries: dict[SlotKey, int] = {}
        self._env_restart_fn: Callable[[], Awaitable[None]] | None = None
        self._env_health_probe_fn: Callable[[], Awaitable[bool]] | None = None
        self._env_restart_threshold_seconds = 300.0
        self._env_restart_cooldown_seconds = 300.0
        self._env_health_probe_interval_seconds = 60.0
        self._env_reconciliation_pending = False
        self._env_restart_counts = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
        }

    def configure_env_watchdog(
        self,
        *,
        restart_fn: Callable[[], Awaitable[None]],
        health_probe_fn: Callable[[], Awaitable[bool]],
        threshold_seconds: float,
        cooldown_seconds: float,
        probe_interval_seconds: float,
    ) -> None:
        if min(threshold_seconds, cooldown_seconds, probe_interval_seconds) <= 0:
            raise ValueError("v2 environment watchdog intervals must be positive")
        self._env_restart_fn = restart_fn
        self._env_health_probe_fn = health_probe_fn
        self._env_restart_threshold_seconds = float(threshold_seconds)
        self._env_restart_cooldown_seconds = float(cooldown_seconds)
        self._env_health_probe_interval_seconds = float(probe_interval_seconds)

    async def start(self) -> None:
        if self._started:
            return
        await self._validate_env_master_capabilities()
        self._stopping = False
        await self.lease_manager.reconcile_startup()
        await self.scheduler.start()
        await self.lease_manager.start()
        await self.coordinator.resume_pending_replacements()
        for batch in self.store.list_batches(
            states=[
                BatchState.PLANNED,
                BatchState.RUNNING,
                BatchState.RECOVERING,
            ]
        ):
            await self.scheduler.enqueue_batch(self.store.list_slots(batch.batch_id))
        self.supervisor.start("batch-deadline", self._batch_deadline_loop)
        self.supervisor.start(
            "task-group-replacement",
            self._replacement_loop,
        )
        if self._env_restart_fn is not None and self._env_health_probe_fn is not None:
            self.supervisor.start(
                "environment-watchdog",
                self._environment_watchdog_loop,
            )
        self._started = True
        self._started_at = self.clock()

    async def stop(self) -> None:
        self._stopping = True
        await self.supervisor.stop()
        await self.lease_manager.stop()
        await self.scheduler.stop()
        close_session = getattr(self.env_client, "close_session", None)
        if callable(close_session):
            await close_session()
        self._started = False

    async def _validate_env_master_capabilities(self) -> None:
        status_fn = getattr(self.env_client, "status", None)
        list_leases = getattr(self.env_client, "list_leases", None)
        if not callable(status_fn) or not callable(list_leases):
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "Env Master does not expose the Rollout Host v2 lease protocol",
                {
                    "required": [
                        "/status protocol capabilities",
                        "/leases",
                        "worker incarnation fencing",
                        "lease fence tokens",
                    ]
                },
            )
        probe_error: BaseException | None = None
        status: Any = None
        leases: Any = None
        for attempt in range(1, 6):
            try:
                status = await status_fn()
                leases = await list_leases()
                probe_error = None
                break
            except Exception as exc:
                probe_error = exc
                if attempt < 5:
                    await asyncio.sleep(self.config.close_retry_seconds)
        if probe_error is not None:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "Env Master v2 capability probe failed",
                {
                    "error": (
                        f"{type(probe_error).__name__}: {probe_error}"
                    )
                },
            ) from probe_error
        master = status.get("master") if isinstance(status, Mapping) else None
        protocol = master.get("lease_protocol") if isinstance(master, Mapping) else None
        capabilities = (
            set(map(str, protocol.get("capabilities") or []))
            if isinstance(protocol, Mapping)
            else set()
        )
        required = {
            "leases",
            "worker_incarnation",
            "fence_token",
            "allocation_cancellation",
        }
        missing = sorted(required - capabilities)
        if (
            not isinstance(protocol, Mapping)
            or int(protocol.get("version") or 0) < 2
            or protocol.get("durable_state") is not True
            or missing
            or not isinstance(leases, list)
        ):
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "Env Master is not Rollout Host v2 compatible",
                {
                    "required_version": 2,
                    "durable_state_required": True,
                    "durable_state": (
                        protocol.get("durable_state")
                        if isinstance(protocol, Mapping)
                        else None
                    ),
                    "missing_capabilities": missing,
                    "leases_response_type": type(leases).__name__,
                },
            )

    async def _batch_deadline_loop(self) -> None:
        active_states = [
            BatchState.PLANNED,
            BatchState.RUNNING,
            BatchState.DRAINING,
            BatchState.RECOVERING,
        ]
        while not self._stopping:
            now = self.clock()
            next_delay = min(30.0, self.config.batch_deadline_seconds)
            active_batches = self.store.list_batches(states=active_states)
            active_ids = {batch.batch_id for batch in active_batches}
            for key in list(self._infra_retries):
                if key.batch_id not in active_ids:
                    self._infra_retries.pop(key, None)
            for batch in active_batches:
                remaining = self.config.batch_deadline_seconds - (
                    now - batch.created_at
                )
                if remaining <= 0:
                    logger.error(
                        "failing overdue v2 batch %s after %.1fs",
                        batch.batch_id,
                        now - batch.created_at,
                    )
                    await self.coordinator.fail_expired_batch(batch.batch_id)
                else:
                    next_delay = min(next_delay, remaining)
            await asyncio.sleep(max(0.01, next_delay))

    async def _replacement_loop(self) -> None:
        while not self._stopping:
            await self.coordinator.resume_pending_replacements()
            await asyncio.sleep(
                max(0.01, self.config.supervisor_retry_seconds)
            )

    async def _environment_watchdog_loop(self) -> None:
        if self._env_restart_fn is None or self._env_health_probe_fn is None:
            return
        unhealthy_since: float | None = None
        last_restart = float("-inf")
        while not self._stopping:
            now = self.clock()
            try:
                healthy = bool(await self._env_health_probe_fn())
            except Exception:
                healthy = False
                logger.exception("v2 Env Master health probe failed")
            if healthy and self._env_reconciliation_pending:
                try:
                    await self.lease_manager.reconcile_startup()
                except Exception:
                    logger.exception(
                        "v2 Env Master post-restart reconciliation retry failed"
                    )
                else:
                    self._env_reconciliation_pending = False
                    self._env_restart_counts["successes"] += 1
                    unhealthy_since = None
            elif healthy:
                unhealthy_since = None
            else:
                unhealthy_since = (
                    now if unhealthy_since is None else unhealthy_since
                )
                if (
                    now - unhealthy_since >= self._env_restart_threshold_seconds
                    and now - last_restart >= self._env_restart_cooldown_seconds
                ):
                    self._env_restart_counts["attempts"] += 1
                    last_restart = now
                    try:
                        await self._env_restart_fn()
                        self._env_reconciliation_pending = True
                        await self.lease_manager.reconcile_startup()
                    except Exception:
                        self._env_restart_counts["failures"] += 1
                        logger.exception(
                            "v2 Env Master restart or reconciliation failed"
                        )
                    else:
                        self._env_reconciliation_pending = False
                        self._env_restart_counts["successes"] += 1
                        unhealthy_since = None
            await asyncio.sleep(self._env_health_probe_interval_seconds)

    async def _schedule_retry(self, slot: SlotRecord) -> None:
        batch = self.store.get_batch(slot.key.batch_id)
        if self.store.has_allocation_hold(slot.key):
            return
        if batch.state in {
            BatchState.RUNNING,
            BatchState.DRAINING,
            BatchState.RECOVERING,
        }:
            await self.scheduler.enqueue(slot, retry=True)

    async def _run_slot(self, scheduled: SlotRecord) -> RunOutcome:
        current = self.store.get_slot(scheduled.key)
        batch = self.store.get_batch(current.key.batch_id)
        if (
            current.task_group_revision != scheduled.task_group_revision
            or current.state not in {SlotState.PENDING, SlotState.RETRY_PENDING}
            or batch.state
            not in {
                BatchState.PLANNED,
                BatchState.RUNNING,
                BatchState.RECOVERING,
            }
        ):
            return RunOutcome()
        attempt = self.store.begin_attempt(
            current.key,
            worker_id=f"actor:{current.key.actor_id}",
            expected_version=current.version,
        )
        attempt_base = int(current.metadata.get("attempt_base", 0))
        attempt_ordinal = attempt.attempt - attempt_base
        allocation_key = (
            f"{current.key.batch_id.run_id}:{current.key.batch_id.train_step_id}:"
            f"{current.key.task_index}:{current.key.actor_id}:{attempt.attempt}"
        )
        fence_token = f"fence:{allocation_key}"
        operation_prefix = f"op:{allocation_key}"
        owner = {
            "component": "rollout-host-v2",
            **current.key.to_dict(),
            "attempt": attempt.attempt,
            "task_group_revision": current.task_group_revision,
        }
        lease_id: str | None = None
        lease_registered = False
        started = time.monotonic()
        try:
            self.store.begin_allocation(
                current.key,
                attempt.attempt,
                allocation_key=allocation_key,
                fence_token=fence_token,
                owner=owner,
            )
            allocation = await _call_optional_keywords(
                self.env_client.allocate,
                allocation_key,
                allocation_key=allocation_key,
                owner=owner,
                fence_token=fence_token,
                generation=attempt.attempt,
            )
            lease_id = str(allocation["lease_id"])
            actual_fence = str(allocation.get("fence_token") or fence_token)
            self.store.register_attempt_lease(
                current.key,
                attempt.attempt,
                lease_id=lease_id,
                fence_token=actual_fence,
                allocation_key=allocation_key,
                owner=owner,
            )
            lease_registered = True
            task = _task_from_slot(current)
            reset_started = time.monotonic()
            initial_obs = await _call_optional_keywords(
                self.env_client.reset,
                lease_id,
                task.task_config,
                operation_id=f"{operation_prefix}:reset",
                fence_token=actual_fence,
            )
            reset_seconds = time.monotonic() - reset_started
            actor_started = time.monotonic()
            actor = self.actor_clients[current.key.actor_id]
            actor_result = await asyncio.wait_for(
                _call_optional_keywords(
                    actor.run_task,
                    task,
                    lease_id,
                    initial_obs,
                    fence_token=actual_fence,
                    operation_prefix=operation_prefix,
                ),
                timeout=self.config.actor_timeout_seconds,
            )
            actor_seconds = time.monotonic() - actor_started
            evaluate_started = time.monotonic()
            score = await _call_optional_keywords(
                self.env_client.evaluate,
                lease_id,
                operation_id=f"{operation_prefix}:evaluate",
                fence_token=actual_fence,
            )
            evaluate_seconds = time.monotonic() - evaluate_started
            self.store.record_attempt_timing(
                current.key,
                attempt.attempt,
                queue_wait_seconds=max(
                    0.0,
                    float(attempt.started_at) - float(current.updated_at),
                ),
                actor_seconds=actor_seconds,
                total_seconds=time.monotonic() - started,
            )
            current_batch = self.store.get_batch(current.key.batch_id)
            if current_batch.state not in {
                BatchState.PLANNED,
                BatchState.RUNNING,
                BatchState.RECOVERING,
            }:
                raise ProtocolError(
                    ErrorCode.FENCED,
                    "actor result arrived after its batch stopped accepting work",
                    {
                        **current.key.batch_id.to_dict(),
                        "batch_state": current_batch.state.value,
                    },
                )
            payload = {
                "slot": current.key.to_dict(),
                "task_id": current.task_id,
                "task_group_id": current.task_group_id,
                "task_group_revision": current.task_group_revision,
                "manifest_revision": current.manifest_revision,
                "actor_id": current.key.actor_id,
                "attempt": attempt.attempt,
                "lease": {
                    "lease_id": lease_id,
                    "fence_token": actual_fence,
                    "allocation_key": allocation_key,
                    "owner": owner,
                },
                "prompt": current.metadata.get("prompt"),
                "task_config": current.metadata.get("task_config"),
                "trajectory": actor_result.get("trajectory", []),
                "screenshots": actor_result.get("screenshots", []),
                "num_steps": actor_result.get("num_steps", 0),
                "native_score": float(score),
                "timing": {
                    "reset_seconds": reset_seconds,
                    "actor_seconds": actor_seconds,
                    "evaluate_seconds": evaluate_seconds,
                    "total_seconds": time.monotonic() - started,
                },
            }
            artifact = await asyncio.to_thread(self.artifacts.write_json, payload)
            self.store.finish_attempt(
                current.key,
                attempt.attempt,
                succeeded=True,
                actor_artifact=artifact,
            )
            self._infra_retries.pop(current.key, None)
            await self.broker.notify_ready()
            return RunOutcome(lease_id=lease_id, retain_capacity=True)
        except asyncio.CancelledError:
            if lease_registered and lease_id is not None:
                try:
                    batch_state = self.store.get_batch(
                        current.key.batch_id
                    ).state
                    retryable = batch_state not in {
                        BatchState.FAILED,
                        BatchState.INVALID,
                        BatchState.ABANDONED,
                        BatchState.IN_DOUBT,
                        BatchState.COMMITTED,
                    }
                    intent = self.store.enqueue_attempt_close(
                        lease_id,
                        action="RETRY" if retryable else "NONE",
                        reason=(
                            "host-shutdown"
                            if retryable
                            else "batch-failed"
                        ),
                    )
                    self.lease_manager.enqueue(intent)
                except Exception:
                    logger.exception("failed to quarantine cancelled lease %s", lease_id)
                return RunOutcome(
                    lease_id=lease_id,
                    retain_capacity=True,
                )
            return await self._fail_attempt(
                current,
                attempt.attempt,
                attempt_ordinal,
                lease_id,
                asyncio.CancelledError(),
                allocation_key=allocation_key,
                lease_registered=False,
            )
        except ProtocolError as exc:
            if exc.code == ErrorCode.FENCED:
                logger.warning("late slot attempt fenced for %s", current.key)
            return await self._fail_attempt(
                current,
                attempt.attempt,
                attempt_ordinal,
                lease_id,
                exc,
                allocation_key=allocation_key,
                lease_registered=lease_registered,
            )
        except Exception as exc:
            logger.warning(
                "slot attempt %s/%d failed: %s",
                current.key,
                attempt.attempt,
                exc,
            )
            return await self._fail_attempt(
                current,
                attempt.attempt,
                attempt_ordinal,
                lease_id,
                exc,
                allocation_key=allocation_key,
                lease_registered=lease_registered,
            )

    async def _fail_attempt(
        self,
        slot: SlotRecord,
        attempt: int,
        attempt_ordinal: int,
        lease_id: str | None,
        exc: BaseException,
        *,
        allocation_key: str,
        lease_registered: bool,
    ) -> RunOutcome:
        retryable = attempt_ordinal < self.config.max_attempts
        if not retryable and _is_infra_error(exc):

            used = self._infra_retries.get(slot.key, 0)
            if used < self.config.infra_retry_budget:
                self._infra_retries[slot.key] = used + 1
                retryable = True
                logger.warning(
                    "slot %s attempt %d hit infra failure (%s); granting "
                    "infra retry %d/%d",
                    slot.key,
                    attempt,
                    exc,
                    used + 1,
                    self.config.infra_retry_budget,
                )
        if not retryable:
            self._infra_retries.pop(slot.key, None)
        try:
            self.store.finish_attempt(
                slot.key,
                attempt,
                succeeded=False,
                retryable=retryable,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        except ProtocolError as finish_error:
            if finish_error.code != ErrorCode.FENCED:
                raise
        if not lease_registered:
            current_batch = self.store.get_batch(slot.key.batch_id)
            should_retry = retryable and current_batch.state in {
                BatchState.PLANNED,
                BatchState.RUNNING,
                BatchState.RECOVERING,
            }
            pending = self.store.quarantine_allocation(
                allocation_key,
                action="RETRY" if should_retry else "NONE",
                reason=(
                    "allocation-outcome-unknown"
                    if should_retry
                    else "allocation-cancelled-by-batch"
                ),
            )
            self.lease_manager.enqueue_allocation(pending)
            if not retryable:
                await self.coordinator.handle_exhausted(
                    slot.key,
                    slot.task_group_revision,
                )
            return RunOutcome(
                lease_id=self.lease_manager.allocation_hold_id(allocation_key),
                retain_capacity=True,
            )
        if retryable and lease_id is not None:
            intent = self.store.enqueue_attempt_close(
                lease_id,
                action="RETRY",
                reason="attempt-failed",
            )
            self.lease_manager.enqueue(intent)
            return RunOutcome(lease_id=lease_id, retain_capacity=True)
        current = self.store.get_slot(slot.key)
        if retryable and current.state == SlotState.RETRY_PENDING:
            await self.scheduler.enqueue(current, retry=True)
        elif not retryable:
            await self.coordinator.handle_exhausted(
                slot.key,
                slot.task_group_revision,
            )
        return RunOutcome(
            lease_id=lease_id,
            retain_capacity=lease_id is not None,
        )

    async def ack(
        self,
        *,
        claim_id: str,
        generation: int,
        attempt: int | None = None,
        request_id: str,
        judge_payload: Any | None = None,
        judge_artifact: ArtifactRef | None = None,
    ) -> dict[str, Any]:
        if attempt is not None:
            lease = self.store.get_claim_lease(claim_id)
            if lease.attempt != int(attempt):
                raise ProtocolError(
                    ErrorCode.FENCED,
                    "acknowledgement attempt fence does not match claim lease",
                    {
                        "claim_id": claim_id,
                        "expected_attempt": lease.attempt,
                        "actual_attempt": int(attempt),
                    },
                )
        artifact = judge_artifact or await asyncio.to_thread(self.artifacts.write_json, judge_payload)

        await asyncio.to_thread(self.artifacts.read, artifact)
        result = self.store.record_ack_artifact(
            claim_id,
            generation,
            artifact,
            request_id=request_id,
        )
        intent = self.store.get_close_intent(result.intent_id)
        self.lease_manager.enqueue(intent)
        return {
            "claim_token": claim_id,
            "generation": generation,
            "intent_id": result.intent_id,
            "judge_artifact": artifact.to_dict(),
        }

    async def nack(
        self,
        *,
        claim_id: str,
        generation: int,
        request_id: str,
        reason: str,
    ) -> dict[str, Any]:
        intent = self.store.nack_claim(
            claim_id,
            generation,
            reason=reason,
            request_id=request_id,
        )
        response = {
            "claim_token": claim_id,
            "generation": generation,
            "intent_id": intent.intent_id,
        }
        self.lease_manager.enqueue(intent)
        return response

    async def renew(
        self,
        *,
        claim_id: str,
        generation: int,
        ttl_seconds: float,
        consumer_id: str | None,
        request_id: str,
    ) -> dict[str, Any]:
        claim = self.store.renew_claim(
            claim_id,
            generation,
            ttl_seconds=ttl_seconds,
            consumer_id=consumer_id,
            request_id=request_id,
        )
        self.lease_manager.claim_deadline_changed()
        return {
            "claim_token": claim.claim_id,
            "generation": claim.generation,
            "expires_at": claim.expires_at,
        }

    def status(self) -> dict[str, Any]:
        tasks = getattr(self.task_sampler, "tasks", None)
        scheduler = self.scheduler.status()
        used_envs = int(scheduler["live_leases"]) + sum(
            int(value) for value in scheduler["running"].values()
        )
        capacity = max(1, int(scheduler["env_capacity"]))
        try:
            observed = self.store.observability_snapshot(now=self.clock())
        except Exception as exc:
            logger.exception("failed to compute Rollout Host v2 status aggregates")
            observed = {
                "per_actor": {},
                "batches": [],
                "timings_seconds": {},
                "throughput": {},
                "claim_age_seconds": {},
                "lease_age_seconds": {},
                "close_outbox": {},
                "recovery_depth": {},
                "status_error": f"{type(exc).__name__}: {exc}",
            }
        for actor_id in self.actor_clients:
            observed.setdefault("per_actor", {}).setdefault(
                actor_id,
                {
                    "pending": 0,
                    "running": 0,
                    "ready": 0,
                    "claimed": 0,
                    "done": 0,
                    "retry": 0,
                    "failed": 0,
                },
            )
        try:
            storage = self.store.storage_status()
        except Exception as exc:
            storage = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "mode": "v2",
            "started": self._started,
            "task_count": len(tasks) if tasks is not None else None,
            "actor_count": len(self.actor_clients),
            "actor_order": list(self.actor_clients),
            "per_actor": observed.get("per_actor", {}),
            "scheduler": scheduler,
            "vm_utilization": {
                "used": min(capacity, used_envs),
                "capacity": capacity,
                "fraction": min(1.0, used_envs / capacity),
            },
            "batches": observed.get("batches", []),
            "throughput": observed.get("throughput", {}),
            "timings_seconds": observed.get("timings_seconds", {}),
            "claim_age_seconds": observed.get("claim_age_seconds", {}),
            "lease_age_seconds": observed.get("lease_age_seconds", {}),
            "storage": storage,
            "close_outbox": observed.get("close_outbox", {}),
            "recovery_depth": observed.get("recovery_depth", {}),
            "supervisor_restarts": dict(self.supervisor.restart_counts),
            "environment_restarts": dict(self._env_restart_counts),
            "configuration": {
                "mode": "v2",
                "max_slot_attempts": self.config.max_attempts,
                "max_group_replacements": self.config.max_task_replacements,
                "actor_timeout_seconds": self.config.actor_timeout_seconds,
                "batch_deadline_seconds": self.config.batch_deadline_seconds,
                "claim_ttl_seconds": self.config.claim_ttl_seconds,
                "one_active_batch": self.config.one_active_batch,
                "future_prefetch": False,
                "actor_concurrency": dict(scheduler["actor_limits"]),
                "environment_watchdog": {
                    "enabled": self._env_restart_fn is not None,
                    "failure_threshold_seconds": self._env_restart_threshold_seconds,
                    "cooldown_seconds": self._env_restart_cooldown_seconds,
                    "probe_interval_seconds": self._env_health_probe_interval_seconds,
                },
            },
            **(
                {"status_error": observed["status_error"]}
                if observed.get("status_error")
                else {}
            ),
        }

def _task_from_slot(slot: SlotRecord) -> TaskRecord:
    return TaskRecord(
        task_id=slot.task_id,
        batch_id=(
            f"{slot.key.batch_id.run_id}:{slot.key.batch_id.train_step_id}"
        ),
        sample_index=slot.key.task_index,
        prompt=slot.metadata.get("prompt", {}),
        task_config=dict(slot.metadata.get("task_config") or {}),
        step_id=slot.key.batch_id.train_step_id,
        metadata=dict(slot.metadata.get("task_metadata") or {}),
    )

def _task_to_dict(task: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "batch_id": task.batch_id,
        "sample_index": task.sample_index,
        "prompt": task.prompt,
        "task_config": dict(task.task_config),
        "step_id": task.step_id,
        "metadata": dict(task.metadata),
    }

def _lease_to_dict(lease: LeaseRecord) -> dict[str, Any]:
    return {
        "lease_id": lease.lease_id,
        "fence_token": lease.fence_token,
        "allocation_key": lease.allocation_key,
        "owner": dict(lease.owner),
        "status": lease.status.value,
        "attempt": lease.attempt,
    }

async def _call_optional_keywords(
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:

    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        accepted = kwargs
    else:
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted = (
            kwargs
            if accepts_var_kwargs
            else {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
        )
    result = operation(*args, **accepted)
    if inspect.isawaitable(result):
        return await result
    return result
