from __future__ import annotations

import asyncio
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guirl.env_service.artifact_store import ArtifactStore
from guirl.env_service.batch_models import (
    BatchId,
    BatchState,
    ErrorCode,
    LeaseRecord,
    LeaseStatus,
    ManifestSlot,
    ProtocolError,
    ReplacementStatus,
    SlotKey,
    SlotState,
)
from guirl.env_service.master_client import BatchSet, TaskRecord
from guirl.env_service.rollout_host_v2 import (
    FairScheduler,
    LoopSupervisor,
    RunOutcome,
    RolloutHostV2,
    RolloutHostV2Config,
)
from guirl.env_service.state_store import SQLiteStateStore
from guirl.master_datasource.data_source import RolloutDataSource

@pytest.fixture
def anyio_backend():
    return "asyncio"

class FakeSampler:
    def __init__(self, task_count: int) -> None:
        self.tasks = [self._task(f"task-{index}", index) for index in range(task_count)]
        self.sample_batch_calls = 0
        self.sample_next_calls = 0

    @staticmethod
    def _task(task_id: str, index: int) -> TaskRecord:
        return TaskRecord(
            task_id=task_id,
            batch_id="fake",
            sample_index=index,
            prompt={"instruction": f"do {task_id}"},
            task_config={"instruction": f"do {task_id}", "task_id": task_id},
            metadata={"source": "fake"},
        )

    async def sample_batch(self, step_id: int, count: int) -> BatchSet:
        self.sample_batch_calls += 1
        return BatchSet(
            step_id=step_id,
            batch_id=f"batch-{step_id}",
            tasks=list(self.tasks[:count]),
            metadata={"sample_call": self.sample_batch_calls},
        )

    async def sample_next(self, count: int) -> BatchSet:
        self.sample_next_calls += 1
        tasks = [
            self._task(f"replacement-{self.sample_next_calls}-{index}", index)
            for index in range(count)
        ]
        return BatchSet(
            step_id=10_000 + self.sample_next_calls,
            batch_id=f"replacement-{self.sample_next_calls}",
            tasks=tasks,
        )

class FakeEnv:
    def __init__(
        self,
        close_failures: int = 0,
        evaluate_failures: int = 0,
    ) -> None:
        self.allocations: dict[str, dict] = {}
        self.live: dict[str, dict] = {}
        self.closed: list[str] = []
        self.cancelled_allocations: list[str] = []
        self.operations: list[tuple[str, str | None, str | None]] = []
        self.close_failures = close_failures
        self.evaluate_failures = evaluate_failures
        self._condition = asyncio.Condition()

    async def allocate(
        self,
        episode_id: str,
        *,
        allocation_key: str,
        owner: dict,
        fence_token: str,
        generation: int,
    ):
        if allocation_key in self.allocations:
            return self.allocations[allocation_key]
        lease_id = f"lease-{len(self.allocations) + 1}"
        value = {
            "lease_id": lease_id,
            "allocation_key": allocation_key,
            "owner": owner,
            "fence_token": fence_token,
            "generation": generation,
        }
        self.allocations[allocation_key] = value
        self.live[lease_id] = value
        async with self._condition:
            self._condition.notify_all()
        return value

    async def reset(
        self,
        lease_id: str,
        task_config: dict,
        *,
        operation_id: str,
        fence_token: str,
    ):
        self.operations.append(("reset", operation_id, fence_token))
        return {"screenshot": b"reset", "terminal": ""}

    async def evaluate(
        self,
        lease_id: str,
        *,
        operation_id: str,
        fence_token: str,
    ):
        self.operations.append(("evaluate", operation_id, fence_token))
        if self.evaluate_failures:
            self.evaluate_failures -= 1
            raise RuntimeError("injected broken GT")
        return 1.0

    async def close_strict(
        self,
        lease_id: str,
        *,
        operation_id: str,
        fence_token: str,
    ):
        self.operations.append(("close", operation_id, fence_token))
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected close failure")
        self.live.pop(lease_id, None)
        self.closed.append(lease_id)
        async with self._condition:
            self._condition.notify_all()
        return {"ok": True}

    async def cancel_allocation(
        self,
        allocation_key: str,
        *,
        operation_id: str,
        fence_token: str,
        generation: int,
    ):
        value = self.allocations.get(allocation_key)
        if value is not None:
            assert value["fence_token"] == fence_token
            assert int(value["generation"]) == int(generation)
            lease_id = str(value["lease_id"])
            self.live.pop(lease_id, None)
            self.closed.append(lease_id)
        self.cancelled_allocations.append(allocation_key)
        async with self._condition:
            self._condition.notify_all()
        return {"ok": True, "status": "CANCELLED"}

    async def list_leases(self):
        return list(self.live.values())

    async def status(self):
        return {
            "ok": True,
            "master": {
                "lease_protocol": {
                    "version": 2,
                    "durable_state": True,
                    "capabilities": [
                        "leases",
                        "worker_incarnation",
                        "fence_token",
                        "allocation_cancellation",
                    ],
                }
            },
        }

    async def wait_allocations(self, count: int) -> None:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: len(self.allocations) >= count
                ),
                2,
            )

    async def wait_closed(self, count: int) -> None:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: len(self.closed) >= count),
                2,
            )

class FakeActor:
    def __init__(
        self,
        name: str,
        *,
        block: bool = False,
        fail_first: bool = False,
        fail_task_prefix: str | None = None,
    ) -> None:
        self.name = name
        self.block = block
        self.fail_first = fail_first
        self.fail_task_prefix = fail_task_prefix
        self.calls: list[str] = []
        self.release = asyncio.Event()
        self._condition = asyncio.Condition()

    async def run_task(
        self,
        task: TaskRecord,
        lease_id: str,
        initial_obs: dict,
        *,
        fence_token: str,
        operation_prefix: str,
    ):
        self.calls.append(task.task_id)
        async with self._condition:
            self._condition.notify_all()
        if self.block:
            await self.release.wait()
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("injected actor retry")
        if self.fail_task_prefix and task.task_id.startswith(self.fail_task_prefix):
            raise RuntimeError("injected task-group replacement")
        return {
            "trajectory": [{"actor": self.name, "task": task.task_id}],
            "screenshots": ["image"],
            "num_steps": 1,
        }

    async def wait_calls(self, count: int) -> None:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: len(self.calls) >= count),
                2,
            )

def make_host(
    tmp_path: Path,
    *,
    sampler: FakeSampler,
    env: FakeEnv,
    actors: dict[str, FakeActor],
    capacity: int,
    max_attempts: int = 2,
    replacements: int = 2,
    batch_deadline_seconds: float = 6 * 60 * 60,
    failure_policy: str = "replace_task_group",
    infra_retry_budget: int = 0,
) -> RolloutHostV2:
    return RolloutHostV2(
        store=SQLiteStateStore(tmp_path / "state.db"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=sampler,
        env_client=env,
        actor_clients=actors,
        config=RolloutHostV2Config(
            env_capacity=capacity,
            actor_concurrency={name: 16 for name in actors},
            max_attempts=max_attempts,
            max_task_replacements=replacements,
            failure_policy=failure_policy,
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
            batch_deadline_seconds=batch_deadline_seconds,
            infra_retry_budget=infra_retry_budget,
        ),
    )

@pytest.mark.anyio
async def test_fixed_manifest_four_unequal_actors_no_fast_ninth(tmp_path: Path):
    sampler = FakeSampler(8)
    env = FakeEnv()
    fast = FakeActor("fast")
    slow = {
        name: FakeActor(name, block=True)
        for name in ("slow-a", "slow-b", "slow-c")
    }
    actors = {"fast": fast, **slow}
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors=actors,
        capacity=32,
    )
    await host.start()
    try:
        manifest = await host.coordinator.acquire_batch(
            run_id="run",
            train_step_id=1,
            tasks_per_step=8,
            request_id="acquire-1",
            config_fingerprint="cfg",
        )
        replay = await host.coordinator.acquire_batch(
            run_id="run",
            train_step_id=1,
            tasks_per_step=8,
            request_id="acquire-1",
            config_fingerprint="cfg",
        )
        await fast.wait_calls(8)
        for actor in slow.values():
            await actor.wait_calls(8)
        assert replay == manifest
        assert sampler.sample_batch_calls == 1
        assert len(manifest.slots) == 8 * 4
        assert [slot.actor_id for slot in manifest.slots[:4]] == list(actors)
        assert len(fast.calls) == 8
        async with asyncio.timeout(5):
            while any(
                slot.state != SlotState.READY for slot in host.store.list_slots(BatchId("run", 1))
                if slot.key.actor_id == "fast"
            ):
                await asyncio.sleep(0.01)
        slots = host.store.list_slots(BatchId("run", 1))
        assert {
            slot.state for slot in slots if slot.key.actor_id == "fast"
        } == {SlotState.READY}
        assert {
            slot.state for slot in slots if slot.key.actor_id != "fast"
        } == {SlotState.RUNNING}
        for actor in slow.values():
            actor.release.set()
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_capacity_streams_and_ack_immediately_refills(tmp_path: Path):
    sampler = FakeSampler(4)
    env = FakeEnv(close_failures=1)
    actors = {"a": FakeActor("a"), "b": FakeActor("b")}
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors=actors,
        capacity=2,
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="stream",
            train_step_id=2,
            tasks_per_step=4,
            request_id="acquire-stream",
            config_fingerprint="cfg",
        )
        await env.wait_allocations(2)
        assert len(env.allocations) == 2
        assert {
            item["owner"]["actor_id"]
            for item in list(env.allocations.values())[:2]
        } == {"a", "b"}
        first = await host.broker.pull_ready(
            BatchId("stream", 2),
            consumer="trainer",
            limit=1,
            request_id="pull-1",
            ttl_seconds=30,
            wait_seconds=2,
        )
        entry = first["entries"][0]
        await host.ack(
            claim_id=entry["claim_token"],
            generation=entry["generation"],
            request_id="ack-1",
            judge_payload={"reward": 1},
        )
        await env.wait_allocations(3)
        assert len(env.operations) >= 1
        close_intents = host.store.list_close_outbox(status="CLOSED")
        assert close_intents
        assert close_intents[0].close_attempts >= 1
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_retry_then_ready_and_pull_replay_is_identical(tmp_path: Path):
    sampler = FakeSampler(1)
    env = FakeEnv()
    actor = FakeActor("a", fail_first=True)
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors={"a": actor},
        capacity=1,
        max_attempts=2,
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="retry",
            train_step_id=3,
            tasks_per_step=1,
            request_id="acquire-retry",
            config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            BatchId("retry", 3),
            consumer="trainer",
            limit=1,
            request_id="pull-retry",
            ttl_seconds=30,
            wait_seconds=2,
        )
        replay = await host.broker.pull_ready(
            BatchId("retry", 3),
            consumer="trainer",
            limit=1,
            request_id="pull-retry",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert result == replay
        assert result["entries"][0]["attempt"] == 2
        assert len(actor.calls) == 2
        assert len(env.closed) == 1
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_broken_gt_retries_without_publishing_bad_label(tmp_path: Path):
    sampler = FakeSampler(1)
    env = FakeEnv(evaluate_failures=1)
    actor = FakeActor("a")
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors={"a": actor},
        capacity=1,
        max_attempts=2,
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="broken-gt",
            train_step_id=30,
            tasks_per_step=1,
            request_id="acquire-broken-gt",
            config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            BatchId("broken-gt", 30),
            consumer="trainer",
            limit=1,
            request_id="pull-broken-gt",
            ttl_seconds=30,
            wait_seconds=2,
        )

        assert result["entries"][0]["attempt"] == 2
        assert result["entries"][0]["actor_payload"]["native_score"] == 1.0
        assert len(actor.calls) == 2
        assert len(
            [
                operation
                for operation in env.operations
                if operation[0] == "evaluate"
            ]
        ) == 2
        assert len(env.closed) == 1
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_exhaustion_replaces_whole_group_and_reruns_all_actors(
    tmp_path: Path,
):
    sampler = FakeSampler(1)
    env = FakeEnv()
    actors = {
        "fails-original": FakeActor(
            "fails-original",
            fail_task_prefix="task-",
        ),
        "succeeds": FakeActor("succeeds"),
    }
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors=actors,
        capacity=2,
        max_attempts=1,
        replacements=1,
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="replace",
            train_step_id=4,
            tasks_per_step=1,
            request_id="acquire-replace",
            config_fingerprint="cfg",
        )

        async with asyncio.timeout(5):
            while not all(
                slot.task_group_revision == 2 and slot.state == SlotState.READY
                for slot in host.store.list_slots(BatchId("replace", 4))
            ):
                await asyncio.sleep(0.01)
        result = await host.broker.pull_ready(
            BatchId("replace", 4),
            consumer="trainer",
            limit=2,
            request_id="pull-replacement",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert len(result["entries"]) == 2
        assert {item["task_id"] for item in result["entries"]} == {
            "replacement-1-0"
        }
        assert {item["task_group_revision"] for item in result["entries"]} == {2}
        assert sampler.sample_next_calls == 1
        assert actors["succeeds"].calls == ["task-0", "replacement-1-0"]
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_claim_expiry_closes_and_reruns_instead_of_redelivery(
    tmp_path: Path,
):
    now = [0.0]
    sampler = FakeSampler(1)
    env = FakeEnv()
    actor = FakeActor("a")
    host = RolloutHostV2(
        store=SQLiteStateStore(tmp_path / "state.db", clock=lambda: now[0]),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=sampler,
        env_client=env,
        actor_clients={"a": actor},
        config=RolloutHostV2Config(
            env_capacity=1,
            actor_concurrency=1,
            max_attempts=3,
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
        ),
        clock=lambda: now[0],
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="expiry",
            train_step_id=5,
            tasks_per_step=1,
            request_id="acquire-expiry",
            config_fingerprint="cfg",
        )
        first = await host.broker.pull_ready(
            BatchId("expiry", 5),
            consumer="trainer",
            limit=1,
            request_id="pull-expiring",
            ttl_seconds=5,
            wait_seconds=2,
        )
        first_entry = first["entries"][0]
        now[0] = 6
        await host.lease_manager.recover_expired_claims(now=6)
        await actor.wait_calls(2)
        second = await host.broker.pull_ready(
            BatchId("expiry", 5),
            consumer="trainer",
            limit=1,
            request_id="pull-after-expiry",
            ttl_seconds=5,
            wait_seconds=2,
        )
        second_entry = second["entries"][0]
        assert second_entry["attempt"] == 2
        assert second_entry["lease"]["lease_id"] != first_entry["lease"]["lease_id"]
        assert host.store.get_claim(first_entry["claim_token"]).status.value == "EXPIRED"
        assert first_entry["lease"]["lease_id"] in env.closed
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_seal_reports_missing_then_commit_requires_sealed(tmp_path: Path):
    sampler = FakeSampler(1)
    env = FakeEnv()
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("seal", 6)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-seal",
            config_fingerprint="cfg",
        )
        with pytest.raises(ProtocolError) as incomplete:
            host.coordinator.seal_batch(batch_id)
        assert incomplete.value.code == ErrorCode.BATCH_INCOMPLETE
        assert incomplete.value.details["missing_slots"]
        with pytest.raises(ProtocolError) as not_sealed:
            host.coordinator.commit_batch(batch_id)
        assert not_sealed.value.code == ErrorCode.INVALID_TRANSITION

        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-seal",
            ttl_seconds=30,
            wait_seconds=2,
        )
        entry = result["entries"][0]
        with pytest.raises(ProtocolError) as wrong_attempt:
            await host.ack(
                claim_id=entry["claim_token"],
                generation=entry["generation"],
                attempt=entry["attempt"] + 1,
                request_id="ack-seal-wrong-attempt",
                judge_payload={"reward": 1},
            )
        assert wrong_attempt.value.code == ErrorCode.FENCED
        ack = await host.ack(
            claim_id=entry["claim_token"],
            generation=entry["generation"],
            attempt=entry["attempt"],
            request_id="ack-seal",
            judge_payload={"reward": 1},
        )
        await host.lease_manager.wait_closed(
            host.store.get_close_intent(ack["intent_id"])
        )
        sealed = host.coordinator.seal_batch(batch_id)
        assert sealed.state == BatchState.SEALED
        committed = host.coordinator.commit_batch(batch_id)
        assert committed.state == BatchState.COMMITTED
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_datasource_ack_copies_summary_into_host_filesystem(
    tmp_path: Path,
    monkeypatch,
):
    host_root = tmp_path / "host-filesystem"
    trainer_root = tmp_path / "trainer-filesystem"
    host_root.mkdir()
    sampler = FakeSampler(1)
    env = FakeEnv()
    host = make_host(
        host_root,
        sampler=sampler,
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("artifact-copy", 0)
    try:
        manifest = await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-artifact-copy",
            config_fingerprint="cfg",
        )
        claimed = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-artifact-copy",
            ttl_seconds=30,
            wait_seconds=2,
        )
        entry = claimed["entries"][0]

        monkeypatch.setattr(
            RolloutDataSource, "_healthcheck", lambda self: None
        )
        data_source = RolloutDataSource(
            Namespace(
                guirl_rollout_host_v2=True,
                guirl_rollout_master_url="http://host",
                guirl_v2_trainer_state_dir=str(trainer_root),
                guirl_v2_run_id=batch_id.run_id,
                guirl_v2_client_id="trainer",
                guirl_v2_claim_ttl_seconds=30,
                guirl_v2_slots_per_actor=1,
                rollout_batch_size=1,
                global_batch_size=1,
                n_samples_per_prompt=1,
            )
        )
        data_source.accumulator.refresh_manifest(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            manifest_revision=manifest.revision,
            payload_hash=manifest.payload_hash,
            slots=[slot.to_dict() for slot in manifest.slots],
        )
        data_source.accumulator.record_claim(
            batch_id.run_id, batch_id.train_step_id, entry
        )
        persisted = data_source.accumulator.persist_result(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            entry=entry,
            generated_group={"full": "sample-pickle-stays-local"},
            judge_summary={"reward": 1, "verdict": "success"},
            source_group_id=0,
        )
        local_group_uri = data_source.accumulator.pending_acks(
            batch_id.run_id, batch_id.train_step_id
        )[0]["group_uri"]

        loop = asyncio.get_running_loop()
        sent_payloads = []

        def post_to_real_host(endpoint, payload, timeout=None):
            del timeout
            assert endpoint == "/v2/ack"
            sent_payloads.append(payload)
            future = asyncio.run_coroutine_threadsafe(
                host.ack(
                    claim_id=payload["claim_token"],
                    generation=payload["generation"],
                    attempt=payload["attempt"],
                    request_id=payload["request_id"],
                    judge_payload=payload["judge_payload"],
                ),
                loop,
            )
            return {"ok": True, **future.result(timeout=5)}

        monkeypatch.setattr(data_source, "_post_json", post_to_real_host)
        response = await asyncio.to_thread(
            data_source.ack,
            batch_id.train_step_id,
            entry,
            persisted["judge_artifact"],
        )

        assert "judge_artifact" not in sent_payloads[0]
        assert "file://" not in str(sent_payloads[0]["judge_payload"])
        host_ref = response["judge_artifact"]
        assert host.artifacts.read_json(host_ref["uri"]) == sent_payloads[0][
            "judge_payload"
        ]
        assert str(host_root) in host_ref["uri"]
        assert str(trainer_root) not in host_ref["uri"]
        assert data_source.accumulator.host_judge_artifact(
            batch_id.run_id, batch_id.train_step_id, 0, "a"
        ) == host_ref
        local_pickle = Path(str(local_group_uri).removeprefix("file://"))
        assert local_pickle.exists()
        assert str(trainer_root) in str(local_pickle)
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_sealed_batch_can_be_fenced_in_doubt(tmp_path: Path):
    sampler = FakeSampler(1)
    env = FakeEnv()
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("in-doubt", 60)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-in-doubt",
            config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-in-doubt",
            ttl_seconds=30,
            wait_seconds=2,
        )
        entry = result["entries"][0]
        ack = await host.ack(
            claim_id=entry["claim_token"],
            generation=entry["generation"],
            request_id="ack-in-doubt",
            judge_payload={"reward": 1},
        )
        await host.lease_manager.wait_closed(
            host.store.get_close_intent(ack["intent_id"])
        )
        host.coordinator.seal_batch(batch_id)
        fenced = host.coordinator.mark_in_doubt(batch_id, "optimizer failed")
        assert fenced.state == BatchState.IN_DOUBT
        assert host.coordinator.mark_in_doubt(
            batch_id, "response replay"
        ).state == BatchState.IN_DOUBT
        with pytest.raises(ProtocolError) as cannot_commit:
            host.coordinator.commit_batch(batch_id)
        assert cannot_commit.value.code == ErrorCode.INVALID_TRANSITION
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_manifest_conflict_marks_batch_invalid(tmp_path: Path):
    sampler = FakeSampler(1)
    env = FakeEnv()
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("invalid", 7)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-invalid",
            config_fingerprint="cfg",
        )
        with host.store._lock:
            host.store._connection.execute(
                """
                UPDATE slots SET task_id='conflicting-task'
                WHERE run_id=? AND train_step_id=?
                """,
                (batch_id.run_id, batch_id.train_step_id),
            )
        with pytest.raises(ProtocolError) as conflict:
            host.coordinator.seal_batch(batch_id)
        assert conflict.value.code == ErrorCode.VALIDATION_ERROR
        assert host.store.get_batch(batch_id).state == BatchState.INVALID
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_startup_reconciles_interrupted_real_lease(tmp_path: Path):
    store = SQLiteStateStore(tmp_path / "state.db")
    batch_id = BatchId("restart", 8)
    store.create_batch(
        batch_id,
        [
            ManifestSlot(
                0,
                "a",
                "task-0",
                "restart:8:task:0",
                {
                    "prompt": {"instruction": "do task-0"},
                    "task_config": {"instruction": "do task-0"},
                    "attempt_base": 0,
                },
            )
        ],
        request_id="create-restart",
        initial_state=BatchState.RUNNING,
    )
    key = store.list_slots(batch_id)[0].key
    attempt = store.begin_attempt(key)
    store.register_attempt_lease(
        key,
        attempt.attempt,
        lease_id="persisted-lease",
        fence_token="persisted-fence",
        allocation_key="persisted-allocation",
        owner={"component": "rollout-host-v2"},
    )
    env = FakeEnv()
    persisted = {
        "lease_id": "persisted-lease",
        "fence_token": "persisted-fence",
        "allocation_key": "persisted-allocation",
        "owner": {"component": "rollout-host-v2"},
    }
    env.allocations["persisted-allocation"] = persisted
    env.live["persisted-lease"] = persisted
    actor = FakeActor("a")
    host = RolloutHostV2(
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=env,
        actor_clients={"a": actor},
        config=RolloutHostV2Config(
            env_capacity=1,
            actor_concurrency=1,
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
        ),
    )
    await host.start()
    try:
        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-restarted",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert result["entries"][0]["attempt"] == 2
        assert "persisted-lease" in env.closed
        assert store.list_attempts(key)[0].status.value == "FAILED"
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_supervisor_restarts_failed_loop_without_host_death():
    supervisor = LoopSupervisor(retry_seconds=0)
    healthy = asyncio.Event()
    stop = asyncio.Event()
    calls = 0

    async def loop():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected loop crash")
        healthy.set()
        await stop.wait()

    supervisor.start("test", loop)
    await asyncio.wait_for(healthy.wait(), 1)
    assert calls == 2
    assert supervisor.restart_counts == {"test": 1}
    stop.set()
    await supervisor.stop()

@pytest.mark.anyio
async def test_default_fair_share_and_empty_status_are_robust(tmp_path: Path):
    actors = {name: FakeActor(name) for name in ("a", "b", "c", "d")}
    host = RolloutHostV2(
        store=SQLiteStateStore(tmp_path / "state.db"),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=FakeEnv(),
        actor_clients=actors,
        config=RolloutHostV2Config(env_capacity=10),
    )
    await host.start()
    try:
        status = host.status()
        assert status["scheduler"]["actor_limits"] == {
            "a": 3,
            "b": 3,
            "c": 3,
            "d": 3,
        }
        assert status["configuration"]["max_slot_attempts"] == 2
        assert status["configuration"]["max_group_replacements"] == 2
        assert status["configuration"]["actor_timeout_seconds"] == 1200
        assert status["configuration"]["batch_deadline_seconds"] == 21600
        assert status["configuration"]["future_prefetch"] is False
        assert status["storage"]["synchronous"] == "FULL"
        assert status["batches"] == []
        assert all(
            actor["pending"] == actor["done"] == 0
            for actor in status["per_actor"].values()
        )
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_startup_rejects_env_master_without_v2_fencing(tmp_path: Path):
    env = FakeEnv()

    async def legacy_status():
        return {"ok": True, "master": {"worker_count": 1}}

    env.status = legacy_status
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    with pytest.raises(ProtocolError, match="not Rollout Host v2 compatible") as error:
        await host.start()
    assert error.value.code == ErrorCode.VALIDATION_ERROR

@pytest.mark.anyio
async def test_status_has_actor_timings_utilization_and_batch_completion(
    tmp_path: Path,
):
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=FakeEnv(),
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="status",
            train_step_id=9,
            tasks_per_step=1,
            request_id="acquire-status",
            config_fingerprint="cfg",
        )
        for _ in range(100):
            if host.store.get_slot(
                host.store.list_slots(BatchId("status", 9))[0].key
            ).state == SlotState.READY:
                break
            await asyncio.sleep(0.01)
        status = host.status()
        assert status["per_actor"]["a"]["ready"] == 1
        assert status["timings_seconds"]["queue_wait"]["count"] == 1
        assert status["timings_seconds"]["actor_execution"]["count"] == 1
        assert status["vm_utilization"]["fraction"] == 1.0
        assert status["batches"][0]["step_completion"]["fraction"] == 0.0
        assert "checkpoint" in status["storage"]
        assert "pending_close_intents" in status["recovery_depth"]
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_supervised_batch_deadline_fails_and_drains(tmp_path: Path):
    actor = FakeActor("slow", block=True)
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=FakeEnv(),
        actors={"slow": actor},
        capacity=1,
        batch_deadline_seconds=0.05,
    )
    await host.start()
    batch_id = BatchId("deadline", 10)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-deadline",
            config_fingerprint="cfg",
        )
        await actor.wait_calls(1)
        for _ in range(100):
            if host.store.get_batch(batch_id).state == BatchState.FAILED:
                break
            await asyncio.sleep(0.01)
        assert host.store.get_batch(batch_id).state == BatchState.FAILED
        assert host.status()["configuration"]["batch_deadline_seconds"] == 0.05
        actor.release.set()
    finally:
        actor.release.set()
        await host.stop()

@pytest.mark.anyio
async def test_failed_batch_drains_and_rejects_pull(tmp_path: Path):
    actor = FakeActor("slow", block=True)
    env = FakeEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"slow": actor},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("manual-fail", 11)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-manual-fail",
            config_fingerprint="cfg",
        )
        await env.wait_allocations(1)
        failed = await host.coordinator.fail_batch(
            batch_id,
            "operator-test",
        )
        assert failed.state == BatchState.FAILED
        await env.wait_closed(1)
        for _ in range(100):
            if host.store.get_lease(
                next(iter(env.closed))
            ).status == LeaseStatus.CLOSED:
                break
            await asyncio.sleep(0.01)
        assert host.store.list_slots(batch_id)[0].state == (
            SlotState.CANCELED
        )
        assert {
            lease.status for lease in host.store.list_leases(
                batch_id=batch_id
            )
        } == {LeaseStatus.CLOSED}
        with pytest.raises(
            ProtocolError,
            match="does not accept result pulls",
        ):
            await host.broker.pull_ready(
                batch_id,
                consumer="trainer",
                limit=1,
                request_id="pull-after-fail",
                ttl_seconds=30,
            )
        assert host.scheduler.status()["available_capacity"] == 1
    finally:
        actor.release.set()
        await host.stop()

@pytest.mark.anyio
@pytest.mark.parametrize(
    ("remote_status", "expected_local_status"),
    [
        ("CLOSED", LeaseStatus.CLOSED),
        ("LOST", LeaseStatus.LOST),
    ],
)
async def test_startup_reconciles_remote_terminal_status_by_id(
    tmp_path: Path,
    remote_status: str,
    expected_local_status: LeaseStatus,
):
    store = SQLiteStateStore(tmp_path / "state.db")
    batch_id = BatchId(f"remote-{remote_status.lower()}", 12)
    store.create_batch(
        batch_id,
        [
            ManifestSlot(
                0,
                "a",
                "task-0",
                f"{batch_id.run_id}:task:0",
                {
                    "prompt": {"instruction": "do task-0"},
                    "task_config": {"instruction": "do task-0"},
                    "attempt_base": 0,
                },
            )
        ],
        request_id=f"create-{remote_status}",
        initial_state=BatchState.RUNNING,
    )
    key = store.list_slots(batch_id)[0].key
    attempt = store.begin_attempt(key)
    store.register_attempt_lease(
        key,
        attempt.attempt,
        lease_id="terminal-remote-lease",
        fence_token="terminal-remote-fence",
        allocation_key="terminal-remote-allocation",
        owner={
            "component": "rollout-host-v2",
            "task_group_revision": 1,
        },
    )
    env = FakeEnv()
    remote = {
        "lease_id": "terminal-remote-lease",
        "status": remote_status,
        "fence_token": "terminal-remote-fence",
        "allocation_key": "terminal-remote-allocation",
        "owner": {
            "component": "rollout-host-v2",
            "task_group_revision": 1,
        },
    }
    env.allocations["terminal-remote-allocation"] = remote
    env.live["terminal-remote-lease"] = remote
    host = RolloutHostV2(
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=env,
        actor_clients={"a": FakeActor("a")},
        config=RolloutHostV2Config(
            env_capacity=1,
            actor_concurrency=1,
            max_attempts=2,
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
        ),
    )
    await host.start()
    try:
        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id=f"pull-{remote_status}",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert result["entries"][0]["attempt"] == 2
        assert host.store.get_lease(
            "terminal-remote-lease"
        ).status == expected_local_status
        assert "terminal-remote-lease" not in env.closed
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_two_actors_exhaust_same_revision_only_replace_once(
    tmp_path: Path,
):
    sampler = FakeSampler(1)
    actors = {
        "a": FakeActor("a", fail_task_prefix="task-"),
        "b": FakeActor("b", fail_task_prefix="task-"),
    }
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=FakeEnv(),
        actors=actors,
        capacity=2,
        max_attempts=1,
        replacements=1,
    )
    await host.start()
    batch_id = BatchId("coalesced", 13)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-coalesced",
            config_fingerprint="cfg",
        )
        async with asyncio.timeout(5):
            while not all(
                slot.task_group_revision == 2 and slot.state == SlotState.READY
                for slot in host.store.list_slots(batch_id)
            ):
                await asyncio.sleep(0.01)
        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=2,
            request_id="pull-coalesced",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert len(result["entries"]) == 2
        assert sampler.sample_next_calls == 1
        assert {
            item["task_group_revision"] for item in result["entries"]
        } == {2}
        intents = host.store.list_replacement_intents()
        assert len(intents) == 1
        assert intents[0].status == ReplacementStatus.COMPLETED
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_replacement_resumes_after_restart_between_stages(
    tmp_path: Path,
):
    class BlockingCloseEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.allow_close = asyncio.Event()

        async def close_strict(
            self,
            lease_id: str,
            *,
            operation_id: str,
            fence_token: str,
        ):
            self.close_started.set()
            await self.allow_close.wait()
            return await super().close_strict(
                lease_id,
                operation_id=operation_id,
                fence_token=fence_token,
            )

    env = BlockingCloseEnv()
    first_sampler = FakeSampler(1)
    first = make_host(
        tmp_path,
        sampler=first_sampler,
        env=env,
        actors={"a": FakeActor("a", fail_task_prefix="task-")},
        capacity=1,
        max_attempts=1,
        replacements=1,
    )
    await first.start()
    batch_id = BatchId("replacement-restart", 14)
    await first.coordinator.acquire_batch(
        run_id=batch_id.run_id,
        train_step_id=batch_id.train_step_id,
        tasks_per_step=1,
        request_id="acquire-replacement-restart",
        config_fingerprint="cfg",
    )
    await asyncio.wait_for(env.close_started.wait(), 2)
    intent = first.store.list_replacement_intents()[0]
    assert intent.status == ReplacementStatus.CLOSING
    assert intent.replacement_task is not None
    assert first_sampler.sample_next_calls == 1
    await first.stop()
    first.store.close()

    env.allow_close.set()
    second_sampler = FakeSampler(1)
    second = make_host(
        tmp_path,
        sampler=second_sampler,
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
        max_attempts=1,
        replacements=1,
    )
    await second.start()
    try:
        result = await second.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-after-replacement-restart",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert result["entries"][0]["task_id"] == "replacement-1-0"
        assert second_sampler.sample_next_calls == 0
        assert second.store.list_replacement_intents()[0].status == (
            ReplacementStatus.COMPLETED
        )
    finally:
        await second.stop()

@pytest.mark.anyio
async def test_restart_at_attempt_limit_never_starts_extra_attempt(
    tmp_path: Path,
):
    store = SQLiteStateStore(tmp_path / "state.db")
    batch_id = BatchId("attempt-limit", 15)
    store.create_batch(
        batch_id,
        [
            ManifestSlot(
                0,
                "a",
                "task-0",
                "attempt-limit:task:0",
                {
                    "prompt": {"instruction": "do task-0"},
                    "task_config": {"instruction": "do task-0"},
                    "attempt_base": 0,
                },
            )
        ],
        request_id="create-attempt-limit",
        initial_state=BatchState.RUNNING,
    )
    key = store.list_slots(batch_id)[0].key
    attempt = store.begin_attempt(key)
    store.register_attempt_lease(
        key,
        attempt.attempt,
        lease_id="attempt-limit-lease",
        fence_token="attempt-limit-fence",
        allocation_key="attempt-limit-allocation",
        owner={
            "component": "rollout-host-v2",
            "task_group_revision": 1,
        },
    )
    env = FakeEnv()
    remote = {
        "lease_id": "attempt-limit-lease",
        "status": "ACTIVE",
        "fence_token": "attempt-limit-fence",
        "allocation_key": "attempt-limit-allocation",
        "owner": {
            "component": "rollout-host-v2",
            "task_group_revision": 1,
        },
    }
    env.allocations["attempt-limit-allocation"] = remote
    env.live["attempt-limit-lease"] = remote
    actor = FakeActor("a")
    host = RolloutHostV2(
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=env,
        actor_clients={"a": actor},
        config=RolloutHostV2Config(
            env_capacity=1,
            actor_concurrency=1,
            max_attempts=1,
            failure_policy="fail_batch",
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
        ),
    )
    await host.start()
    try:
        for _ in range(200):
            if store.get_batch(batch_id).state == BatchState.FAILED:
                break
            await asyncio.sleep(0.01)
        assert store.get_batch(batch_id).state == BatchState.FAILED
        assert len(store.list_attempts(key)) == 1
        assert store.list_attempts(key)[0].status.value == "FAILED"
        assert actor.calls == []
    finally:
        await host.stop()
    store.close()

    reopened = SQLiteStateStore(tmp_path / "state.db")
    assert len(reopened.list_attempts(key)) == 1
    assert reopened.get_batch(batch_id).state == BatchState.FAILED
    reopened.close()

@pytest.mark.anyio
async def test_restart_resolves_running_attempt_without_registered_lease(
    tmp_path: Path,
):
    store = SQLiteStateStore(tmp_path / "state.db")
    batch_id = BatchId("pre-lease-crash", 17)
    store.create_batch(
        batch_id,
        [
            ManifestSlot(
                0,
                "a",
                "task-0",
                "pre-lease-crash:task:0",
                {"attempt_base": 0},
            )
        ],
        request_id="create-pre-lease-crash",
        initial_state=BatchState.RUNNING,
    )
    key = store.list_slots(batch_id)[0].key
    store.begin_attempt(key)
    actor = FakeActor("a")
    host = RolloutHostV2(
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=FakeEnv(),
        actor_clients={"a": actor},
        config=RolloutHostV2Config(
            env_capacity=1,
            max_attempts=1,
            failure_policy="fail_batch",
            close_retry_seconds=0,
            supervisor_retry_seconds=0,
        ),
    )
    await host.start()
    try:
        assert store.get_batch(batch_id).state == BatchState.FAILED
        assert store.get_slot(key).state == SlotState.FAILED
        assert store.list_attempts(key)[0].status.value == "FAILED"
        assert actor.calls == []
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_sealed_and_in_doubt_batches_ignore_execution_deadline(
    tmp_path: Path,
):
    clock = lambda: 100.0
    store = SQLiteStateStore(tmp_path / "state.db", clock=clock)
    for index, state in enumerate(
        (BatchState.SEALED, BatchState.IN_DOUBT)
    ):
        batch_id = BatchId(f"deadline-{state.value.lower()}", index)
        store.create_batch(
            batch_id,
            [
                ManifestSlot(
                    0,
                    "a",
                    f"task-{index}",
                    f"group-{index}",
                )
            ],
            request_id=f"create-{state.value}",
        )
        with store._lock:
            store._connection.execute(
                """
                UPDATE batches SET state=?, created_at=0, updated_at=0
                WHERE run_id=? AND train_step_id=?
                """,
                (state.value, batch_id.run_id, batch_id.train_step_id),
            )
    host = RolloutHostV2(
        store=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        task_sampler=FakeSampler(1),
        env_client=FakeEnv(),
        actor_clients={"a": FakeActor("a")},
        config=RolloutHostV2Config(
            env_capacity=1,
            batch_deadline_seconds=0.01,
            supervisor_retry_seconds=0,
        ),
        clock=clock,
    )
    await host.start()
    try:
        await asyncio.sleep(0.05)
        assert {
            batch.state for batch in store.list_batches()
        } == {BatchState.SEALED, BatchState.IN_DOUBT}
        with pytest.raises(
            ProtocolError,
            match="optimizer-outcome resolution",
        ):
            await host.coordinator.recover_batch(
                BatchId("deadline-in_doubt", 1),
                request_id="recover-in-doubt",
            )
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_scheduler_capacity_stays_capped_after_oversubscribed_recovery(
    tmp_path: Path,
):
    store = SQLiteStateStore(tmp_path / "state.db")
    batch_id = BatchId("capacity-shrink", 16)
    store.create_batch(
        batch_id,
        [ManifestSlot(0, "a", "task", "group")],
        request_id="create-capacity-shrink",
    )
    slot = store.list_slots(batch_id)[0]
    leases = tuple(
        LeaseRecord(
            lease_id=f"lease-{index}",
            key=slot.key,
            attempt=index,
            fence_token=f"fence-{index}",
            allocation_key=f"allocation-{index}",
            owner={},
            status=LeaseStatus.ACTIVE,
            claim_id=None,
            created_at=0,
            updated_at=0,
        )
        for index in range(3)
    )
    ran = asyncio.Event()

    async def runner(_slot):
        ran.set()
        return RunOutcome()

    scheduler = FairScheduler(
        actor_ids=["a"],
        env_capacity=1,
        actor_concurrency=1,
        runner=runner,
        initial_leases=leases,
    )
    await scheduler.start()
    try:
        await scheduler.enqueue(slot)
        await asyncio.sleep(0.02)
        assert not ran.is_set()
        assert scheduler.status()["available_capacity"] == 0
        await scheduler.release_lease("lease-0")
        await scheduler.release_lease("lease-1")
        await asyncio.sleep(0.02)
        assert not ran.is_set()
        await scheduler.release_lease("lease-2")
        await asyncio.wait_for(ran.wait(), 1)
        for _ in range(100):
            if scheduler.status()["available_capacity"] == 1:
                break
            await asyncio.sleep(0.01)
        assert scheduler.status()["available_capacity"] == 1
        assert scheduler.status()["capacity_in_use"] == 0
    finally:
        await scheduler.stop()
        store.close()

@pytest.mark.anyio
async def test_unknown_allocate_outcome_is_cancelled_before_retry(
    tmp_path: Path,
):
    class UnknownOnceEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.failed = False

        async def allocate(self, *args, **kwargs):
            value = await super().allocate(*args, **kwargs)
            if not self.failed:
                self.failed = True
                raise RuntimeError("allocate response was lost")
            return value

    env = UnknownOnceEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    try:
        batch_id = BatchId("unknown-allocation", 18)
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-unknown-allocation",
            config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            batch_id=batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-after-allocation-cleanup",
            ttl_seconds=30,
            wait_seconds=2,
        )
        assert result["entries"][0]["attempt"] == 2
        first_key = "unknown-allocation:18:0:a:1"
        assert first_key in env.cancelled_allocations
        cleanup = host.store.get_allocation_cleanup(first_key)
        assert cleanup["status"] == "CLOSED"
        assert all(
            value["allocation_key"] != first_key
            for value in env.live.values()
        )
    finally:
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_remote_lost_close_terminalizes_local_outbox(
    tmp_path: Path,
):
    class LostCloseEnv(FakeEnv):
        async def close_strict(
            self,
            lease_id: str,
            *,
            operation_id: str,
            fence_token: str,
        ):
            self.operations.append(("close", operation_id, fence_token))
            self.live.pop(lease_id, None)
            self.closed.append(lease_id)
            async with self._condition:
                self._condition.notify_all()
            return {"ok": True, "status": "LOST"}

    env = LostCloseEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    try:
        batch_id = BatchId("lost-close", 19)
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-lost-close",
            config_fingerprint="cfg",
        )
        claim = await host.broker.pull_ready(
            batch_id=batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-lost-close",
            ttl_seconds=30,
            wait_seconds=2,
        )
        entry = claim["entries"][0]
        await host.ack(
            claim_id=entry["claim_token"],
            generation=entry["generation"],
            request_id="ack-lost-close",
            judge_payload={"reward": 1},
        )
        await env.wait_closed(1)
        await asyncio.sleep(0)
        lease = host.store.list_leases(batch_id=batch_id)[0]
        assert lease.status == LeaseStatus.LOST
        assert host.store.list_close_outbox() == ()
        assert host.store.get_slot(lease.key).state == SlotState.DONE
        assert host.scheduler.status()["capacity_in_use"] == 0
    finally:
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_startup_retries_transient_inventory_failure(tmp_path: Path):
    class FlakyInventoryEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.list_calls = 0

        async def list_leases(self):
            self.list_calls += 1
            if self.list_calls <= 2:
                raise RuntimeError("inventory temporarily unavailable")
            return await super().list_leases()

    env = FlakyInventoryEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    try:
        assert env.list_calls == 4
    finally:
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_runtime_env_restart_reconciles_live_lease(tmp_path: Path):
    env = FakeEnv()
    actor = FakeActor("a", block=True)
    healthy = True
    restarted = asyncio.Event()

    async def health_probe():
        return healthy

    async def restart_env():
        nonlocal healthy
        env.live.clear()
        healthy = True
        restarted.set()

    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": actor},
        capacity=1,
    )
    host._env_restart_fn = restart_env
    host._env_health_probe_fn = health_probe
    host._env_health_probe_interval_seconds = 0.01
    host._env_restart_threshold_seconds = 0
    host._env_restart_cooldown_seconds = 0
    await host.start()
    try:
        batch_id = BatchId("runtime-reconcile", 20)
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-runtime-reconcile",
            config_fingerprint="cfg",
        )
        await env.wait_allocations(1)
        await actor.wait_calls(1)
        healthy = False
        await asyncio.wait_for(restarted.wait(), 2)
        for _ in range(100):
            leases = host.store.list_leases(batch_id=batch_id)
            if leases and leases[0].status == LeaseStatus.LOST:
                break
            await asyncio.sleep(0.01)
        assert host.store.list_leases(batch_id=batch_id)[0].status == LeaseStatus.LOST
        assert host._env_restart_counts["successes"] == 1
    finally:
        actor.release.set()
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_failed_batch_waits_for_unknown_allocation_cleanup(
    tmp_path: Path,
):
    class BlockingUnknownEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.allocate_started = asyncio.Event()
            self.allocate_release = asyncio.Event()
            self.cancel_started = asyncio.Event()
            self.cancel_release = asyncio.Event()

        async def allocate(self, *args, **kwargs):
            value = await super().allocate(*args, **kwargs)
            self.allocate_started.set()
            await self.allocate_release.wait()
            return value

        async def cancel_allocation(self, *args, **kwargs):
            self.cancel_started.set()
            await self.cancel_release.wait()
            return await super().cancel_allocation(*args, **kwargs)

    env = BlockingUnknownEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    await host.start()
    batch_id = BatchId("failed-unknown", 21)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-failed-unknown",
            config_fingerprint="cfg",
        )
        await asyncio.wait_for(env.allocate_started.wait(), 2)
        failed = await host.coordinator.fail_batch(
            batch_id,
            "operator-failed-unknown",
        )
        assert failed.state == BatchState.FAILED
        await asyncio.wait_for(env.cancel_started.wait(), 2)
        cleanup = host.store.list_allocation_cleanups(
            batch_id=batch_id,
        )[0]
        assert cleanup["status"] == "PENDING"
        assert host.scheduler.status()["capacity_in_use"] == 1
        with pytest.raises(ProtocolError, match="finish draining"):
            host.store.recover_failed_batch(
                batch_id,
                request_id="recover-before-allocation-cleanup",
            )
        env.cancel_release.set()
        for _ in range(100):
            cleanup = host.store.get_allocation_cleanup(
                str(cleanup["allocation_key"])
            )
            if cleanup["status"] == "CLOSED":
                break
            await asyncio.sleep(0.01)
        assert cleanup["status"] == "CLOSED"
        assert host.scheduler.status()["capacity_in_use"] == 0
        assert env.live == {}
    finally:
        env.allocate_release.set()
        env.cancel_release.set()
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_startup_fails_when_inventory_never_recovers(tmp_path: Path):
    class BrokenInventoryEnv(FakeEnv):
        async def list_leases(self):
            raise RuntimeError("inventory unavailable")

    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=BrokenInventoryEnv(),
        actors={"a": FakeActor("a")},
        capacity=1,
    )
    with pytest.raises(
        ProtocolError,
        match="capability probe failed",
    ):
        await host.start()
    host.store.close()

@pytest.mark.anyio
async def test_infra_errors_do_not_exhaust_slot_attempts(tmp_path: Path):

    class InfraFlakyActor(FakeActor):
        def __init__(self, name: str, failures: int) -> None:
            super().__init__(name)
            self.failures = failures

        async def run_task(self, task, lease_id, initial_obs, *, fence_token, operation_prefix):
            self.calls.append(task.task_id)
            async with self._condition:
                self._condition.notify_all()
            if len(self.calls) <= self.failures:
                raise RuntimeError(
                    f"/step failed: Lease master-lease-{len(self.calls)} is not active"
                )
            return {
                "trajectory": [{"actor": self.name, "task": task.task_id}],
                "screenshots": ["image"],
                "num_steps": 1,
            }

    actor = InfraFlakyActor("a", failures=4)
    sampler = FakeSampler(1)
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=FakeEnv(),
        actors={"a": actor},
        capacity=1,
        max_attempts=1,
        replacements=0,
        failure_policy="replace_task_group",
        infra_retry_budget=10,
    )
    await host.start()
    batch_id = BatchId("infra", 1)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-infra",
            config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-infra",
            ttl_seconds=30,
            wait_seconds=5,
        )
        assert len(result["entries"]) == 1
        assert len(actor.calls) == 5
        batch = host.store.get_batch(batch_id)
        assert batch.state not in {BatchState.FAILED, BatchState.INVALID}

        assert list(host.store.list_replacement_intents()) == []
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_non_infra_errors_still_exhaust_slot_attempts(tmp_path: Path):

    sampler = FakeSampler(1)
    actor = FakeActor("a", fail_task_prefix="task-")
    host = make_host(
        tmp_path,
        sampler=sampler,
        env=FakeEnv(),
        actors={"a": actor},
        capacity=1,
        max_attempts=1,
        replacements=0,
        failure_policy="fail_batch",
        infra_retry_budget=10,
    )
    await host.start()
    batch_id = BatchId("noninfra", 1)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-noninfra",
            config_fingerprint="cfg",
        )
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            batch = host.store.get_batch(batch_id)
            if batch.state == BatchState.FAILED:
                break
            await asyncio.sleep(0.05)
        assert host.store.get_batch(batch_id).state == BatchState.FAILED
        assert len(actor.calls) == 1
    finally:
        await host.stop()

@pytest.mark.anyio
async def test_abandoned_allocation_cleanup_releases_hold(
    tmp_path: Path, monkeypatch
):

    monkeypatch.setenv("GUI_ENV_CLEANUP_MAX_ATTEMPTS", "3")

    class WedgedCancelEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.cancel_calls = 0
            self.fail_first_allocate = True
            self.allocate_calls = 0

        async def allocate(self, *args, **kwargs):
            self.allocate_calls += 1
            if self.allocate_calls == 1:

                raise RuntimeError("allocate timed out")
            return await super().allocate(*args, **kwargs)

        async def cancel_allocation(self, *args, **kwargs):
            self.cancel_calls += 1
            raise RuntimeError("Read timed out (wedged worker)")

    env = WedgedCancelEnv()
    host = make_host(
        tmp_path,
        sampler=FakeSampler(1),
        env=env,
        actors={"a": FakeActor("a")},
        capacity=1,
        max_attempts=2,
    )
    await host.start()
    batch_id = BatchId("abandoned-cleanup", 1)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id,
            train_step_id=batch_id.train_step_id,
            tasks_per_step=1,
            request_id="acquire-abandoned",
            config_fingerprint="cfg",
        )

        result = await host.broker.pull_ready(
            batch_id,
            consumer="trainer",
            limit=1,
            request_id="pull-abandoned",
            ttl_seconds=30,
            wait_seconds=10,
        )
        assert len(result["entries"]) == 1
        assert env.cancel_calls == 3
        cleanup = host.store.list_allocation_cleanups(batch_id=batch_id)[0]
        assert cleanup["status"] == "CLOSED"
    finally:
        await host.stop()
        host.store.close()

@pytest.mark.anyio
async def test_exhaustion_after_sibling_done_preserves_result(tmp_path):
    failing = FakeActor("fails", block=True, fail_task_prefix="task-")
    sampler = FakeSampler(1)
    host = make_host(
        tmp_path, sampler=sampler, env=FakeEnv(),
        actors={"done": FakeActor("done"), "fails": failing},
        capacity=2, max_attempts=1,
    )
    batch_id = BatchId("done-sibling", 1)
    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id, train_step_id=1, tasks_per_step=1,
            request_id="acquire", config_fingerprint="cfg",
        )
        result = await host.broker.pull_ready(
            batch_id, consumer="trainer", limit=1, request_id="pull",
            ttl_seconds=30, wait_seconds=2,
        )
        entry = result["entries"][0]
        assert entry["actor_id"] == "done"
        ack = await host.ack(
            claim_id=entry["claim_token"], generation=entry["generation"],
            request_id="ack", judge_payload={"reward": 1},
        )
        await host.lease_manager.wait_closed(host.store.get_close_intent(ack["intent_id"]))
        key = SlotKey(batch_id, 0, "done")
        completed = host.store.get_slot(key)
        assert completed.state == SlotState.DONE
        failing.release.set()
        async with asyncio.timeout(5):
            while host.store.get_batch(batch_id).state != BatchState.FAILED:
                await asyncio.sleep(0.01)
        preserved = host.store.get_slot(key)
        assert preserved.state == SlotState.DONE
        assert preserved.artifact_digest == completed.artifact_digest
        assert preserved.task_group_revision == completed.task_group_revision
        assert host.store.get_manifest(batch_id).revision == 1
        assert sampler.sample_next_calls == 0
        assert host.store.list_replacement_intents()[0].status == ReplacementStatus.FAILED
    finally:
        failing.release.set()
        await host.stop()

@pytest.mark.anyio
async def test_slow_close_does_not_hold_other_lease_capacity(tmp_path):
    class SlowCloseEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.slow_lease = None

        async def close_strict(self, lease_id, **kwargs):
            if lease_id == self.slow_lease:
                self.started.set()
                await self.release.wait()
            return await super().close_strict(lease_id, **kwargs)

    env = SlowCloseEnv()
    host = make_host(tmp_path, sampler=FakeSampler(2), env=env,
                     actors={"a": FakeActor("a")}, capacity=2)
    await host.start()
    batch_id = BatchId("parallel-close", 1)
    try:
        await host.coordinator.acquire_batch(
            run_id=batch_id.run_id, train_step_id=1, tasks_per_step=2,
            request_id="acquire", config_fingerprint="cfg",
        )
        entries = []
        for index in range(2):
            result = await host.broker.pull_ready(
                batch_id, consumer="trainer", limit=1, request_id=f"pull-{index}",
                ttl_seconds=30, wait_seconds=2,
            )
            entries.extend(result["entries"])
        env.slow_lease = entries[0]["lease"]["lease_id"]
        acks = []
        for index, entry in enumerate(entries):
            acks.append(await host.ack(
                claim_id=entry["claim_token"], generation=entry["generation"],
                request_id=f"ack-{index}", judge_payload={"reward": 1},
            ))
        await asyncio.wait_for(env.started.wait(), 2)
        fast_intent = host.store.get_close_intent(acks[1]["intent_id"])
        await asyncio.wait_for(host.lease_manager.wait_closed(fast_intent), 1)
        assert not env.release.is_set()
        assert fast_intent.intent_id not in host.lease_manager._completion

        await asyncio.wait_for(host.lease_manager.wait_closed(fast_intent), 1)
        assert fast_intent.intent_id not in host.lease_manager._completion
    finally:
        env.release.set()
        await host.stop()

@pytest.mark.anyio
async def test_artifact_write_does_not_block_event_loop(tmp_path, monkeypatch):
    import threading

    host = make_host(tmp_path, sampler=FakeSampler(1), env=FakeEnv(),
                     actors={"a": FakeActor("a")}, capacity=1)
    entered, release = threading.Event(), threading.Event()
    original = host.artifacts.write_json
    loop_thread = threading.get_ident()
    worker_threads = []

    def slow_write(payload):
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(payload)

    await host.start()
    try:
        await host.coordinator.acquire_batch(
            run_id="slow-artifact", train_step_id=1, tasks_per_step=1,
            request_id="acquire", config_fingerprint="cfg",
        )
        monkeypatch.setattr(host.artifacts, "write_json", slow_write)
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        assert worker_threads[0] != loop_thread
        assert host.scheduler.status()["capacity_in_use"] == 1
    finally:
        release.set()
        await host.stop()

@pytest.mark.anyio
async def test_scheduler_release_markers_do_not_keep_completed_history():
    started, release = asyncio.Event(), asyncio.Event()
    batch = BatchId("marker-history", 1)
    with SQLiteStateStore(":memory:") as store:
        store.create_batch(batch, [ManifestSlot(0, "a", "t", "g")], request_id="create")
        slot = store.list_slots(batch)[0]
        scheduler = None

        async def runner(_slot):
            await scheduler.release_lease("early", slot.key)
            started.set()
            await release.wait()
            return RunOutcome(lease_id="early", retain_capacity=True)

        scheduler = FairScheduler(actor_ids=["a"], env_capacity=1,
                                  actor_concurrency=1, runner=runner)
        await scheduler.start()
        try:
            await scheduler.enqueue(slot)
            await asyncio.wait_for(started.wait(), 1)
            assert "early" in scheduler._released_before_register
            release.set()
            async with asyncio.timeout(1):
                while scheduler.status()["capacity_in_use"]:
                    await asyncio.sleep(0)
            for index in range(100):
                await scheduler.release_lease(f"finished-{index}", slot.key)
            await asyncio.sleep(0)
            assert not scheduler._released_before_register
            assert scheduler.status()["capacity_in_use"] == 0
        finally:
            release.set()
            await scheduler.stop()
