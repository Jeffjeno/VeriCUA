from __future__ import annotations

import sqlite3
import sys
import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guirl.env_service.artifact_store import ArtifactStore
from guirl.env_service.batch_models import (
    BatchId,
    BatchState,
    ArtifactRef,
    ErrorCode,
    ManifestSlot,
    OutboxStatus,
    ProtocolError,
    SlotKey,
    SlotState,
    validate_batch_transition,
    validate_slot_transition,
)
from guirl.env_service.state_store import SQLiteStateStore

def _manifest() -> list[ManifestSlot]:
    return [
        ManifestSlot(0, "actor-a", "task-0", "group-a", {"seed": 10}),
        ManifestSlot(1, "actor-b", "task-1", "group-a", {"seed": 20}),
    ]

def _create(store: SQLiteStateStore, batch_id: BatchId | None = None) -> BatchId:
    identity = batch_id or BatchId("run-1", 7)
    store.create_batch(identity, _manifest(), request_id=f"create-{identity.run_id}")
    return identity

def _make_ready(store: SQLiteStateStore, key: SlotKey) -> None:
    attempt = store.begin_attempt(key)
    lease_id = (
        f"lease-{key.batch_id.run_id}-{key.batch_id.train_step_id}-"
        f"{key.task_index}-{key.actor_id}-{attempt.attempt}"
    )
    store.register_attempt_lease(
        key,
        attempt.attempt,
        lease_id=lease_id,
        fence_token=f"fence-{lease_id}",
        allocation_key=f"allocation-{lease_id}",
        owner={"test": True},
    )
    store.finish_attempt(
        key,
        attempt.attempt,
        succeeded=True,
        actor_artifact=_actor_ref(lease_id),
    )

def _actor_ref(lease_id: str) -> ArtifactRef:
    payload = f"actor:{lease_id}".encode()
    return ArtifactRef(
        digest=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        uri=f"file:///tmp/{lease_id}.json",
        media_type="application/json",
    )

def _complete_slot(
    store: SQLiteStateStore,
    key: SlotKey,
    artifact_store: ArtifactStore,
    request_suffix: str,
) -> None:
    _make_ready(store, key)
    claim = store.claim_ready_slots(
        key.batch_id,
        consumer_id="trainer",
        limit=1,
        request_id=f"claim-{request_suffix}",
        ttl_seconds=30,
        now=10,
    ).claims[0]
    artifact = artifact_store.write_json({"slot": request_suffix})
    store.record_ack_artifact(
        claim.claim_id,
        claim.generation,
        artifact,
        request_id=f"ack-{request_suffix}",
        now=11,
    )
    store.mark_lease_closed(claim.claim_id, claim.generation, now=12)

def test_legal_and_illegal_transitions(tmp_path: Path) -> None:
    validate_batch_transition(BatchState.PLANNED, BatchState.RUNNING)
    validate_slot_transition(SlotState.RUNNING, SlotState.READY)
    with pytest.raises(ProtocolError) as error:
        validate_batch_transition(BatchState.PLANNED, BatchState.COMMITTED)
    assert error.value.code == ErrorCode.INVALID_TRANSITION
    with pytest.raises(ProtocolError) as error:
        validate_slot_transition(SlotState.PENDING, SlotState.DONE)
    assert error.value.code == ErrorCode.INVALID_TRANSITION

    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        running = store.update_batch_state(batch_id, BatchState.RUNNING, expected_version=1)
        assert running.version == 2
        with pytest.raises(ProtocolError) as error:
            store.update_batch_state(batch_id, BatchState.SEALED, expected_version=2)
        assert error.value.code == ErrorCode.INVALID_TRANSITION
        with pytest.raises(ProtocolError) as error:
            store.update_batch_state(batch_id, BatchState.DRAINING, expected_version=1)
        assert error.value.code == ErrorCode.VERSION_CONFLICT

def test_idempotency_replay_and_payload_conflict(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = BatchId("idempotent", 1)
        first = store.create_batch(batch_id, _manifest(), request_id="request-1")
        replay = store.create_batch(batch_id, _manifest(), request_id="request-1")
        assert replay == first

        changed = list(_manifest())
        changed[0] = ManifestSlot(0, "actor-a", "different-task", "group-a")
        with pytest.raises(ProtocolError) as error:
            store.create_batch(batch_id, changed, request_id="request-1")
        assert error.value.code == ErrorCode.IDEMPOTENCY_CONFLICT

def test_immutable_manifest_and_task_group_revision(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    with SQLiteStateStore(database) as store:
        batch_id = _create(store)
        original = store.get_manifest(batch_id, 1)
        replacement = [
            ManifestSlot(0, "actor-a", "replacement-0", "group-a", {"revision": 2}),
            ManifestSlot(1, "actor-b", "replacement-1", "group-a", {"revision": 2}),
        ]
        result = store.replace_task_group(
            batch_id,
            "group-a",
            replacement,
            request_id="replace-1",
            expected_batch_version=1,
        )
        replay = store.replace_task_group(
            batch_id,
            "group-a",
            replacement,
            request_id="replace-1",
            expected_batch_version=1,
        )
        assert result.manifest_revision == 2
        assert result.task_group_revision == 2
        assert replay == result
        assert store.get_manifest(batch_id, 1) == original
        assert [slot.task_id for slot in store.get_manifest(batch_id).slots] == [
            "replacement-0",
            "replacement-1",
        ]
        assert {slot.task_group_revision for slot in store.list_slots(batch_id)} == {2}
        assert {slot.state for slot in store.list_slots(batch_id)} == {SlotState.PENDING}

    connection = sqlite3.connect(database)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE manifest_revisions SET manifest_json='[]' "
            "WHERE run_id='run-1' AND train_step_id=7 AND revision=1"
        )
    connection.close()

def test_attempt_and_claim_generation_fence_stale_work(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        key = SlotKey(batch_id, 0, "actor-a")
        first = store.begin_attempt(key, worker_id="worker-1")
        second = store.begin_attempt(key, worker_id="worker-2")
        with pytest.raises(ProtocolError) as error:
            store.finish_attempt(key, first.attempt, succeeded=True)
        assert error.value.code == ErrorCode.FENCED
        store.register_attempt_lease(
            key,
            second.attempt,
            lease_id="attempt-fence-lease",
            fence_token="attempt-fence",
            allocation_key="attempt-allocation",
            owner={"test": True},
        )
        store.finish_attempt(
            key,
            second.attempt,
            succeeded=True,
            actor_artifact=_actor_ref("attempt-fence-lease"),
        )

        old_claim = store.claim_ready_slots(
            batch_id,
            consumer_id="trainer",
            limit=1,
            request_id="claim-old",
            ttl_seconds=5,
            now=10,
        ).claims[0]
        store.recover_expired_claims(now=16)
        assert store.get_slot(key).state == SlotState.ACKING
        intent = store.list_close_outbox()[0]
        store.mark_close_intent_terminal(intent.intent_id, lost=False, now=16)
        assert store.get_slot(key).state == SlotState.RETRY_PENDING
        _make_ready(store, key)
        new_claim = store.claim_ready_slots(
            batch_id,
            consumer_id="trainer",
            limit=1,
            request_id="claim-new",
            ttl_seconds=5,
            now=16,
        ).claims[0]
        assert new_claim.generation == old_claim.generation + 1
        with pytest.raises(ProtocolError) as error:
            store.renew_claim(old_claim.claim_id, old_claim.generation, ttl_seconds=5, now=17)
        assert error.value.code == ErrorCode.FENCED

def test_claim_replay_conflict_and_expiry(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        _make_ready(store, SlotKey(batch_id, 0, "actor-a"))
        first = store.claim_ready_slots(
            batch_id,
            consumer_id="consumer-a",
            limit=1,
            request_id="claim-1",
            ttl_seconds=5,
            now=100,
        )
        replay = store.claim_ready_slots(
            batch_id,
            consumer_id="consumer-a",
            limit=1,
            request_id="claim-1",
            ttl_seconds=5,
            now=200,
        )
        assert replay.replayed
        assert replay.claims == first.claims
        with pytest.raises(ProtocolError) as error:
            store.claim_ready_slots(
                batch_id,
                consumer_id="consumer-a",
                limit=1,
                request_id="claim-1",
                ttl_seconds=10,
                now=101,
            )
        assert error.value.code == ErrorCode.IDEMPOTENCY_CONFLICT

        report = store.recover_expired_claims(now=106)
        assert report.expired_claims == 1
        assert report.released_slots == 0
        assert report.quarantined_leases == 1
        key = SlotKey(batch_id, 0, "actor-a")
        assert store.get_slot(key).state == SlotState.ACKING
        intent = store.list_close_outbox()[0]
        assert intent.lease_id.startswith("lease-")
        store.mark_close_intent_terminal(intent.intent_id, lost=False, now=107)
        assert store.get_slot(key).state == SlotState.RETRY_PENDING

def test_ack_artifact_and_close_outbox(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        key = SlotKey(batch_id, 0, "actor-a")
        _make_ready(store, key)
        claim = store.claim_ready_slots(
            batch_id,
            consumer_id="trainer",
            limit=1,
            request_id="claim",
            ttl_seconds=30,
            now=10,
        ).claims[0]
        artifact = ArtifactStore(tmp_path / "artifacts").write_json({"reward": 1})
        ack = store.record_ack_artifact(
            claim.claim_id,
            claim.generation,
            artifact,
            request_id="ack",
            now=11,
        )
        replay = store.record_ack_artifact(
            claim.claim_id,
            claim.generation,
            artifact,
            request_id="ack",
            now=12,
        )
        assert replay.replayed
        assert replay.intent_id == ack.intent_id
        pending = store.list_close_outbox()
        assert len(pending) == 1
        assert pending[0].artifact_digest == artifact.digest
        assert pending[0].lease_id == store.get_claim_lease(claim.claim_id).lease_id
        assert pending[0].fence_token.startswith("fence-lease-")
        assert pending[0].attempt == 1
        assert store.get_slot(key).state == SlotState.ACKING

        done = store.mark_lease_closed(claim.claim_id, claim.generation, now=13)
        assert done.state == SlotState.DONE
        assert store.list_close_outbox() == ()
        assert store.list_close_outbox(status=OutboxStatus.CLOSED)[0].intent_id == ack.intent_id

def test_expected_slot_completeness(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        artifacts = ArtifactStore(tmp_path / "artifacts")
        initial = store.validate_expected_slot_completeness(batch_id)
        assert not initial.complete
        assert initial.expected == 2
        assert len(initial.not_done) == 2

        _complete_slot(store, SlotKey(batch_id, 0, "actor-a"), artifacts, "zero")
        partial = store.validate_expected_slot_completeness(batch_id)
        assert not partial.complete
        assert partial.done == 1

        _complete_slot(store, SlotKey(batch_id, 1, "actor-b"), artifacts, "one")
        complete = store.validate_expected_slot_completeness(batch_id)
        assert complete.complete
        assert complete.done == complete.expected == complete.present == 2

def test_wal_reopen_and_expired_claim_recovery(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    batch_id = BatchId("reopen", 3)
    key = SlotKey(batch_id, 0, "actor-a")
    with SQLiteStateStore(database) as store:
        _create(store, batch_id)
        _make_ready(store, key)
        store.claim_ready_slots(
            batch_id,
            consumer_id="consumer",
            limit=1,
            request_id="claim",
            ttl_seconds=5,
            now=1,
        )
        assert store.journal_mode == "wal"

    with SQLiteStateStore(database) as reopened:
        assert reopened.get_slot(key).state == SlotState.CLAIMED
        report = reopened.recover_expired_claims(now=7)
        assert report.quarantined_leases == 1
        assert reopened.get_slot(key).state == SlotState.ACKING

def test_artifact_digest_json_delete_and_corruption(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.write_bytes(b"payload")
    replay = store.write_bytes(b"payload")
    assert replay == first
    assert first.size == 7
    assert store.verify(first)
    assert store.read(first) == b"payload"

    json_ref = store.write_json({"b": 2, "a": 1})
    assert store.read_json(json_ref) == {"a": 1, "b": 2}
    assert store.delete(json_ref)
    assert not store.delete(json_ref)

    artifact_path = Path(unquote(urlparse(first.uri).path))
    artifact_path.write_bytes(b"corrupt")
    assert not store.verify(first)
    with pytest.raises(ProtocolError) as error:
        store.read(first)
    assert error.value.code == ErrorCode.ARTIFACT_CORRUPT

def test_replacement_cannot_shrink_stable_group_membership(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        with pytest.raises(
            ProtocolError,
            match="preserve exact slot membership",
        ):
            store.replace_task_group(
                batch_id,
                "group-a",
                [
                    ManifestSlot(
                        0,
                        "actor-a",
                        "replacement",
                        "group-a",
                    )
                ],
                request_id="replace-shrunk",
                expected_batch_version=1,
            )
        assert store.get_manifest(batch_id).revision == 1
        assert store.get_batch(batch_id).expected_slots == 2

def test_nack_renew_and_recover_lost_response_replay(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        key = SlotKey(batch_id, 0, "actor-a")
        _make_ready(store, key)
        claim = store.claim_ready_slots(
            batch_id,
            consumer_id="trainer",
            limit=1,
            request_id="claim-idempotent-mutations",
            ttl_seconds=30,
            now=10,
        ).claims[0]

        renewed = store.renew_claim(
            claim.claim_id,
            claim.generation,
            ttl_seconds=20,
            consumer_id="trainer",
            request_id="renew-lost-response",
            now=11,
        )
        renewed_replay = store.renew_claim(
            claim.claim_id,
            claim.generation,
            ttl_seconds=20,
            consumer_id="trainer",
            request_id="renew-lost-response",
            now=25,
        )
        assert renewed_replay == renewed
        with pytest.raises(ProtocolError) as renew_conflict:
            store.renew_claim(
                claim.claim_id,
                claim.generation,
                ttl_seconds=21,
                consumer_id="trainer",
                request_id="renew-lost-response",
                now=12,
            )
        assert renew_conflict.value.code == (
            ErrorCode.IDEMPOTENCY_CONFLICT
        )

        nack = store.nack_claim(
            claim.claim_id,
            claim.generation,
            reason="trainer-retry",
            request_id="nack-lost-response",
            now=12,
        )
        nack_replay = store.nack_claim(
            claim.claim_id,
            claim.generation,
            reason="trainer-retry",
            request_id="nack-lost-response",
            now=13,
        )
        assert nack_replay == nack
        with pytest.raises(ProtocolError) as nack_conflict:
            store.nack_claim(
                claim.claim_id,
                claim.generation,
                reason="different-reason",
                request_id="nack-lost-response",
                now=13,
            )
        assert nack_conflict.value.code == (
            ErrorCode.IDEMPOTENCY_CONFLICT
        )

        recover_batch = BatchId("recover-idempotent", 2)
        store.create_batch(
            recover_batch,
            [ManifestSlot(0, "actor-a", "task", "recover-group")],
            request_id="create-recover-idempotent",
            initial_state=BatchState.RUNNING,
        )
        store.fail_batch_and_drain(
            recover_batch,
            reason="test-failure",
        )
        recovered = store.recover_failed_batch(
            recover_batch,
            request_id="recover-lost-response",
        )
        recovered_replay = store.recover_failed_batch(
            recover_batch,
            request_id="recover-lost-response",
        )
        assert recovered_replay == recovered
        assert recovered.state == BatchState.RUNNING

@pytest.mark.parametrize("acknowledged_state", [SlotState.ACKING, SlotState.DONE])
def test_replacement_preserves_acknowledged_sibling_atomically(tmp_path, acknowledged_state):
    with SQLiteStateStore(tmp_path / "state.db") as store:
        batch_id = _create(store)
        key = SlotKey(batch_id, 0, "actor-a")
        _make_ready(store, key)
        claim = store.claim_ready_slots(
            batch_id, consumer_id="trainer", limit=1, request_id="claim", ttl_seconds=30, now=10,
        ).claims[0]
        artifact = ArtifactStore(tmp_path / "artifacts").write_json({"reward": 1})
        store.record_ack_artifact(claim.claim_id, claim.generation, artifact, request_id="ack", now=11)
        if acknowledged_state == SlotState.DONE:
            store.mark_lease_closed(claim.claim_id, claim.generation, now=12)
        before = store.list_slots(batch_id)
        events = store.list_state_events(batch_id)
        manifest = store.get_manifest(batch_id)
        with pytest.raises(ProtocolError, match="acknowledged slots") as error:
            store.replace_task_group(
                batch_id, "group-a", _manifest(), request_id="replace",
                expected_batch_version=store.get_batch(batch_id).version,
            )
        assert error.value.code == ErrorCode.INVALID_TRANSITION
        assert store.list_slots(batch_id) == before
        assert store.get_manifest(batch_id) == manifest
        assert store.list_state_events(batch_id) == events
