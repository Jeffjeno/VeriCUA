from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Mapping, Sequence, TypeVar

class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

class BatchState(_StringEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    COMPLETE = "COMPLETE"
    SEALED = "SEALED"
    COMMITTED = "COMMITTED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"
    INVALID = "INVALID"
    ABANDONED = "ABANDONED"
    IN_DOUBT = "IN_DOUBT"

class SlotState(_StringEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    READY = "READY"
    CLAIMED = "CLAIMED"
    ACKING = "ACKING"
    DONE = "DONE"
    RETRY_PENDING = "RETRY_PENDING"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"
    CANCELED = "CANCELED"

class AttemptStatus(_StringEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"

class ClaimStatus(_StringEnum):
    ACTIVE = "ACTIVE"
    ACKING = "ACKING"
    ACKED = "ACKED"
    EXPIRED = "EXPIRED"
    LOST = "LOST"
    SUPERSEDED = "SUPERSEDED"

class LeaseStatus(_StringEnum):
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    LOST = "LOST"
    EXPIRED = "EXPIRED"

class OutboxStatus(_StringEnum):
    PENDING = "PENDING"
    CLOSED = "CLOSED"
    LOST = "LOST"

class ReplacementStatus(_StringEnum):
    PENDING = "PENDING"
    CLOSING = "CLOSING"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ErrorCode(_StringEnum):
    INVALID_TRANSITION = "INVALID_TRANSITION"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    BATCH_INCOMPLETE = "BATCH_INCOMPLETE"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    FENCED = "FENCED"
    ARTIFACT_CORRUPT = "ARTIFACT_CORRUPT"
    STORAGE_ERROR = "STORAGE_ERROR"

@dataclass
class ProtocolError(Exception):

    code: ErrorCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }

T = TypeVar("T")

@dataclass(frozen=True)
class Result(Generic[T]):

    value: T | None = None
    error: ProtocolError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def failure(cls, error: ProtocolError) -> "Result[T]":
        return cls(error=error)

@dataclass(frozen=True, order=True)
class BatchId:
    run_id: str
    train_step_id: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "run_id must not be empty")
        if self.train_step_id < 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "train_step_id must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "train_step_id": self.train_step_id}

@dataclass(frozen=True, order=True)
class SlotKey:
    batch_id: BatchId
    task_index: int
    actor_id: str

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "task_index must be non-negative")
        if not self.actor_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "actor_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.batch_id.to_dict(),
            "task_index": self.task_index,
            "actor_id": self.actor_id,
        }

@dataclass(frozen=True)
class ManifestSlot:
    task_index: int
    actor_id: str
    task_id: str
    task_group_id: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "task_index must be non-negative")
        if not self.actor_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "actor_id must not be empty")
        if not self.task_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "task_id must not be empty")
        if not self.task_group_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "task_group_id must not be empty")

    def key(self, batch_id: BatchId) -> SlotKey:
        return SlotKey(batch_id, self.task_index, self.actor_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "task_group_id": self.task_group_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestSlot":
        return cls(
            task_index=int(value["task_index"]),
            actor_id=str(value["actor_id"]),
            task_id=str(value["task_id"]),
            task_group_id=str(value.get("task_group_id", "default")),
            metadata=dict(value.get("metadata") or {}),
        )

SlotSpec = ManifestSlot

@dataclass(frozen=True)
class ManifestRevision:
    batch_id: BatchId
    revision: int
    slots: tuple[ManifestSlot, ...]
    payload_hash: str
    created_at: float

@dataclass(frozen=True)
class BatchRecord:
    batch_id: BatchId
    state: BatchState
    version: int
    manifest_revision: int
    expected_slots: int
    metadata: Mapping[str, Any]
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class SlotRecord:
    key: SlotKey
    task_id: str
    task_group_id: str
    task_group_revision: int
    manifest_revision: int
    state: SlotState
    version: int
    current_attempt: int
    claim_generation: int
    artifact_digest: str | None
    metadata: Mapping[str, Any]
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class AttemptRecord:
    key: SlotKey
    attempt: int
    status: AttemptStatus
    worker_id: str | None
    started_at: float
    finished_at: float | None
    error: Mapping[str, Any] | None = None

@dataclass(frozen=True)
class LeaseRecord:

    lease_id: str
    key: SlotKey
    attempt: int
    fence_token: str
    allocation_key: str
    owner: Mapping[str, Any]
    status: LeaseStatus
    claim_id: str | None
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    key: SlotKey
    generation: int
    status: ClaimStatus
    consumer_id: str
    expires_at: float
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class ClaimResult:
    request_id: str
    claims: tuple[ClaimRecord, ...]
    replayed: bool = False

@dataclass(frozen=True)
class ArtifactRef:
    digest: str
    size: int
    uri: str
    media_type: str = "application/octet-stream"

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "size": self.size,
            "uri": self.uri,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            digest=str(value["digest"]),
            size=int(value["size"]),
            uri=str(value["uri"]),
            media_type=str(value.get("media_type", "application/octet-stream")),
        )

@dataclass(frozen=True)
class AckResult:
    intent_id: str
    claim_id: str
    key: SlotKey
    artifact: ArtifactRef
    replayed: bool = False

@dataclass(frozen=True)
class CloseIntent:
    intent_id: str
    claim_id: str | None
    key: SlotKey
    generation: int | None
    lease_id: str
    fence_token: str
    attempt: int
    operation_id: str
    action: str
    reason: str
    artifact_digest: str | None
    status: OutboxStatus
    close_attempts: int
    last_error: str | None
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class CompletenessReport:
    batch_id: BatchId
    manifest_revision: int
    expected: int
    present: int
    done: int
    missing: tuple[SlotKey, ...]
    not_done: tuple[SlotKey, ...]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.not_done and self.done == self.expected

@dataclass(frozen=True)
class RecoveryReport:
    expired_claims: int
    released_slots: int
    quarantined_leases: int = 0

@dataclass(frozen=True)
class TaskGroupReplacement:
    batch_id: BatchId
    task_group_id: str
    task_group_revision: int
    manifest_revision: int
    superseded_slots: int
    active_slots: tuple[SlotRecord, ...]

@dataclass(frozen=True)
class ReplacementIntent:
    batch_id: BatchId
    task_group_id: str
    triggering_group_revision: int
    status: ReplacementStatus
    replacement_task: Mapping[str, Any] | None
    replacement_slots: tuple[ManifestSlot, ...]
    last_error: str | None
    created_at: float
    updated_at: float

@dataclass(frozen=True)
class BatchInspection:
    batch: BatchRecord
    slots: tuple[SlotRecord, ...]
    counts: Mapping[SlotState, int]
    completeness: CompletenessReport

BATCH_TRANSITIONS: Mapping[BatchState, frozenset[BatchState]] = {
    BatchState.PLANNED: frozenset(
        {BatchState.RUNNING, BatchState.INVALID, BatchState.FAILED, BatchState.ABANDONED}
    ),
    BatchState.RUNNING: frozenset(
        {
            BatchState.DRAINING,
            BatchState.COMPLETE,
            BatchState.RECOVERING,
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
        }
    ),
    BatchState.DRAINING: frozenset(
        {
            BatchState.RUNNING,
            BatchState.COMPLETE,
            BatchState.RECOVERING,
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
        }
    ),
    BatchState.COMPLETE: frozenset(
        {BatchState.SEALED, BatchState.RECOVERING, BatchState.FAILED, BatchState.INVALID, BatchState.IN_DOUBT}
    ),
    BatchState.SEALED: frozenset({BatchState.COMMITTED, BatchState.FAILED, BatchState.IN_DOUBT}),
    BatchState.COMMITTED: frozenset(),
    BatchState.RECOVERING: frozenset(
        {
            BatchState.RUNNING,
            BatchState.DRAINING,
            BatchState.COMPLETE,
            BatchState.FAILED,
            BatchState.INVALID,
            BatchState.ABANDONED,
            BatchState.IN_DOUBT,
        }
    ),
    BatchState.FAILED: frozenset({BatchState.RECOVERING, BatchState.ABANDONED}),
    BatchState.INVALID: frozenset({BatchState.ABANDONED}),
    BatchState.ABANDONED: frozenset(),
    BatchState.IN_DOUBT: frozenset(),
}

SLOT_TRANSITIONS: Mapping[SlotState, frozenset[SlotState]] = {
    SlotState.PENDING: frozenset(
        {SlotState.RUNNING, SlotState.RETRY_PENDING, SlotState.FAILED, SlotState.SUPERSEDED, SlotState.CANCELED}
    ),
    SlotState.RUNNING: frozenset(
        {SlotState.READY, SlotState.RETRY_PENDING, SlotState.FAILED, SlotState.SUPERSEDED, SlotState.CANCELED}
    ),
    SlotState.READY: frozenset(
        {SlotState.CLAIMED, SlotState.RETRY_PENDING, SlotState.FAILED, SlotState.SUPERSEDED, SlotState.CANCELED}
    ),
    SlotState.CLAIMED: frozenset(
        {
            SlotState.ACKING,
            SlotState.READY,
            SlotState.RETRY_PENDING,
            SlotState.FAILED,
            SlotState.SUPERSEDED,
            SlotState.CANCELED,
        }
    ),
    SlotState.ACKING: frozenset(
        {SlotState.DONE, SlotState.RETRY_PENDING, SlotState.FAILED, SlotState.SUPERSEDED, SlotState.CANCELED}
    ),
    SlotState.DONE: frozenset({SlotState.SUPERSEDED}),
    SlotState.RETRY_PENDING: frozenset(
        {SlotState.PENDING, SlotState.RUNNING, SlotState.FAILED, SlotState.SUPERSEDED, SlotState.CANCELED}
    ),
    SlotState.FAILED: frozenset({SlotState.RETRY_PENDING, SlotState.SUPERSEDED, SlotState.CANCELED}),
    SlotState.SUPERSEDED: frozenset(),
    SlotState.CANCELED: frozenset(
        {SlotState.RETRY_PENDING, SlotState.SUPERSEDED}
    ),
}

def validate_batch_transition(current: BatchState, target: BatchState) -> None:
    _validate_transition("batch", current, target, BATCH_TRANSITIONS)

def validate_slot_transition(current: SlotState, target: SlotState) -> None:
    _validate_transition("slot", current, target, SLOT_TRANSITIONS)

def _validate_transition(
    kind: str,
    current: _StringEnum,
    target: _StringEnum,
    transitions: Mapping[Any, frozenset[Any]],
) -> None:
    if current == target:
        return
    if target not in transitions.get(current, frozenset()):
        raise ProtocolError(
            ErrorCode.INVALID_TRANSITION,
            f"illegal {kind} state transition {current.value} -> {target.value}",
            {"kind": kind, "current": current.value, "target": target.value},
        )

def validate_manifest(slots: Sequence[ManifestSlot]) -> tuple[ManifestSlot, ...]:
    if not slots:
        raise ProtocolError(ErrorCode.VALIDATION_ERROR, "manifest must contain at least one slot")
    result = tuple(slots)
    seen: set[tuple[int, str]] = set()
    for slot in result:
        identity = (slot.task_index, slot.actor_id)
        if identity in seen:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "manifest contains duplicate stable slot identity",
                {"task_index": slot.task_index, "actor_id": slot.actor_id},
            )
        seen.add(identity)
    return result
