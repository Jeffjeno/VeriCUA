from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, TypeVar

from .batch_models import (
    AckResult,
    ArtifactRef,
    AttemptRecord,
    AttemptStatus,
    BatchId,
    BatchInspection,
    BatchRecord,
    BatchState,
    ClaimRecord,
    ClaimResult,
    ClaimStatus,
    CloseIntent,
    CompletenessReport,
    ErrorCode,
    LeaseRecord,
    LeaseStatus,
    ManifestRevision,
    ManifestSlot,
    OutboxStatus,
    ProtocolError,
    RecoveryReport,
    ReplacementIntent,
    ReplacementStatus,
    SlotKey,
    SlotRecord,
    SlotState,
    TaskGroupReplacement,
    validate_batch_transition,
    validate_manifest,
    validate_slot_transition,
)

_EnumT = TypeVar("_EnumT")

class SQLiteStateStore:

    SCHEMA_VERSION = 3
    DEFAULT_MAX_INLINE_JSON_BYTES = 256 * 1024
    _SYNCHRONOUS_VALUES = {"OFF", "NORMAL", "FULL", "EXTRA"}

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        synchronous: str = "FULL",
        busy_timeout_ms: int = 5_000,
        max_inline_json_bytes: int = DEFAULT_MAX_INLINE_JSON_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        synchronous_value = synchronous.upper()
        if synchronous_value not in self._SYNCHRONOUS_VALUES:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "unsupported SQLite synchronous setting",
                {"synchronous": synchronous},
            )
        if busy_timeout_ms < 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "busy_timeout_ms must be non-negative")
        if max_inline_json_bytes <= 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "max_inline_json_bytes must be positive")

        self.path = str(path)
        self._clock = clock
        self._max_inline_json_bytes = max_inline_json_bytes
        self._synchronous = synchronous_value
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._busy_errors = 0
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            self._connection.execute(f"PRAGMA synchronous = {synchronous_value}")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteStateStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def journal_mode(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
            return str(row[0]).lower()

    def observability_snapshot(
        self,
        *,
        now: float | None = None,
        recent_attempt_limit: int = 2_048,
        recent_batch_limit: int = 32,
    ) -> dict[str, Any]:

        timestamp = float(self._clock() if now is None else now)
        attempt_limit = max(1, min(int(recent_attempt_limit), 10_000))
        batch_limit = max(1, min(int(recent_batch_limit), 100))
        active_batch_states = (
            BatchState.PLANNED.value,
            BatchState.RUNNING.value,
            BatchState.DRAINING.value,
            BatchState.COMPLETE.value,
            BatchState.SEALED.value,
            BatchState.RECOVERING.value,
            BatchState.IN_DOUBT.value,
        )
        with self._lock:
            focus = self._connection.execute(
                f"""
                SELECT run_id, train_step_id FROM batches
                ORDER BY CASE WHEN state IN ({','.join('?' for _ in active_batch_states)})
                              THEN 0 ELSE 1 END,
                         updated_at DESC
                LIMIT 1
                """,
                active_batch_states,
            ).fetchone()
            actor_rows: list[sqlite3.Row] = []
            if focus is not None:
                actor_rows = self._connection.execute(
                    """
                    SELECT actor_id, state, COUNT(*) AS count
                    FROM slots
                    WHERE run_id=? AND train_step_id=?
                    GROUP BY actor_id, state
                    ORDER BY actor_id, state
                    """,
                    (focus["run_id"], focus["train_step_id"]),
                ).fetchall()
            batch_rows = self._connection.execute(
                """
                WITH recent AS (
                    SELECT * FROM batches ORDER BY updated_at DESC LIMIT ?
                )
                SELECT r.*, s.state AS slot_state, COUNT(s.actor_id) AS slot_count
                FROM recent r
                LEFT JOIN slots s
                  ON s.run_id=r.run_id AND s.train_step_id=r.train_step_id
                GROUP BY r.run_id, r.train_step_id, s.state
                ORDER BY r.updated_at DESC, r.run_id, r.train_step_id
                """,
                (batch_limit,),
            ).fetchall()
            attempt_rows = self._connection.execute(
                """
                SELECT a.actor_id, a.status, a.started_at, a.finished_at,
                       COALESCE(t.queue_wait_seconds,
                                MAX(0.0, a.started_at - s.created_at)) AS queue_wait,
                       COALESCE(t.actor_seconds,
                                CASE WHEN a.finished_at IS NOT NULL
                                     THEN MAX(0.0, a.finished_at-a.started_at)
                                     ELSE NULL END) AS actor_seconds
                FROM attempts a
                JOIN slots s
                  ON s.run_id=a.run_id AND s.train_step_id=a.train_step_id
                 AND s.task_index=a.task_index AND s.actor_id=a.actor_id
                LEFT JOIN attempt_timings t
                  ON t.run_id=a.run_id AND t.train_step_id=a.train_step_id
                 AND t.task_index=a.task_index AND t.actor_id=a.actor_id
                 AND t.attempt=a.attempt
                ORDER BY a.started_at DESC
                LIMIT ?
                """,
                (attempt_limit,),
            ).fetchall()
            claim_rows = self._connection.execute(
                """
                SELECT created_at FROM claims
                WHERE status IN (?, ?)
                ORDER BY created_at DESC LIMIT ?
                """,
                (
                    ClaimStatus.ACTIVE.value,
                    ClaimStatus.ACKING.value,
                    attempt_limit,
                ),
            ).fetchall()
            lease_rows = self._connection.execute(
                """
                SELECT created_at FROM actual_leases
                WHERE status IN (?, ?)
                ORDER BY created_at DESC LIMIT ?
                """,
                (
                    LeaseStatus.ACTIVE.value,
                    LeaseStatus.CLOSING.value,
                    attempt_limit,
                ),
            ).fetchall()
            outbox_rows = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count, COALESCE(SUM(close_attempts), 0) AS attempts
                FROM lease_close_outbox_v2 GROUP BY status
                """
            ).fetchall()
            recovery_row = self._connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM batches WHERE state=?) AS batches,
                  (SELECT COUNT(*) FROM slots WHERE state=?) AS retry_slots
                """,
                (BatchState.RECOVERING.value, SlotState.RETRY_PENDING.value),
            ).fetchone()

        per_actor: dict[str, dict[str, int]] = {}
        state_keys = {
            SlotState.PENDING.value: "pending",
            SlotState.RUNNING.value: "running",
            SlotState.READY.value: "ready",
            SlotState.CLAIMED.value: "claimed",
            SlotState.ACKING.value: "claimed",
            SlotState.DONE.value: "done",
            SlotState.RETRY_PENDING.value: "retry",
            SlotState.FAILED.value: "failed",
        }
        for row in actor_rows:
            values = per_actor.setdefault(
                str(row["actor_id"]),
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
            key = state_keys.get(str(row["state"]))
            if key is not None:
                values[key] += int(row["count"])

        batch_map: dict[tuple[str, int], dict[str, Any]] = {}
        for row in batch_rows:
            identity = (str(row["run_id"]), int(row["train_step_id"]))
            batch = batch_map.setdefault(
                identity,
                {
                    "batch_id": {
                        "run_id": identity[0],
                        "train_step_id": identity[1],
                    },
                    "state": str(row["state"]),
                    "version": int(row["version"]),
                    "manifest_revision": int(row["current_manifest_revision"]),
                    "expected_slots": int(row["expected_slots"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "slot_counts": {},
                },
            )
            if row["slot_state"] is not None:
                batch["slot_counts"][str(row["slot_state"])] = int(row["slot_count"])
        batches = list(batch_map.values())
        for batch in batches:
            done = int(batch["slot_counts"].get(SlotState.DONE.value, 0))
            expected = max(0, int(batch["expected_slots"]))
            batch["step_completion"] = {
                "done": done,
                "expected": expected,
                "fraction": (done / expected) if expected else 0.0,
            }

        queue_waits = [float(row["queue_wait"]) for row in attempt_rows]
        actor_seconds = [
            float(row["actor_seconds"])
            for row in attempt_rows
            if row["actor_seconds"] is not None
        ]
        per_actor_timings: dict[str, dict[str, list[float]]] = {}
        for row in attempt_rows:
            values = per_actor_timings.setdefault(
                str(row["actor_id"]),
                {"queue_wait": [], "actor_execution": []},
            )
            values["queue_wait"].append(float(row["queue_wait"]))
            if row["actor_seconds"] is not None:
                values["actor_execution"].append(float(row["actor_seconds"]))
        timing_by_actor = {
            actor_id: {
                name: self._distribution(values)
                for name, values in timings.items()
            }
            for actor_id, timings in per_actor_timings.items()
        }
        succeeded = sum(
            1 for row in attempt_rows if row["status"] == AttemptStatus.SUCCEEDED.value
        )
        observation_window = (
            max(1.0, timestamp - min(float(row["started_at"]) for row in attempt_rows))
            if attempt_rows
            else 0.0
        )
        claim_ages = [max(0.0, timestamp - float(row["created_at"])) for row in claim_rows]
        lease_ages = [max(0.0, timestamp - float(row["created_at"])) for row in lease_rows]
        actor_throughput: dict[str, dict[str, float | int]] = {}
        for actor_id in {str(row["actor_id"]) for row in attempt_rows}:
            actor_rows = [
                row for row in attempt_rows if str(row["actor_id"]) == actor_id
            ]
            actor_succeeded = sum(
                1
                for row in actor_rows
                if row["status"] == AttemptStatus.SUCCEEDED.value
            )
            actor_window = max(
                1.0,
                timestamp
                - min(float(row["started_at"]) for row in actor_rows),
            )
            actor_throughput[actor_id] = {
                "successful_slots": actor_succeeded,
                "observation_window_seconds": actor_window,
                "slots_per_second": actor_succeeded / actor_window,
            }
        outbox = {
            str(row["status"]).lower(): int(row["count"]) for row in outbox_rows
        }
        outbox["close_attempts"] = sum(int(row["attempts"]) for row in outbox_rows)
        return {
            "focus_batch": (
                {
                    "run_id": str(focus["run_id"]),
                    "train_step_id": int(focus["train_step_id"]),
                }
                if focus is not None
                else None
            ),
            "per_actor": per_actor,
            "batches": batches,
            "timings_seconds": {
                "queue_wait": self._distribution(queue_waits),
                "actor_execution": self._distribution(actor_seconds),
                "by_actor": timing_by_actor,
            },
            "throughput": {
                "successful_slots": succeeded,
                "observation_window_seconds": observation_window,
                "slots_per_second": (
                    succeeded / observation_window if observation_window else 0.0
                ),
                "by_actor": actor_throughput,
            },
            "claim_age_seconds": self._distribution(claim_ages),
            "lease_age_seconds": self._distribution(lease_ages),
            "close_outbox": outbox,
            "recovery_depth": {
                "batches": int(recovery_row["batches"]) if recovery_row else 0,
                "retry_slots": int(recovery_row["retry_slots"]) if recovery_row else 0,
                "pending_close_intents": int(outbox.get("pending", 0)),
            },
        }

    def storage_status(self) -> dict[str, Any]:

        wal_bytes = 0
        if self.path != ":memory:":
            try:
                wal_bytes = os.path.getsize(f"{self.path}-wal")
            except OSError:
                wal_bytes = 0
        checkpoint = {"busy": 0, "log_frames": 0, "checkpointed_frames": 0}
        with self._lock:
            try:
                row = self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if row is not None and len(row) >= 3:
                    checkpoint = {
                        "busy": int(row[0]),
                        "log_frames": int(row[1]),
                        "checkpointed_frames": int(row[2]),
                    }
            except sqlite3.OperationalError as exc:
                if "busy" in str(exc).lower() or "locked" in str(exc).lower():
                    self._busy_errors += 1
                checkpoint["error"] = str(exc)
        return {
            "journal_mode": self.journal_mode,
            "synchronous": self._synchronous,
            "busy_timeout_ms": self._busy_timeout_ms,
            "busy_errors": self._busy_errors,
            "wal_bytes": wal_bytes,
            "checkpoint": checkpoint,
        }

    @staticmethod
    def _distribution(values: Sequence[float]) -> dict[str, float | int]:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}

        def percentile(fraction: float) -> float:
            index = int(round((len(ordered) - 1) * fraction))
            return ordered[max(0, min(index, len(ordered) - 1))]

        return {
            "count": len(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": ordered[-1],
        }

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                self._connection.commit()
            except BaseException as exc:
                if isinstance(exc, sqlite3.OperationalError) and (
                    "busy" in str(exc).lower() or "locked" in str(exc).lower()
                ):
                    self._busy_errors += 1
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                version INTEGER NOT NULL
            );
            INSERT INTO schema_info(version)
            SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);

            CREATE TABLE IF NOT EXISTS batches (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL,
                current_manifest_revision INTEGER NOT NULL,
                expected_slots INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, train_step_id)
            );

            CREATE TABLE IF NOT EXISTS manifest_revisions (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                payload_hash TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, train_step_id, revision),
                FOREIGN KEY (run_id, train_step_id)
                    REFERENCES batches(run_id, train_step_id) ON DELETE CASCADE
            );

            CREATE TRIGGER IF NOT EXISTS manifest_revisions_no_update
            BEFORE UPDATE ON manifest_revisions
            BEGIN
                SELECT RAISE(ABORT, 'manifest revisions are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS manifest_revisions_no_delete
            BEFORE DELETE ON manifest_revisions
            BEGIN
                SELECT RAISE(ABORT, 'manifest revisions are immutable');
            END;

            CREATE TABLE IF NOT EXISTS task_groups (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_group_id TEXT NOT NULL,
                group_revision INTEGER NOT NULL,
                manifest_revision INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                superseded_at REAL,
                replaced_by_revision INTEGER,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, train_step_id, task_group_id, group_revision),
                FOREIGN KEY (run_id, train_step_id, manifest_revision)
                    REFERENCES manifest_revisions(run_id, train_step_id, revision)
            );

            CREATE TABLE IF NOT EXISTS slots (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_group_id TEXT NOT NULL,
                task_group_revision INTEGER NOT NULL,
                manifest_revision INTEGER NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL,
                current_attempt INTEGER NOT NULL DEFAULT 0,
                claim_generation INTEGER NOT NULL DEFAULT 0,
                artifact_digest TEXT,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, train_step_id, task_index, actor_id),
                FOREIGN KEY (run_id, train_step_id)
                    REFERENCES batches(run_id, train_step_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS slots_ready_idx
                ON slots(run_id, train_step_id, manifest_revision, state, task_index, actor_id);

            CREATE TABLE IF NOT EXISTS attempts (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                started_at REAL NOT NULL,
                finished_at REAL,
                error_json TEXT,
                PRIMARY KEY (run_id, train_step_id, task_index, actor_id, attempt),
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id)
                    REFERENCES slots(run_id, train_step_id, task_index, actor_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS attempts_started_idx
                ON attempts(started_at DESC);

            CREATE TABLE IF NOT EXISTS attempt_timings (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                queue_wait_seconds REAL NOT NULL,
                actor_seconds REAL NOT NULL,
                total_seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, train_step_id, task_index, actor_id, attempt),
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id, attempt)
                    REFERENCES attempts(run_id, train_step_id, task_index, actor_id, attempt)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                request_id TEXT NOT NULL,
                status TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (run_id, train_step_id, task_index, actor_id, generation),
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id)
                    REFERENCES slots(run_id, train_step_id, task_index, actor_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS claims_expiry_idx ON claims(status, expires_at);

            CREATE TABLE IF NOT EXISTS idempotency_records (
                namespace TEXT NOT NULL,
                request_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (namespace, request_id)
            );

            CREATE TABLE IF NOT EXISTS leases (
                lease_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_refs (
                digest TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                uri TEXT NOT NULL,
                media_type TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS close_outbox (
                intent_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                artifact_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id),
                FOREIGN KEY (artifact_digest) REFERENCES artifact_refs(digest)
            );
            CREATE INDEX IF NOT EXISTS close_outbox_status_idx
                ON close_outbox(status, created_at);

            -- V2 leases are registered when the actor attempt allocates a real
            -- environment.  They are never synthesized by a trainer claim.
            CREATE TABLE IF NOT EXISTS actual_leases (
                lease_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                fence_token TEXT NOT NULL,
                allocation_key TEXT NOT NULL,
                owner_json TEXT NOT NULL,
                status TEXT NOT NULL,
                claim_id TEXT UNIQUE,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (run_id, train_step_id, task_index, actor_id, attempt),
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id, attempt)
                    REFERENCES attempts(run_id, train_step_id, task_index, actor_id, attempt)
                    ON DELETE CASCADE,
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
            );
            CREATE INDEX IF NOT EXISTS actual_leases_slot_idx
                ON actual_leases(run_id, train_step_id, task_index, actor_id, attempt);
            CREATE INDEX IF NOT EXISTS actual_leases_status_idx
                ON actual_leases(status, created_at);

            -- Allocation is an RPC with an unknown-outcome window before the
            -- returned lease can be registered.  Keep that window durable so
            -- retries, batch failure and Host restart retain capacity until
            -- the Env Master has authoritatively cancelled the allocation key.
            CREATE TABLE IF NOT EXISTS allocation_cleanup_outbox_v2 (
                allocation_key TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                fence_token TEXT NOT NULL,
                owner_json TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                cleanup_attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id, attempt)
                    REFERENCES attempts(run_id, train_step_id, task_index, actor_id, attempt)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS allocation_cleanup_outbox_v2_status_idx
                ON allocation_cleanup_outbox_v2(status, created_at);

            CREATE TABLE IF NOT EXISTS slot_artifacts (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                role TEXT NOT NULL,
                digest TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (
                    run_id, train_step_id, task_index, actor_id, attempt, role
                ),
                FOREIGN KEY (run_id, train_step_id, task_index, actor_id)
                    REFERENCES slots(run_id, train_step_id, task_index, actor_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (digest) REFERENCES artifact_refs(digest)
            );

            CREATE TABLE IF NOT EXISTS lease_close_outbox_v2 (
                intent_id TEXT PRIMARY KEY,
                lease_id TEXT NOT NULL UNIQUE,
                claim_id TEXT,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                generation INTEGER,
                fence_token TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                artifact_digest TEXT,
                status TEXT NOT NULL,
                close_attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (lease_id) REFERENCES actual_leases(lease_id),
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id),
                FOREIGN KEY (artifact_digest) REFERENCES artifact_refs(digest)
            );
            CREATE INDEX IF NOT EXISTS lease_close_outbox_v2_status_idx
                ON lease_close_outbox_v2(status, created_at);

            CREATE TABLE IF NOT EXISTS replacement_intents (
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_group_id TEXT NOT NULL,
                triggering_group_revision INTEGER NOT NULL,
                status TEXT NOT NULL,
                replacement_task_json TEXT,
                replacement_slots_json TEXT NOT NULL DEFAULT '[]',
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (
                    run_id, train_step_id, task_group_id,
                    triggering_group_revision
                ),
                FOREIGN KEY (run_id, train_step_id)
                    REFERENCES batches(run_id, train_step_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS replacement_intents_status_idx
                ON replacement_intents(status, updated_at);

            CREATE TABLE IF NOT EXISTS state_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                train_step_id INTEGER NOT NULL,
                task_index INTEGER,
                actor_id TEXT,
                entity_type TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                entity_version INTEGER NOT NULL,
                reason TEXT,
                details_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS state_events_batch_idx
                ON state_events(run_id, train_step_id, event_id);
            """
        )
        self._connection.execute(
            "UPDATE schema_info SET version=?",
            (self.SCHEMA_VERSION,),
        )

    def create_batch(
        self,
        batch_id: BatchId,
        manifest: Sequence[ManifestSlot],
        *,
        request_id: str,
        metadata: Mapping[str, Any] | None = None,
        initial_state: BatchState = BatchState.PLANNED,
    ) -> BatchRecord:
        if not request_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "request_id must not be empty")
        slots = validate_manifest(manifest)
        state = self._coerce_enum(BatchState, initial_state, "batch state")
        manifest_payload = [slot.to_dict() for slot in slots]
        manifest_json = self._json(manifest_payload, "manifest")
        manifest_hash = self._hash_text(manifest_json)
        metadata_json = self._json(dict(metadata or {}), "batch metadata")
        request_payload = {
            "batch_id": batch_id.to_dict(),
            "manifest_hash": manifest_hash,
            "metadata": json.loads(metadata_json),
            "initial_state": state.value,
        }
        payload_hash = self._payload_hash(request_payload)
        now = float(self._clock())

        with self._transaction() as cursor:
            replay = self._idempotency_lookup(cursor, "create_batch", request_id, payload_hash)
            if replay is not None:
                return self._batch_from_dict(replay["batch"])

            existing = cursor.execute(
                """
                SELECT b.*, m.payload_hash AS manifest_hash
                FROM batches b
                JOIN manifest_revisions m
                  ON m.run_id=b.run_id AND m.train_step_id=b.train_step_id
                 AND m.revision=1
                WHERE b.run_id=? AND b.train_step_id=?
                """,
                (batch_id.run_id, batch_id.train_step_id),
            ).fetchone()
            if existing is not None:
                if existing["manifest_hash"] != manifest_hash or existing["metadata_json"] != metadata_json:
                    raise ProtocolError(
                        ErrorCode.ALREADY_EXISTS,
                        "batch identity already exists with different immutable input",
                        batch_id.to_dict(),
                    )
                record = self._batch_from_row(existing)
                self._idempotency_store(
                    cursor,
                    "create_batch",
                    request_id,
                    payload_hash,
                    {"batch": self._batch_to_dict(record)},
                    now,
                )
                return record

            cursor.execute(
                """
                INSERT INTO batches(
                    run_id, train_step_id, state, version,
                    current_manifest_revision, expected_slots, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, 1, ?, ?, ?, ?)
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    state.value,
                    len(slots),
                    metadata_json,
                    now,
                    now,
                ),
            )
            cursor.execute(
                """
                INSERT INTO manifest_revisions(
                    run_id, train_step_id, revision, payload_hash, manifest_json, created_at
                ) VALUES (?, ?, 1, ?, ?, ?)
                """,
                (batch_id.run_id, batch_id.train_step_id, manifest_hash, manifest_json, now),
            )
            groups: dict[str, list[ManifestSlot]] = {}
            for slot in slots:
                groups.setdefault(slot.task_group_id, []).append(slot)
                cursor.execute(
                    """
                    INSERT INTO slots(
                        run_id, train_step_id, task_index, actor_id, task_id,
                        task_group_id, task_group_revision, manifest_revision,
                        state, version, current_attempt, claim_generation,
                        artifact_digest, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, 1, 0, 0, NULL, ?, ?, ?)
                    """,
                    (
                        batch_id.run_id,
                        batch_id.train_step_id,
                        slot.task_index,
                        slot.actor_id,
                        slot.task_id,
                        slot.task_group_id,
                        SlotState.PENDING.value,
                        self._json(dict(slot.metadata), "slot metadata"),
                        now,
                        now,
                    ),
                )
                self._event(
                    cursor,
                    batch_id,
                    "slot",
                    SlotState.PENDING.value,
                    1,
                    now,
                    key=slot.key(batch_id),
                    reason="manifest-created",
                )
            for group_id, group_slots in groups.items():
                self._insert_task_group(cursor, batch_id, group_id, 1, 1, group_slots, now)
            self._event(
                cursor,
                batch_id,
                "batch",
                state.value,
                1,
                now,
                reason="manifest-created",
            )
            row = self._require_batch_row(cursor, batch_id)
            record = self._batch_from_row(row)
            self._idempotency_store(
                cursor,
                "create_batch",
                request_id,
                payload_hash,
                {"batch": self._batch_to_dict(record)},
                now,
            )
            return record

    def get_batch(self, batch_id: BatchId) -> BatchRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM batches WHERE run_id=? AND train_step_id=?",
                (batch_id.run_id, batch_id.train_step_id),
            ).fetchone()
        if row is None:
            raise self._not_found("batch", batch_id.to_dict())
        return self._batch_from_row(row)

    def list_batches(
        self,
        *,
        states: Sequence[BatchState] | None = None,
    ) -> tuple[BatchRecord, ...]:
        query = "SELECT * FROM batches"
        parameters: list[Any] = []
        if states:
            values = [
                self._coerce_enum(BatchState, state, "batch state").value
                for state in states
            ]
            query += f" WHERE state IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query += " ORDER BY created_at, run_id, train_step_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._batch_from_row(row) for row in rows)

    def get_manifest(self, batch_id: BatchId, revision: int | None = None) -> ManifestRevision:
        with self._lock:
            if revision is None:
                batch_row = self._connection.execute(
                    "SELECT current_manifest_revision FROM batches WHERE run_id=? AND train_step_id=?",
                    (batch_id.run_id, batch_id.train_step_id),
                ).fetchone()
                if batch_row is None:
                    raise self._not_found("batch", batch_id.to_dict())
                revision = int(batch_row["current_manifest_revision"])
            row = self._connection.execute(
                """
                SELECT * FROM manifest_revisions
                WHERE run_id=? AND train_step_id=? AND revision=?
                """,
                (batch_id.run_id, batch_id.train_step_id, revision),
            ).fetchone()
        if row is None:
            raise self._not_found("manifest revision", {**batch_id.to_dict(), "revision": revision})
        slots = tuple(ManifestSlot.from_dict(value) for value in json.loads(row["manifest_json"]))
        return ManifestRevision(
            batch_id=batch_id,
            revision=int(row["revision"]),
            slots=slots,
            payload_hash=str(row["payload_hash"]),
            created_at=float(row["created_at"]),
        )

    def update_batch_state(
        self,
        batch_id: BatchId,
        target: BatchState,
        *,
        expected_version: int,
        reason: str | None = None,
    ) -> BatchRecord:
        target_state = self._coerce_enum(BatchState, target, "batch state")
        now = float(self._clock())
        with self._transaction() as cursor:
            row = self._require_batch_row(cursor, batch_id)
            current = BatchState(row["state"])
            self._check_version("batch", expected_version, int(row["version"]), batch_id.to_dict())
            validate_batch_transition(current, target_state)
            if current == target_state:
                return self._batch_from_row(row)
            version = int(row["version"]) + 1
            cursor.execute(
                """
                UPDATE batches SET state=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND version=?
                """,
                (
                    target_state.value,
                    version,
                    now,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise self._version_conflict("batch", expected_version, batch_id.to_dict())
            self._event(
                cursor,
                batch_id,
                "batch",
                target_state.value,
                version,
                now,
                from_state=current.value,
                reason=reason,
            )
            return self._batch_from_row(self._require_batch_row(cursor, batch_id))

    def fail_batch_and_drain(
        self,
        batch_id: BatchId,
        *,
        reason: str,
    ) -> tuple[BatchRecord, tuple[CloseIntent, ...]]:

        timestamp = float(self._clock())
        with self._transaction() as cursor:
            batch_row = self._require_batch_row(cursor, batch_id)
            current_batch = BatchState(batch_row["state"])
            if current_batch == BatchState.COMMITTED:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "a committed batch cannot be failed",
                    {"state": current_batch.value, **batch_id.to_dict()},
                )
            if current_batch != BatchState.FAILED:
                validate_batch_transition(current_batch, BatchState.FAILED)
                batch_version = int(batch_row["version"]) + 1
                cursor.execute(
                    """
                    UPDATE batches SET state=?, version=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND version=?
                    """,
                    (
                        BatchState.FAILED.value,
                        batch_version,
                        timestamp,
                        batch_id.run_id,
                        batch_id.train_step_id,
                        int(batch_row["version"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise self._version_conflict(
                        "batch", int(batch_row["version"]), batch_id.to_dict()
                    )
                self._event(
                    cursor,
                    batch_id,
                    "batch",
                    BatchState.FAILED.value,
                    batch_version,
                    timestamp,
                    from_state=current_batch.value,
                    reason=reason,
                )

            slot_rows = cursor.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=?
                  AND manifest_revision=(
                    SELECT current_manifest_revision FROM batches
                    WHERE run_id=? AND train_step_id=?
                  )
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    batch_id.run_id,
                    batch_id.train_step_id,
                ),
            ).fetchall()
            cancellable = {
                SlotState.PENDING,
                SlotState.RUNNING,
                SlotState.READY,
                SlotState.CLAIMED,
                SlotState.ACKING,
                SlotState.RETRY_PENDING,
            }
            for slot_row in slot_rows:
                key = self._key_from_row(slot_row)
                current = SlotState(slot_row["state"])
                cursor.execute(
                    """
                    UPDATE attempts SET status=?, finished_at=?, error_json=?
                    WHERE run_id=? AND train_step_id=? AND task_index=?
                      AND actor_id=? AND status=?
                    """,
                    (
                        AttemptStatus.FAILED.value,
                        timestamp,
                        self._json(
                            {"type": "BatchFailed", "message": reason},
                            "attempt error",
                        ),
                        *self._key_values(key),
                        AttemptStatus.RUNNING.value,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE claims SET status=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=?
                      AND actor_id=? AND status IN (?, ?)
                    """,
                    (
                        ClaimStatus.LOST.value,
                        timestamp,
                        *self._key_values(key),
                        ClaimStatus.ACTIVE.value,
                        ClaimStatus.ACKING.value,
                    ),
                )
                if current not in cancellable:
                    continue
                validate_slot_transition(current, SlotState.CANCELED)
                version = int(slot_row["version"]) + 1
                cursor.execute(
                    """
                    UPDATE slots
                    SET state=?, claim_generation=claim_generation+1,
                        version=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=?
                      AND actor_id=?
                    """,
                    (
                        SlotState.CANCELED.value,
                        version,
                        timestamp,
                        *self._key_values(key),
                    ),
                )
                self._event(
                    cursor,
                    batch_id,
                    "slot",
                    SlotState.CANCELED.value,
                    version,
                    timestamp,
                    key=key,
                    from_state=current.value,
                    reason=reason,
                )

            lease_rows = cursor.execute(
                """
                SELECT * FROM actual_leases
                WHERE run_id=? AND train_step_id=? AND status IN (?, ?)
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    LeaseStatus.ACTIVE.value,
                    LeaseStatus.CLOSING.value,
                ),
            ).fetchall()
            for lease_row in lease_rows:
                existing = cursor.execute(
                    "SELECT * FROM lease_close_outbox_v2 WHERE lease_id=?",
                    (lease_row["lease_id"],),
                ).fetchone()
                if existing is None:
                    cursor.execute(
                        """
                        INSERT INTO lease_close_outbox_v2(
                            intent_id, lease_id, claim_id, run_id,
                            train_step_id, task_index, actor_id, attempt,
                            generation, fence_token, operation_id, action,
                            reason, artifact_digest, status, close_attempts,
                            last_error, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NONE',
                                  ?, NULL, ?, 0, NULL, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            str(lease_row["lease_id"]),
                            lease_row["claim_id"],
                            batch_id.run_id,
                            batch_id.train_step_id,
                            int(lease_row["task_index"]),
                            str(lease_row["actor_id"]),
                            int(lease_row["attempt"]),
                            None,
                            str(lease_row["fence_token"]),
                            f"close:{lease_row['lease_id']}:{lease_row['attempt']}",
                            reason,
                            OutboxStatus.PENDING.value,
                            timestamp,
                            timestamp,
                        ),
                    )
                elif OutboxStatus(existing["status"]) == OutboxStatus.PENDING:
                    cursor.execute(
                        """
                        UPDATE lease_close_outbox_v2
                        SET action='NONE', reason=?, updated_at=?
                        WHERE intent_id=?
                        """,
                        (reason, timestamp, existing["intent_id"]),
                    )
                cursor.execute(
                    """
                    UPDATE actual_leases SET status=?, updated_at=?
                    WHERE lease_id=?
                    """,
                    (
                        LeaseStatus.CLOSING.value,
                        timestamp,
                        lease_row["lease_id"],
                    ),
                )

            cursor.execute(
                """
                UPDATE allocation_cleanup_outbox_v2
                SET action='NONE', reason=?, status='PENDING', updated_at=?
                WHERE run_id=? AND train_step_id=? AND status<> 'CLOSED'
                """,
                (
                    reason,
                    timestamp,
                    batch_id.run_id,
                    batch_id.train_step_id,
                ),
            )

            intent_rows = cursor.execute(
                """
                SELECT * FROM lease_close_outbox_v2
                WHERE run_id=? AND train_step_id=? AND status=?
                ORDER BY created_at, intent_id
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    OutboxStatus.PENDING.value,
                ),
            ).fetchall()
            return (
                self._batch_from_row(self._require_batch_row(cursor, batch_id)),
                tuple(self._close_intent_from_row(row) for row in intent_rows),
            )

    def recover_failed_batch(
        self,
        batch_id: BatchId,
        *,
        request_id: str,
        reason: str = "operator-recovery",
    ) -> BatchRecord:

        if not request_id:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "request_id must not be empty",
            )
        payload_hash = self._payload_hash(
            {"batch_id": batch_id.to_dict(), "reason": reason}
        )
        timestamp = float(self._clock())
        with self._transaction() as cursor:
            replay = self._idempotency_lookup(
                cursor, "recover_failed_batch", request_id, payload_hash
            )
            if replay is not None:
                return self._batch_from_dict(replay["batch"])
            batch_row = self._require_batch_row(cursor, batch_id)
            state = BatchState(batch_row["state"])
            if state == BatchState.IN_DOUBT:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "IN_DOUBT requires explicit optimizer-outcome resolution",
                    {"state": state.value, **batch_id.to_dict()},
                )
            if state != BatchState.FAILED:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "only a FAILED batch may be recovered",
                    {"state": state.value, **batch_id.to_dict()},
                )
            live = cursor.execute(
                """
                SELECT (
                    SELECT COUNT(*) FROM actual_leases
                    WHERE run_id=? AND train_step_id=? AND status IN (?, ?)
                ) + (
                    SELECT COUNT(*) FROM allocation_cleanup_outbox_v2
                    WHERE run_id=? AND train_step_id=?
                      AND status IN ('ALLOCATING', 'PENDING')
                ) AS count
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    LeaseStatus.ACTIVE.value,
                    LeaseStatus.CLOSING.value,
                    batch_id.run_id,
                    batch_id.train_step_id,
                ),
            ).fetchone()
            if int(live["count"] if live is not None else 0):
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "FAILED batch must finish draining before recovery",
                    batch_id.to_dict(),
                )
            current_revision = int(batch_row["current_manifest_revision"])
            slot_rows = cursor.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=? AND manifest_revision=?
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    current_revision,
                ),
            ).fetchall()
            recoverable = {
                SlotState.CANCELED,
                SlotState.FAILED,
                SlotState.RETRY_PENDING,
                SlotState.PENDING,
                SlotState.DONE,
            }
            invalid = [
                self._key_from_row(row).to_dict()
                for row in slot_rows
                if SlotState(row["state"]) not in recoverable
            ]
            if invalid:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "FAILED batch contains undrained slots",
                    {"slots": invalid},
                )
            validate_batch_transition(state, BatchState.RECOVERING)
            recovering_version = int(batch_row["version"]) + 1
            cursor.execute(
                """
                UPDATE batches SET state=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND version=?
                """,
                (
                    BatchState.RECOVERING.value,
                    recovering_version,
                    timestamp,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    int(batch_row["version"]),
                ),
            )
            self._event(
                cursor,
                batch_id,
                "batch",
                BatchState.RECOVERING.value,
                recovering_version,
                timestamp,
                from_state=BatchState.FAILED.value,
                reason=reason,
            )
            for row in slot_rows:
                current = SlotState(row["state"])
                if current not in {SlotState.CANCELED, SlotState.FAILED}:
                    continue
                validate_slot_transition(current, SlotState.RETRY_PENDING)
                version = int(row["version"]) + 1
                key = self._key_from_row(row)
                cursor.execute(
                    """
                    UPDATE slots SET state=?, version=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=?
                      AND actor_id=?
                    """,
                    (
                        SlotState.RETRY_PENDING.value,
                        version,
                        timestamp,
                        *self._key_values(key),
                    ),
                )
                self._event(
                    cursor,
                    batch_id,
                    "slot",
                    SlotState.RETRY_PENDING.value,
                    version,
                    timestamp,
                    key=key,
                    from_state=current.value,
                    reason=reason,
                )
            validate_batch_transition(
                BatchState.RECOVERING, BatchState.RUNNING
            )
            running_version = recovering_version + 1
            cursor.execute(
                """
                UPDATE batches SET state=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND version=?
                """,
                (
                    BatchState.RUNNING.value,
                    running_version,
                    timestamp,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    recovering_version,
                ),
            )
            self._event(
                cursor,
                batch_id,
                "batch",
                BatchState.RUNNING.value,
                running_version,
                timestamp,
                from_state=BatchState.RECOVERING.value,
                reason="recovery-scheduled",
            )
            result = self._batch_from_row(
                self._require_batch_row(cursor, batch_id)
            )
            self._idempotency_store(
                cursor,
                "recover_failed_batch",
                request_id,
                payload_hash,
                {"batch": self._batch_to_dict(result)},
                timestamp,
            )
            return result

    def get_slot(self, key: SlotKey) -> SlotRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                self._key_values(key),
            ).fetchone()
        if row is None:
            raise self._not_found("slot", key.to_dict())
        return self._slot_from_row(row)

    def list_slots(
        self,
        batch_id: BatchId,
        *,
        states: Sequence[SlotState] | None = None,
    ) -> tuple[SlotRecord, ...]:
        parameters: list[Any] = [batch_id.run_id, batch_id.train_step_id]
        query = "SELECT * FROM slots WHERE run_id=? AND train_step_id=?"
        if states:
            values = [self._coerce_enum(SlotState, state, "slot state").value for state in states]
            query += f" AND state IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query += " ORDER BY task_index, actor_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._slot_from_row(row) for row in rows)

    def slot_counts(self, batch_id: BatchId) -> dict[SlotState, int]:
        self.get_batch(batch_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT state, COUNT(*) AS count FROM slots
                WHERE run_id=? AND train_step_id=? GROUP BY state
                """,
                (batch_id.run_id, batch_id.train_step_id),
            ).fetchall()
        return {SlotState(row["state"]): int(row["count"]) for row in rows}

    def list_attempts(self, key: SlotKey) -> tuple[AttemptRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM attempts
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                ORDER BY attempt
                """,
                self._key_values(key),
            ).fetchall()
        return tuple(
            AttemptRecord(
                key=key,
                attempt=int(row["attempt"]),
                status=AttemptStatus(row["status"]),
                worker_id=row["worker_id"],
                started_at=float(row["started_at"]),
                finished_at=(
                    float(row["finished_at"]) if row["finished_at"] is not None else None
                ),
                error=(
                    dict(json.loads(row["error_json"]))
                    if row["error_json"] is not None
                    else None
                ),
            )
            for row in rows
        )

    def update_slot_state(
        self,
        key: SlotKey,
        target: SlotState,
        *,
        expected_version: int,
        expected_attempt: int | None = None,
        expected_claim_generation: int | None = None,
        reason: str | None = None,
    ) -> SlotRecord:
        target_state = self._coerce_enum(SlotState, target, "slot state")
        now = float(self._clock())
        with self._transaction() as cursor:
            row = self._require_slot_row(cursor, key)
            self._check_version("slot", expected_version, int(row["version"]), key.to_dict())
            self._check_fences(row, key, expected_attempt, expected_claim_generation)
            current = SlotState(row["state"])
            validate_slot_transition(current, target_state)
            if current == target_state:
                return self._slot_from_row(row)
            version = int(row["version"]) + 1
            cursor.execute(
                """
                UPDATE slots SET state=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=? AND version=?
                """,
                (target_state.value, version, now, *self._key_values(key), expected_version),
            )
            if cursor.rowcount != 1:
                raise self._version_conflict("slot", expected_version, key.to_dict())
            self._event(
                cursor,
                key.batch_id,
                "slot",
                target_state.value,
                version,
                now,
                key=key,
                from_state=current.value,
                reason=reason,
            )
            return self._slot_from_row(self._require_slot_row(cursor, key))

    def begin_attempt(
        self,
        key: SlotKey,
        *,
        worker_id: str | None = None,
        expected_version: int | None = None,
    ) -> AttemptRecord:
        now = float(self._clock())
        with self._transaction() as cursor:
            row = self._require_slot_row(cursor, key)
            if expected_version is not None:
                self._check_version("slot", expected_version, int(row["version"]), key.to_dict())
            current = SlotState(row["state"])
            if current not in {SlotState.PENDING, SlotState.RETRY_PENDING, SlotState.RUNNING}:
                validate_slot_transition(current, SlotState.RUNNING)
            previous_attempt = int(row["current_attempt"])
            if current == SlotState.RUNNING and previous_attempt:
                cursor.execute(
                    """
                    UPDATE attempts SET status=?, finished_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                      AND attempt=? AND status=?
                    """,
                    (
                        AttemptStatus.SUPERSEDED.value,
                        now,
                        *self._key_values(key),
                        previous_attempt,
                        AttemptStatus.RUNNING.value,
                    ),
                )
            attempt = previous_attempt + 1
            version = int(row["version"]) + 1
            cursor.execute(
                """
                UPDATE slots SET state=?, current_attempt=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (SlotState.RUNNING.value, attempt, version, now, *self._key_values(key)),
            )
            cursor.execute(
                """
                INSERT INTO attempts(
                    run_id, train_step_id, task_index, actor_id, attempt,
                    status, worker_id, started_at, finished_at, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (*self._key_values(key), attempt, AttemptStatus.RUNNING.value, worker_id, now),
            )
            self._event(
                cursor,
                key.batch_id,
                "slot",
                SlotState.RUNNING.value,
                version,
                now,
                key=key,
                from_state=current.value,
                reason="attempt-started",
                details={"attempt": attempt},
            )
            return AttemptRecord(
                key=key,
                attempt=attempt,
                status=AttemptStatus.RUNNING,
                worker_id=worker_id,
                started_at=now,
                finished_at=None,
            )

    def finish_attempt(
        self,
        key: SlotKey,
        attempt: int,
        *,
        succeeded: bool,
        retryable: bool = True,
        actor_artifact: ArtifactRef | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> AttemptRecord:
        now = float(self._clock())
        error_json = self._json(dict(error), "attempt error") if error is not None else None
        with self._transaction() as cursor:
            slot_row = self._require_slot_row(cursor, key)
            if int(slot_row["current_attempt"]) != attempt:
                raise self._fenced(
                    "attempt is stale",
                    {**key.to_dict(), "expected_attempt": int(slot_row["current_attempt"]), "attempt": attempt},
                )
            attempt_row = cursor.execute(
                """
                SELECT * FROM attempts
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=? AND attempt=?
                """,
                (*self._key_values(key), attempt),
            ).fetchone()
            if attempt_row is None:
                raise self._not_found("attempt", {**key.to_dict(), "attempt": attempt})
            if AttemptStatus(attempt_row["status"]) != AttemptStatus.RUNNING:
                raise self._fenced(
                    "attempt has already finished",
                    {**key.to_dict(), "attempt": attempt, "status": attempt_row["status"]},
                )
            current = SlotState(slot_row["state"])
            target = SlotState.READY if succeeded else (SlotState.RETRY_PENDING if retryable else SlotState.FAILED)
            validate_slot_transition(current, target)
            if succeeded and actor_artifact is not None:
                self._register_artifact(cursor, actor_artifact, now)
                cursor.execute(
                    """
                    INSERT INTO slot_artifacts(
                        run_id, train_step_id, task_index, actor_id, attempt,
                        role, digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'actor', ?, ?)
                    ON CONFLICT(
                        run_id, train_step_id, task_index, actor_id, attempt, role
                    ) DO UPDATE SET digest=excluded.digest
                    """,
                    (*self._key_values(key), attempt, actor_artifact.digest, now),
                )
            status = AttemptStatus.SUCCEEDED if succeeded else AttemptStatus.FAILED
            cursor.execute(
                """
                UPDATE attempts SET status=?, finished_at=?, error_json=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND attempt=? AND status=?
                """,
                (
                    status.value,
                    now,
                    error_json,
                    *self._key_values(key),
                    attempt,
                    AttemptStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise self._fenced("attempt completion lost its fence", {**key.to_dict(), "attempt": attempt})
            version = int(slot_row["version"]) + 1
            cursor.execute(
                """
                UPDATE slots SET state=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND current_attempt=?
                """,
                (target.value, version, now, *self._key_values(key), attempt),
            )
            if cursor.rowcount != 1:
                raise self._fenced("slot attempt fence changed", {**key.to_dict(), "attempt": attempt})
            self._event(
                cursor,
                key.batch_id,
                "slot",
                target.value,
                version,
                now,
                key=key,
                from_state=current.value,
                reason="attempt-finished",
                details={"attempt": attempt, "status": status.value},
            )
            return AttemptRecord(
                key=key,
                attempt=attempt,
                status=status,
                worker_id=attempt_row["worker_id"],
                started_at=float(attempt_row["started_at"]),
                finished_at=now,
                error=dict(error) if error is not None else None,
            )

    def record_attempt_timing(
        self,
        key: SlotKey,
        attempt: int,
        *,
        queue_wait_seconds: float,
        actor_seconds: float,
        total_seconds: float,
    ) -> None:
        values = (
            float(queue_wait_seconds),
            float(actor_seconds),
            float(total_seconds),
        )
        if any(value < 0 for value in values):
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "attempt timings must be non-negative",
                {**key.to_dict(), "attempt": attempt},
            )
        now = float(self._clock())
        with self._transaction() as cursor:
            attempt_row = cursor.execute(
                """
                SELECT 1 FROM attempts
                WHERE run_id=? AND train_step_id=? AND task_index=?
                  AND actor_id=? AND attempt=?
                """,
                (*self._key_values(key), attempt),
            ).fetchone()
            if attempt_row is None:
                raise self._not_found(
                    "attempt",
                    {**key.to_dict(), "attempt": attempt},
                )
            cursor.execute(
                """
                INSERT INTO attempt_timings(
                    run_id, train_step_id, task_index, actor_id, attempt,
                    queue_wait_seconds, actor_seconds, total_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    run_id, train_step_id, task_index, actor_id, attempt
                ) DO UPDATE SET
                    queue_wait_seconds=excluded.queue_wait_seconds,
                    actor_seconds=excluded.actor_seconds,
                    total_seconds=excluded.total_seconds
                """,
                (*self._key_values(key), attempt, *values, now),
            )

    def register_attempt_lease(
        self,
        key: SlotKey,
        attempt: int,
        *,
        lease_id: str,
        fence_token: str,
        allocation_key: str,
        owner: Mapping[str, Any],
        now: float | None = None,
    ) -> LeaseRecord:

        if not lease_id or not fence_token or not allocation_key:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "lease_id, fence_token and allocation_key must not be empty",
            )
        timestamp = float(self._clock() if now is None else now)
        owner_json = self._json(dict(owner), "lease owner")
        with self._transaction() as cursor:
            slot_row = self._require_slot_row(cursor, key)
            if (
                int(slot_row["current_attempt"]) != attempt
                or SlotState(slot_row["state"]) != SlotState.RUNNING
            ):
                raise self._fenced(
                    "attempt no longer owns the slot",
                    {**key.to_dict(), "attempt": attempt},
                )
            existing = cursor.execute(
                """
                SELECT * FROM actual_leases
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND attempt=?
                """,
                (*self._key_values(key), attempt),
            ).fetchone()
            if existing is not None:
                if (
                    existing["lease_id"] != lease_id
                    or existing["fence_token"] != fence_token
                    or existing["allocation_key"] != allocation_key
                    or existing["owner_json"] != owner_json
                ):
                    raise ProtocolError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "attempt lease is already registered with different data",
                        {**key.to_dict(), "attempt": attempt},
                    )
                cursor.execute(
                    "DELETE FROM allocation_cleanup_outbox_v2 WHERE allocation_key=?",
                    (allocation_key,),
                )
                return self._lease_from_row(existing)
            collision = cursor.execute(
                "SELECT * FROM actual_leases WHERE lease_id=?",
                (lease_id,),
            ).fetchone()
            if collision is not None:
                raise ProtocolError(
                    ErrorCode.IDEMPOTENCY_CONFLICT,
                    "environment lease is already owned by another attempt",
                    {"lease_id": lease_id},
                )
            cursor.execute(
                """
                INSERT INTO actual_leases(
                    lease_id, run_id, train_step_id, task_index, actor_id,
                    attempt, fence_token, allocation_key, owner_json, status,
                    claim_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    lease_id,
                    *self._key_values(key),
                    attempt,
                    fence_token,
                    allocation_key,
                    owner_json,
                    LeaseStatus.ACTIVE.value,
                    timestamp,
                    timestamp,
                ),
            )
            cursor.execute(
                "DELETE FROM allocation_cleanup_outbox_v2 WHERE allocation_key=?",
                (allocation_key,),
            )
            return self._lease_from_row(
                cursor.execute(
                    "SELECT * FROM actual_leases WHERE lease_id=?",
                    (lease_id,),
                ).fetchone()
            )

    def begin_allocation(
        self,
        key: SlotKey,
        attempt: int,
        *,
        allocation_key: str,
        fence_token: str,
        owner: Mapping[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]:

        if not allocation_key or not fence_token:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "allocation_key and fence_token must not be empty",
            )
        timestamp = float(self._clock() if now is None else now)
        owner_json = self._json(dict(owner), "allocation owner")
        operation_id = f"cancel-allocation:{allocation_key}"
        with self._transaction() as cursor:
            slot_row = self._require_slot_row(cursor, key)
            if (
                int(slot_row["current_attempt"]) != int(attempt)
                or SlotState(slot_row["state"]) != SlotState.RUNNING
            ):
                raise self._fenced(
                    "attempt no longer owns the allocation",
                    {**key.to_dict(), "attempt": attempt},
                )
            existing = cursor.execute(
                """
                SELECT * FROM allocation_cleanup_outbox_v2
                WHERE allocation_key=?
                """,
                (allocation_key,),
            ).fetchone()
            if existing is not None:
                if (
                    self._key_from_row(existing) != key
                    or int(existing["attempt"]) != int(attempt)
                    or str(existing["fence_token"]) != fence_token
                    or str(existing["owner_json"]) != owner_json
                ):
                    raise ProtocolError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "allocation key is already owned by different work",
                        {"allocation_key": allocation_key},
                    )
                return self._allocation_cleanup_from_row(existing)
            cursor.execute(
                """
                INSERT INTO allocation_cleanup_outbox_v2(
                    allocation_key, run_id, train_step_id, task_index,
                    actor_id, attempt, fence_token, owner_json, operation_id,
                    action, reason, status, cleanup_attempts, last_error,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NONE',
                          'allocation-in-flight', 'ALLOCATING', 0, NULL, ?, ?)
                """,
                (
                    allocation_key,
                    *self._key_values(key),
                    int(attempt),
                    fence_token,
                    owner_json,
                    operation_id,
                    timestamp,
                    timestamp,
                ),
            )
            return self._allocation_cleanup_from_row(
                cursor.execute(
                    """
                    SELECT * FROM allocation_cleanup_outbox_v2
                    WHERE allocation_key=?
                    """,
                    (allocation_key,),
                ).fetchone()
            )

    def quarantine_allocation(
        self,
        allocation_key: str,
        *,
        action: str,
        reason: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        if action not in {"RETRY", "NONE"}:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "allocation cleanup action must be RETRY or NONE",
                {"action": action},
            )
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            row = cursor.execute(
                """
                SELECT * FROM allocation_cleanup_outbox_v2
                WHERE allocation_key=?
                """,
                (allocation_key,),
            ).fetchone()
            if row is None:
                raise self._not_found(
                    "allocation cleanup",
                    {"allocation_key": allocation_key},
                )
            if str(row["status"]) == "CLOSED":
                return self._allocation_cleanup_from_row(row)
            cursor.execute(
                """
                UPDATE allocation_cleanup_outbox_v2
                SET action=?, reason=?, status='PENDING', updated_at=?
                WHERE allocation_key=?
                """,
                (action, str(reason), timestamp, allocation_key),
            )
            return self._allocation_cleanup_from_row(
                cursor.execute(
                    """
                    SELECT * FROM allocation_cleanup_outbox_v2
                    WHERE allocation_key=?
                    """,
                    (allocation_key,),
                ).fetchone()
            )

    def list_allocation_cleanups(
        self,
        *,
        statuses: Sequence[str] | None = None,
        batch_id: BatchId | None = None,
    ) -> tuple[dict[str, Any], ...]:
        query = "SELECT * FROM allocation_cleanup_outbox_v2 WHERE 1=1"
        parameters: list[Any] = []
        if statuses:
            normalized = [str(status).upper() for status in statuses]
            query += f" AND status IN ({','.join('?' for _ in normalized)})"
            parameters.extend(normalized)
        if batch_id is not None:
            query += " AND run_id=? AND train_step_id=?"
            parameters.extend([batch_id.run_id, batch_id.train_step_id])
        query += " ORDER BY created_at, allocation_key"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._allocation_cleanup_from_row(row) for row in rows)

    def get_allocation_cleanup(self, allocation_key: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM allocation_cleanup_outbox_v2
                WHERE allocation_key=?
                """,
                (allocation_key,),
            ).fetchone()
        if row is None:
            raise self._not_found(
                "allocation cleanup",
                {"allocation_key": allocation_key},
            )
        return self._allocation_cleanup_from_row(row)

    def has_allocation_hold(
        self,
        key: SlotKey,
        attempt: int | None = None,
    ) -> bool:
        query = """
            SELECT 1 FROM allocation_cleanup_outbox_v2
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
              AND status IN ('ALLOCATING', 'PENDING')
        """
        parameters: list[Any] = list(self._key_values(key))
        if attempt is not None:
            query += " AND attempt=?"
            parameters.append(int(attempt))
        query += " LIMIT 1"
        with self._lock:
            return self._connection.execute(query, parameters).fetchone() is not None

    def record_allocation_cleanup_failure(
        self,
        allocation_key: str,
        error: str,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            row = cursor.execute(
                """
                SELECT * FROM allocation_cleanup_outbox_v2
                WHERE allocation_key=?
                """,
                (allocation_key,),
            ).fetchone()
            if row is None:
                raise self._not_found(
                    "allocation cleanup",
                    {"allocation_key": allocation_key},
                )
            if str(row["status"]) == "CLOSED":
                return self._allocation_cleanup_from_row(row)
            cursor.execute(
                """
                UPDATE allocation_cleanup_outbox_v2
                SET cleanup_attempts=cleanup_attempts+1, last_error=?,
                    status='PENDING', updated_at=?
                WHERE allocation_key=?
                """,
                (str(error), timestamp, allocation_key),
            )
            return self._allocation_cleanup_from_row(
                cursor.execute(
                    """
                    SELECT * FROM allocation_cleanup_outbox_v2
                    WHERE allocation_key=?
                    """,
                    (allocation_key,),
                ).fetchone()
            )

    def mark_allocation_cleanup_terminal(
        self,
        allocation_key: str,
        *,
        now: float | None = None,
    ) -> tuple[dict[str, Any], SlotRecord]:
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            row = cursor.execute(
                """
                SELECT * FROM allocation_cleanup_outbox_v2
                WHERE allocation_key=?
                """,
                (allocation_key,),
            ).fetchone()
            if row is None:
                raise self._not_found(
                    "allocation cleanup",
                    {"allocation_key": allocation_key},
                )
            if str(row["status"]) != "CLOSED":
                cursor.execute(
                    """
                    UPDATE allocation_cleanup_outbox_v2
                    SET status='CLOSED', cleanup_attempts=cleanup_attempts+1,
                        last_error=NULL, updated_at=?
                    WHERE allocation_key=?
                    """,
                    (timestamp, allocation_key),
                )
                row = cursor.execute(
                    """
                    SELECT * FROM allocation_cleanup_outbox_v2
                    WHERE allocation_key=?
                    """,
                    (allocation_key,),
                ).fetchone()
            key = self._key_from_row(row)
            return (
                self._allocation_cleanup_from_row(row),
                self._slot_from_row(self._require_slot_row(cursor, key)),
            )

    def get_attempt_lease(
        self,
        key: SlotKey,
        attempt: int | None = None,
    ) -> LeaseRecord:
        slot = self.get_slot(key)
        selected_attempt = slot.current_attempt if attempt is None else attempt
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM actual_leases
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND attempt=?
                """,
                (*self._key_values(key), selected_attempt),
            ).fetchone()
        if row is None:
            raise self._not_found(
                "attempt lease",
                {**key.to_dict(), "attempt": selected_attempt},
            )
        return self._lease_from_row(row)

    def get_claim_lease(self, claim_id: str) -> LeaseRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM actual_leases WHERE claim_id=?",
                (claim_id,),
            ).fetchone()
        if row is None:
            raise self._not_found("claim lease", {"claim_id": claim_id})
        return self._lease_from_row(row)

    def get_lease(self, lease_id: str) -> LeaseRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM actual_leases WHERE lease_id=?",
                (lease_id,),
            ).fetchone()
        if row is None:
            raise self._not_found("attempt lease", {"lease_id": lease_id})
        return self._lease_from_row(row)

    def list_leases(
        self,
        *,
        batch_id: BatchId | None = None,
        statuses: Sequence[LeaseStatus] | None = None,
    ) -> tuple[LeaseRecord, ...]:
        query = "SELECT * FROM actual_leases WHERE 1=1"
        parameters: list[Any] = []
        if batch_id is not None:
            query += " AND run_id=? AND train_step_id=?"
            parameters.extend([batch_id.run_id, batch_id.train_step_id])
        if statuses:
            values = [
                self._coerce_enum(LeaseStatus, status, "lease status").value
                for status in statuses
            ]
            query += f" AND status IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query += " ORDER BY created_at, lease_id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._lease_from_row(row) for row in rows)

    def get_slot_artifact(
        self,
        key: SlotKey,
        *,
        role: str = "actor",
        attempt: int | None = None,
    ) -> ArtifactRef:
        slot = self.get_slot(key)
        selected_attempt = slot.current_attempt if attempt is None else attempt
        with self._lock:
            row = self._connection.execute(
                """
                SELECT a.* FROM slot_artifacts sa
                JOIN artifact_refs a ON a.digest=sa.digest
                WHERE sa.run_id=? AND sa.train_step_id=? AND sa.task_index=?
                  AND sa.actor_id=? AND sa.attempt=? AND sa.role=?
                """,
                (*self._key_values(key), selected_attempt, role),
            ).fetchone()
        if row is None:
            raise self._not_found(
                "slot artifact",
                {**key.to_dict(), "attempt": selected_attempt, "role": role},
            )
        return self._artifact_from_row(row)

    def claim_ready_slots(
        self,
        batch_id: BatchId,
        *,
        consumer_id: str,
        limit: int,
        request_id: str,
        ttl_seconds: float,
        now: float | None = None,
        store_empty_replay: bool = True,
    ) -> ClaimResult:
        if not consumer_id or not request_id:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "consumer_id and request_id must not be empty",
            )
        if limit <= 0 or ttl_seconds <= 0:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "claim limit and ttl_seconds must be positive",
            )
        timestamp = float(self._clock() if now is None else now)
        payload_hash = self._payload_hash(
            {
                "batch_id": batch_id.to_dict(),
                "consumer_id": consumer_id,
                "limit": limit,
                "ttl_seconds": ttl_seconds,
            }
        )
        with self._transaction() as cursor:
            batch_row = self._require_batch_row(cursor, batch_id)
            batch_state = BatchState(batch_row["state"])
            if batch_state in {
                BatchState.FAILED,
                BatchState.INVALID,
                BatchState.ABANDONED,
                BatchState.IN_DOUBT,
                BatchState.COMMITTED,
            }:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "batch does not accept result claims",
                    {"state": batch_state.value, **batch_id.to_dict()},
                )
            replay = self._idempotency_lookup(cursor, "claim_ready_slots", request_id, payload_hash)
            if replay is not None:
                return ClaimResult(
                    request_id=request_id,
                    claims=tuple(self._claim_from_dict(item) for item in replay["claims"]),
                    replayed=True,
                )
            self._recover_expired_claims(cursor, timestamp, batch_id)
            rows = cursor.execute(
                """
                SELECT s.* FROM slots s
                JOIN actual_leases l
                  ON l.run_id=s.run_id AND l.train_step_id=s.train_step_id
                 AND l.task_index=s.task_index AND l.actor_id=s.actor_id
                 AND l.attempt=s.current_attempt
                JOIN slot_artifacts a
                  ON a.run_id=s.run_id AND a.train_step_id=s.train_step_id
                 AND a.task_index=s.task_index AND a.actor_id=s.actor_id
                 AND a.attempt=s.current_attempt AND a.role='actor'
                WHERE s.run_id=? AND s.train_step_id=?
                  AND s.manifest_revision=? AND s.state=?
                  AND l.status=? AND l.claim_id IS NULL
                ORDER BY s.task_index, s.actor_id
                LIMIT ?
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    int(batch_row["current_manifest_revision"]),
                    SlotState.READY.value,
                    LeaseStatus.ACTIVE.value,
                    limit,
                ),
            ).fetchall()
            claims: list[ClaimRecord] = []
            expires_at = timestamp + ttl_seconds
            for row in rows:
                key = self._key_from_row(row)
                generation = int(row["claim_generation"]) + 1
                version = int(row["version"]) + 1
                claim_id = uuid.uuid4().hex
                cursor.execute(
                    """
                    UPDATE slots
                    SET state=?, claim_generation=?, version=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                      AND state=? AND claim_generation=?
                    """,
                    (
                        SlotState.CLAIMED.value,
                        generation,
                        version,
                        timestamp,
                        *self._key_values(key),
                        SlotState.READY.value,
                        int(row["claim_generation"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise self._fenced("ready slot changed while claiming", key.to_dict())
                cursor.execute(
                    """
                    INSERT INTO claims(
                        claim_id, run_id, train_step_id, task_index, actor_id,
                        generation, request_id, status, consumer_id, expires_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_id,
                        *self._key_values(key),
                        generation,
                        request_id,
                        ClaimStatus.ACTIVE.value,
                        consumer_id,
                        expires_at,
                        timestamp,
                        timestamp,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE actual_leases SET claim_id=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=?
                      AND actor_id=? AND attempt=? AND status=? AND claim_id IS NULL
                    """,
                    (
                        claim_id,
                        timestamp,
                        *self._key_values(key),
                        int(row["current_attempt"]),
                        LeaseStatus.ACTIVE.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise self._fenced(
                        "real attempt lease changed while claiming",
                        key.to_dict(),
                    )
                claim = ClaimRecord(
                    claim_id=claim_id,
                    key=key,
                    generation=generation,
                    status=ClaimStatus.ACTIVE,
                    consumer_id=consumer_id,
                    expires_at=expires_at,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                claims.append(claim)
                self._event(
                    cursor,
                    batch_id,
                    "slot",
                    SlotState.CLAIMED.value,
                    version,
                    timestamp,
                    key=key,
                    from_state=SlotState.READY.value,
                    reason="claimed",
                    details={"claim_id": claim_id, "generation": generation},
                )
            response = {"claims": [self._claim_to_dict(claim) for claim in claims]}
            if claims or store_empty_replay:
                self._idempotency_store(
                    cursor,
                    "claim_ready_slots",
                    request_id,
                    payload_hash,
                    response,
                    timestamp,
                )
            return ClaimResult(request_id=request_id, claims=tuple(claims))

    def renew_claim(
        self,
        claim_id: str,
        generation: int,
        *,
        ttl_seconds: float,
        consumer_id: str | None = None,
        request_id: str | None = None,
        now: float | None = None,
    ) -> ClaimRecord:
        if ttl_seconds <= 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "ttl_seconds must be positive")
        if request_id is not None and not request_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "request_id must not be empty")
        timestamp = float(self._clock() if now is None else now)
        payload_hash = self._payload_hash(
            {
                "claim_id": claim_id,
                "generation": generation,
                "ttl_seconds": ttl_seconds,
                "consumer_id": consumer_id,
            }
        )
        with self._transaction() as cursor:
            if request_id is not None:
                replay = self._idempotency_lookup(
                    cursor, "renew_claim", request_id, payload_hash
                )
                if replay is not None:
                    return self._claim_from_dict(replay["claim"])
            self._recover_expired_claims(cursor, timestamp, None)
            claim_row = self._require_claim_row(cursor, claim_id)
            key = self._key_from_row(claim_row)
            if (
                int(claim_row["generation"]) != generation
                or ClaimStatus(claim_row["status"]) != ClaimStatus.ACTIVE
                or (consumer_id is not None and claim_row["consumer_id"] != consumer_id)
            ):
                raise self._fenced("claim renewal fence does not match", {"claim_id": claim_id})
            slot_row = self._require_slot_row(cursor, key)
            if int(slot_row["claim_generation"]) != generation or SlotState(slot_row["state"]) != SlotState.CLAIMED:
                raise self._fenced("slot no longer belongs to claim", {"claim_id": claim_id})
            expires_at = timestamp + ttl_seconds
            cursor.execute(
                "UPDATE claims SET expires_at=?, updated_at=? WHERE claim_id=?",
                (expires_at, timestamp, claim_id),
            )
            updated = self._require_claim_row(cursor, claim_id)
            result = self._claim_from_row(updated)
            if request_id is not None:
                self._idempotency_store(
                    cursor,
                    "renew_claim",
                    request_id,
                    payload_hash,
                    {"claim": self._claim_to_dict(result)},
                    timestamp,
                )
            return result

    def get_claim(self, claim_id: str) -> ClaimRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM claims WHERE claim_id=?",
                (claim_id,),
            ).fetchone()
        if row is None:
            raise self._not_found("claim", {"claim_id": claim_id})
        return self._claim_from_row(row)

    def next_claim_expiry(self) -> float | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT MIN(expires_at) AS expires_at FROM claims WHERE status=?",
                (ClaimStatus.ACTIVE.value,),
            ).fetchone()
        value = row["expires_at"] if row is not None else None
        return float(value) if value is not None else None

    def recover_expired_claims(
        self,
        *,
        now: float | None = None,
        batch_id: BatchId | None = None,
    ) -> RecoveryReport:
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            return self._recover_expired_claims(cursor, timestamp, batch_id)

    def record_ack_artifact(
        self,
        claim_id: str,
        generation: int,
        artifact: ArtifactRef,
        *,
        request_id: str,
        now: float | None = None,
    ) -> AckResult:
        if not request_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "request_id must not be empty")
        if (
            artifact.size < 0
            or len(artifact.digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact.digest)
            or not artifact.uri
        ):
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "invalid artifact reference")
        timestamp = float(self._clock() if now is None else now)
        payload_hash = self._payload_hash(
            {
                "claim_id": claim_id,
                "generation": generation,
                "artifact": artifact.to_dict(),
            }
        )
        self.recover_expired_claims(now=timestamp)
        with self._transaction() as cursor:
            replay = self._idempotency_lookup(cursor, "record_ack_artifact", request_id, payload_hash)
            if replay is not None:
                return self._ack_from_dict(replay, replayed=True)
            claim_row = self._require_claim_row(cursor, claim_id)
            key = self._key_from_row(claim_row)
            if (
                int(claim_row["generation"]) != generation
                or ClaimStatus(claim_row["status"]) != ClaimStatus.ACTIVE
            ):
                raise self._fenced("claim acknowledgement fence does not match", {"claim_id": claim_id})
            slot_row = self._require_slot_row(cursor, key)
            if (
                int(slot_row["claim_generation"]) != generation
                or SlotState(slot_row["state"]) != SlotState.CLAIMED
            ):
                raise self._fenced("slot no longer belongs to claim", {"claim_id": claim_id})
            existing_artifact = cursor.execute(
                "SELECT * FROM artifact_refs WHERE digest=?",
                (artifact.digest,),
            ).fetchone()
            if existing_artifact is not None and (
                int(existing_artifact["size"]) != artifact.size
                or existing_artifact["uri"] != artifact.uri
            ):
                raise ProtocolError(
                    ErrorCode.IDEMPOTENCY_CONFLICT,
                    "artifact digest is already registered with different metadata",
                    {"digest": artifact.digest},
                )
            cursor.execute(
                """
                INSERT OR IGNORE INTO artifact_refs(digest, size, uri, media_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (artifact.digest, artifact.size, artifact.uri, artifact.media_type, timestamp),
            )
            cursor.execute(
                """
                INSERT INTO slot_artifacts(
                    run_id, train_step_id, task_index, actor_id, attempt,
                    role, digest, created_at
                ) VALUES (?, ?, ?, ?, ?, 'judge', ?, ?)
                ON CONFLICT(
                    run_id, train_step_id, task_index, actor_id, attempt, role
                ) DO UPDATE SET digest=excluded.digest
                """,
                (
                    *self._key_values(key),
                    int(slot_row["current_attempt"]),
                    artifact.digest,
                    timestamp,
                ),
            )
            lease_row = cursor.execute(
                "SELECT * FROM actual_leases WHERE claim_id=?",
                (claim_id,),
            ).fetchone()
            if lease_row is None:
                raise self._fenced(
                    "claim has no real environment lease",
                    {"claim_id": claim_id},
                )
            version = int(slot_row["version"]) + 1
            intent_id = uuid.uuid4().hex
            cursor.execute(
                """
                UPDATE slots SET state=?, artifact_digest=?, version=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND claim_generation=? AND state=?
                """,
                (
                    SlotState.ACKING.value,
                    artifact.digest,
                    version,
                    timestamp,
                    *self._key_values(key),
                    generation,
                    SlotState.CLAIMED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise self._fenced("slot changed while recording acknowledgement", {"claim_id": claim_id})
            cursor.execute(
                "UPDATE claims SET status=?, updated_at=? WHERE claim_id=?",
                (ClaimStatus.ACKING.value, timestamp, claim_id),
            )
            cursor.execute(
                "UPDATE actual_leases SET status=?, updated_at=? WHERE claim_id=?",
                (LeaseStatus.CLOSING.value, timestamp, claim_id),
            )
            operation_id = f"close:{lease_row['lease_id']}:{lease_row['attempt']}"
            cursor.execute(
                """
                INSERT INTO lease_close_outbox_v2(
                    intent_id, lease_id, claim_id, run_id, train_step_id,
                    task_index, actor_id, attempt, generation, fence_token,
                    operation_id, action, reason, artifact_digest, status,
                    close_attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DONE',
                          'trainer-ack', ?, ?, 0, NULL, ?, ?)
                """,
                (
                    intent_id,
                    str(lease_row["lease_id"]),
                    claim_id,
                    *self._key_values(key),
                    int(lease_row["attempt"]),
                    generation,
                    str(lease_row["fence_token"]),
                    operation_id,
                    artifact.digest,
                    OutboxStatus.PENDING.value,
                    timestamp,
                    timestamp,
                ),
            )
            self._event(
                cursor,
                key.batch_id,
                "slot",
                SlotState.ACKING.value,
                version,
                timestamp,
                key=key,
                from_state=SlotState.CLAIMED.value,
                reason="artifact-acknowledged",
                details={"claim_id": claim_id, "intent_id": intent_id, "digest": artifact.digest},
            )
            result = AckResult(intent_id=intent_id, claim_id=claim_id, key=key, artifact=artifact)
            response = self._ack_to_dict(result)
            self._idempotency_store(
                cursor,
                "record_ack_artifact",
                request_id,
                payload_hash,
                response,
                timestamp,
            )
            return result

    record_ack = record_ack_artifact

    def list_close_outbox(
        self,
        *,
        status: OutboxStatus = OutboxStatus.PENDING,
        limit: int = 100,
    ) -> tuple[CloseIntent, ...]:
        outbox_status = self._coerce_enum(OutboxStatus, status, "outbox status")
        if limit <= 0:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM lease_close_outbox_v2 WHERE status=?
                ORDER BY created_at, intent_id LIMIT ?
                """,
                (outbox_status.value, limit),
            ).fetchall()
        return tuple(self._close_intent_from_row(row) for row in rows)

    def get_close_intent(self, intent_id: str) -> CloseIntent:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                (intent_id,),
            ).fetchone()
        if row is None:
            raise self._not_found("close intent", {"intent_id": intent_id})
        return self._close_intent_from_row(row)

    def record_close_failure(
        self,
        intent_id: str,
        error: str,
        *,
        now: float | None = None,
    ) -> CloseIntent:
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                (intent_id,),
            ).fetchone()
            if row is None:
                raise self._not_found("close intent", {"intent_id": intent_id})
            if OutboxStatus(row["status"]) != OutboxStatus.PENDING:
                return self._close_intent_from_row(row)
            cursor.execute(
                """
                UPDATE lease_close_outbox_v2
                SET close_attempts=close_attempts+1, last_error=?, updated_at=?
                WHERE intent_id=?
                """,
                (str(error), timestamp, intent_id),
            )
            return self._close_intent_from_row(
                cursor.execute(
                    "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                    (intent_id,),
                ).fetchone()
            )

    def mark_lease_closed(
        self,
        claim_id: str,
        generation: int,
        *,
        now: float | None = None,
    ) -> SlotRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT intent_id FROM lease_close_outbox_v2
                WHERE claim_id=? AND generation=?
                """,
                (claim_id, generation),
            ).fetchone()
        if row is None:
            raise self._not_found(
                "close intent",
                {"claim_id": claim_id, "generation": generation},
            )
        return self.mark_close_intent_terminal(
            str(row["intent_id"]),
            lost=False,
            now=now,
        )

    def mark_lease_lost(
        self,
        claim_id: str,
        generation: int,
        *,
        now: float | None = None,
    ) -> SlotRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT intent_id FROM lease_close_outbox_v2
                WHERE claim_id=? AND generation=?
                """,
                (claim_id, generation),
            ).fetchone()
        if row is None:
            raise self._not_found(
                "close intent",
                {"claim_id": claim_id, "generation": generation},
            )
        return self.mark_close_intent_terminal(
            str(row["intent_id"]),
            lost=True,
            now=now,
        )

    def enqueue_attempt_close(
        self,
        lease_id: str,
        *,
        action: str = "RETRY",
        reason: str = "attempt-failed",
        now: float | None = None,
    ) -> CloseIntent:
        if action not in {"RETRY", "NONE"}:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "attempt close action must be RETRY or NONE",
                {"action": action},
            )
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            lease_row = cursor.execute(
                "SELECT * FROM actual_leases WHERE lease_id=?",
                (lease_id,),
            ).fetchone()
            if lease_row is None:
                raise self._not_found("attempt lease", {"lease_id": lease_id})
            existing = cursor.execute(
                "SELECT * FROM lease_close_outbox_v2 WHERE lease_id=?",
                (lease_id,),
            ).fetchone()
            if existing is not None:
                return self._close_intent_from_row(existing)
            key = self._key_from_row(lease_row)
            intent_id = uuid.uuid4().hex
            operation_id = f"close:{lease_id}:{lease_row['attempt']}"
            cursor.execute(
                """
                UPDATE actual_leases SET status=?, updated_at=? WHERE lease_id=?
                """,
                (LeaseStatus.CLOSING.value, timestamp, lease_id),
            )
            cursor.execute(
                """
                INSERT INTO lease_close_outbox_v2(
                    intent_id, lease_id, claim_id, run_id, train_step_id,
                    task_index, actor_id, attempt, generation, fence_token,
                    operation_id, action, reason, artifact_digest, status,
                    close_attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?,
                          NULL, ?, 0, NULL, ?, ?)
                """,
                (
                    intent_id,
                    lease_id,
                    *self._key_values(key),
                    int(lease_row["attempt"]),
                    str(lease_row["fence_token"]),
                    operation_id,
                    action,
                    reason,
                    OutboxStatus.PENDING.value,
                    timestamp,
                    timestamp,
                ),
            )
            return self._close_intent_from_row(
                cursor.execute(
                    "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                    (intent_id,),
                ).fetchone()
            )

    def nack_claim(
        self,
        claim_id: str,
        generation: int,
        *,
        reason: str = "trainer-nack",
        request_id: str | None = None,
        now: float | None = None,
    ) -> CloseIntent:
        if request_id is not None and not request_id:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "request_id must not be empty")
        timestamp = float(self._clock() if now is None else now)
        payload_hash = self._payload_hash(
            {
                "claim_id": claim_id,
                "generation": generation,
                "reason": reason,
            }
        )
        with self._transaction() as cursor:
            if request_id is not None:
                replay = self._idempotency_lookup(
                    cursor, "nack_claim", request_id, payload_hash
                )
                if replay is not None:
                    return self._close_intent_from_dict(replay["intent"])
            claim_row = self._require_claim_row(cursor, claim_id)
            if (
                int(claim_row["generation"]) != generation
                or ClaimStatus(claim_row["status"]) != ClaimStatus.ACTIVE
            ):
                raise self._fenced("claim nack fence does not match", {"claim_id": claim_id})
            intent = self._quarantine_claim(cursor, claim_row, timestamp, reason)
            if request_id is not None:
                self._idempotency_store(
                    cursor,
                    "nack_claim",
                    request_id,
                    payload_hash,
                    {"intent": self._close_intent_to_dict(intent)},
                    timestamp,
                )
            return intent

    def mark_close_intent_terminal(
        self,
        intent_id: str,
        *,
        lost: bool,
        now: float | None = None,
    ) -> SlotRecord:
        timestamp = float(self._clock() if now is None else now)
        with self._transaction() as cursor:
            intent = cursor.execute(
                "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                (intent_id,),
            ).fetchone()
            if intent is None:
                raise self._not_found("close intent", {"intent_id": intent_id})
            key = self._key_from_row(intent)
            terminal_outbox = OutboxStatus.LOST if lost else OutboxStatus.CLOSED
            if OutboxStatus(intent["status"]) in {OutboxStatus.CLOSED, OutboxStatus.LOST}:
                return self._slot_from_row(self._require_slot_row(cursor, key))
            lease_status = LeaseStatus.LOST if lost else LeaseStatus.CLOSED
            cursor.execute(
                "UPDATE actual_leases SET status=?, updated_at=? WHERE lease_id=?",
                (lease_status.value, timestamp, intent["lease_id"]),
            )
            cursor.execute(
                """
                UPDATE lease_close_outbox_v2
                SET status=?, close_attempts=close_attempts+1, updated_at=?
                WHERE intent_id=?
                """,
                (terminal_outbox.value, timestamp, intent_id),
            )
            slot_row = self._require_slot_row(cursor, key)
            current = SlotState(slot_row["state"])
            action = str(intent["action"])
            target = current
            if action == "DONE":
                target = SlotState.DONE
                if intent["claim_id"]:
                    cursor.execute(
                        "UPDATE claims SET status=?, updated_at=? WHERE claim_id=?",
                        (ClaimStatus.ACKED.value, timestamp, intent["claim_id"]),
                    )
            elif action == "RETRY":
                target = SlotState.RETRY_PENDING
                if intent["claim_id"]:
                    cursor.execute(
                        """
                        UPDATE claims SET status=CASE
                            WHEN status=? THEN ? ELSE status END, updated_at=?
                        WHERE claim_id=?
                        """,
                        (
                            ClaimStatus.ACTIVE.value,
                            ClaimStatus.LOST.value,
                            timestamp,
                            intent["claim_id"],
                        ),
                    )
            if target != current:
                validate_slot_transition(current, target)
                version = int(slot_row["version"]) + 1
                cursor.execute(
                    """
                    UPDATE slots SET state=?, version=?, updated_at=?
                    WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                    """,
                    (target.value, version, timestamp, *self._key_values(key)),
                )
                self._event(
                    cursor,
                    key.batch_id,
                    "slot",
                    target.value,
                    version,
                    timestamp,
                    key=key,
                    from_state=current.value,
                    reason="lease-lost" if lost else "lease-closed",
                    details={
                        "intent_id": intent_id,
                        "lease_id": intent["lease_id"],
                        "action": action,
                    },
                )
            return self._slot_from_row(self._require_slot_row(cursor, key))

    def begin_replacement_intent(
        self,
        batch_id: BatchId,
        task_group_id: str,
        triggering_group_revision: int,
    ) -> ReplacementIntent | None:

        timestamp = float(self._clock())
        with self._transaction() as cursor:
            existing = cursor.execute(
                """
                SELECT * FROM replacement_intents
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                  AND triggering_group_revision=?
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    triggering_group_revision,
                ),
            ).fetchone()
            if existing is not None:
                return self._replacement_intent_from_row(existing)
            batch = self._require_batch_row(cursor, batch_id)
            if BatchState(batch["state"]) not in {
                BatchState.RUNNING,
                BatchState.DRAINING,
                BatchState.RECOVERING,
            }:
                return None
            group_row = cursor.execute(
                """
                SELECT MAX(task_group_revision) AS revision
                FROM slots
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                """,
                (batch_id.run_id, batch_id.train_step_id, task_group_id),
            ).fetchone()
            current_revision = int(group_row["revision"] or 0)
            if current_revision != triggering_group_revision:
                return None
            exhausted = cursor.execute(
                """
                SELECT 1 FROM slots
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                  AND task_group_revision=? AND state=?
                LIMIT 1
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    triggering_group_revision,
                    SlotState.FAILED.value,
                ),
            ).fetchone()
            if exhausted is None:
                return None
            cursor.execute(
                """
                INSERT INTO replacement_intents(
                    run_id, train_step_id, task_group_id,
                    triggering_group_revision, status,
                    replacement_task_json, replacement_slots_json,
                    last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, '[]', NULL, ?, ?)
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    triggering_group_revision,
                    ReplacementStatus.PENDING.value,
                    timestamp,
                    timestamp,
                ),
            )
            return self._replacement_intent_from_row(
                cursor.execute(
                    """
                    SELECT * FROM replacement_intents
                    WHERE run_id=? AND train_step_id=? AND task_group_id=?
                      AND triggering_group_revision=?
                    """,
                    (
                        batch_id.run_id,
                        batch_id.train_step_id,
                        task_group_id,
                        triggering_group_revision,
                    ),
                ).fetchone()
            )

    def store_replacement_payload(
        self,
        intent: ReplacementIntent,
        *,
        replacement_task: Mapping[str, Any],
        replacement_slots: Sequence[ManifestSlot],
    ) -> ReplacementIntent:
        slots = tuple(replacement_slots)
        task_json = self._json(dict(replacement_task), "replacement task")
        slots_json = self._json(
            [slot.to_dict() for slot in slots],
            "replacement slots",
        )
        timestamp = float(self._clock())
        with self._transaction() as cursor:
            row = self._require_replacement_intent_row(cursor, intent)
            status = ReplacementStatus(row["status"])
            if status != ReplacementStatus.PENDING:
                if (
                    row["replacement_task_json"] == task_json
                    and row["replacement_slots_json"] == slots_json
                ):
                    return self._replacement_intent_from_row(row)
                raise ProtocolError(
                    ErrorCode.IDEMPOTENCY_CONFLICT,
                    "replacement payload changed after it was persisted",
                    {
                        **intent.batch_id.to_dict(),
                        "task_group_id": intent.task_group_id,
                        "triggering_group_revision": (
                            intent.triggering_group_revision
                        ),
                    },
                )
            cursor.execute(
                """
                UPDATE replacement_intents
                SET status=?, replacement_task_json=?,
                    replacement_slots_json=?, last_error=NULL, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                  AND triggering_group_revision=?
                """,
                (
                    ReplacementStatus.CLOSING.value,
                    task_json,
                    slots_json,
                    timestamp,
                    intent.batch_id.run_id,
                    intent.batch_id.train_step_id,
                    intent.task_group_id,
                    intent.triggering_group_revision,
                ),
            )
            return self._replacement_intent_from_row(
                self._require_replacement_intent_row(cursor, intent)
            )

    def update_replacement_status(
        self,
        intent: ReplacementIntent,
        target: ReplacementStatus,
        *,
        error: str | None = None,
    ) -> ReplacementIntent:
        target_status = self._coerce_enum(
            ReplacementStatus, target, "replacement status"
        )
        timestamp = float(self._clock())
        with self._transaction() as cursor:
            row = self._require_replacement_intent_row(cursor, intent)
            current = ReplacementStatus(row["status"])
            if current == target_status:
                return self._replacement_intent_from_row(row)
            allowed = {
                ReplacementStatus.PENDING: {
                    ReplacementStatus.CLOSING,
                    ReplacementStatus.FAILED,
                },
                ReplacementStatus.CLOSING: {
                    ReplacementStatus.READY_TO_COMMIT,
                    ReplacementStatus.FAILED,
                },
                ReplacementStatus.READY_TO_COMMIT: {
                    ReplacementStatus.COMPLETED,
                    ReplacementStatus.FAILED,
                },
                ReplacementStatus.FAILED: {
                    ReplacementStatus.PENDING,
                },
                ReplacementStatus.COMPLETED: set(),
            }
            if target_status not in allowed[current]:
                raise ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "invalid replacement intent transition",
                    {"current": current.value, "target": target_status.value},
                )
            cursor.execute(
                """
                UPDATE replacement_intents
                SET status=?, last_error=?, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                  AND triggering_group_revision=?
                """,
                (
                    target_status.value,
                    error,
                    timestamp,
                    intent.batch_id.run_id,
                    intent.batch_id.train_step_id,
                    intent.task_group_id,
                    intent.triggering_group_revision,
                ),
            )
            return self._replacement_intent_from_row(
                self._require_replacement_intent_row(cursor, intent)
            )

    def list_replacement_intents(
        self,
        *,
        statuses: Sequence[ReplacementStatus] | None = None,
    ) -> tuple[ReplacementIntent, ...]:
        query = "SELECT * FROM replacement_intents"
        parameters: list[Any] = []
        if statuses:
            values = [
                self._coerce_enum(
                    ReplacementStatus, status, "replacement status"
                ).value
                for status in statuses
            ]
            query += f" WHERE status IN ({','.join('?' for _ in values)})"
            parameters.extend(values)
        query += (
            " ORDER BY created_at, run_id, train_step_id, task_group_id,"
            " triggering_group_revision"
        )
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._replacement_intent_from_row(row) for row in rows)

    def list_exhausted_slots(self) -> tuple[SlotRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT s.* FROM slots s
                JOIN batches b
                  ON b.run_id=s.run_id AND b.train_step_id=s.train_step_id
                WHERE s.state=? AND b.state IN (?, ?, ?)
                ORDER BY s.updated_at, s.run_id, s.train_step_id,
                         s.task_index, s.actor_id
                """,
                (
                    SlotState.FAILED.value,
                    BatchState.RUNNING.value,
                    BatchState.DRAINING.value,
                    BatchState.RECOVERING.value,
                ),
            ).fetchall()
        return tuple(self._slot_from_row(row) for row in rows)

    def replacement_group_leases_terminal(
        self,
        intent: ReplacementIntent,
    ) -> bool:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT (
                    SELECT COUNT(*)
                    FROM actual_leases l
                    JOIN slots s
                      ON s.run_id=l.run_id AND s.train_step_id=l.train_step_id
                     AND s.task_index=l.task_index AND s.actor_id=l.actor_id
                    WHERE l.run_id=? AND l.train_step_id=?
                      AND s.task_group_id=?
                      AND l.status IN (?, ?)
                ) + (
                    SELECT COUNT(*)
                    FROM allocation_cleanup_outbox_v2 a
                    JOIN slots s
                      ON s.run_id=a.run_id AND s.train_step_id=a.train_step_id
                     AND s.task_index=a.task_index AND s.actor_id=a.actor_id
                    WHERE a.run_id=? AND a.train_step_id=?
                      AND s.task_group_id=?
                      AND a.status IN ('ALLOCATING', 'PENDING')
                ) AS count
                """,
                (
                    intent.batch_id.run_id,
                    intent.batch_id.train_step_id,
                    intent.task_group_id,
                    LeaseStatus.ACTIVE.value,
                    LeaseStatus.CLOSING.value,
                    intent.batch_id.run_id,
                    intent.batch_id.train_step_id,
                    intent.task_group_id,
                ),
            ).fetchone()
        return int(row["count"] if row is not None else 0) == 0

    def replace_task_group(
        self,
        batch_id: BatchId,
        task_group_id: str,
        replacement_slots: Sequence[ManifestSlot],
        *,
        request_id: str,
        expected_batch_version: int,
    ) -> TaskGroupReplacement:
        if not task_group_id or not request_id:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "task_group_id and request_id must not be empty",
            )
        replacements = tuple(replacement_slots)
        if not replacements:
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "replacement task group must not be empty")
        for slot in replacements:
            if slot.task_group_id != task_group_id:
                raise ProtocolError(
                    ErrorCode.VALIDATION_ERROR,
                    "replacement slot belongs to a different task group",
                    {"expected": task_group_id, "actual": slot.task_group_id},
                )
        payload_hash = self._payload_hash(
            {
                "batch_id": batch_id.to_dict(),
                "task_group_id": task_group_id,
                "slots": [slot.to_dict() for slot in replacements],
                "expected_batch_version": expected_batch_version,
            }
        )
        now = float(self._clock())
        with self._transaction() as cursor:
            replay = self._idempotency_lookup(cursor, "replace_task_group", request_id, payload_hash)
            if replay is not None:
                return self._replacement_from_dict(replay)
            batch_row = self._require_batch_row(cursor, batch_id)
            self._check_version(
                "batch",
                expected_batch_version,
                int(batch_row["version"]),
                batch_id.to_dict(),
            )
            old_manifest_row = cursor.execute(
                """
                SELECT * FROM manifest_revisions
                WHERE run_id=? AND train_step_id=? AND revision=?
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    int(batch_row["current_manifest_revision"]),
                ),
            ).fetchone()
            if old_manifest_row is None:
                raise self._not_found("current manifest", batch_id.to_dict())
            old_manifest = tuple(
                ManifestSlot.from_dict(value) for value in json.loads(old_manifest_row["manifest_json"])
            )
            old_group_slots = tuple(
                slot for slot in old_manifest if slot.task_group_id == task_group_id
            )
            if not old_group_slots:
                raise self._not_found(
                    "task group",
                    {**batch_id.to_dict(), "task_group_id": task_group_id},
                )
            old_identities = {
                (slot.task_index, slot.actor_id) for slot in old_group_slots
            }
            replacement_identities = {
                (slot.task_index, slot.actor_id) for slot in replacements
            }
            if replacement_identities != old_identities or (
                len(replacement_identities) != len(replacements)
            ):
                raise ProtocolError(
                    ErrorCode.VALIDATION_ERROR,
                    "replacement task group must preserve exact slot membership",
                    {
                        "expected": sorted(old_identities),
                        "actual": sorted(replacement_identities),
                    },
                )
            new_manifest = validate_manifest(
                [slot for slot in old_manifest if slot.task_group_id != task_group_id] + list(replacements)
            )
            current_manifest_revision = int(batch_row["current_manifest_revision"])
            new_manifest_revision = current_manifest_revision + 1
            manifest_json = self._json([slot.to_dict() for slot in new_manifest], "manifest")
            manifest_hash = self._hash_text(manifest_json)
            cursor.execute(
                """
                INSERT INTO manifest_revisions(
                    run_id, train_step_id, revision, payload_hash, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    new_manifest_revision,
                    manifest_hash,
                    manifest_json,
                    now,
                ),
            )
            old_group_row = cursor.execute(
                """
                SELECT MAX(group_revision) AS revision FROM task_groups
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                """,
                (batch_id.run_id, batch_id.train_step_id, task_group_id),
            ).fetchone()
            old_group_revision = int(old_group_row["revision"] or 0)
            if old_group_revision <= 0:
                raise self._not_found(
                    "task group revision",
                    {**batch_id.to_dict(), "task_group_id": task_group_id},
                )
            new_group_revision = old_group_revision + 1
            cursor.execute(
                """
                UPDATE task_groups SET superseded_at=?, replaced_by_revision=?
                WHERE run_id=? AND train_step_id=? AND task_group_id=? AND group_revision=?
                """,
                (
                    now,
                    new_group_revision,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    old_group_revision,
                ),
            )
            self._insert_task_group(
                cursor,
                batch_id,
                task_group_id,
                new_group_revision,
                new_manifest_revision,
                replacements,
                now,
            )
            old_rows = cursor.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                """,
                (batch_id.run_id, batch_id.train_step_id, task_group_id),
            ).fetchall()
            old_by_identity = {
                (int(row["task_index"]), str(row["actor_id"])): row for row in old_rows
            }

            for row in old_rows:
                current = SlotState(row["state"])
                if current in {SlotState.ACKING, SlotState.DONE}:
                    raise ProtocolError(
                        ErrorCode.INVALID_TRANSITION,
                        "cannot replace a task group with acknowledged slots",
                        {"task_group_id": task_group_id, "state": current.value},
                    )
                validate_slot_transition(current, SlotState.SUPERSEDED)
            for row in old_rows:
                key = self._key_from_row(row)
                self._fence_slot_work(cursor, key, int(row["claim_generation"]), now)
                if (key.task_index, key.actor_id) not in replacement_identities:
                    current = SlotState(row["state"])
                    if current != SlotState.SUPERSEDED:
                        validate_slot_transition(current, SlotState.SUPERSEDED)
                        version = int(row["version"]) + 1
                        cursor.execute(
                            """
                            UPDATE slots
                            SET state=?, claim_generation=claim_generation+1,
                                version=?, updated_at=?
                            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                            """,
                            (SlotState.SUPERSEDED.value, version, now, *self._key_values(key)),
                        )
                        self._event(
                            cursor,
                            batch_id,
                            "slot",
                            SlotState.SUPERSEDED.value,
                            version,
                            now,
                            key=key,
                            from_state=current.value,
                            reason="task-group-replaced",
                        )
            for slot in replacements:
                identity = (slot.task_index, slot.actor_id)
                key = slot.key(batch_id)
                existing = old_by_identity.get(identity)
                if existing is None:
                    collision = cursor.execute(
                        """
                        SELECT task_group_id FROM slots
                        WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                        """,
                        self._key_values(key),
                    ).fetchone()
                    if collision is not None:
                        raise ProtocolError(
                            ErrorCode.VALIDATION_ERROR,
                            "replacement stable identity belongs to another task group",
                            {**key.to_dict(), "task_group_id": collision["task_group_id"]},
                        )
                    cursor.execute(
                        """
                        INSERT INTO slots(
                            run_id, train_step_id, task_index, actor_id, task_id,
                            task_group_id, task_group_revision, manifest_revision,
                            state, version, current_attempt, claim_generation,
                            artifact_digest, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 0, NULL, ?, ?, ?)
                        """,
                        (
                            *self._key_values(key),
                            slot.task_id,
                            task_group_id,
                            new_group_revision,
                            new_manifest_revision,
                            SlotState.PENDING.value,
                            self._json(dict(slot.metadata), "slot metadata"),
                            now,
                            now,
                        ),
                    )
                    self._event(
                        cursor,
                        batch_id,
                        "slot",
                        SlotState.PENDING.value,
                        1,
                        now,
                        key=key,
                        reason="task-group-replacement-created",
                    )
                else:
                    old_state = SlotState(existing["state"])

                    version = int(existing["version"]) + 2
                    cursor.execute(
                        """
                        UPDATE slots
                        SET task_id=?, task_group_revision=?, manifest_revision=?,
                            state=?, version=?, claim_generation=claim_generation+1,
                            artifact_digest=NULL, metadata_json=?, updated_at=?
                        WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                        """,
                        (
                            slot.task_id,
                            new_group_revision,
                            new_manifest_revision,
                            SlotState.PENDING.value,
                            version,
                            self._json(dict(slot.metadata), "slot metadata"),
                            now,
                            *self._key_values(key),
                        ),
                    )
                    self._event(
                        cursor,
                        batch_id,
                        "slot",
                        SlotState.SUPERSEDED.value,
                        version - 1,
                        now,
                        key=key,
                        from_state=old_state.value,
                        reason="task-group-replaced",
                    )
                    self._event(
                        cursor,
                        batch_id,
                        "slot",
                        SlotState.PENDING.value,
                        version,
                        now,
                        key=key,
                        reason="task-group-replacement-created",
                    )
            cursor.execute(
                """
                UPDATE slots SET manifest_revision=?, version=version+1, updated_at=?
                WHERE run_id=? AND train_step_id=? AND task_group_id<>? AND state<>?
                """,
                (
                    new_manifest_revision,
                    now,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    SlotState.SUPERSEDED.value,
                ),
            )
            cursor.execute(
                """
                UPDATE batches
                SET current_manifest_revision=?, expected_slots=?, version=version+1, updated_at=?
                WHERE run_id=? AND train_step_id=? AND version=?
                """,
                (
                    new_manifest_revision,
                    len(new_manifest),
                    now,
                    batch_id.run_id,
                    batch_id.train_step_id,
                    expected_batch_version,
                ),
            )
            if cursor.rowcount != 1:
                raise self._version_conflict("batch", expected_batch_version, batch_id.to_dict())
            active_rows = cursor.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=? AND task_group_id=?
                  AND task_group_revision=? AND state<>?
                ORDER BY task_index, actor_id
                """,
                (
                    batch_id.run_id,
                    batch_id.train_step_id,
                    task_group_id,
                    new_group_revision,
                    SlotState.SUPERSEDED.value,
                ),
            ).fetchall()
            result = TaskGroupReplacement(
                batch_id=batch_id,
                task_group_id=task_group_id,
                task_group_revision=new_group_revision,
                manifest_revision=new_manifest_revision,
                superseded_slots=len(old_rows),
                active_slots=tuple(self._slot_from_row(row) for row in active_rows),
            )
            response = self._replacement_to_dict(result)
            self._idempotency_store(
                cursor,
                "replace_task_group",
                request_id,
                payload_hash,
                response,
                now,
            )
            return result

    def validate_expected_slot_completeness(self, batch_id: BatchId) -> CompletenessReport:
        manifest = self.get_manifest(batch_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM slots
                WHERE run_id=? AND train_step_id=? AND manifest_revision=?
                """,
                (batch_id.run_id, batch_id.train_step_id, manifest.revision),
            ).fetchall()
        by_identity = {
            (int(row["task_index"]), str(row["actor_id"])): row for row in rows
        }
        missing: list[SlotKey] = []
        not_done: list[SlotKey] = []
        done = 0
        for slot in manifest.slots:
            key = slot.key(batch_id)
            row = by_identity.get((slot.task_index, slot.actor_id))
            if row is None:
                missing.append(key)
            elif SlotState(row["state"]) == SlotState.DONE and row["artifact_digest"]:
                done += 1
            else:
                not_done.append(key)
        return CompletenessReport(
            batch_id=batch_id,
            manifest_revision=manifest.revision,
            expected=len(manifest.slots),
            present=len(manifest.slots) - len(missing),
            done=done,
            missing=tuple(missing),
            not_done=tuple(not_done),
        )

    validate_completeness = validate_expected_slot_completeness

    def inspect_batch(self, batch_id: BatchId) -> BatchInspection:
        batch = self.get_batch(batch_id)
        slots = self.list_slots(batch_id)
        return BatchInspection(
            batch=batch,
            slots=slots,
            counts=self.slot_counts(batch_id),
            completeness=self.validate_expected_slot_completeness(batch_id),
        )

    def list_state_events(self, batch_id: BatchId) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM state_events
                WHERE run_id=? AND train_step_id=? ORDER BY event_id
                """,
                (batch_id.run_id, batch_id.train_step_id),
            ).fetchall()
        return tuple(
            {
                "event_id": int(row["event_id"]),
                "entity_type": row["entity_type"],
                "task_index": row["task_index"],
                "actor_id": row["actor_id"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "entity_version": int(row["entity_version"]),
                "reason": row["reason"],
                "details": json.loads(row["details_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        )

    def get_idempotent_response(
        self,
        namespace: str,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        payload_hash = self._payload_hash(dict(payload))
        with self._transaction() as cursor:
            return self._idempotency_lookup(
                cursor,
                namespace,
                request_id,
                payload_hash,
            )

    def put_idempotent_response(
        self,
        namespace: str,
        request_id: str,
        payload: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload_hash = self._payload_hash(dict(payload))
        now = float(self._clock())
        with self._transaction() as cursor:
            replay = self._idempotency_lookup(
                cursor,
                namespace,
                request_id,
                payload_hash,
            )
            if replay is not None:
                return replay
            self._idempotency_store(
                cursor,
                namespace,
                request_id,
                payload_hash,
                response,
                now,
            )
            return dict(response)

    def _recover_expired_claims(
        self,
        cursor: sqlite3.Cursor,
        now: float,
        batch_id: BatchId | None,
    ) -> RecoveryReport:
        query = "SELECT * FROM claims WHERE status=? AND expires_at<=?"
        parameters: list[Any] = [ClaimStatus.ACTIVE.value, now]
        if batch_id is not None:
            query += " AND run_id=? AND train_step_id=?"
            parameters.extend([batch_id.run_id, batch_id.train_step_id])
        rows = cursor.execute(query, parameters).fetchall()
        released = 0
        quarantined = 0
        for row in rows:
            quarantined += self._expire_claim(cursor, row, now)
        return RecoveryReport(
            expired_claims=len(rows),
            released_slots=released,
            quarantined_leases=quarantined,
        )

    def _expire_claim(self, cursor: sqlite3.Cursor, claim_row: sqlite3.Row, now: float) -> int:
        self._quarantine_claim(cursor, claim_row, now, "claim-expired")
        return 1

    def _quarantine_claim(
        self,
        cursor: sqlite3.Cursor,
        claim_row: sqlite3.Row,
        now: float,
        reason: str,
    ) -> CloseIntent:
        claim_id = str(claim_row["claim_id"])
        generation = int(claim_row["generation"])
        key = self._key_from_row(claim_row)
        slot_row = self._require_slot_row(cursor, key)
        if (
            int(slot_row["claim_generation"]) != generation
            or SlotState(slot_row["state"]) != SlotState.CLAIMED
        ):
            raise self._fenced("slot no longer belongs to claim", {"claim_id": claim_id})
        lease_row = cursor.execute(
            "SELECT * FROM actual_leases WHERE claim_id=?",
            (claim_id,),
        ).fetchone()
        if lease_row is None:
            raise self._fenced(
                "claim has no real environment lease to quarantine",
                {"claim_id": claim_id},
            )
        existing = cursor.execute(
            "SELECT * FROM lease_close_outbox_v2 WHERE lease_id=?",
            (lease_row["lease_id"],),
        ).fetchone()
        if existing is not None:
            return self._close_intent_from_row(existing)
        claim_status = (
            ClaimStatus.EXPIRED if reason == "claim-expired" else ClaimStatus.LOST
        )
        cursor.execute(
            "UPDATE claims SET status=?, updated_at=? WHERE claim_id=?",
            (claim_status.value, now, claim_id),
        )
        cursor.execute(
            "UPDATE actual_leases SET status=?, updated_at=? WHERE lease_id=?",
            (LeaseStatus.CLOSING.value, now, lease_row["lease_id"]),
        )
        version = int(slot_row["version"]) + 1
        cursor.execute(
            """
            UPDATE slots SET state=?, version=?, updated_at=?
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
            """,
            (SlotState.ACKING.value, version, now, *self._key_values(key)),
        )
        intent_id = uuid.uuid4().hex
        operation_id = f"close:{lease_row['lease_id']}:{lease_row['attempt']}"
        cursor.execute(
            """
            INSERT INTO lease_close_outbox_v2(
                intent_id, lease_id, claim_id, run_id, train_step_id,
                task_index, actor_id, attempt, generation, fence_token,
                operation_id, action, reason, artifact_digest, status,
                close_attempts, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'RETRY', ?, NULL,
                      ?, 0, NULL, ?, ?)
            """,
            (
                intent_id,
                str(lease_row["lease_id"]),
                claim_id,
                *self._key_values(key),
                int(lease_row["attempt"]),
                generation,
                str(lease_row["fence_token"]),
                operation_id,
                reason,
                OutboxStatus.PENDING.value,
                now,
                now,
            ),
        )
        self._event(
            cursor,
            key.batch_id,
            "slot",
            SlotState.ACKING.value,
            version,
            now,
            key=key,
            from_state=SlotState.CLAIMED.value,
            reason=reason,
            details={
                "claim_id": claim_id,
                "generation": generation,
                "lease_id": lease_row["lease_id"],
            },
        )
        return self._close_intent_from_row(
            cursor.execute(
                "SELECT * FROM lease_close_outbox_v2 WHERE intent_id=?",
                (intent_id,),
            ).fetchone()
        )

    def _fence_slot_work(
        self,
        cursor: sqlite3.Cursor,
        key: SlotKey,
        generation: int,
        now: float,
    ) -> None:
        cursor.execute(
            """
            UPDATE attempts SET status=?, finished_at=?
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=? AND status=?
            """,
            (
                AttemptStatus.SUPERSEDED.value,
                now,
                *self._key_values(key),
                AttemptStatus.RUNNING.value,
            ),
        )
        claims = cursor.execute(
            """
            SELECT claim_id FROM claims
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
              AND generation=? AND status IN (?, ?)
            """,
            (
                *self._key_values(key),
                generation,
                ClaimStatus.ACTIVE.value,
                ClaimStatus.ACKING.value,
            ),
        ).fetchall()
        for claim in claims:
            claim_id = claim["claim_id"]
            cursor.execute(
                "UPDATE claims SET status=?, updated_at=? WHERE claim_id=?",
                (ClaimStatus.SUPERSEDED.value, now, claim_id),
            )
            cursor.execute(
                "UPDATE leases SET status=?, updated_at=? WHERE claim_id=?",
                (LeaseStatus.LOST.value, now, claim_id),
            )
            cursor.execute(
                "UPDATE close_outbox SET status=?, updated_at=? WHERE claim_id=?",
                (OutboxStatus.LOST.value, now, claim_id),
            )

    def _insert_task_group(
        self,
        cursor: sqlite3.Cursor,
        batch_id: BatchId,
        group_id: str,
        group_revision: int,
        manifest_revision: int,
        slots: Sequence[ManifestSlot],
        now: float,
    ) -> None:
        payload_json = self._json([slot.to_dict() for slot in slots], "task group")
        cursor.execute(
            """
            INSERT INTO task_groups(
                run_id, train_step_id, task_group_id, group_revision,
                manifest_revision, payload_json, payload_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id.run_id,
                batch_id.train_step_id,
                group_id,
                group_revision,
                manifest_revision,
                payload_json,
                self._hash_text(payload_json),
                now,
            ),
        )

    def _register_artifact(
        self,
        cursor: sqlite3.Cursor,
        artifact: ArtifactRef,
        now: float,
    ) -> None:
        if (
            artifact.size < 0
            or len(artifact.digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact.digest)
            or not artifact.uri
        ):
            raise ProtocolError(ErrorCode.VALIDATION_ERROR, "invalid artifact reference")
        existing = cursor.execute(
            "SELECT * FROM artifact_refs WHERE digest=?",
            (artifact.digest,),
        ).fetchone()
        if existing is not None and (
            int(existing["size"]) != artifact.size
            or existing["uri"] != artifact.uri
            or existing["media_type"] != artifact.media_type
        ):
            raise ProtocolError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "artifact digest is already registered with different metadata",
                {"digest": artifact.digest},
            )
        cursor.execute(
            """
            INSERT OR IGNORE INTO artifact_refs(digest, size, uri, media_type, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                artifact.digest,
                artifact.size,
                artifact.uri,
                artifact.media_type,
                now,
            ),
        )

    def _idempotency_lookup(
        self,
        cursor: sqlite3.Cursor,
        namespace: str,
        request_id: str,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        row = cursor.execute(
            """
            SELECT payload_hash, response_json FROM idempotency_records
            WHERE namespace=? AND request_id=?
            """,
            (namespace, request_id),
        ).fetchone()
        if row is None:
            return None
        if row["payload_hash"] != payload_hash:
            raise ProtocolError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                "request ID was already used with a different payload",
                {"namespace": namespace, "request_id": request_id},
            )
        return dict(json.loads(row["response_json"]))

    def _idempotency_store(
        self,
        cursor: sqlite3.Cursor,
        namespace: str,
        request_id: str,
        payload_hash: str,
        response: Mapping[str, Any],
        now: float,
    ) -> None:
        response_json = self._json(dict(response), "idempotency response")
        cursor.execute(
            """
            INSERT INTO idempotency_records(
                namespace, request_id, payload_hash, response_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (namespace, request_id, payload_hash, response_json, now),
        )

    def _event(
        self,
        cursor: sqlite3.Cursor,
        batch_id: BatchId,
        entity_type: str,
        to_state: str,
        version: int,
        now: float,
        *,
        key: SlotKey | None = None,
        from_state: str | None = None,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO state_events(
                run_id, train_step_id, task_index, actor_id, entity_type,
                from_state, to_state, entity_version, reason, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id.run_id,
                batch_id.train_step_id,
                key.task_index if key else None,
                key.actor_id if key else None,
                entity_type,
                from_state,
                to_state,
                version,
                reason,
                self._json(dict(details or {}), "state event"),
                now,
            ),
        )

    def _require_batch_row(self, cursor: sqlite3.Cursor, batch_id: BatchId) -> sqlite3.Row:
        row = cursor.execute(
            "SELECT * FROM batches WHERE run_id=? AND train_step_id=?",
            (batch_id.run_id, batch_id.train_step_id),
        ).fetchone()
        if row is None:
            raise self._not_found("batch", batch_id.to_dict())
        return row

    def _require_slot_row(self, cursor: sqlite3.Cursor, key: SlotKey) -> sqlite3.Row:
        row = cursor.execute(
            """
            SELECT * FROM slots
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
            """,
            self._key_values(key),
        ).fetchone()
        if row is None:
            raise self._not_found("slot", key.to_dict())
        return row

    @staticmethod
    def _require_claim_row(cursor: sqlite3.Cursor, claim_id: str) -> sqlite3.Row:
        row = cursor.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        if row is None:
            raise ProtocolError(ErrorCode.NOT_FOUND, "claim was not found", {"claim_id": claim_id})
        return row

    @staticmethod
    def _require_replacement_intent_row(
        cursor: sqlite3.Cursor,
        intent: ReplacementIntent,
    ) -> sqlite3.Row:
        row = cursor.execute(
            """
            SELECT * FROM replacement_intents
            WHERE run_id=? AND train_step_id=? AND task_group_id=?
              AND triggering_group_revision=?
            """,
            (
                intent.batch_id.run_id,
                intent.batch_id.train_step_id,
                intent.task_group_id,
                intent.triggering_group_revision,
            ),
        ).fetchone()
        if row is None:
            raise ProtocolError(
                ErrorCode.NOT_FOUND,
                "replacement intent was not found",
                {
                    **intent.batch_id.to_dict(),
                    "task_group_id": intent.task_group_id,
                    "triggering_group_revision": (
                        intent.triggering_group_revision
                    ),
                },
            )
        return row

    @staticmethod
    def _batch_from_row(row: sqlite3.Row) -> BatchRecord:
        return BatchRecord(
            batch_id=BatchId(str(row["run_id"]), int(row["train_step_id"])),
            state=BatchState(row["state"]),
            version=int(row["version"]),
            manifest_revision=int(row["current_manifest_revision"]),
            expected_slots=int(row["expected_slots"]),
            metadata=dict(json.loads(row["metadata_json"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _slot_from_row(row: sqlite3.Row) -> SlotRecord:
        key = SQLiteStateStore._key_from_row(row)
        return SlotRecord(
            key=key,
            task_id=str(row["task_id"]),
            task_group_id=str(row["task_group_id"]),
            task_group_revision=int(row["task_group_revision"]),
            manifest_revision=int(row["manifest_revision"]),
            state=SlotState(row["state"]),
            version=int(row["version"]),
            current_attempt=int(row["current_attempt"]),
            claim_generation=int(row["claim_generation"]),
            artifact_digest=row["artifact_digest"],
            metadata=dict(json.loads(row["metadata_json"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _claim_from_row(row: sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            claim_id=str(row["claim_id"]),
            key=SQLiteStateStore._key_from_row(row),
            generation=int(row["generation"]),
            status=ClaimStatus(row["status"]),
            consumer_id=str(row["consumer_id"]),
            expires_at=float(row["expires_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _lease_from_row(row: sqlite3.Row) -> LeaseRecord:
        return LeaseRecord(
            lease_id=str(row["lease_id"]),
            key=SQLiteStateStore._key_from_row(row),
            attempt=int(row["attempt"]),
            fence_token=str(row["fence_token"]),
            allocation_key=str(row["allocation_key"]),
            owner=dict(json.loads(row["owner_json"])),
            status=LeaseStatus(row["status"]),
            claim_id=str(row["claim_id"]) if row["claim_id"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _allocation_cleanup_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "allocation_key": str(row["allocation_key"]),
            "key": SQLiteStateStore._key_from_row(row),
            "attempt": int(row["attempt"]),
            "fence_token": str(row["fence_token"]),
            "owner": dict(json.loads(row["owner_json"])),
            "operation_id": str(row["operation_id"]),
            "action": str(row["action"]),
            "reason": str(row["reason"]),
            "status": str(row["status"]),
            "cleanup_attempts": int(row["cleanup_attempts"]),
            "last_error": row["last_error"],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> ArtifactRef:
        return ArtifactRef(
            digest=str(row["digest"]),
            size=int(row["size"]),
            uri=str(row["uri"]),
            media_type=str(row["media_type"]),
        )

    @staticmethod
    def _close_intent_from_row(row: sqlite3.Row) -> CloseIntent:
        return CloseIntent(
            intent_id=str(row["intent_id"]),
            claim_id=str(row["claim_id"]) if row["claim_id"] is not None else None,
            key=SQLiteStateStore._key_from_row(row),
            generation=int(row["generation"]) if row["generation"] is not None else None,
            lease_id=str(row["lease_id"]),
            fence_token=str(row["fence_token"]),
            attempt=int(row["attempt"]),
            operation_id=str(row["operation_id"]),
            action=str(row["action"]),
            reason=str(row["reason"]),
            artifact_digest=(
                str(row["artifact_digest"])
                if row["artifact_digest"] is not None
                else None
            ),
            status=OutboxStatus(row["status"]),
            close_attempts=int(row["close_attempts"]),
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _replacement_intent_from_row(row: sqlite3.Row) -> ReplacementIntent:
        task_json = row["replacement_task_json"]
        slots_json = row["replacement_slots_json"]
        return ReplacementIntent(
            batch_id=BatchId(
                str(row["run_id"]),
                int(row["train_step_id"]),
            ),
            task_group_id=str(row["task_group_id"]),
            triggering_group_revision=int(
                row["triggering_group_revision"]
            ),
            status=ReplacementStatus(row["status"]),
            replacement_task=(
                dict(json.loads(task_json))
                if task_json is not None
                else None
            ),
            replacement_slots=tuple(
                ManifestSlot.from_dict(value)
                for value in json.loads(slots_json or "[]")
            ),
            last_error=(
                str(row["last_error"])
                if row["last_error"] is not None
                else None
            ),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _close_intent_to_dict(intent: CloseIntent) -> dict[str, Any]:
        return {
            "intent_id": intent.intent_id,
            "claim_id": intent.claim_id,
            "key": intent.key.to_dict(),
            "generation": intent.generation,
            "lease_id": intent.lease_id,
            "fence_token": intent.fence_token,
            "attempt": intent.attempt,
            "operation_id": intent.operation_id,
            "action": intent.action,
            "reason": intent.reason,
            "artifact_digest": intent.artifact_digest,
            "status": intent.status.value,
            "close_attempts": intent.close_attempts,
            "last_error": intent.last_error,
            "created_at": intent.created_at,
            "updated_at": intent.updated_at,
        }

    @staticmethod
    def _close_intent_from_dict(value: Mapping[str, Any]) -> CloseIntent:
        key_value = value["key"]
        return CloseIntent(
            intent_id=str(value["intent_id"]),
            claim_id=(
                str(value["claim_id"])
                if value.get("claim_id") is not None
                else None
            ),
            key=SlotKey(
                BatchId(
                    str(key_value["run_id"]),
                    int(key_value["train_step_id"]),
                ),
                int(key_value["task_index"]),
                str(key_value["actor_id"]),
            ),
            generation=(
                int(value["generation"])
                if value.get("generation") is not None
                else None
            ),
            lease_id=str(value["lease_id"]),
            fence_token=str(value["fence_token"]),
            attempt=int(value["attempt"]),
            operation_id=str(value["operation_id"]),
            action=str(value["action"]),
            reason=str(value["reason"]),
            artifact_digest=(
                str(value["artifact_digest"])
                if value.get("artifact_digest") is not None
                else None
            ),
            status=OutboxStatus(value["status"]),
            close_attempts=int(value["close_attempts"]),
            last_error=(
                str(value["last_error"])
                if value.get("last_error") is not None
                else None
            ),
            created_at=float(value["created_at"]),
            updated_at=float(value["updated_at"]),
        )

    @staticmethod
    def _key_from_row(row: sqlite3.Row) -> SlotKey:
        return SlotKey(
            BatchId(str(row["run_id"]), int(row["train_step_id"])),
            int(row["task_index"]),
            str(row["actor_id"]),
        )

    @staticmethod
    def _key_values(key: SlotKey) -> tuple[Any, ...]:
        return (
            key.batch_id.run_id,
            key.batch_id.train_step_id,
            key.task_index,
            key.actor_id,
        )

    @staticmethod
    def _batch_to_dict(record: BatchRecord) -> dict[str, Any]:
        return {
            "batch_id": record.batch_id.to_dict(),
            "state": record.state.value,
            "version": record.version,
            "manifest_revision": record.manifest_revision,
            "expected_slots": record.expected_slots,
            "metadata": dict(record.metadata),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _batch_from_dict(value: Mapping[str, Any]) -> BatchRecord:
        batch = value["batch_id"]
        return BatchRecord(
            batch_id=BatchId(str(batch["run_id"]), int(batch["train_step_id"])),
            state=BatchState(value["state"]),
            version=int(value["version"]),
            manifest_revision=int(value["manifest_revision"]),
            expected_slots=int(value["expected_slots"]),
            metadata=dict(value.get("metadata") or {}),
            created_at=float(value["created_at"]),
            updated_at=float(value["updated_at"]),
        )

    @staticmethod
    def _slot_to_dict(record: SlotRecord) -> dict[str, Any]:
        return {
            "key": record.key.to_dict(),
            "task_id": record.task_id,
            "task_group_id": record.task_group_id,
            "task_group_revision": record.task_group_revision,
            "manifest_revision": record.manifest_revision,
            "state": record.state.value,
            "version": record.version,
            "current_attempt": record.current_attempt,
            "claim_generation": record.claim_generation,
            "artifact_digest": record.artifact_digest,
            "metadata": dict(record.metadata),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _slot_from_dict(value: Mapping[str, Any]) -> SlotRecord:
        key_value = value["key"]
        key = SlotKey(
            BatchId(str(key_value["run_id"]), int(key_value["train_step_id"])),
            int(key_value["task_index"]),
            str(key_value["actor_id"]),
        )
        return SlotRecord(
            key=key,
            task_id=str(value["task_id"]),
            task_group_id=str(value["task_group_id"]),
            task_group_revision=int(value["task_group_revision"]),
            manifest_revision=int(value["manifest_revision"]),
            state=SlotState(value["state"]),
            version=int(value["version"]),
            current_attempt=int(value["current_attempt"]),
            claim_generation=int(value["claim_generation"]),
            artifact_digest=value.get("artifact_digest"),
            metadata=dict(value.get("metadata") or {}),
            created_at=float(value["created_at"]),
            updated_at=float(value["updated_at"]),
        )

    @staticmethod
    def _claim_to_dict(record: ClaimRecord) -> dict[str, Any]:
        return {
            "claim_id": record.claim_id,
            "key": record.key.to_dict(),
            "generation": record.generation,
            "status": record.status.value,
            "consumer_id": record.consumer_id,
            "expires_at": record.expires_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _claim_from_dict(value: Mapping[str, Any]) -> ClaimRecord:
        key_value = value["key"]
        return ClaimRecord(
            claim_id=str(value["claim_id"]),
            key=SlotKey(
                BatchId(str(key_value["run_id"]), int(key_value["train_step_id"])),
                int(key_value["task_index"]),
                str(key_value["actor_id"]),
            ),
            generation=int(value["generation"]),
            status=ClaimStatus(value["status"]),
            consumer_id=str(value["consumer_id"]),
            expires_at=float(value["expires_at"]),
            created_at=float(value["created_at"]),
            updated_at=float(value["updated_at"]),
        )

    @staticmethod
    def _ack_to_dict(result: AckResult) -> dict[str, Any]:
        return {
            "intent_id": result.intent_id,
            "claim_id": result.claim_id,
            "key": result.key.to_dict(),
            "artifact": result.artifact.to_dict(),
        }

    @staticmethod
    def _ack_from_dict(value: Mapping[str, Any], *, replayed: bool) -> AckResult:
        key_value = value["key"]
        return AckResult(
            intent_id=str(value["intent_id"]),
            claim_id=str(value["claim_id"]),
            key=SlotKey(
                BatchId(str(key_value["run_id"]), int(key_value["train_step_id"])),
                int(key_value["task_index"]),
                str(key_value["actor_id"]),
            ),
            artifact=ArtifactRef.from_dict(value["artifact"]),
            replayed=replayed,
        )

    @staticmethod
    def _replacement_to_dict(result: TaskGroupReplacement) -> dict[str, Any]:
        return {
            "batch_id": result.batch_id.to_dict(),
            "task_group_id": result.task_group_id,
            "task_group_revision": result.task_group_revision,
            "manifest_revision": result.manifest_revision,
            "superseded_slots": result.superseded_slots,
            "active_slots": [SQLiteStateStore._slot_to_dict(slot) for slot in result.active_slots],
        }

    @staticmethod
    def _replacement_from_dict(value: Mapping[str, Any]) -> TaskGroupReplacement:
        batch = value["batch_id"]
        return TaskGroupReplacement(
            batch_id=BatchId(str(batch["run_id"]), int(batch["train_step_id"])),
            task_group_id=str(value["task_group_id"]),
            task_group_revision=int(value["task_group_revision"]),
            manifest_revision=int(value["manifest_revision"]),
            superseded_slots=int(value["superseded_slots"]),
            active_slots=tuple(
                SQLiteStateStore._slot_from_dict(slot) for slot in value["active_slots"]
            ),
        )

    def _json(self, value: Any, label: str) -> str:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                f"{label} is not JSON serializable",
                {"error": str(exc)},
            ) from exc
        size = len(encoded.encode("utf-8"))
        if size > self._max_inline_json_bytes:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                f"{label} is too large for inline SQLite storage; use the artifact spool",
                {"size": size, "limit": self._max_inline_json_bytes},
            )
        return encoded

    def _payload_hash(self, value: Any) -> str:
        return self._hash_text(self._json(value, "idempotency payload"))

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_enum(enum_type: type[_EnumT], value: Any, label: str) -> _EnumT:
        try:
            return enum_type(value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                f"invalid {label}",
                {"value": str(value)},
            ) from exc

    @staticmethod
    def _check_version(
        entity: str,
        expected: int,
        actual: int,
        identity: Mapping[str, Any],
    ) -> None:
        if expected != actual:
            raise ProtocolError(
                ErrorCode.VERSION_CONFLICT,
                f"{entity} version does not match",
                {**identity, "expected_version": expected, "actual_version": actual},
            )

    @staticmethod
    def _check_fences(
        row: sqlite3.Row,
        key: SlotKey,
        expected_attempt: int | None,
        expected_claim_generation: int | None,
    ) -> None:
        if expected_attempt is not None and int(row["current_attempt"]) != expected_attempt:
            raise SQLiteStateStore._fenced(
                "attempt fence does not match",
                {
                    **key.to_dict(),
                    "expected_attempt": expected_attempt,
                    "actual_attempt": int(row["current_attempt"]),
                },
            )
        if (
            expected_claim_generation is not None
            and int(row["claim_generation"]) != expected_claim_generation
        ):
            raise SQLiteStateStore._fenced(
                "claim generation fence does not match",
                {
                    **key.to_dict(),
                    "expected_generation": expected_claim_generation,
                    "actual_generation": int(row["claim_generation"]),
                },
            )

    @staticmethod
    def _not_found(entity: str, details: Mapping[str, Any]) -> ProtocolError:
        return ProtocolError(ErrorCode.NOT_FOUND, f"{entity} was not found", details)

    @staticmethod
    def _version_conflict(
        entity: str,
        expected: int,
        details: Mapping[str, Any],
    ) -> ProtocolError:
        return ProtocolError(
            ErrorCode.VERSION_CONFLICT,
            f"{entity} compare-and-swap failed",
            {**details, "expected_version": expected},
        )

    @staticmethod
    def _fenced(message: str, details: Mapping[str, Any]) -> ProtocolError:
        return ProtocolError(ErrorCode.FENCED, message, details)
