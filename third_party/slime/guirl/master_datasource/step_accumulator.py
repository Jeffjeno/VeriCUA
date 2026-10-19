from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

class BatchInvalidError(RuntimeError):

    code = "BATCH_INVALID"

_JOURNAL_STATES = {
    "SEALED",
    "OPTIMIZING",
    "OPTIMIZER_SUCCEEDED",
    "CHECKPOINTED",
    "COMMITTING",
    "COMMITTED",
    "IN_DOUBT",
}
_JOURNAL_TRANSITIONS = {
    None: {"SEALED"},
    "SEALED": {"SEALED", "OPTIMIZING", "IN_DOUBT"},
    "OPTIMIZING": {"OPTIMIZER_SUCCEEDED", "IN_DOUBT"},
    "OPTIMIZER_SUCCEEDED": {"CHECKPOINTED", "IN_DOUBT"},
    "CHECKPOINTED": {"COMMITTING", "IN_DOUBT"},
    "COMMITTING": {"COMMITTED", "IN_DOUBT"},
    "COMMITTED": {"COMMITTED"},
    "IN_DOUBT": {"IN_DOUBT"},
}
_JOURNAL_TIMESTAMP_COLUMNS = {
    "SEALED": "sealed_at",
    "OPTIMIZING": "optimizing_at",
    "OPTIMIZER_SUCCEEDED": "optimizer_succeeded_at",
    "CHECKPOINTED": "checkpointed_at",
    "COMMITTING": "committing_at",
    "COMMITTED": "committed_at",
    "IN_DOUBT": "in_doubt_at",
}

class StepAccumulator:

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        artifact_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root is not None else self.root / "artifacts"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.root / "accumulator.db",
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    @property
    def journal_mode(self) -> str:
        return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS trainer_batches (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    manifest_revision INTEGER NOT NULL,
                    payload_hash TEXT NOT NULL,
                    actor_order_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'RUNNING',
                    PRIMARY KEY (run_id, train_step_id)
                );

                CREATE TABLE IF NOT EXISTS expected_slots (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    task_index INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_group_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (run_id, train_step_id, task_index, actor_id),
                    FOREIGN KEY (run_id, train_step_id)
                        REFERENCES trainer_batches(run_id, train_step_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS trainer_claims (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    task_index INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_group_id TEXT NOT NULL,
                    task_group_revision INTEGER NOT NULL,
                    manifest_revision INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    claim_token TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    entry_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'CLAIMED',
                    PRIMARY KEY (run_id, train_step_id, task_index, actor_id),
                    FOREIGN KEY (run_id, train_step_id, task_index, actor_id)
                        REFERENCES expected_slots(run_id, train_step_id, task_index, actor_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS accepted_results (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    task_index INTEGER NOT NULL,
                    actor_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_group_id TEXT NOT NULL,
                    task_group_revision INTEGER NOT NULL,
                    manifest_revision INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    claim_token TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    source_group_id TEXT NOT NULL,
                    group_digest TEXT NOT NULL,
                    group_size INTEGER NOT NULL,
                    group_uri TEXT NOT NULL,
                    summary_digest TEXT NOT NULL,
                    summary_size INTEGER NOT NULL,
                    summary_uri TEXT NOT NULL,
                    acked INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, train_step_id, task_index, actor_id),
                    FOREIGN KEY (run_id, train_step_id, task_index, actor_id)
                        REFERENCES expected_slots(run_id, train_step_id, task_index, actor_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS request_sequences (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    next_value INTEGER NOT NULL,
                    PRIMARY KEY (run_id, train_step_id, operation)
                );

                CREATE TABLE IF NOT EXISTS pending_requests (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    payload_json TEXT,
                    created_at REAL,
                    PRIMARY KEY (run_id, train_step_id, operation)
                );

                CREATE TABLE IF NOT EXISTS optimizer_journal (
                    run_id TEXT NOT NULL,
                    train_step_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    sealed_at REAL,
                    optimizing_at REAL,
                    optimizer_succeeded_at REAL,
                    checkpointed_at REAL,
                    committing_at REAL,
                    committed_at REAL,
                    in_doubt_at REAL,
                    updated_at REAL NOT NULL,
                    error TEXT,
                    checkpoint_marker_json TEXT,
                    PRIMARY KEY (run_id, train_step_id),
                    FOREIGN KEY (run_id, train_step_id)
                        REFERENCES trainer_batches(run_id, train_step_id)
                        ON DELETE CASCADE
                );
                """
            )
            self._ensure_column("pending_requests", "payload_json", "TEXT")
            self._ensure_column("pending_requests", "created_at", "REAL")
            self._ensure_column(
                "accepted_results", "host_summary_digest", "TEXT"
            )
            self._ensure_column(
                "accepted_results", "host_summary_size", "INTEGER"
            )
            self._ensure_column("accepted_results", "host_summary_uri", "TEXT")

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
        if column not in columns:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    @staticmethod
    def slot_identity(value: Mapping[str, Any]) -> tuple[int, str]:
        source = value.get("slot") or value
        return int(source["task_index"]), str(source["actor_id"])

    def refresh_manifest(
        self,
        *,
        run_id: str,
        train_step_id: int,
        manifest_revision: int,
        payload_hash: str,
        slots: Sequence[Mapping[str, Any]],
    ) -> bool:

        normalized = [
            {
                "task_index": int(slot["task_index"]),
                "actor_id": str(slot["actor_id"]),
                "task_id": str(slot["task_id"]),
                "task_group_id": str(slot.get("task_group_id", "default")),
            }
            for slot in slots
        ]
        if not normalized:
            raise BatchInvalidError("BATCH_INVALID: empty v2 manifest")
        identities = [(slot["task_index"], slot["actor_id"]) for slot in normalized]
        if len(identities) != len(set(identities)):
            raise BatchInvalidError("BATCH_INVALID: duplicate stable slot in manifest")
        actor_order = list(dict.fromkeys(slot["actor_id"] for slot in normalized))
        actor_rank = {actor_id: index for index, actor_id in enumerate(actor_order)}
        normalized.sort(key=lambda slot: (slot["task_index"], actor_rank[slot["actor_id"]]))

        deleted_refs: list[str] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                batch = self._connection.execute(
                    """
                    SELECT * FROM trainer_batches
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                ).fetchone()
                old_rows = self._connection.execute(
                    """
                    SELECT * FROM expected_slots
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                ).fetchall()
                old = {
                    (int(row["task_index"]), str(row["actor_id"])): {
                        "task_id": str(row["task_id"]),
                        "task_group_id": str(row["task_group_id"]),
                    }
                    for row in old_rows
                }
                new = {
                    (slot["task_index"], slot["actor_id"]): {
                        "task_id": slot["task_id"],
                        "task_group_id": slot["task_group_id"],
                    }
                    for slot in normalized
                }
                if batch is not None:
                    current_revision = int(batch["manifest_revision"])
                    if manifest_revision < current_revision:
                        raise BatchInvalidError(
                            "BATCH_INVALID: Host returned a stale manifest revision "
                            f"{manifest_revision} < {current_revision}"
                        )
                    if manifest_revision == current_revision:
                        if str(batch["payload_hash"]) != payload_hash or old != new:
                            raise BatchInvalidError(
                                "BATCH_INVALID: manifest revision replay changed payload"
                            )
                        self._connection.execute("COMMIT")
                        return False

                changed_groups = {
                    spec["task_group_id"]
                    for identity, spec in old.items()
                    if identity not in new or new[identity] != spec
                }
                changed_groups.update(
                    spec["task_group_id"]
                    for identity, spec in new.items()
                    if identity not in old or old[identity] != spec
                )
                if changed_groups:
                    placeholders = ",".join("?" for _ in changed_groups)
                    deleted = self._connection.execute(
                        f"""
                        SELECT group_uri, summary_uri FROM accepted_results
                        WHERE run_id=? AND train_step_id=?
                          AND task_group_id IN ({placeholders})
                        """,
                        (run_id, train_step_id, *sorted(changed_groups)),
                    ).fetchall()
                    deleted_refs.extend(
                        uri
                        for row in deleted
                        for uri in (str(row["group_uri"]), str(row["summary_uri"]))
                    )
                    self._connection.execute(
                        f"""
                        DELETE FROM accepted_results
                        WHERE run_id=? AND train_step_id=?
                          AND task_group_id IN ({placeholders})
                        """,
                        (run_id, train_step_id, *sorted(changed_groups)),
                    )
                    self._connection.execute(
                        f"""
                        DELETE FROM trainer_claims
                        WHERE run_id=? AND train_step_id=?
                          AND task_group_id IN ({placeholders})
                        """,
                        (run_id, train_step_id, *sorted(changed_groups)),
                    )

                self._connection.execute(
                    """
                    INSERT INTO trainer_batches(
                        run_id, train_step_id, manifest_revision, payload_hash,
                        actor_order_json, state
                    ) VALUES (?, ?, ?, ?, ?, 'RUNNING')
                    ON CONFLICT(run_id, train_step_id) DO UPDATE SET
                        manifest_revision=excluded.manifest_revision,
                        payload_hash=excluded.payload_hash,
                        actor_order_json=excluded.actor_order_json,
                        state=CASE
                            WHEN trainer_batches.state='COMMITTED' THEN 'COMMITTED'
                            ELSE 'RUNNING'
                        END
                    """,
                    (
                        run_id,
                        train_step_id,
                        manifest_revision,
                        payload_hash,
                        self._json(actor_order),
                    ),
                )
                self._connection.executemany(
                    """
                    INSERT INTO expected_slots(
                        run_id, train_step_id, task_index, actor_id, task_id,
                        task_group_id, ordinal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, train_step_id, task_index, actor_id)
                    DO UPDATE SET
                        task_id=excluded.task_id,
                        task_group_id=excluded.task_group_id,
                        ordinal=excluded.ordinal
                    """,
                    [
                        (
                            run_id,
                            train_step_id,
                            slot["task_index"],
                            slot["actor_id"],
                            slot["task_id"],
                            slot["task_group_id"],
                            ordinal,
                        )
                        for ordinal, slot in enumerate(normalized)
                    ],
                )
                removed = set(old) - set(new)
                self._connection.executemany(
                    """
                    DELETE FROM expected_slots
                    WHERE run_id=? AND train_step_id=?
                      AND task_index=? AND actor_id=?
                    """,
                    [
                        (run_id, train_step_id, task_index, actor_id)
                        for task_index, actor_id in removed
                    ],
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        self._delete_unreferenced(deleted_refs)
        return True

    def record_claim(self, run_id: str, train_step_id: int, entry: Mapping[str, Any]) -> str:

        deleted_refs: list[str] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                status, removed = self._record_claim_in_transaction(
                    run_id, train_step_id, entry
                )
                deleted_refs.extend(removed)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        self._delete_unreferenced(deleted_refs)
        return status

    def record_claims_and_complete_request(
        self,
        run_id: str,
        train_step_id: int,
        entries: Sequence[Mapping[str, Any]],
        request_id: str,
    ) -> list[str]:

        deleted_refs: list[str] = []
        statuses: list[str] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                pending = self._connection.execute(
                    """
                    SELECT request_id FROM pending_requests
                    WHERE run_id=? AND train_step_id=? AND operation='pull'
                    """,
                    (run_id, train_step_id),
                ).fetchone()
                if pending is None or str(pending["request_id"]) != request_id:
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: pull response does not match "
                        f"the outstanding request {request_id}"
                    )
                for entry in entries:
                    status, removed = self._record_claim_in_transaction(
                        run_id, train_step_id, entry
                    )
                    statuses.append(status)
                    deleted_refs.extend(removed)
                deleted = self._connection.execute(
                    """
                    DELETE FROM pending_requests
                    WHERE run_id=? AND train_step_id=? AND operation='pull'
                      AND request_id=?
                    """,
                    (run_id, train_step_id, request_id),
                )
                if deleted.rowcount != 1:
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: outstanding pull request changed "
                        "while recording its response"
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        self._delete_unreferenced(deleted_refs)
        return statuses

    def _record_claim_in_transaction(
        self,
        run_id: str,
        train_step_id: int,
        entry: Mapping[str, Any],
    ) -> tuple[str, list[str]]:
        task_index, actor_id = self.slot_identity(entry)
        claim_token = str(entry["claim_token"])
        generation = int(entry["generation"])
        task_id = str(entry["task_id"])
        task_group_id = str(entry.get("task_group_id", "default"))
        deleted_refs: list[str] = []
        expected = self._connection.execute(
            """
            SELECT * FROM expected_slots
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
            """,
            (run_id, train_step_id, task_index, actor_id),
        ).fetchone()
        if expected is None or (
            str(expected["task_id"]) != task_id
            or str(expected["task_group_id"]) != task_group_id
        ):
            return "SUPERSEDED", deleted_refs
        current = self._connection.execute(
            """
            SELECT * FROM trainer_claims
            WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
            """,
            (run_id, train_step_id, task_index, actor_id),
        ).fetchone()
        if current is not None:
            if (
                str(current["claim_token"]) == claim_token
                and int(current["generation"]) == generation
            ):
                return str(current["state"]), deleted_refs
            if int(current["generation"]) > generation:
                return "SUPERSEDED", deleted_refs
            accepted = self._connection.execute(
                """
                SELECT group_uri, summary_uri FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
            if accepted is not None:
                deleted_refs.extend(
                    [str(accepted["group_uri"]), str(accepted["summary_uri"])]
                )
                self._connection.execute(
                    """
                    DELETE FROM accepted_results
                    WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                    """,
                    (run_id, train_step_id, task_index, actor_id),
                )
        self._connection.execute(
            """
            INSERT INTO trainer_claims(
                run_id, train_step_id, task_index, actor_id, task_id,
                task_group_id, task_group_revision, manifest_revision,
                attempt, claim_token, generation, entry_json, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLAIMED')
            ON CONFLICT(run_id, train_step_id, task_index, actor_id) DO UPDATE SET
                task_id=excluded.task_id,
                task_group_id=excluded.task_group_id,
                task_group_revision=excluded.task_group_revision,
                manifest_revision=excluded.manifest_revision,
                attempt=excluded.attempt,
                claim_token=excluded.claim_token,
                generation=excluded.generation,
                entry_json=excluded.entry_json,
                state='CLAIMED'
            """,
            (
                run_id,
                train_step_id,
                task_index,
                actor_id,
                task_id,
                task_group_id,
                int(entry.get("task_group_revision", 1)),
                int(entry.get("manifest_revision", 1)),
                int(entry.get("attempt", 1)),
                claim_token,
                generation,
                self._json(dict(entry)),
            ),
        )
        return "CLAIMED", deleted_refs

    def recoverable_claims(self, run_id: str, train_step_id: int) -> list[dict[str, Any]]:

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.entry_json FROM trainer_claims c
                LEFT JOIN accepted_results a
                  ON a.run_id=c.run_id AND a.train_step_id=c.train_step_id
                 AND a.task_index=c.task_index AND a.actor_id=c.actor_id
                WHERE c.run_id=? AND c.train_step_id=? AND c.state='CLAIMED'
                  AND a.claim_token IS NULL
                ORDER BY c.task_index, c.actor_id
                """,
                (run_id, train_step_id),
            ).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def pending_acks(self, run_id: str, train_step_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT a.*, c.entry_json FROM accepted_results a
                JOIN trainer_claims c
                  ON c.run_id=a.run_id AND c.train_step_id=a.train_step_id
                 AND c.task_index=a.task_index AND c.actor_id=a.actor_id
                WHERE a.run_id=? AND a.train_step_id=? AND a.acked=0
                ORDER BY a.task_index, a.actor_id
                """,
                (run_id, train_step_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_renewal_and_complete_request(
        self,
        run_id: str,
        train_step_id: int,
        entry: Mapping[str, Any],
        request_id: str,
        *,
        expires_at: float,
    ) -> dict[str, Any]:

        task_index, actor_id = self.slot_identity(entry)
        claim_token = str(entry["claim_token"])
        generation = int(entry["generation"])
        renewed_expiry = float(expires_at)
        if renewed_expiry <= 0:
            raise RuntimeError("GUIRL_V2_STATE_ERROR: invalid renewed claim expiry")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                pending = self._connection.execute(
                    """
                    SELECT operation FROM pending_requests
                    WHERE run_id=? AND train_step_id=? AND request_id=?
                    """,
                    (run_id, train_step_id, request_id),
                ).fetchone()
                if pending is None or not str(pending["operation"]).startswith(
                    f"renew:{claim_token}:{generation}:"
                ):
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: renewal response does not match "
                        f"the outstanding request {request_id}"
                    )
                claim = self._connection.execute(
                    """
                    SELECT claim_token, generation, entry_json FROM trainer_claims
                    WHERE run_id=? AND train_step_id=?
                      AND task_index=? AND actor_id=?
                    """,
                    (run_id, train_step_id, task_index, actor_id),
                ).fetchone()
                if claim is None or (
                    str(claim["claim_token"]) != claim_token
                    or int(claim["generation"]) != generation
                ):
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: renewed claim no longer matches "
                        "the durable slot fence"
                    )
                durable_entry = json.loads(str(claim["entry_json"]))
                if not isinstance(durable_entry, dict):
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: durable claim entry is invalid"
                    )
                durable_entry["expires_at"] = renewed_expiry
                updated = self._connection.execute(
                    """
                    UPDATE trainer_claims SET entry_json=?
                    WHERE run_id=? AND train_step_id=?
                      AND task_index=? AND actor_id=?
                      AND claim_token=? AND generation=?
                    """,
                    (
                        self._json(durable_entry),
                        run_id,
                        train_step_id,
                        task_index,
                        actor_id,
                        claim_token,
                        generation,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: durable claim changed while "
                        "recording renewal"
                    )
                deleted = self._connection.execute(
                    """
                    DELETE FROM pending_requests
                    WHERE run_id=? AND train_step_id=? AND request_id=?
                    """,
                    (run_id, train_step_id, request_id),
                )
                if deleted.rowcount != 1:
                    raise RuntimeError(
                        "GUIRL_V2_STATE_ERROR: renewal request changed while "
                        "recording its response"
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return durable_entry

    def persist_result(
        self,
        *,
        run_id: str,
        train_step_id: int,
        entry: Mapping[str, Any],
        generated_group: Any,
        judge_summary: Mapping[str, Any],
        source_group_id: int | str,
    ) -> dict[str, Any]:

        task_index, actor_id = self.slot_identity(entry)
        group_ref = self._write_bytes(
            pickle.dumps(generated_group, protocol=pickle.HIGHEST_PROTOCOL),
            media_type="application/x-python-pickle",
        )
        summary = dict(judge_summary)

        summary["sample_serialization"] = {
            "digest": group_ref["digest"],
            "size": group_ref["size"],
            "media_type": group_ref["media_type"],
        }
        summary_ref = self._write_bytes(
            self._json(summary).encode("utf-8"),
            media_type="application/json",
        )

        with self._lock:
            expected = self._connection.execute(
                """
                SELECT * FROM expected_slots
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
            if expected is None or (
                str(expected["task_id"]) != str(entry["task_id"])
                or str(expected["task_group_id"])
                != str(entry.get("task_group_id", "default"))
            ):
                self._delete_unreferenced([group_ref["uri"], summary_ref["uri"]])
                return {"status": "SUPERSEDED"}
            claim = self._connection.execute(
                """
                SELECT claim_token, generation FROM trainer_claims
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
            if claim is None or (
                str(claim["claim_token"]) != str(entry["claim_token"])
                or int(claim["generation"]) != int(entry["generation"])
            ):
                self._delete_unreferenced([group_ref["uri"], summary_ref["uri"]])
                return {"status": "SUPERSEDED"}
            existing = self._connection.execute(
                """
                SELECT * FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
            if existing is not None:
                if str(existing["group_digest"]) == group_ref["digest"]:
                    self._delete_unreferenced([group_ref["uri"], summary_ref["uri"]])
                    return {
                        "status": "ACKED" if int(existing["acked"]) else "PERSISTED",
                        "judge_artifact": self._summary_ref(existing),
                        "group_digest": str(existing["group_digest"]),
                    }
                self._connection.execute(
                    """
                    UPDATE trainer_batches SET state='INVALID'
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                )
                self._delete_unreferenced([group_ref["uri"], summary_ref["uri"]])
                raise BatchInvalidError(
                    "BATCH_INVALID: conflicting Judge artifact for slot "
                    f"({task_index}, {actor_id})"
                )
            self._connection.execute(
                """
                INSERT INTO accepted_results(
                    run_id, train_step_id, task_index, actor_id, task_id,
                    task_group_id, task_group_revision, manifest_revision,
                    attempt, claim_token, generation, source_group_id,
                    group_digest, group_size, group_uri,
                    summary_digest, summary_size, summary_uri, acked
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    train_step_id,
                    task_index,
                    actor_id,
                    str(entry["task_id"]),
                    str(entry.get("task_group_id", "default")),
                    int(entry.get("task_group_revision", 1)),
                    int(entry.get("manifest_revision", 1)),
                    int(entry.get("attempt", 1)),
                    str(entry["claim_token"]),
                    int(entry["generation"]),
                    str(source_group_id),
                    group_ref["digest"],
                    group_ref["size"],
                    group_ref["uri"],
                    summary_ref["digest"],
                    summary_ref["size"],
                    summary_ref["uri"],
                ),
            )
            self._connection.execute(
                """
                UPDATE trainer_claims SET state='PERSISTED'
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            )
        return {
            "status": "PERSISTED",
            "judge_artifact": summary_ref,
            "group_digest": group_ref["digest"],
        }

    def mark_acked(
        self,
        run_id: str,
        train_step_id: int,
        task_index: int,
        actor_id: str,
        *,
        claim_token: str,
        generation: int,
        host_judge_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        host_digest = None
        host_size = None
        host_uri = None
        if host_judge_artifact is not None:
            host_digest = str(host_judge_artifact["digest"])
            host_size = int(host_judge_artifact["size"])
            host_uri = str(host_judge_artifact["uri"])
            if not host_digest or host_size < 0 or not host_uri:
                raise BatchInvalidError(
                    "BATCH_INVALID: Host returned an invalid Judge artifact"
                )
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE accepted_results SET
                    acked=1,
                    host_summary_digest=?,
                    host_summary_size=?,
                    host_summary_uri=?
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                  AND claim_token=? AND generation=?
                """,
                (
                    host_digest,
                    host_size,
                    host_uri,
                    run_id,
                    train_step_id,
                    task_index,
                    actor_id,
                    claim_token,
                    generation,
                ),
            )
            if cursor.rowcount != 1:
                raise BatchInvalidError(
                    "BATCH_INVALID: ack response does not match persisted claim fence"
                )
            self._connection.execute(
                """
                UPDATE trainer_claims SET state='ACKED'
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            )

    def load_judge_payload(
        self,
        run_id: str,
        train_step_id: int,
        task_index: int,
        actor_id: str,
        summary_ref: Mapping[str, Any],
    ) -> dict[str, Any]:

        with self._lock:
            row = self._connection.execute(
                """
                SELECT summary_digest, summary_size, summary_uri
                FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
        if row is None:
            raise BatchInvalidError(
                "BATCH_INVALID: no persisted Judge summary for acknowledgement"
            )
        expected = self._summary_ref(row)
        for field in ("digest", "size", "uri"):
            if str(summary_ref[field]) != str(expected[field]):
                raise BatchInvalidError(
                    "BATCH_INVALID: acknowledgement Judge summary does not "
                    "match trainer spool"
                )
        payload = self._read_ref(expected)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BatchInvalidError(
                "BATCH_INVALID: trainer Judge summary is not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise BatchInvalidError(
                "BATCH_INVALID: trainer Judge summary must be a JSON object"
            )
        return value

    def host_judge_artifact(
        self,
        run_id: str,
        train_step_id: int,
        task_index: int,
        actor_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT host_summary_digest, host_summary_size, host_summary_uri
                FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
        if row is None or row["host_summary_digest"] is None:
            return None
        return {
            "digest": str(row["host_summary_digest"]),
            "size": int(row["host_summary_size"]),
            "uri": str(row["host_summary_uri"]),
            "media_type": "application/json",
        }

    def discard_slot(
        self,
        run_id: str,
        train_step_id: int,
        task_index: int,
        actor_id: str,
    ) -> None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT group_uri, summary_uri FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
            self._connection.execute(
                """
                DELETE FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            )
            self._connection.execute(
                """
                DELETE FROM trainer_claims
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            )
        if row is not None:
            self._delete_unreferenced([str(row["group_uri"]), str(row["summary_uri"])])

    def accepted_keys(self, run_id: str, train_step_id: int) -> set[tuple[int, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT task_index, actor_id FROM accepted_results
                WHERE run_id=? AND train_step_id=? AND acked=1
                """,
                (run_id, train_step_id),
            ).fetchall()
        return {(int(row["task_index"]), str(row["actor_id"])) for row in rows}

    def expected_keys(self, run_id: str, train_step_id: int) -> set[tuple[int, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT task_index, actor_id FROM expected_slots
                WHERE run_id=? AND train_step_id=?
                """,
                (run_id, train_step_id),
            ).fetchall()
        return {(int(row["task_index"]), str(row["actor_id"])) for row in rows}

    def entry_is_expected(
        self,
        run_id: str,
        train_step_id: int,
        entry: Mapping[str, Any],
    ) -> bool:
        task_index, actor_id = self.slot_identity(entry)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT task_id, task_group_id FROM expected_slots
                WHERE run_id=? AND train_step_id=? AND task_index=? AND actor_id=?
                """,
                (run_id, train_step_id, task_index, actor_id),
            ).fetchone()
        return row is not None and (
            str(row["task_id"]) == str(entry.get("task_id"))
            and str(row["task_group_id"])
            == str(entry.get("task_group_id", "default"))
        )

    def is_complete(self, run_id: str, train_step_id: int) -> bool:
        expected = self.expected_keys(run_id, train_step_id)
        return bool(expected) and expected == self.accepted_keys(run_id, train_step_id)

    def ordered_output(self, run_id: str, train_step_id: int) -> list[Any]:

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT a.group_digest, a.group_size, a.group_uri
                FROM expected_slots e
                JOIN accepted_results a
                  ON a.run_id=e.run_id AND a.train_step_id=e.train_step_id
                 AND a.task_index=e.task_index AND a.actor_id=e.actor_id
                WHERE e.run_id=? AND e.train_step_id=? AND a.acked=1
                ORDER BY e.ordinal
                """,
                (run_id, train_step_id),
            ).fetchall()
            expected_count = int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) FROM expected_slots
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                ).fetchone()[0]
            )
        if len(rows) != expected_count or expected_count == 0:
            raise RuntimeError(
                f"partial v2 batch cannot be returned: {len(rows)}/{expected_count} slots"
            )
        output = []
        for row in rows:
            payload = self._read_ref(
                {
                    "digest": str(row["group_digest"]),
                    "size": int(row["group_size"]),
                    "uri": str(row["group_uri"]),
                }
            )
            output.append(pickle.loads(payload))
        return output

    def mark_state(self, run_id: str, train_step_id: int, state: str) -> None:
        if state in _JOURNAL_STATES:
            self._transition_journal(run_id, train_step_id, state)
            return
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE trainer_batches SET state=?
                WHERE run_id=? AND train_step_id=?
                """,
                (state, run_id, train_step_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"unknown trainer batch {run_id}/{train_step_id}")

    def state(self, run_id: str, train_step_id: int) -> str | None:
        journal = self.journal(run_id, train_step_id)
        if journal is not None:
            return str(journal["state"])
        with self._lock:
            row = self._connection.execute(
                """
                SELECT state FROM trainer_batches
                WHERE run_id=? AND train_step_id=?
                """,
                (run_id, train_step_id),
            ).fetchone()
        return None if row is None else str(row["state"])

    def journal(
        self, run_id: str, train_step_id: int
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM optimizer_journal
                WHERE run_id=? AND train_step_id=?
                """,
                (run_id, train_step_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        marker = result.pop("checkpoint_marker_json", None)
        result["checkpoint_marker"] = (
            None if marker is None else json.loads(str(marker))
        )
        return result

    def begin_update(self, run_id: str, train_step_id: int) -> None:
        self._transition_journal(
            run_id, train_step_id, "OPTIMIZING", expected={"SEALED"}
        )

    def optimizer_succeeded(self, run_id: str, train_step_id: int) -> None:
        self._transition_journal(
            run_id,
            train_step_id,
            "OPTIMIZER_SUCCEEDED",
            expected={"OPTIMIZING"},
        )

    def checkpoint_succeeded(
        self,
        run_id: str,
        train_step_id: int,
        checkpoint_marker: Mapping[str, Any],
    ) -> None:
        marker = dict(checkpoint_marker)
        if not marker:
            raise RuntimeError("v2 checkpoint marker must not be empty")
        self._transition_journal(
            run_id,
            train_step_id,
            "CHECKPOINTED",
            expected={"OPTIMIZER_SUCCEEDED"},
            checkpoint_marker=marker,
        )

    def begin_commit(self, run_id: str, train_step_id: int) -> None:
        self._transition_journal(
            run_id, train_step_id, "COMMITTING", expected={"CHECKPOINTED"}
        )

    def mark_committed(
        self,
        run_id: str,
        train_step_id: int,
        *,
        host_proved_committed: bool = False,
    ) -> None:
        expected: set[Any] = (
            _JOURNAL_STATES | {None}
            if host_proved_committed
            else {"COMMITTING"}
        )
        self._transition_journal(
            run_id,
            train_step_id,
            "COMMITTED",
            expected=expected,
            allow_host_resolution=host_proved_committed,
        )

    def mark_optimizer_in_doubt(
        self,
        run_id: str,
        train_step_id: int,
        error: str,
    ) -> None:
        self._transition_journal(
            run_id,
            train_step_id,
            "IN_DOUBT",
            expected={
                "OPTIMIZING",
                "OPTIMIZER_SUCCEEDED",
                "CHECKPOINTED",
                "COMMITTING",
                "IN_DOUBT",
            },
            error=str(error),
        )

    def assert_replay_safe(self, run_id: str, train_step_id: int) -> None:
        journal = self.journal(run_id, train_step_id)
        state = None if journal is None else str(journal["state"])
        if state == "SEALED":
            return
        if state is None:
            raise RuntimeError(
                "GUIRL_V2_OPERATOR_RESOLUTION_REQUIRED: Host batch is SEALED "
                "but the trainer optimizer journal is missing, so the trainer "
                "cannot prove that optimization never started"
            )
        raise RuntimeError(
            "GUIRL_V2_OPERATOR_RESOLUTION_REQUIRED: refusing automatic replay "
            f"for optimizer journal state {state}"
        )

    def _transition_journal(
        self,
        run_id: str,
        train_step_id: int,
        target: str,
        *,
        expected: set[str] | None = None,
        error: str | None = None,
        checkpoint_marker: Mapping[str, Any] | None = None,
        allow_host_resolution: bool = False,
    ) -> None:
        if target not in _JOURNAL_STATES:
            raise ValueError(f"unknown optimizer journal state {target}")
        timestamp = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                batch = self._connection.execute(
                    """
                    SELECT 1 FROM trainer_batches
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                ).fetchone()
                if batch is None:
                    raise RuntimeError(
                        f"unknown trainer batch {run_id}/{train_step_id}"
                    )
                row = self._connection.execute(
                    """
                    SELECT state FROM optimizer_journal
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (run_id, train_step_id),
                ).fetchone()
                current = None if row is None else str(row["state"])
                if current == target:
                    self._connection.execute(
                        """
                        UPDATE trainer_batches SET state=?
                        WHERE run_id=? AND train_step_id=?
                        """,
                        (target, run_id, train_step_id),
                    )
                    self._connection.execute("COMMIT")
                    return
                valid = (
                    current in expected
                    if expected is not None
                    else target in _JOURNAL_TRANSITIONS.get(current, set())
                )
                if not valid:
                    qualifier = " after Host COMMITTED proof" if allow_host_resolution else ""
                    raise RuntimeError(
                        "GUIRL_V2_OPERATOR_RESOLUTION_REQUIRED: illegal optimizer "
                        f"journal transition {current} -> {target}{qualifier}"
                    )
                timestamp_column = _JOURNAL_TIMESTAMP_COLUMNS[target]
                marker_json = (
                    None
                    if checkpoint_marker is None
                    else self._json(dict(checkpoint_marker))
                )
                if row is None:
                    self._connection.execute(
                        f"""
                        INSERT INTO optimizer_journal(
                            run_id, train_step_id, state, {timestamp_column},
                            updated_at, error, checkpoint_marker_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            train_step_id,
                            target,
                            timestamp,
                            timestamp,
                            error,
                            marker_json,
                        ),
                    )
                else:
                    self._connection.execute(
                        f"""
                        UPDATE optimizer_journal SET
                            state=?,
                            {timestamp_column}=?,
                            updated_at=?,
                            error=COALESCE(?, error),
                            checkpoint_marker_json=COALESCE(
                                ?, checkpoint_marker_json
                            )
                        WHERE run_id=? AND train_step_id=?
                        """,
                        (
                            target,
                            timestamp,
                            timestamp,
                            error,
                            marker_json,
                            run_id,
                            train_step_id,
                        ),
                    )
                self._connection.execute(
                    """
                    UPDATE trainer_batches SET state=?
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (target, run_id, train_step_id),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def clear_revision(
        self,
        run_id: str,
        train_step_id: int,
        *,
        terminal_state: str,
    ) -> None:

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT group_uri, summary_uri FROM accepted_results
                WHERE run_id=? AND train_step_id=?
                """,
                (run_id, train_step_id),
            ).fetchall()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "DELETE FROM accepted_results WHERE run_id=? AND train_step_id=?",
                    (run_id, train_step_id),
                )
                self._connection.execute(
                    "DELETE FROM trainer_claims WHERE run_id=? AND train_step_id=?",
                    (run_id, train_step_id),
                )
                self._connection.execute(
                    """
                    UPDATE trainer_batches SET state=?
                    WHERE run_id=? AND train_step_id=?
                    """,
                    (terminal_state, run_id, train_step_id),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        self._delete_unreferenced(
            uri
            for row in rows
            for uri in (str(row["group_uri"]), str(row["summary_uri"]))
        )

    def next_request_id(
        self,
        run_id: str,
        train_step_id: int,
        operation: str,
    ) -> str:

        request_id = self.reserve_request_id(run_id, train_step_id, operation)
        self.complete_request(request_id)
        return request_id

    def reserve_request_id(
        self,
        run_id: str,
        train_step_id: int,
        operation: str,
    ) -> str:

        request_id, _ = self.reserve_request(
            run_id, train_step_id, operation, {}
        )
        return request_id

    def reserve_request(
        self,
        run_id: str,
        train_step_id: int,
        operation: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:

        payload_json = self._json(dict(payload))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                pending = self._connection.execute(
                    """
                    SELECT request_id, payload_json FROM pending_requests
                    WHERE run_id=? AND train_step_id=? AND operation=?
                    """,
                    (run_id, train_step_id, operation),
                ).fetchone()
                if pending is not None:
                    stored_payload = pending["payload_json"]
                    if stored_payload is None:
                        raise RuntimeError(
                            "GUIRL_V2_OPERATOR_RESOLUTION_REQUIRED: outstanding "
                            "request predates durable payload journaling"
                        )
                    loaded = json.loads(str(stored_payload))
                    if not isinstance(loaded, dict):
                        raise RuntimeError(
                            "GUIRL_V2_STATE_ERROR: pending request payload is invalid"
                        )
                    self._connection.execute("COMMIT")
                    return str(pending["request_id"]), loaded
                row = self._connection.execute(
                    """
                    SELECT next_value FROM request_sequences
                    WHERE run_id=? AND train_step_id=? AND operation=?
                    """,
                    (run_id, train_step_id, operation),
                ).fetchone()
                value = 0 if row is None else int(row["next_value"])
                self._connection.execute(
                    """
                    INSERT INTO request_sequences(
                        run_id, train_step_id, operation, next_value
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id, train_step_id, operation)
                    DO UPDATE SET next_value=excluded.next_value
                    """,
                    (run_id, train_step_id, operation, value + 1),
                )
                request_id = f"{operation}:{run_id}:{train_step_id}:{value}"
                self._connection.execute(
                    """
                    INSERT INTO pending_requests(
                        run_id, train_step_id, operation, request_id,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        train_step_id,
                        operation,
                        request_id,
                        payload_json,
                        time.time(),
                    ),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return request_id, dict(payload)

    def complete_request(self, request_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM pending_requests WHERE request_id=?",
                (request_id,),
            )

    @staticmethod
    def _summary_ref(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "digest": str(row["summary_digest"]),
            "size": int(row["summary_size"]),
            "uri": str(row["summary_uri"]),
            "media_type": "application/json",
        }

    def _write_bytes(self, payload: bytes, *, media_type: str) -> dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        path = self.artifact_root / digest[:2] / digest[2:4] / f"{digest}.blob"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{digest}.",
                    suffix=".tmp",
                    dir=str(path.parent),
                )
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_name, path)
                    self._fsync_directory(path.parent)
                except BaseException:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
                    raise
            ref = {
                "digest": digest,
                "size": len(payload),
                "uri": path.as_uri(),
                "media_type": media_type,
            }
            self._read_ref(ref)
            return ref

    def _read_ref(self, ref: Mapping[str, Any]) -> bytes:
        path = self._path_from_uri(str(ref["uri"]))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(ref["digest"]) or len(payload) != int(ref["size"]):
            raise BatchInvalidError(
                f"BATCH_INVALID: corrupt trainer artifact {ref['uri']}"
            )
        return payload

    def _delete_unreferenced(self, uris: Iterable[str]) -> None:
        for uri in set(uris):
            with self._lock:
                count = int(
                    self._connection.execute(
                        """
                        SELECT
                          (SELECT COUNT(*) FROM accepted_results WHERE group_uri=?)
                          + (SELECT COUNT(*) FROM accepted_results WHERE summary_uri=?)
                        """,
                        (uri, uri),
                    ).fetchone()[0]
                )
            if count:
                continue
            path = self._path_from_uri(uri)
            try:
                path.unlink()
                self._fsync_directory(path.parent)
            except FileNotFoundError:
                pass

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _path_from_uri(uri: str) -> Path:
        parsed = urlparse(uri)
        return Path(unquote(parsed.path) if parsed.scheme else uri)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
