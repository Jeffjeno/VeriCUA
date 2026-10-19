from __future__ import annotations

import argparse
import atexit
from collections import deque
import hashlib
import json
import logging
import sqlite3
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request

logger = logging.getLogger("gui.env_master_server")
app = Flask(__name__)

@dataclass
class WorkerRecord:
    worker_id: str
    pid: int
    base_url: str
    capacity: int
    status: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    active_leases: int = 0
    reserved_allocations: int = 0
    incarnation: str = ""
    lease_inventory: tuple[dict[str, Any], ...] = ()

@dataclass
class LeaseRecord:
    master_lease_id: str
    worker_id: str
    worker_base_url: str
    worker_lease_id: str
    episode_id: str
    env_id: str | None = None
    allocation_key: str | None = None
    allocation_payload_hash: str = ""
    owner: dict[str, Any] = field(default_factory=dict)
    status: str = "ACTIVE"
    worker_incarnation: str = ""
    generation: int = 1
    fence_token: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closing_at: float | None = None
    closed_at: float | None = None
    lost_at: float | None = None
    close_operation_id: str | None = None
    close_status: str | None = None
    close_error: str | None = None
    close_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_lease_id": self.master_lease_id,
            "lease_id": self.master_lease_id,
            "worker_id": self.worker_id,
            "worker_base_url": self.worker_base_url,
            "worker_lease_id": self.worker_lease_id,
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "allocation_key": self.allocation_key,
            "owner": dict(self.owner),
            "status": self.status,
            "worker_incarnation": self.worker_incarnation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closing_at": self.closing_at,
            "closed_at": self.closed_at,
            "lost_at": self.lost_at,
            "close_operation_id": self.close_operation_id,
            "close_status": self.close_status,
            "close_error": self.close_error,
            "fence_token": self.fence_token,
            "generation": self.generation,
            "fencing": {
                "token": self.fence_token,
                "generation": self.generation,
            },
            "close": {
                "operation_id": self.close_operation_id,
                "status": self.close_status,
                "error": self.close_error,
                "attempts": self.close_attempts,
            },
        }

@dataclass
class AllocateRequestRecord:
    episode_id: str
    request_key: str = ""
    allocation_key: str | None = None
    payload_hash: str = ""
    owner: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    dispatching_worker_id: str | None = None
    dispatching_worker_incarnation: str | None = None
    assigned_lease_id: str | None = None
    assigned_env_id: str | None = None
    assigned_reused: bool = False
    last_error: str | None = None
    waiter_count: int = 0
    closed: bool = False
    pending_master_lease_id: str = ""
    pending_fence_token: str = ""
    pending_generation: int = 1

class LeaseProtocolError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})

def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

class SQLiteLeaseStore:

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS env_master_leases (
                    master_lease_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    worker_base_url TEXT NOT NULL,
                    worker_lease_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    env_id TEXT,
                    allocation_key TEXT,
                    allocation_payload_hash TEXT NOT NULL,
                    owner_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_incarnation TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    fence_token TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    closing_at REAL,
                    closed_at REAL,
                    lost_at REAL,
                    close_operation_id TEXT,
                    close_status TEXT,
                    close_error TEXT,
                    close_attempts INTEGER NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(env_master_leases)"
                ).fetchall()
            }
            lease_migrations = {
                "env_id": "TEXT",
                "allocation_key": "TEXT",
                "allocation_payload_hash": "TEXT NOT NULL DEFAULT ''",
                "owner_json": "TEXT NOT NULL DEFAULT '{}'",
                "status": "TEXT NOT NULL DEFAULT 'ACTIVE'",
                "worker_incarnation": "TEXT NOT NULL DEFAULT ''",
                "generation": "INTEGER NOT NULL DEFAULT 1",
                "fence_token": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "REAL NOT NULL DEFAULT 0",
                "closing_at": "REAL",
                "closed_at": "REAL",
                "lost_at": "REAL",
                "close_operation_id": "TEXT",
                "close_status": "TEXT",
                "close_error": "TEXT",
                "close_attempts": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in lease_migrations.items():
                if column not in columns:
                    self._connection.execute(
                        "ALTER TABLE env_master_leases "
                        f"ADD COLUMN {column} {declaration}"
                    )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS env_master_leases_allocation_idx "
                "ON env_master_leases(allocation_key, generation)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS env_master_allocation_intents (
                    request_key TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    allocation_key TEXT,
                    payload_hash TEXT NOT NULL,
                    owner_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dispatching_worker_id TEXT,
                    dispatching_worker_incarnation TEXT,
                    assigned_lease_id TEXT,
                    assigned_env_id TEXT,
                    assigned_reused INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    pending_master_lease_id TEXT NOT NULL,
                    pending_fence_token TEXT NOT NULL,
                    pending_generation INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            intent_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(env_master_allocation_intents)"
                ).fetchall()
            }
            if "dispatching_worker_incarnation" not in intent_columns:
                self._connection.execute(
                    "ALTER TABLE env_master_allocation_intents "
                    "ADD COLUMN dispatching_worker_incarnation TEXT"
                )

    @property
    def journal_mode(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
            return str(row[0]).lower()

    def load(self) -> list[LeaseRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM env_master_leases ORDER BY created_at, master_lease_id"
            ).fetchall()
        return [
            LeaseRecord(
                master_lease_id=str(row["master_lease_id"]),
                worker_id=str(row["worker_id"]),
                worker_base_url=str(row["worker_base_url"]),
                worker_lease_id=str(row["worker_lease_id"]),
                episode_id=str(row["episode_id"]),
                env_id=row["env_id"],
                allocation_key=row["allocation_key"],
                allocation_payload_hash=str(row["allocation_payload_hash"]),
                owner=dict(json.loads(row["owner_json"])),
                status=str(row["status"]),
                worker_incarnation=str(row["worker_incarnation"]),
                generation=int(row["generation"]),
                fence_token=str(row["fence_token"]),
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
                closing_at=row["closing_at"],
                closed_at=row["closed_at"],
                lost_at=row["lost_at"],
                close_operation_id=row["close_operation_id"],
                close_status=row["close_status"],
                close_error=row["close_error"],
                close_attempts=int(row["close_attempts"]),
            )
            for row in rows
        ]

    def save(self, lease: LeaseRecord) -> None:
        values = (
            lease.master_lease_id,
            lease.worker_id,
            lease.worker_base_url,
            lease.worker_lease_id,
            lease.episode_id,
            lease.env_id,
            lease.allocation_key,
            lease.allocation_payload_hash,
            json.dumps(lease.owner, sort_keys=True, separators=(",", ":")),
            lease.status,
            lease.worker_incarnation,
            lease.generation,
            lease.fence_token,
            lease.created_at,
            lease.updated_at,
            lease.closing_at,
            lease.closed_at,
            lease.lost_at,
            lease.close_operation_id,
            lease.close_status,
            lease.close_error,
            lease.close_attempts,
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO env_master_leases(
                    master_lease_id, worker_id, worker_base_url, worker_lease_id,
                    episode_id, env_id, allocation_key, allocation_payload_hash, owner_json,
                    status, worker_incarnation, generation, fence_token, created_at,
                    updated_at, closing_at, closed_at, lost_at, close_operation_id,
                    close_status, close_error, close_attempts
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(master_lease_id) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    worker_base_url=excluded.worker_base_url,
                    worker_lease_id=excluded.worker_lease_id,
                    episode_id=excluded.episode_id,
                    env_id=excluded.env_id,
                    allocation_key=excluded.allocation_key,
                    allocation_payload_hash=excluded.allocation_payload_hash,
                    owner_json=excluded.owner_json,
                    status=excluded.status,
                    worker_incarnation=excluded.worker_incarnation,
                    generation=excluded.generation,
                    fence_token=excluded.fence_token,
                    updated_at=excluded.updated_at,
                    closing_at=excluded.closing_at,
                    closed_at=excluded.closed_at,
                    lost_at=excluded.lost_at,
                    close_operation_id=excluded.close_operation_id,
                    close_status=excluded.close_status,
                    close_error=excluded.close_error,
                    close_attempts=excluded.close_attempts
                """,
                values,
            )

    def load_intents(self) -> list[AllocateRequestRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM env_master_allocation_intents
                ORDER BY created_at, request_key
                """
            ).fetchall()
        return [
            AllocateRequestRecord(
                episode_id=str(row["episode_id"]),
                request_key=str(row["request_key"]),
                allocation_key=row["allocation_key"],
                payload_hash=str(row["payload_hash"]),
                owner=dict(json.loads(row["owner_json"])),
                created_at=float(row["created_at"]),
                status=str(row["status"]),
                dispatching_worker_id=row["dispatching_worker_id"],
                dispatching_worker_incarnation=(
                    row["dispatching_worker_incarnation"]
                ),
                assigned_lease_id=row["assigned_lease_id"],
                assigned_env_id=row["assigned_env_id"],
                assigned_reused=bool(row["assigned_reused"]),
                last_error=row["last_error"],
                pending_master_lease_id=str(
                    row["pending_master_lease_id"]
                ),
                pending_fence_token=str(row["pending_fence_token"]),
                pending_generation=int(row["pending_generation"]),
            )
            for row in rows
        ]

    def save_intent(self, intent: AllocateRequestRecord) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO env_master_allocation_intents(
                    request_key, episode_id, allocation_key, payload_hash,
                    owner_json, status, dispatching_worker_id,
                    dispatching_worker_incarnation,
                    assigned_lease_id, assigned_env_id, assigned_reused,
                    last_error, pending_master_lease_id,
                    pending_fence_token, pending_generation,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_key) DO UPDATE SET
                    episode_id=excluded.episode_id,
                    allocation_key=excluded.allocation_key,
                    payload_hash=excluded.payload_hash,
                    owner_json=excluded.owner_json,
                    status=excluded.status,
                    dispatching_worker_id=excluded.dispatching_worker_id,
                    dispatching_worker_incarnation=
                        excluded.dispatching_worker_incarnation,
                    assigned_lease_id=excluded.assigned_lease_id,
                    assigned_env_id=excluded.assigned_env_id,
                    assigned_reused=excluded.assigned_reused,
                    last_error=excluded.last_error,
                    pending_master_lease_id=excluded.pending_master_lease_id,
                    pending_fence_token=excluded.pending_fence_token,
                    pending_generation=excluded.pending_generation,
                    updated_at=excluded.updated_at
                """,
                (
                    intent.request_key,
                    intent.episode_id,
                    intent.allocation_key,
                    intent.payload_hash,
                    json.dumps(
                        intent.owner,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    intent.status,
                    intent.dispatching_worker_id,
                    intent.dispatching_worker_incarnation,
                    intent.assigned_lease_id,
                    intent.assigned_env_id,
                    int(intent.assigned_reused),
                    intent.last_error,
                    intent.pending_master_lease_id,
                    intent.pending_fence_token,
                    intent.pending_generation,
                    intent.created_at,
                    now,
                ),
            )

    def save_lease_and_intent(
        self,
        lease: LeaseRecord,
        intent: AllocateRequestRecord,
    ) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self.save(lease)
                self.save_intent(intent)
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def delete_intent(self, request_key: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                DELETE FROM env_master_allocation_intents
                WHERE request_key=?
                """,
                (request_key,),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

class WorkerRegistry:
    def __init__(
        self,
        heartbeat_timeout_seconds: int,
        proxy_timeout_seconds: int,
        state_db: str | None = None,
    ) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.proxy_timeout_seconds = proxy_timeout_seconds
        self._workers: dict[str, WorkerRecord] = {}
        self._leases: dict[str, LeaseRecord] = {}
        self._lock = threading.RLock()
        self._allocate_cv = threading.Condition(self._lock)
        self._allocate_requests: dict[str, AllocateRequestRecord] = {}
        self._allocation_queue: deque[str] = deque()
        self._store = SQLiteLeaseStore(state_db) if state_db else None
        if self._store is not None:
            for lease in self._store.load():
                self._leases[lease.master_lease_id] = lease
            for allocate_request in self._store.load_intents():
                if allocate_request.status == "dispatching":

                    allocate_request.status = "pending"
                    self._store.save_intent(allocate_request)
                self._allocate_requests[
                    allocate_request.request_key
                ] = allocate_request
                if allocate_request.status == "pending":
                    self._allocation_queue.append(
                        allocate_request.request_key
                    )
            for lease in self._leases.values():
                if lease.status in {"ACTIVE", "CLOSING"}:
                    request_key = lease.allocation_key or lease.episode_id
                    if request_key not in self._allocate_requests:
                        allocate_request = AllocateRequestRecord(
                            episode_id=lease.episode_id,
                            request_key=request_key,
                            allocation_key=lease.allocation_key,
                            payload_hash=lease.allocation_payload_hash,
                            owner=dict(lease.owner),
                            status="assigned",
                            assigned_lease_id=lease.master_lease_id,
                            assigned_env_id=(
                                lease.env_id or lease.worker_lease_id
                            ),
                            pending_master_lease_id=lease.master_lease_id,
                            pending_fence_token=lease.fence_token,
                            pending_generation=lease.generation,
                        )
                        self._allocate_requests[
                            request_key
                        ] = allocate_request
                        self._store.save_intent(allocate_request)

    def close_store(self) -> None:
        if self._store is not None:
            self._store.close()

    def _save_lease_locked(self, lease: LeaseRecord) -> None:
        if self._store is not None:
            self._store.save(lease)

    def _save_allocate_request_locked(
        self,
        allocate_request: AllocateRequestRecord,
    ) -> None:
        if self._store is not None:
            self._store.save_intent(allocate_request)

    @staticmethod
    def _inventory_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        inventory = payload.get("active_leases")
        if inventory is None:
            inventory = payload.get("lease_inventory")
        if not isinstance(inventory, list):
            return ()
        return tuple(dict(item) for item in inventory if isinstance(item, dict))

    def _worker_id_from_payload(self, payload: dict[str, Any]) -> str:
        worker_id = payload.get("worker_id")
        if worker_id:
            return str(worker_id)
        pid = payload.get("pid")
        base_url = payload.get("base_url", "")
        if pid is None:
            raise KeyError("worker_id is required")
        if not base_url:
            matches = [worker.worker_id for worker in self._workers.values() if worker.pid == int(pid)]
            if len(matches) == 1:
                return matches[0]
        return f"legacy:{pid}:{base_url}"

    def _refresh_worker_health_locked(self) -> None:
        now = time.time()
        stale_worker_ids = [
            worker_id
            for worker_id, worker in self._workers.items()
            if now - worker.last_heartbeat > self.heartbeat_timeout_seconds
        ]
        for worker_id in stale_worker_ids:
            logger.warning("Worker %s marked stale after heartbeat timeout", worker_id)

    def register_worker(self, payload: dict[str, Any]) -> WorkerRecord:
        worker_id = self._worker_id_from_payload(payload)
        pid = int(payload["pid"])
        base_url = str(payload["base_url"]).rstrip("/")
        capacity = int(payload.get("capacity", 0))
        status = dict(payload.get("status") or {})
        incarnation = str(payload.get("incarnation") or payload.get("worker_incarnation") or "")
        inventory = self._inventory_from_payload(payload)
        now = time.time()

        with self._allocate_cv:
            worker = WorkerRecord(
                worker_id=worker_id,
                pid=pid,
                base_url=base_url,
                capacity=capacity,
                status=status,
                registered_at=now,
                last_heartbeat=now,
                incarnation=incarnation,
                lease_inventory=inventory,
            )
            self._workers[worker_id] = worker
            if incarnation or "active_leases" in payload or "lease_inventory" in payload:
                self._reconcile_worker_locked(worker, inventory)
            logger.info("Registered worker_id=%s pid=%s base_url=%s", worker_id, pid, base_url)
            self._allocate_cv.notify_all()
            return worker

    def update_worker_heartbeat(self, payload: dict[str, Any]) -> WorkerRecord:
        worker_id = self._worker_id_from_payload(payload)
        status = dict(payload.get("status") or {})
        incarnation = str(payload.get("incarnation") or payload.get("worker_incarnation") or "")
        inventory = self._inventory_from_payload(payload)
        now = time.time()

        with self._allocate_cv:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise LeaseProtocolError(
                    "unknown_worker",
                    f"Unknown worker_id: {worker_id}",
                    404,
                    {"worker_id": worker_id},
                )
            if incarnation and worker.incarnation and incarnation != worker.incarnation:
                raise LeaseProtocolError(
                    "stale_worker_incarnation",
                    "Worker incarnation changed; re-registration is required",
                    409,
                    {
                        "worker_id": worker_id,
                        "registered_incarnation": worker.incarnation,
                        "reported_incarnation": incarnation,
                    },
                )
            was_healthy = now - worker.last_heartbeat <= self.heartbeat_timeout_seconds
            previous_available = self._available_capacity_locked(worker)
            if "pid" in payload:
                worker.pid = int(payload["pid"])
            worker.status = status
            worker.last_heartbeat = now
            if incarnation:
                worker.incarnation = incarnation
            if "active_leases" in payload or "lease_inventory" in payload:
                worker.lease_inventory = inventory
                self._reconcile_worker_locked(worker, inventory)
            current_available = self._available_capacity_locked(worker)
            if not was_healthy or current_available != previous_available:
                self._allocate_cv.notify_all()
            return worker

    def _reconcile_worker_locked(
        self,
        worker: WorkerRecord,
        inventory: tuple[dict[str, Any], ...],
    ) -> dict[str, int]:
        adopted = 0
        for item in inventory:
            allocation_key = item.get("allocation_key")
            worker_lease_id = item.get("lease_id") or item.get(
                "worker_lease_id"
            )
            if not allocation_key or not worker_lease_id:
                continue
            allocate_request = self._allocate_requests.get(
                str(allocation_key)
            )
            if (
                allocate_request is None
                or allocate_request.status not in {"pending", "dispatching"}
                or (
                    allocate_request.dispatching_worker_id is not None
                    and allocate_request.dispatching_worker_id
                    != worker.worker_id
                )
            ):
                continue
            item_owner = item.get("owner") or {}
            if (
                not isinstance(item_owner, dict)
                or dict(item_owner) != allocate_request.owner
                or str(item.get("episode_id") or "")
                != allocate_request.episode_id
                or str(item.get("fence_token") or "")
                != allocate_request.pending_fence_token
                or int(item.get("generation") or 0)
                != allocate_request.pending_generation
            ):
                logger.warning(
                    "Refusing unsafe allocation adoption key=%s worker=%s",
                    allocation_key,
                    worker.worker_id,
                )
                continue
            collision = next(
                (
                    lease
                    for lease in self._leases.values()
                    if lease.worker_id == worker.worker_id
                    and lease.worker_lease_id == str(worker_lease_id)
                    and lease.status in {"ACTIVE", "CLOSING"}
                ),
                None,
            )
            if collision is not None:
                continue
            lease = self._create_master_lease_locked(
                worker,
                str(worker_lease_id),
                allocate_request.episode_id,
                env_id=(
                    str(item["env_id"])
                    if item.get("env_id") is not None
                    else str(worker_lease_id)
                ),
                master_lease_id=(
                    allocate_request.pending_master_lease_id
                ),
                allocation_key=allocate_request.allocation_key,
                allocation_payload_hash=allocate_request.payload_hash,
                owner=allocate_request.owner,
                generation=allocate_request.pending_generation,
                fence_token=allocate_request.pending_fence_token,
                persist=False,
            )
            if allocate_request.dispatching_worker_id == worker.worker_id:
                worker.reserved_allocations = max(
                    0, worker.reserved_allocations - 1
                )
            allocate_request.status = "assigned"
            allocate_request.dispatching_worker_id = None
            allocate_request.assigned_lease_id = lease.master_lease_id
            allocate_request.assigned_env_id = (
                lease.env_id or lease.worker_lease_id
            )
            allocate_request.assigned_reused = True
            allocate_request.last_error = None
            try:
                self._allocation_queue.remove(
                    allocate_request.request_key
                )
            except ValueError:
                pass
            if self._store is not None:
                self._store.save_lease_and_intent(
                    lease,
                    allocate_request,
                )
            adopted += 1

        for allocate_request in self._allocate_requests.values():
            if (
                allocate_request.status == "pending"
                and allocate_request.dispatching_worker_id == worker.worker_id
                and allocate_request.dispatching_worker_incarnation
                and worker.incarnation
                and allocate_request.dispatching_worker_incarnation
                != worker.incarnation
            ):
                allocate_request.dispatching_worker_incarnation = (
                    worker.incarnation
                )
                self._save_allocate_request_locked(allocate_request)
                if allocate_request.request_key not in self._allocation_queue:
                    self._allocation_queue.append(allocate_request.request_key)

        inventory_ids = {
            str(item.get("lease_id") or item.get("worker_lease_id"))
            for item in inventory
            if item.get("lease_id") or item.get("worker_lease_id")
        }
        reconciled = 0
        lost = 0
        now = time.time()
        for lease in self._leases.values():
            if lease.worker_id != worker.worker_id or lease.status not in {"ACTIVE", "CLOSING"}:
                continue
            incarnation_matches = bool(
                worker.incarnation
                and lease.worker_incarnation
                and worker.incarnation == lease.worker_incarnation
            )
            inventory_matches = lease.worker_lease_id in inventory_ids
            if incarnation_matches and lease.status == "CLOSING" and not inventory_matches:

                lease.worker_base_url = worker.base_url
                lease.updated_at = now
                self._save_lease_locked(lease)
                reconciled += 1
                continue
            if not incarnation_matches or not inventory_matches:
                lease.status = "LOST"
                lease.lost_at = now
                lease.updated_at = now
                lease.close_status = "LOST"
                lease.close_error = (
                    "worker incarnation changed"
                    if not incarnation_matches
                    else "lease absent from worker inventory"
                )
                self._save_lease_locked(lease)
                lost += 1
                request_key = lease.allocation_key or lease.episode_id
                allocate_request = self._allocate_requests.get(request_key)
                if (
                    allocate_request is not None
                    and allocate_request.assigned_lease_id == lease.master_lease_id
                ):
                    self._remove_allocate_request_locked(request_key)
                continue
            lease.worker_base_url = worker.base_url
            lease.updated_at = now
            self._save_lease_locked(lease)
            reconciled += 1
        worker.active_leases = sum(
            1
            for lease in self._leases.values()
            if lease.worker_id == worker.worker_id and lease.status in {"ACTIVE", "CLOSING"}
        )
        return {
            "reconciled": reconciled,
            "lost": lost,
            "adopted": adopted,
        }

    def reconcile_worker(self, payload: dict[str, Any]) -> dict[str, int]:
        worker_id = self._worker_id_from_payload(payload)
        with self._allocate_cv:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise LeaseProtocolError(
                    "unknown_worker",
                    f"Unknown worker_id: {worker_id}",
                    404,
                    {"worker_id": worker_id},
                )
            incarnation = str(payload.get("incarnation") or payload.get("worker_incarnation") or "")
            if incarnation:
                worker.incarnation = incarnation
            inventory = self._inventory_from_payload(payload)
            worker.lease_inventory = inventory
            result = self._reconcile_worker_locked(worker, inventory)
            self._allocate_cv.notify_all()
            return result

    def get_worker(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            self._refresh_worker_health_locked()
            worker = self._workers.get(worker_id)
            if worker is None:
                raise KeyError(f"Unknown worker_id: {worker_id}")
            if time.time() - worker.last_heartbeat > self.heartbeat_timeout_seconds:
                raise RuntimeError(f"Worker {worker_id} is stale")
            return worker

    def _max_envs_locked(self, worker: WorkerRecord) -> int:
        return int(worker.status.get("max_envs", worker.capacity))

    def _reported_busy_envs_locked(self, worker: WorkerRecord) -> int:
        return int(worker.status.get("busy_envs", 0))

    def _available_capacity_locked(self, worker: WorkerRecord) -> int:
        max_envs = self._max_envs_locked(worker)
        busy_envs = max(self._reported_busy_envs_locked(worker), worker.active_leases)
        return max(0, max_envs - busy_envs - worker.reserved_allocations)

    def _get_allocation_candidates_locked(self) -> list[WorkerRecord]:
        self._refresh_worker_health_locked()
        now = time.time()
        candidates = [
            worker
            for worker in self._workers.values()
            if now - worker.last_heartbeat <= self.heartbeat_timeout_seconds
            and self._available_capacity_locked(worker) > 0
        ]
        candidates.sort(
            key=lambda worker: (
                self._available_capacity_locked(worker),
                worker.last_heartbeat,
            ),
            reverse=True,
        )
        return candidates

    def get_allocation_candidates(self) -> list[WorkerRecord]:
        with self._lock:
            return self._get_allocation_candidates_locked()

    def _create_master_lease_locked(
        self,
        worker: WorkerRecord,
        worker_lease_id: str,
        episode_id: str,
        *,
        env_id: str | None = None,
        master_lease_id: str | None = None,
        allocation_key: str | None = None,
        allocation_payload_hash: str = "",
        owner: dict[str, Any] | None = None,
        generation: int = 1,
        fence_token: str | None = None,
        persist: bool = True,
    ) -> LeaseRecord:
        master_lease_id = master_lease_id or f"master-lease-{uuid.uuid4().hex[:16]}"
        lease = LeaseRecord(
            master_lease_id=master_lease_id,
            worker_id=worker.worker_id,
            worker_base_url=worker.base_url,
            worker_lease_id=worker_lease_id,
            episode_id=episode_id,
            env_id=env_id,
            allocation_key=allocation_key,
            allocation_payload_hash=allocation_payload_hash,
            owner=dict(owner or {}),
            worker_incarnation=worker.incarnation,
            generation=generation,
            fence_token=fence_token or uuid.uuid4().hex,
        )
        self._leases[master_lease_id] = lease
        if persist:
            self._save_lease_locked(lease)
        return lease

    def create_master_lease(
        self,
        worker: WorkerRecord,
        worker_lease_id: str,
        episode_id: str,
        *,
        env_id: str | None = None,
        allocation_key: str | None = None,
        allocation_payload_hash: str = "",
        owner: dict[str, Any] | None = None,
        generation: int = 1,
        fence_token: str | None = None,
    ) -> LeaseRecord:
        with self._lock:
            return self._create_master_lease_locked(
                worker,
                worker_lease_id,
                episode_id,
                env_id=env_id,
                allocation_key=allocation_key,
                allocation_payload_hash=allocation_payload_hash,
                owner=owner,
                generation=generation,
                fence_token=fence_token,
            )

    def get_lease(self, master_lease_id: str) -> LeaseRecord:
        with self._lock:
            lease = self._leases.get(master_lease_id)
            if lease is None:
                raise KeyError(f"Unknown lease_id: {master_lease_id}")
            return lease

    def list_leases(self, status: str | None = None) -> list[LeaseRecord]:
        with self._lock:
            leases = list(self._leases.values())
            if status:
                leases = [lease for lease in leases if lease.status == status.upper()]
            leases.sort(key=lambda lease: (lease.created_at, lease.master_lease_id))
            return leases

    def lease_routing_error(self, lease: LeaseRecord) -> LeaseProtocolError | None:
        with self._lock:
            worker = self._workers.get(lease.worker_id)
            if worker is None:
                return LeaseProtocolError(
                    "lease_recovering",
                    "Lease route is waiting for worker re-registration",
                    503,
                    {"lease_id": lease.master_lease_id, "worker_id": lease.worker_id},
                )
            if (
                lease.worker_incarnation
                and worker.incarnation != lease.worker_incarnation
            ):
                return LeaseProtocolError(
                    "fenced",
                    "Lease belongs to a different worker incarnation",
                    409,
                    {
                        "lease_id": lease.master_lease_id,
                        "lease_incarnation": lease.worker_incarnation,
                        "worker_incarnation": worker.incarnation,
                    },
                )
            if time.time() - worker.last_heartbeat > self.heartbeat_timeout_seconds:
                return LeaseProtocolError(
                    "worker_stale",
                    "Lease worker heartbeat is stale",
                    503,
                    {"lease_id": lease.master_lease_id, "worker_id": lease.worker_id},
                )
            return None

    def _remove_allocate_request_locked(self, request_key: str) -> None:
        self._allocate_requests.pop(request_key, None)
        if self._store is not None:
            self._store.delete_intent(request_key)
        try:
            self._allocation_queue.remove(request_key)
        except ValueError:
            pass

    def _note_allocate_locked(self, worker: WorkerRecord, reused: bool) -> None:
        status = dict(worker.status)
        total_envs = int(status.get("total_envs", 0))
        busy_envs = int(status.get("busy_envs", 0))
        leases = int(status.get("leases", 0))
        max_envs = self._max_envs_locked(worker)
        if reused:
            total_envs = max(total_envs, busy_envs + 1)
        else:
            total_envs = min(max_envs, max(total_envs + 1, busy_envs + 1))
        busy_envs = min(max_envs, busy_envs + 1)
        leases += 1
        status["total_envs"] = total_envs
        status["busy_envs"] = busy_envs
        status["idle_envs"] = max(0, total_envs - busy_envs)
        status["leases"] = leases
        status["max_envs"] = max_envs
        worker.status = status

    def _note_close_locked(self, worker: WorkerRecord) -> None:
        status = dict(worker.status)
        total_envs = int(status.get("total_envs", 0))
        busy_envs = max(0, int(status.get("busy_envs", 0)) - 1)
        leases = max(0, int(status.get("leases", 0)) - 1)
        status["total_envs"] = total_envs
        status["busy_envs"] = busy_envs
        status["idle_envs"] = max(0, total_envs - busy_envs)
        status["leases"] = leases
        worker.status = status

    def note_allocate(self, worker_id: str, reused: bool) -> None:
        with self._allocate_cv:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            self._note_allocate_locked(worker, reused)

    def note_close(self, worker_id: str) -> None:
        with self._allocate_cv:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            worker.active_leases = max(0, worker.active_leases - 1)
            self._note_close_locked(worker)
            self._allocate_cv.notify_all()

    def begin_allocate_request(
        self,
        episode_id: str,
        allocation_key: str | None = None,
        owner: dict[str, Any] | None = None,
        payload_hash: str = "",
        fence_token: str | None = None,
        generation: int | None = None,
    ) -> str:
        request_key = allocation_key or episode_id
        with self._allocate_cv:
            allocate_request = self._allocate_requests.get(request_key)
            if (
                allocate_request is not None
                and allocate_request.payload_hash
                and payload_hash
                and allocate_request.payload_hash != payload_hash
            ):
                raise LeaseProtocolError(
                    "idempotency_conflict",
                    "allocation key was already used with a different payload",
                    409,
                    {"allocation_key": request_key},
                )
            if allocate_request is not None and allocate_request.status == "assigned":
                lease = self._leases.get(str(allocate_request.assigned_lease_id))
                if lease is None or lease.status not in {"ACTIVE", "CLOSING"}:
                    self._remove_allocate_request_locked(request_key)
                    allocate_request = None
            if allocate_request is None:
                previous_generations = [
                    lease.generation
                    for lease in self._leases.values()
                    if (lease.allocation_key or lease.episode_id) == request_key
                ]
                allocate_request = AllocateRequestRecord(
                    episode_id=episode_id,
                    request_key=request_key,
                    allocation_key=allocation_key,
                    payload_hash=payload_hash,
                    owner=dict(owner or {}),
                    pending_master_lease_id=f"master-lease-{uuid.uuid4().hex[:16]}",
                    pending_fence_token=(
                        str(fence_token)
                        if fence_token is not None
                        else uuid.uuid4().hex
                    ),
                    pending_generation=(
                        int(generation)
                        if generation is not None
                        else max(previous_generations, default=0) + 1
                    ),
                )
                self._allocate_requests[request_key] = allocate_request

                self._save_allocate_request_locked(allocate_request)
                self._allocation_queue.append(request_key)
                logger.info(
                    "Queued allocate request episode_id=%s allocation_key=%s pending_requests=%s",
                    episode_id,
                    request_key,
                    len(self._allocation_queue),
                )
                self._allocate_cv.notify_all()
            allocate_request.waiter_count += 1
            return request_key

    def end_allocate_request(self, request_key: str) -> None:
        with self._allocate_cv:
            allocate_request = self._allocate_requests.get(request_key)
            if allocate_request is None:
                return
            allocate_request.waiter_count = max(0, allocate_request.waiter_count - 1)
            if allocate_request.closed and allocate_request.waiter_count == 0:
                self._remove_allocate_request_locked(request_key)
                self._allocate_cv.notify_all()

    def _can_dispatch_request_locked(self, request_key: str) -> bool:
        for queued_request_key in self._allocation_queue:
            allocate_request = self._allocate_requests.get(queued_request_key)
            if allocate_request is None:
                continue
            if queued_request_key == request_key:
                return allocate_request.status == "pending"
            if allocate_request.status == "pending":
                return False
        return False

    def wait_for_allocate_assignment(
        self,
        request_key: str,
    ) -> tuple[str, WorkerRecord | dict[str, Any]]:
        with self._allocate_cv:
            while True:
                allocate_request = self._allocate_requests.get(request_key)
                if allocate_request is None:
                    raise KeyError(f"Unknown allocate request: {request_key}")

                if allocate_request.status == "assigned":
                    lease = self._leases.get(str(allocate_request.assigned_lease_id))
                    return (
                        "assigned",
                        {
                            "ok": True,
                            "lease_id": allocate_request.assigned_lease_id,
                            "env_id": allocate_request.assigned_env_id,
                            "reused": allocate_request.assigned_reused,
                            "worker_id": lease.worker_id if lease else None,
                            "worker_incarnation": lease.worker_incarnation if lease else None,
                            "fence_token": lease.fence_token if lease else None,
                            "generation": lease.generation if lease else None,
                        },
                    )

                if allocate_request.status in {"canceling", "cancelled"}:
                    raise LeaseProtocolError(
                        "allocation_cancelled",
                        "Allocation was cancelled",
                        409,
                        {"allocation_key": request_key},
                    )

                if allocate_request.status == "dispatching":
                    self._allocate_cv.wait()
                    continue

                if not self._can_dispatch_request_locked(request_key):
                    self._allocate_cv.wait()
                    continue

                candidates = self._get_allocation_candidates_locked()
                if allocate_request.dispatching_worker_id is not None:
                    candidates = [
                        worker
                        for worker in candidates
                        if worker.worker_id
                        == allocate_request.dispatching_worker_id
                        and (
                            not allocate_request.dispatching_worker_incarnation
                            or not worker.incarnation
                            or worker.incarnation
                            == allocate_request.dispatching_worker_incarnation
                        )
                    ]
                if not candidates:
                    self._allocate_cv.wait()
                    continue

                worker = candidates[0]
                worker.reserved_allocations += 1
                allocate_request.status = "dispatching"
                allocate_request.dispatching_worker_id = worker.worker_id
                allocate_request.dispatching_worker_incarnation = (
                    worker.incarnation
                )
                allocate_request.last_error = None
                self._save_allocate_request_locked(allocate_request)
                return "dispatch", worker

    def requeue_allocate_request(self, request_key: str, worker_id: str, error: str) -> None:
        with self._allocate_cv:
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.reserved_allocations = max(0, worker.reserved_allocations - 1)

            allocate_request = self._allocate_requests.get(request_key)
            if (
                allocate_request is not None
                and allocate_request.status in {"pending", "dispatching"}
            ):
                allocate_request.status = "pending"
                allocate_request.last_error = error
                self._save_allocate_request_locked(allocate_request)

            self._allocate_cv.notify_all()

    def fulfill_allocate_request(
        self,
        request_key: str,
        worker_id: str,
        worker_lease_id: str,
        env_id: str | None,
        reused: bool,
    ) -> dict[str, Any]:
        with self._allocate_cv:
            allocate_request = self._allocate_requests.get(request_key)
            if allocate_request is None:
                raise KeyError(f"Unknown allocate request: {request_key}")

            worker = self._workers.get(worker_id)
            if worker is None:
                raise KeyError(f"Unknown worker_id: {worker_id}")

            if (
                allocate_request.status in {"assigned", "canceling"}
                and allocate_request.assigned_lease_id
            ):
                existing = self._leases.get(
                    str(allocate_request.assigned_lease_id)
                )
                if (
                    existing is not None
                    and existing.worker_id == worker_id
                    and existing.worker_lease_id == str(worker_lease_id)
                ):
                    return {
                        "ok": True,
                        "lease_id": existing.master_lease_id,
                        "env_id": existing.env_id,
                        "reused": True,
                        "worker_id": existing.worker_id,
                        "worker_incarnation": existing.worker_incarnation,
                        "fence_token": existing.fence_token,
                        "generation": existing.generation,
                    }
                raise LeaseProtocolError(
                    "allocation_already_assigned",
                    "Allocation was already assigned to a different worker lease",
                    409,
                    {"allocation_key": request_key},
                )
            if allocate_request.status in {"canceling", "cancelled"}:
                raise LeaseProtocolError(
                    "allocation_cancelled",
                    "Allocation was cancelled while the worker request was in flight",
                    409,
                    {"allocation_key": request_key},
                )
            if (
                allocate_request.status != "dispatching"
                or allocate_request.dispatching_worker_id != worker_id
                or (
                    allocate_request.dispatching_worker_incarnation
                    and worker.incarnation
                    and allocate_request.dispatching_worker_incarnation
                    != worker.incarnation
                )
            ):
                raise LeaseProtocolError(
                    "allocation_dispatch_fenced",
                    "Worker allocation response lost its dispatch affinity",
                    409,
                    {"allocation_key": request_key, "worker_id": worker_id},
                )

            worker.reserved_allocations = max(0, worker.reserved_allocations - 1)
            worker.active_leases += 1
            lease = self._create_master_lease_locked(
                worker,
                worker_lease_id,
                allocate_request.episode_id,
                env_id=env_id,
                master_lease_id=allocate_request.pending_master_lease_id,
                allocation_key=allocate_request.allocation_key,
                allocation_payload_hash=allocate_request.payload_hash,
                owner=allocate_request.owner,
                generation=allocate_request.pending_generation,
                fence_token=allocate_request.pending_fence_token,
                persist=False,
            )
            self._note_allocate_locked(worker, reused)

            allocate_request.status = "assigned"
            allocate_request.dispatching_worker_id = None
            allocate_request.assigned_lease_id = lease.master_lease_id
            allocate_request.assigned_env_id = env_id
            allocate_request.assigned_reused = bool(reused)
            allocate_request.last_error = None
            if self._store is not None:
                self._store.save_lease_and_intent(
                    lease,
                    allocate_request,
                )

            try:
                self._allocation_queue.remove(request_key)
            except ValueError:
                pass

            self._allocate_cv.notify_all()
            return {
                "ok": True,
                "lease_id": lease.master_lease_id,
                "env_id": env_id,
                "reused": bool(reused),
                "worker_id": lease.worker_id,
                "worker_incarnation": lease.worker_incarnation,
                "fence_token": lease.fence_token,
                "generation": lease.generation,
            }

    def begin_cancel_allocation(
        self,
        request_key: str,
        *,
        fence_token: str,
        generation: int,
    ) -> dict[str, Any]:

        with self._allocate_cv:
            allocate_request = self._allocate_requests.get(request_key)
            if allocate_request is None:
                lease = next(
                    (
                        item
                        for item in self._leases.values()
                        if (item.allocation_key or item.episode_id) == request_key
                        and item.generation == int(generation)
                    ),
                    None,
                )
                if lease is None:
                    return {"kind": "terminal", "status": "CANCELLED"}
                if lease.fence_token != str(fence_token):
                    raise LeaseProtocolError(
                        "fenced",
                        "Allocation fence token does not match",
                        409,
                        {"allocation_key": request_key},
                    )
                return {"kind": "lease", "lease": lease}
            if (
                allocate_request.pending_fence_token != str(fence_token)
                or allocate_request.pending_generation != int(generation)
            ):
                raise LeaseProtocolError(
                    "fenced",
                    "Allocation cancellation fence does not match",
                    409,
                    {"allocation_key": request_key},
                )
            if allocate_request.status == "cancelled":
                return {"kind": "terminal", "status": "CANCELLED"}
            if allocate_request.status == "assigned":
                lease = self._leases.get(
                    str(allocate_request.assigned_lease_id)
                )
                if lease is None or lease.status in {"CLOSED", "LOST"}:
                    allocate_request.status = "cancelled"
                    self._save_allocate_request_locked(allocate_request)
                    return {
                        "kind": "terminal",
                        "status": lease.status if lease is not None else "CANCELLED",
                    }
                allocate_request.status = "canceling"
                self._save_allocate_request_locked(allocate_request)
                self._allocate_cv.notify_all()
                return {"kind": "lease", "lease": lease}

            worker_id = allocate_request.dispatching_worker_id
            if worker_id is None:
                allocate_request.status = "cancelled"
                try:
                    self._allocation_queue.remove(request_key)
                except ValueError:
                    pass
                self._save_allocate_request_locked(allocate_request)
                self._allocate_cv.notify_all()
                return {"kind": "terminal", "status": "CANCELLED"}

            worker = self._workers.get(worker_id)
            if worker is None:
                allocate_request.status = "canceling"
                self._save_allocate_request_locked(allocate_request)
                self._allocate_cv.notify_all()
                raise LeaseProtocolError(
                    "allocation_recovering",
                    "Pinned allocation worker has not re-registered",
                    503,
                    {"allocation_key": request_key, "worker_id": worker_id},
                )
            if (
                allocate_request.dispatching_worker_incarnation
                and worker.incarnation
                and allocate_request.dispatching_worker_incarnation
                != worker.incarnation
            ):
                allocate_request.status = "cancelled"
                self._save_allocate_request_locked(allocate_request)
                self._allocate_cv.notify_all()
                return {"kind": "terminal", "status": "LOST"}

            allocate_request.status = "canceling"
            try:
                self._allocation_queue.remove(request_key)
            except ValueError:
                pass
            self._save_allocate_request_locked(allocate_request)
            self._allocate_cv.notify_all()
            return {
                "kind": "worker",
                "worker": worker,
                "allocation_key": request_key,
            }

    def complete_cancel_allocation(self, request_key: str) -> None:
        with self._allocate_cv:
            allocate_request = self._allocate_requests.get(request_key)
            if allocate_request is None:
                return
            worker = (
                self._workers.get(allocate_request.dispatching_worker_id)
                if allocate_request.dispatching_worker_id
                else None
            )
            if worker is not None:
                worker.reserved_allocations = max(
                    0, worker.reserved_allocations - 1
                )
            allocate_request.status = "cancelled"
            allocate_request.last_error = None
            self._save_allocate_request_locked(allocate_request)
            try:
                self._allocation_queue.remove(request_key)
            except ValueError:
                pass
            self._allocate_cv.notify_all()

    def begin_close(self, lease: LeaseRecord, operation_id: str | None) -> str:
        with self._allocate_cv:
            current = self._leases.get(lease.master_lease_id)
            if current is None:
                return "CLOSED"
            if current.status == "CLOSED":
                return "CLOSED"
            if current.status == "LOST":
                return "LOST"
            now = time.time()
            if current.status == "ACTIVE":
                current.status = "CLOSING"
                current.closing_at = now
            current.updated_at = now
            current.close_operation_id = operation_id or current.close_operation_id
            current.close_status = "IN_PROGRESS"
            current.close_error = None
            current.close_attempts += 1
            self._save_lease_locked(current)
            return current.status

    def note_close_failure(self, lease: LeaseRecord, error: str) -> None:
        with self._allocate_cv:
            current = self._leases.get(lease.master_lease_id)
            if current is None or current.status in {"CLOSED", "LOST"}:
                return
            current.status = "CLOSING"
            current.close_status = "RETRYABLE_FAILURE"
            current.close_error = error
            current.updated_at = time.time()
            self._save_lease_locked(current)

    def complete_close(self, lease: LeaseRecord) -> None:
        with self._allocate_cv:
            current = self._leases.get(lease.master_lease_id)
            if current is None or current.status == "CLOSED":
                return
            now = time.time()
            current.status = "CLOSED"
            current.close_status = "SUCCEEDED"
            current.close_error = None
            current.closed_at = now
            current.updated_at = now
            self._save_lease_locked(current)

            worker = self._workers.get(current.worker_id)
            if worker is not None:
                worker.active_leases = max(0, worker.active_leases - 1)
                self._note_close_locked(worker)

            request_key = current.allocation_key or current.episode_id
            allocate_request = self._allocate_requests.get(request_key)
            if (
                allocate_request is not None
                and allocate_request.assigned_lease_id == current.master_lease_id
            ):
                allocate_request.closed = True
                if (
                    allocate_request.waiter_count == 0
                    and allocate_request.status
                    not in {"canceling", "cancelled"}
                ):
                    self._remove_allocate_request_locked(request_key)

            self._allocate_cv.notify_all()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_worker_health_locked()
            now = time.time()
            workers = []
            for worker in self._workers.values():
                heartbeat_age = now - worker.last_heartbeat
                workers.append(
                    {
                        "worker_id": worker.worker_id,
                        "incarnation": worker.incarnation,
                        "pid": worker.pid,
                        "base_url": worker.base_url,
                        "capacity": worker.capacity,
                        "status": worker.status,
                        "active_leases": worker.active_leases,
                        "reserved_allocations": worker.reserved_allocations,
                        "available_capacity": self._available_capacity_locked(worker),
                        "last_heartbeat": worker.last_heartbeat,
                        "heartbeat_age_seconds": heartbeat_age,
                        "healthy": heartbeat_age <= self.heartbeat_timeout_seconds,
                        "lease_inventory": list(worker.lease_inventory),
                    }
                )
            return {
                "lease_protocol": {
                    "version": 2,
                    "capabilities": [
                        "leases",
                        "worker_incarnation",
                        "fence_token",
                        "idempotent_allocation",
                        "allocation_cancellation",
                        *(["durable_routes"] if self._store is not None else []),
                    ],
                    "durable_state": self._store is not None,
                },
                "worker_count": len(workers),
                "healthy_workers": sum(1 for worker in workers if worker["healthy"]),
                "active_leases": sum(
                    1 for lease in self._leases.values() if lease.status in {"ACTIVE", "CLOSING"}
                ),
                "lease_state_counts": {
                    state: sum(1 for lease in self._leases.values() if lease.status == state)
                    for state in ("ACTIVE", "CLOSING", "CLOSED", "LOST")
                },
                "pending_allocate_requests": len(self._allocation_queue),
                "tracked_allocate_requests": len(self._allocate_requests),
                "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
                "workers": workers,
            }

    def snapshot_workers(self) -> list[WorkerRecord]:
        with self._lock:
            return list(self._workers.values())

REGISTRY: WorkerRegistry | None = None
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_DONE = False

def _json_error(message: str, status_code: int) -> tuple[dict[str, Any], int]:
    return {"ok": False, "error": message}, status_code

def _protocol_error(exc: LeaseProtocolError) -> tuple[dict[str, Any], int]:
    return {
        "ok": False,
        "error": str(exc),
        "code": exc.code,
        "details": exc.details,
    }, exc.status_code

def _require_registry() -> WorkerRegistry:
    if REGISTRY is None:
        raise RuntimeError("Registry is not initialized")
    return REGISTRY

def _cleanup_master(reason: str) -> None:
    global REGISTRY, _CLEANUP_DONE

    with _CLEANUP_LOCK:
        if _CLEANUP_DONE:
            return
        _CLEANUP_DONE = True
        registry = REGISTRY
        REGISTRY = None

    if registry is None:
        logger.info("Master cleanup skipped; registry not initialized (%s)", reason)
        return

    logger.warning("Master cleanup triggered due to %s; shutting down all workers", reason)
    workers = registry.snapshot_workers()
    shutdown_timeout = min(10, max(1, registry.proxy_timeout_seconds))
    for worker in workers:
        try:
            response = requests.post(
                f"{worker.base_url.rstrip('/')}/shutdown",
                json={"reason": f"master shutdown: {reason}"},
                timeout=shutdown_timeout,
            )
            response.raise_for_status()
            logger.info("Shutdown requested for worker_id=%s pid=%s", worker.worker_id, worker.pid)
        except Exception as exc:
            logger.exception(
                "Failed to shutdown worker_id=%s pid=%s during master cleanup: %s",
                worker.worker_id,
                worker.pid,
                exc,
            )
    registry.close_store()

def _handle_termination_signal(signum: int, _frame: Any) -> None:
    signame = signal.Signals(signum).name
    logger.warning("Received %s; shutting down all workers", signame)
    _cleanup_master(f"signal {signame}")
    raise SystemExit(128 + signum)

def _proxy_post(
    worker_base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[dict[str, Any], int]:
    response = requests.post(
        f"{worker_base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout_seconds,
    )
    try:
        body = response.json()
    except ValueError:
        body = {
            "ok": False,
            "error": response.text.strip() or f"Worker returned HTTP {response.status_code}",
        }
    return body, response.status_code

def _proxy_lease_request(path: str, data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    registry = _require_registry()
    lease_id = data.get("lease_id")
    if not lease_id:
        return _json_error("lease_id is required", 400)

    try:
        lease = registry.get_lease(str(lease_id))
    except KeyError:
        if path == "/close":
            return {"ok": True, "already_closed": True}, 200
        return _protocol_error(
            LeaseProtocolError(
                "lease_not_found",
                f"Unknown lease_id: {lease_id}",
                404,
                {"lease_id": lease_id},
            )
        )
    except Exception as exc:
        return _json_error(str(exc), 500)

    supplied_fence = data.get("fence_token")
    v2_owned = bool(
        lease.allocation_key
        or lease.owner.get("component") == "rollout-host-v2"
    )
    if v2_owned and supplied_fence is None:
        return _protocol_error(
            LeaseProtocolError(
                "fence_required",
                "Lease fence token is required",
                409,
                {
                    "lease_id": lease_id,
                    "generation": lease.generation,
                },
            )
        )
    if (
        supplied_fence is not None
        and str(supplied_fence) != lease.fence_token
    ):
        return _protocol_error(
            LeaseProtocolError(
                "fenced",
                "Lease fence token does not match",
                409,
                {
                    "lease_id": lease_id,
                    "generation": lease.generation,
                },
            )
        )

    if path == "/close" and lease.status == "CLOSED":
        if data.get("operation_id") is None and data.get("fence_token") is None:
            return {"ok": True, "already_closed": True}, 200
        return {
            "ok": True,
            "already_closed": True,
            "status": "CLOSED",
            "lease": lease.to_dict(),
        }, 200
    if lease.status == "LOST":
        if path == "/close":
            return {
                "ok": True,
                "already_closed": True,
                "status": "LOST",
                "lease": lease.to_dict(),
            }, 200
        return _protocol_error(
            LeaseProtocolError(
                "lease_lost",
                f"Lease {lease_id} is lost and cannot be routed",
                409,
                {"lease_id": lease_id, "status": lease.status},
            )
        )
    if path != "/close" and lease.status != "ACTIVE":
        return _protocol_error(
            LeaseProtocolError(
                "lease_not_active",
                f"Lease {lease_id} is not active",
                409,
                {"lease_id": lease_id, "status": lease.status},
            )
        )

    routing_error = registry.lease_routing_error(lease)
    if routing_error is not None:
        return _protocol_error(routing_error)

    if path == "/close":
        close_state = registry.begin_close(lease, data.get("operation_id"))
        if close_state == "CLOSED":
            return {"ok": True, "already_closed": True, "status": "CLOSED"}, 200
        if close_state == "LOST":
            return _protocol_error(
                LeaseProtocolError("lease_lost", f"Lease {lease_id} is lost", 409)
            )

    payload = dict(data)
    payload["lease_id"] = lease.worker_lease_id
    if supplied_fence is not None:
        payload["fence_token"] = lease.fence_token
    try:
        body, status_code = _proxy_post(
            lease.worker_base_url,
            path,
            payload,
            registry.proxy_timeout_seconds,
        )
    except Exception as exc:
        if path == "/close":
            logger.warning(
                "Failed to proxy /close to worker for lease %s: %s; close remains retryable",
                lease_id, exc,
            )
            registry.note_close_failure(lease, str(exc))
            return {
                "ok": False,
                "error": f"Failed to proxy close to worker: {exc}",
                "code": "close_transport_failed",
                "retryable": True,
                "status": "CLOSING",
            }, 502
        return _json_error(f"Failed to proxy to worker: {exc}", 502)

    if path == "/close":
        if not (200 <= status_code < 300) or not body.get("ok", False):
            error = str(body.get("error") or f"worker returned HTTP {status_code}")
            registry.note_close_failure(lease, error)
            return {
                "ok": False,
                "error": error,
                "code": "close_worker_failed",
                "retryable": True,
                "status": "CLOSING",
                "worker_status": status_code,
            }, 502
        registry.complete_close(lease)
        if data.get("operation_id") is not None or supplied_fence is not None:
            body = dict(body)
            body["status"] = "CLOSED"
            body["lease_id"] = lease.master_lease_id

    return body, status_code

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.get("/status")
def status():
    try:
        registry = _require_registry()
        return jsonify({"ok": True, "master": registry.summary()})
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

@app.get("/leases")
def list_leases():
    try:
        registry = _require_registry()
        leases = [
            lease.to_dict()
            for lease in registry.list_leases(request.args.get("status"))
        ]
        return jsonify({"ok": True, "leases": leases, "count": len(leases)})
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

@app.get("/leases/<master_lease_id>")
def get_lease(master_lease_id: str):
    try:
        lease = _require_registry().get_lease(master_lease_id)
        return jsonify({"ok": True, "lease": lease.to_dict()})
    except KeyError:
        body, status_code = _protocol_error(
            LeaseProtocolError(
                "lease_not_found",
                f"Unknown lease_id: {master_lease_id}",
                404,
                {"lease_id": master_lease_id},
            )
        )
        return jsonify(body), status_code
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

@app.post("/register")
def register_worker():
    data = request.get_json(force=True, silent=True) or {}
    for required_key in ("base_url",):
        if required_key not in data:
            body, status_code = _json_error(f"{required_key} is required", 400)
            return jsonify(body), status_code

    try:
        registry = _require_registry()
        worker = registry.register_worker(data)
    except LeaseProtocolError as exc:
        body, status_code = _protocol_error(exc)
        return jsonify(body), status_code
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

    return jsonify(
        {
            "ok": True,
            "worker_id": worker.worker_id,
            "pid": worker.pid,
            "base_url": worker.base_url,
            "incarnation": worker.incarnation,
        }
    )

@app.post("/reconcile")
def reconcile_worker():
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = _require_registry().reconcile_worker(data)
        return jsonify({"ok": True, **result})
    except LeaseProtocolError as exc:
        body, status_code = _protocol_error(exc)
        return jsonify(body), status_code
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

@app.post("/heartbeat")
def heartbeat():
    data = request.get_json(force=True, silent=True) or {}

    if "lease_id" in data:
        body, status_code = _proxy_lease_request("/heartbeat", data)
        return jsonify(body), status_code

    if "worker_id" not in data and "pid" not in data:
        body, status_code = _json_error("worker_id is required", 400)
        return jsonify(body), status_code

    try:
        registry = _require_registry()
        worker = registry.update_worker_heartbeat(data)
    except LeaseProtocolError as exc:
        body, status_code = _protocol_error(exc)
        return jsonify(body), status_code
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

    return jsonify(
        {
            "ok": True,
            "worker_id": worker.worker_id,
            "pid": worker.pid,
            "last_heartbeat": worker.last_heartbeat,
        }
    )

@app.post("/allocate")
def allocate():
    data = request.get_json(force=True, silent=True) or {}
    episode_id = str(data.get("episode_id", "")).strip()
    if not episode_id:
        body, status_code = _json_error("episode_id is required", 400)
        return jsonify(body), status_code

    try:
        registry = _require_registry()
    except Exception as exc:
        body, status_code = _json_error(str(exc), 500)
        return jsonify(body), status_code

    allocation_key_value = data.get("allocation_key", data.get("idempotency_key"))
    allocation_key = (
        str(allocation_key_value).strip()
        if allocation_key_value is not None
        else None
    )
    if allocation_key_value is not None and not allocation_key:
        body, status_code = _protocol_error(
            LeaseProtocolError(
                "validation_error",
                "allocation_key must not be empty",
                400,
            )
        )
        return jsonify(body), status_code
    owner_value = data.get("owner") or {}
    if not isinstance(owner_value, dict):
        body, status_code = _protocol_error(
            LeaseProtocolError("validation_error", "owner must be an object", 400)
        )
        return jsonify(body), status_code
    owner = dict(owner_value)
    fence_token_value = data.get("fence_token")
    fence_token = (
        str(fence_token_value)
        if fence_token_value is not None
        else None
    )
    generation_value = data.get("generation")
    try:
        generation = (
            int(generation_value)
            if generation_value is not None
            else None
        )
        if generation is not None and generation <= 0:
            raise ValueError("generation must be positive")
    except (TypeError, ValueError) as exc:
        body, status_code = _protocol_error(
            LeaseProtocolError("validation_error", str(exc), 400)
        )
        return jsonify(body), status_code
    try:
        payload_hash = _canonical_hash(
            {
                "episode_id": episode_id,
                "owner": owner,
                "fence_token": fence_token,
                "generation": generation,
            }
        )
        request_key = registry.begin_allocate_request(
            episode_id,
            allocation_key=allocation_key,
            owner=owner,
            payload_hash=payload_hash,
            fence_token=fence_token,
            generation=generation,
        )
    except LeaseProtocolError as exc:
        body, status_code = _protocol_error(exc)
        return jsonify(body), status_code
    except (TypeError, ValueError) as exc:
        body, status_code = _protocol_error(
            LeaseProtocolError("validation_error", str(exc), 400)
        )
        return jsonify(body), status_code
    try:
        try:
            while True:
                dispatch_state, payload = registry.wait_for_allocate_assignment(request_key)
                if dispatch_state == "assigned":
                    return jsonify(payload)

                worker = payload
                if not isinstance(worker, WorkerRecord):
                    raise RuntimeError("dispatcher returned an invalid worker assignment")
                allocate_request = registry._allocate_requests.get(request_key)
                if allocate_request is None:
                    raise RuntimeError(f"allocate request disappeared: {request_key}")
                worker_payload: dict[str, Any] = {
                    "episode_id": episode_id,
                    "allocation_key": request_key,
                    "owner": owner,
                    "master_lease_id": allocate_request.pending_master_lease_id,
                    "fence_token": allocate_request.pending_fence_token,
                    "generation": allocate_request.pending_generation,
                    "v2_owned": bool(
                        allocate_request.allocation_key
                        or allocate_request.owner.get("component")
                        == "rollout-host-v2"
                    ),
                }

                try:
                    body, status_code = _proxy_post(
                        worker.base_url,
                        "/allocate",
                        worker_payload,
                        registry.proxy_timeout_seconds,
                    )
                except Exception as exc:
                    error = f"Worker {worker.worker_id} allocate failed: {exc}"
                    logger.warning(error)
                    registry.requeue_allocate_request(request_key, worker.worker_id, error)
                    continue

                if not body.get("ok", False):
                    if status_code == 409:
                        registry.requeue_allocate_request(request_key, worker.worker_id, str(body))
                        conflict = LeaseProtocolError(
                            str(body.get("code") or "worker_allocation_conflict"),
                            str(body.get("error") or "worker allocation conflict"),
                            409,
                            dict(body.get("details") or {}),
                        )
                        conflict_body, conflict_status = _protocol_error(conflict)
                        return jsonify(conflict_body), conflict_status
                    error = f"Worker {worker.worker_id} allocate rejected: {body.get('error', body)}"
                    logger.warning(error)
                    registry.requeue_allocate_request(request_key, worker.worker_id, error)
                    continue

                try:
                    response_body = registry.fulfill_allocate_request(
                        request_key=request_key,
                        worker_id=worker.worker_id,
                        worker_lease_id=body["lease_id"],
                        env_id=body.get("env_id"),
                        reused=bool(body.get("reused", False)),
                    )
                except LeaseProtocolError as exc:
                    if exc.code in {
                        "allocation_cancelled",
                        "allocation_dispatch_fenced",
                        "allocation_already_assigned",
                    }:
                        conflict_body, conflict_status = _protocol_error(exc)
                        return jsonify(conflict_body), conflict_status
                    error = (
                        "Failed to finalize allocate for "
                        f"episode_id={episode_id}: {exc}"
                    )
                    logger.exception(error)
                    registry.requeue_allocate_request(
                        request_key,
                        worker.worker_id,
                        error,
                    )
                    continue
                except Exception as exc:
                    error = f"Failed to finalize allocate for episode_id={episode_id}: {exc}"
                    logger.exception(error)
                    registry.requeue_allocate_request(request_key, worker.worker_id, error)
                    continue

                return jsonify(response_body)
        except LeaseProtocolError as exc:
            body, status_code = _protocol_error(exc)
            return jsonify(body), status_code
        except Exception as exc:
            body, status_code = _json_error(str(exc), 500)
            return jsonify(body), status_code
    finally:
        registry.end_allocate_request(request_key)

@app.post("/cancel_allocation")
def cancel_allocation():
    data = request.get_json(force=True, silent=True) or {}
    allocation_key = str(data.get("allocation_key") or "").strip()
    fence_token = str(data.get("fence_token") or "")
    operation_id = str(
        data.get("operation_id") or f"cancel-allocation:{allocation_key}"
    )
    try:
        generation = int(data["generation"])
    except (KeyError, TypeError, ValueError):
        body, status_code = _json_error(
            "allocation_key, fence_token and positive generation are required",
            400,
        )
        return jsonify(body), status_code
    if not allocation_key or not fence_token or generation <= 0:
        body, status_code = _json_error(
            "allocation_key, fence_token and positive generation are required",
            400,
        )
        return jsonify(body), status_code

    registry = _require_registry()
    try:
        plan = registry.begin_cancel_allocation(
            allocation_key,
            fence_token=fence_token,
            generation=generation,
        )
        if plan["kind"] == "terminal":
            return jsonify({
                "ok": True,
                "allocation_key": allocation_key,
                "status": plan.get("status", "CANCELLED"),
            })
        if plan["kind"] == "lease":
            lease = plan["lease"]
            body, status_code = _proxy_lease_request(
                "/close",
                {
                    "lease_id": lease.master_lease_id,
                    "fence_token": fence_token,
                    "operation_id": operation_id,
                },
            )
            if (
                body.get("code") == "lease_lost"
                or body.get("status") == "LOST"
            ):
                registry.complete_cancel_allocation(allocation_key)
                return jsonify({
                    "ok": True,
                    "allocation_key": allocation_key,
                    "status": "LOST",
                })
            if not (200 <= status_code < 300) or not body.get("ok", False):
                return jsonify(body), status_code
            registry.complete_cancel_allocation(allocation_key)
            return jsonify({
                "ok": True,
                "allocation_key": allocation_key,
                "status": "CANCELLED",
            })

        worker = plan["worker"]
        body, status_code = _proxy_post(
            worker.base_url,
            "/cancel_allocation",
            {
                "allocation_key": allocation_key,
                "fence_token": fence_token,
                "generation": generation,
                "operation_id": operation_id,
            },
            registry.proxy_timeout_seconds,
        )
        if not (200 <= status_code < 300) or not body.get("ok", False):
            return jsonify(body), status_code
        registry.complete_cancel_allocation(allocation_key)
        return jsonify({
            "ok": True,
            "allocation_key": allocation_key,
            "status": str(body.get("status") or "CANCELLED"),
        })
    except LeaseProtocolError as exc:
        body, status_code = _protocol_error(exc)
        return jsonify(body), status_code
    except Exception as exc:
        body, status_code = _json_error(str(exc), 502)
        return jsonify(body), status_code

@app.post("/reset")
def reset():
    body, status_code = _proxy_lease_request("/reset", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/get_obs")
def get_obs():
    body, status_code = _proxy_lease_request("/get_obs", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/prepare_for_probe")
def prepare_for_probe():
    body, status_code = _proxy_lease_request("/prepare_for_probe", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/step")
def step():
    body, status_code = _proxy_lease_request("/step", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/evaluate")
def evaluate():
    body, status_code = _proxy_lease_request("/evaluate", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/evaluate_inline")
def evaluate_inline():
    body, status_code = _proxy_lease_request("/evaluate_inline", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/start_recording")
def start_recording():
    body, status_code = _proxy_lease_request("/start_recording", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/end_recording")
def end_recording():
    body, status_code = _proxy_lease_request("/end_recording", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

@app.post("/close")
def close():
    body, status_code = _proxy_lease_request("/close", request.get_json(force=True, silent=True) or {})
    return jsonify(body), status_code

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI env master server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9900)
    parser.add_argument(
        "--heartbeat-timeout-seconds",
        type=int,
        default=15,
        help="Worker is considered stale after this many seconds without heartbeat",
    )
    parser.add_argument(
        "--proxy-timeout-seconds",
        type=int,
        default=300,
        help="Timeout used when proxying requests to workers",
    )
    parser.add_argument(
        "--state-db",
        default=None,
        help="Optional SQLite path for durable lease routes; unset keeps legacy in-memory state",
    )
    return parser.parse_args()

def main() -> None:
    global REGISTRY

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    atexit.register(_cleanup_master, "process exit")
    REGISTRY = WorkerRegistry(
        heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
        proxy_timeout_seconds=args.proxy_timeout_seconds,
        state_db=args.state_db,
    )
    logger.info(
        "Starting GUI env master server on %s:%s heartbeat_timeout=%ss proxy_timeout=%ss state_db=%s",
        args.host,
        args.port,
        args.heartbeat_timeout_seconds,
        args.proxy_timeout_seconds,
        args.state_db or "memory",
    )
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        _cleanup_master("server shutdown")

if __name__ == "__main__":
    main()
