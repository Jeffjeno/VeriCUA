from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
import hashlib
import json
import logging
import os
import signal
import sqlite3
import tempfile
import sys
import threading
import time
import uuid
from typing import Any

from flask import Flask, jsonify, request

logger = logging.getLogger("guirl.env_rpc_worker")
app = Flask(__name__)

_ENV_CLIENT = None
_WORKER_INFO: dict[str, Any] = {}
_MASTER_URL: str = ""
_HEARTBEAT_THREAD: threading.Thread | None = None
_SHUTDOWN = False
_PROCESS_INCARNATION = f"incarnation-{uuid.uuid4().hex}"
_STATE_LOCK = threading.RLock()
_WORKER_ALLOCATIONS: dict[str, dict[str, Any]] = {}
_LEASE_ALLOCATIONS: dict[str, dict[str, Any]] = {}
_OPERATION_RESULTS: dict[
    str,
    OrderedDict[tuple[str, str], tuple[str, dict[str, Any], int]],
] = {}

_CACHEABLE_OPERATION_ENDPOINTS = frozenset({
    "/reset", "/step", "/prepare_for_probe", "/evaluate", "/close",
})
_OPERATION_RESULT_CACHE_SIZE = max(
    1, int(os.environ.get("GUI_ENV_OPERATION_CACHE_SIZE", "64"))
)

_ENV_OP_TIMEOUT_S = float(os.environ.get("GUI_ENV_OP_TIMEOUT_S", "600"))
_ENV_OP_EXECUTOR: ThreadPoolExecutor | None = None
_LEASE_OP_LOCKS: dict[str, threading.Lock] = {}
_INFLIGHT_OPS: dict[str, dict[str, Any]] = {}

_INFLIGHT_BY_OPID: dict[tuple[str, str, str], tuple[str, Any]] = {}

_TOMBSTONE_DIR = None
_TOMBSTONE_DB = None

_TOMBSTONE_MAX = max(1, int(os.environ.get("GUI_ENV_TOMBSTONE_MAX", "4096")))
_TOMBSTONE_TTL_S = max(0.0, float(os.environ.get("GUI_ENV_TOMBSTONE_TTL_S", "86400")))
_PENDING_ALLOCATIONS: dict[str, tuple[str, str, Any]] = {}
_ALLOC_EXECUTOR: ThreadPoolExecutor | None = None

def _tombstones():
    global _TOMBSTONE_DIR, _TOMBSTONE_DB
    if _TOMBSTONE_DB is None:
        _TOMBSTONE_DIR = tempfile.TemporaryDirectory(prefix="guirl-worker-tombstones-")
        _TOMBSTONE_DB = sqlite3.connect(
            os.path.join(_TOMBSTONE_DIR.name, "closed.sqlite"), check_same_thread=False,
            isolation_level=None,
        )
        _TOMBSTONE_DB.execute("PRAGMA cache_size=-1024")

        _TOMBSTONE_DB.execute("PRAGMA synchronous=OFF")
        _TOMBSTONE_DB.execute("PRAGMA journal_mode=MEMORY")
        _TOMBSTONE_DB.execute(
            "CREATE TABLE closed (allocation_key TEXT PRIMARY KEY, lease_id TEXT, payload TEXT, closed_at REAL NOT NULL)"
        )
        _TOMBSTONE_DB.execute("CREATE INDEX closed_lease ON closed(lease_id)")
        _TOMBSTONE_DB.execute("CREATE INDEX closed_at_idx ON closed(closed_at)")
    return _TOMBSTONE_DB

def _closed_record(column, value):

    if _TOMBSTONE_DB is None:
        return None
    if _TOMBSTONE_TTL_S > 0:
        cutoff = time.time() - _TOMBSTONE_TTL_S
        _TOMBSTONE_DB.execute("DELETE FROM closed WHERE closed_at < ?", (cutoff,))
    row = _TOMBSTONE_DB.execute(
        f"SELECT payload FROM closed WHERE {column}=? ORDER BY rowid DESC LIMIT 1", (value,),
    ).fetchone()
    return json.loads(row[0]) if row else None

def _retire_lease(lease_id):
    allocation = _LEASE_ALLOCATIONS.get(lease_id)
    if allocation is None or allocation["status"] != "CLOSED" or allocation.get("cleanup_pending"):
        return
    if any(op["lease_id"] == lease_id for op in _INFLIGHT_OPS.values()):
        return
    close_results = [
        [list(key), value] for key, value in _OPERATION_RESULTS.get(lease_id, {}).items()
        if key[0] == "/close"
    ]
    db = _tombstones()
    closed_at = float(allocation.get("closed_at") or time.time())
    db.execute(
        "INSERT OR REPLACE INTO closed(allocation_key, lease_id, payload, closed_at) VALUES (?, ?, ?, ?)",
        (allocation["allocation_key"], lease_id, json.dumps({
            "allocation": allocation, "close_results": close_results,
        }), closed_at),
    )

    db.execute(
        "DELETE FROM closed WHERE rowid NOT IN "
        "(SELECT rowid FROM closed ORDER BY closed_at DESC, rowid DESC LIMIT ?)",
        (_TOMBSTONE_MAX,),
    )
    if _WORKER_ALLOCATIONS.get(allocation["allocation_key"]) is allocation:
        _WORKER_ALLOCATIONS.pop(allocation["allocation_key"], None)
    _LEASE_ALLOCATIONS.pop(lease_id, None)
    _OPERATION_RESULTS.pop(lease_id, None)
    _LEASE_OP_LOCKS.pop(lease_id, None)

_HEARTBEAT_LAST_BEAT: float = time.time()

_HEARTBEAT_STALL_S = float(os.environ.get("GUI_ENV_HEARTBEAT_STALL_S", "60"))

def _get_op_executor() -> ThreadPoolExecutor:
    global _ENV_OP_EXECUTOR
    if _ENV_OP_EXECUTOR is None:
        _ENV_OP_EXECUTOR = ThreadPoolExecutor(
            max_workers=int(os.environ.get("GUI_ENV_OP_MAX_WORKERS", "8")),
            thread_name_prefix="env-op",
        )
    return _ENV_OP_EXECUTOR

def _lease_op_lock(lease_id: str) -> threading.Lock:
    with _STATE_LOCK:
        lock = _LEASE_OP_LOCKS.get(lease_id)
        if lock is None:
            lock = threading.Lock()
            _LEASE_OP_LOCKS[lease_id] = lock
        return lock

def _wedged_ops(now: float | None = None) -> list[dict[str, Any]]:

    current = time.time() if now is None else now
    with _STATE_LOCK:
        return [
            dict(op)
            for op in _INFLIGHT_OPS.values()
            if current - op["started_at"] > _ENV_OP_TIMEOUT_S
        ]

def _json_error(endpoint: str, lease_id: str | None, exc: Exception) -> tuple:
    logger.exception("%s failed for lease_id=%s: %s", endpoint, lease_id, exc)
    return jsonify({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }), 500

def _structured_error(
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> tuple:
    return jsonify({
        "ok": False,
        "error": message,
        "code": code,
        "details": dict(details or {}),
    }), status_code

def _payload_hash(payload: dict[str, Any], *, exclude: set[str] | None = None) -> str:
    excluded = exclude or set()
    canonical = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def _active_lease_inventory() -> list[dict[str, Any]]:
    client = _ENV_CLIENT
    sessions = getattr(client, "_sessions", {}) if client is not None else {}
    with _STATE_LOCK:
        inventory: list[dict[str, Any]] = []
        seen: set[str] = set()
        for lease_id, allocation in _LEASE_ALLOCATIONS.items():
            if allocation.get("status") != "ACTIVE" or lease_id not in sessions:
                continue
            inventory.append({
                "lease_id": lease_id,
                "episode_id": allocation["episode_id"],
                "allocation_key": allocation["allocation_key"],
                "owner": dict(allocation.get("owner") or {}),
                "fence_token": allocation.get("fence_token"),
                "generation": allocation.get("generation", 1),
                "created_at": allocation["created_at"],
            })
            seen.add(lease_id)
        for lease_id in sessions:
            if lease_id not in seen:
                inventory.append({"lease_id": str(lease_id), "legacy": True})
        inventory.sort(key=lambda item: str(item["lease_id"]))
        return inventory

def _execute_mutation(
    endpoint: str,
    payload: dict[str, Any],
    lease_id: str | None,
    callback,
) -> tuple:
    operation_id_value = payload.get("operation_id")
    operation_id = str(operation_id_value) if operation_id_value is not None else None
    try:
        request_hash = _payload_hash(payload, exclude={"operation_id"})
    except (TypeError, ValueError) as exc:
        return _structured_error("validation_error", str(exc), 400)

    with _STATE_LOCK:
        allocation = _LEASE_ALLOCATIONS.get(str(lease_id))
        retired = _closed_record("lease_id", str(lease_id)) if allocation is None else None
        if retired is not None:
            allocation = retired["allocation"]
        supplied_fence = payload.get("fence_token")
        if (
            allocation is not None
            and allocation.get("v2_owned")
            and supplied_fence is None
        ):
            return _structured_error(
                "fence_required",
                "Lease fence token is required",
                409,
                {"lease_id": lease_id},
            )
        if (
            supplied_fence is not None
            and allocation is not None
            and allocation.get("fence_token")
            and str(supplied_fence) != str(allocation["fence_token"])
        ):
            return _structured_error(
                "fenced",
                "Lease fence token does not match",
                409,
                {
                    "lease_id": lease_id,
                    "generation": allocation.get("generation", 1),
                },
            )

        cache_key = (
            (endpoint, operation_id)
            if operation_id is not None and endpoint in _CACHEABLE_OPERATION_ENDPOINTS
            else None
        )
        cache = (
            OrderedDict((tuple(key), value) for key, value in retired["close_results"])
            if retired is not None else _OPERATION_RESULTS.get(str(lease_id), {})
        )
        if cache_key is not None and cache_key in cache:
            previous_hash, body, status_code = cache[cache_key]
            if previous_hash != request_hash:
                return _structured_error(
                    "idempotency_conflict",
                    "operation_id was already used with a different payload",
                    409,
                    {
                        "lease_id": lease_id,
                        "operation_id": operation_id,
                        "endpoint": endpoint,
                    },
                )
            cache.move_to_end(cache_key)
            return jsonify(body), status_code

        inflight_key = (str(lease_id), endpoint, operation_id) if operation_id is not None else None
        joined = _INFLIGHT_BY_OPID.get(inflight_key) if inflight_key is not None else None
        future = None
        if joined is not None:
            previous_hash, future = joined
            if previous_hash != request_hash:
                return _structured_error(
                    "idempotency_conflict",
                    "operation_id is in flight with a different payload",
                    409,
                    {
                        "lease_id": lease_id,
                        "operation_id": operation_id,
                        "endpoint": endpoint,
                    },
                )

        if future is None:
            if allocation is not None and allocation.get("status") != "ACTIVE":
                return _structured_error("lease_not_active", "Lease is closed", 409)
            op_token = f"{endpoint}:{lease_id}:{uuid.uuid4().hex[:8]}"
            op_record = {
                "token": op_token,
                "endpoint": endpoint,
                "lease_id": str(lease_id),
                "started_at": time.time(),
            }

            def _run() -> tuple[dict[str, Any], int]:
                try:
                    with _lease_op_lock(str(lease_id)):

                        with _STATE_LOCK:
                            active = allocation is None or allocation.get("status") == "ACTIVE"
                        if not active:
                            body, status_code = {
                                "ok": False, "code": "lease_not_active", "error": "Lease is closed",
                            }, 409
                        else:
                            body, status_code = callback(), 200
                except Exception as exc:
                    logger.exception("%s failed for lease_id=%s", endpoint, lease_id)
                    body, status_code = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
                with _STATE_LOCK:
                    if cache_key is not None:
                        cache = _OPERATION_RESULTS.setdefault(str(lease_id), OrderedDict())
                        cache[cache_key] = (request_hash, dict(body), status_code)
                        cache.move_to_end(cache_key)
                        if endpoint == "/close" and status_code == 200 and body.get("ok"):
                            _OPERATION_RESULTS[str(lease_id)] = OrderedDict([(cache_key, cache[cache_key])])
                        else:
                            while len(cache) > _OPERATION_RESULT_CACHE_SIZE:
                                cache.popitem(last=False)
                    _INFLIGHT_OPS.pop(op_token, None)
                    if inflight_key is not None:
                        _INFLIGHT_BY_OPID.pop(inflight_key, None)
                    _retire_lease(str(lease_id))
                return body, status_code

            _INFLIGHT_OPS[op_token] = op_record
            try:
                future = _get_op_executor().submit(_run)
            except Exception:
                _INFLIGHT_OPS.pop(op_token, None)
                raise
            if inflight_key is not None:
                _INFLIGHT_BY_OPID[inflight_key] = (request_hash, future)

    try:
        body, status_code = future.result(timeout=_ENV_OP_TIMEOUT_S)
    except FutureTimeoutError:

        logger.error(
            "%s wedged for lease_id=%s (> %.0fs); reporting env_op_timeout "
            "and flagging worker unhealthy",
            endpoint,
            lease_id,
            _ENV_OP_TIMEOUT_S,
        )
        return _structured_error(
            "env_op_timeout",
            f"{endpoint} did not complete within {_ENV_OP_TIMEOUT_S:.0f}s",
            500,
            {"lease_id": lease_id, "endpoint": endpoint},
        )
    except Exception as exc:
        logger.exception("%s failed for lease_id=%s: %s", endpoint, lease_id, exc)
        body = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        status_code = 500

    return jsonify(body), status_code

def _encode_screenshot(obs: dict[str, Any]) -> dict[str, Any]:

    out = dict(obs)
    screenshot = out.pop("screenshot", None)
    if isinstance(screenshot, (bytes, bytearray)):
        out["screenshot_b64"] = base64.b64encode(screenshot).decode("utf-8")
    elif screenshot is None:
        out["screenshot_b64"] = None
    else:
        out["screenshot_b64"] = screenshot
    return out

def _get_client():

    global _ENV_CLIENT
    if _ENV_CLIENT is None:
        raise RuntimeError("Worker not initialized - call init_worker() first")
    return _ENV_CLIENT

@app.route("/healthz", methods=["GET"])
def healthz():

    now = time.time()
    wedged = _wedged_ops(now)
    heartbeat_stalled = (
        _HEARTBEAT_THREAD is not None
        and now - _HEARTBEAT_LAST_BEAT > _HEARTBEAT_STALL_S
    )
    if wedged or heartbeat_stalled:
        return jsonify({
            "ok": False,
            "worker_id": _WORKER_INFO.get("worker_id", ""),
            "incarnation": _PROCESS_INCARNATION,
            "wedged_ops": [
                {
                    "endpoint": op["endpoint"],
                    "lease_id": op["lease_id"],
                    "age_seconds": round(now - op["started_at"], 1),
                }
                for op in wedged
            ],
            "heartbeat_stalled": heartbeat_stalled,
        }), 503
    return jsonify({
        "ok": True,
        "worker_id": _WORKER_INFO.get("worker_id", ""),
        "incarnation": _PROCESS_INCARNATION,
    })

@app.route("/status", methods=["GET"])
def status():
    client = _get_client()
    active = len(client._sessions)
    return jsonify({
        "ok": True,
        "worker_id": _WORKER_INFO.get("worker_id", ""),
        "max_envs": 1,
        "busy_envs": active,
        "idle_envs": max(0, 1 - active),
        "incarnation": _PROCESS_INCARNATION,
        "active_leases": _active_lease_inventory(),
    })

@app.route("/leases", methods=["GET"])
def leases():
    inventory = _active_lease_inventory()
    return jsonify({
        "ok": True,
        "worker_id": _WORKER_INFO.get("worker_id", ""),
        "incarnation": _PROCESS_INCARNATION,
        "leases": inventory,
        "count": len(inventory),
    })

@app.route("/allocate", methods=["POST"])
def allocate():
    payload = request.get_json(force=True) or {}
    episode_id = str(payload.get("episode_id", str(uuid.uuid4())))
    owner = payload.get("owner") or {}
    if not isinstance(owner, dict):
        return _structured_error("validation_error", "owner must be an object", 400)
    allocation_key = str(
        payload.get("allocation_key")
        or payload.get("idempotency_key")
        or episode_id
    )
    inferred_v2 = any(
        key in payload
        for key in (
            "allocation_key",
            "idempotency_key",
            "generation",
            "fence_token",
        )
    ) or owner.get("component") == "rollout-host-v2"
    uses_v2_fence = bool(payload.get("v2_owned", inferred_v2))
    try:
        request_hash = _payload_hash(
            {
                "episode_id": episode_id,
                "owner": owner,
                "fence_token": payload.get("fence_token"),
                "generation": int(payload.get("generation", 1)),
                "v2_owned": uses_v2_fence,
            }
        )
    except (TypeError, ValueError) as exc:
        return _structured_error("validation_error", str(exc), 400)
    with _STATE_LOCK:
        existing = _WORKER_ALLOCATIONS.get(allocation_key)
        if existing is None:
            retired = _closed_record("allocation_key", allocation_key)
            existing = retired["allocation"] if retired else None
        pending = _PENDING_ALLOCATIONS.get(allocation_key)
        if pending is not None:
            if pending[0] != request_hash:
                return _structured_error("idempotency_conflict", "allocation payload changed", 409)
            future = pending[2]
        else:
            future = None
        if existing is not None:
            if existing["status"] == "ACTIVE":
                if existing["payload_hash"] != request_hash:
                    return _structured_error(
                        "idempotency_conflict",
                        "allocation key was already used with a different payload",
                        409,
                        {"allocation_key": allocation_key},
                    )
                return jsonify({
                    "ok": True,
                    "lease_id": existing["lease_id"],
                    "env_id": existing["env_id"],
                    "worker_id": _WORKER_INFO.get("worker_id", ""),
                    "incarnation": _PROCESS_INCARNATION,
                    "fence_token": existing.get("fence_token"),
                    "generation": existing.get("generation", 1),
                    "replayed": True,
                })
            if (
                uses_v2_fence
                and int(payload.get("generation", 1)) <= int(existing.get("generation", 1))
            ):
                return _structured_error(
                    "fenced",
                    "closed allocation key requires a newer generation",
                    409,
                    {
                        "allocation_key": allocation_key,
                        "previous_generation": existing.get("generation", 1),
                    },
                )
        conflicting = next(
            (
                allocation
                for allocation in _LEASE_ALLOCATIONS.values()
                if allocation.get("episode_id") == episode_id
                and (allocation.get("status") == "ACTIVE" or allocation.get("cleanup_pending"))
            ),
            None,
        )
        if conflicting is not None:
            return _structured_error(
                "episode_conflict",
                "episode_id already has an active allocation",
                409,
                {
                    "episode_id": episode_id,
                    "allocation_key": conflicting["allocation_key"],
                },
            )

        if future is None:
            if any(item[1] == episode_id for item in _PENDING_ALLOCATIONS.values()):
                return _structured_error("episode_conflict", "Episode allocation is pending", 409)
            op_token = f"/allocate:{episode_id}:{uuid.uuid4().hex}"
            _INFLIGHT_OPS[op_token] = {
                "endpoint": "/allocate", "lease_id": episode_id, "started_at": time.time(),
            }

            def perform_allocate():
                try:
                    result = _get_client().allocate(episode_id)
                    lease_id = str(result["lease_id"])
                    allocation = {
                        "allocation_key": allocation_key, "episode_id": episode_id,
                        "payload_hash": request_hash, "lease_id": lease_id,
                        "env_id": str(result.get("env_id", lease_id)),
                        "fence_token": payload.get("fence_token"),
                        "generation": int(payload.get("generation", 1)), "owner": dict(owner),
                        "v2_owned": uses_v2_fence, "created_at": time.time(), "status": "ACTIVE",
                    }
                    with _STATE_LOCK:
                        _WORKER_ALLOCATIONS[allocation_key] = allocation
                        _LEASE_ALLOCATIONS[lease_id] = allocation
                    return {
                        "ok": True, "lease_id": lease_id, "env_id": allocation["env_id"],
                        "worker_id": _WORKER_INFO.get("worker_id", ""),
                        "incarnation": _PROCESS_INCARNATION,
                        "fence_token": allocation["fence_token"], "generation": allocation["generation"],
                    }
                finally:
                    with _STATE_LOCK:
                        _INFLIGHT_OPS.pop(op_token, None)
                        _PENDING_ALLOCATIONS.pop(allocation_key, None)

            global _ALLOC_EXECUTOR
            if _ALLOC_EXECUTOR is None:
                _ALLOC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="env-allocate")
            try:
                future = _ALLOC_EXECUTOR.submit(perform_allocate)
            except Exception:
                _INFLIGHT_OPS.pop(op_token, None)
                raise
            _PENDING_ALLOCATIONS[allocation_key] = (request_hash, episode_id, future)
    try:
        return jsonify(future.result(timeout=_ENV_OP_TIMEOUT_S))
    except FutureTimeoutError:
        return _structured_error("env_op_timeout", "Allocation did not complete in time", 500)
    except Exception as exc:
        return _json_error("/allocate", episode_id, exc)

@app.route("/reset", methods=["POST"])
def reset():
    payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    task_config = payload.get("task_config")
    def execute() -> dict[str, Any]:
        client = _get_client()
        obs = client.reset(lease_id, task_config)
        return {"ok": True, "observation": _encode_screenshot(obs)}

    return _execute_mutation("/reset", payload, lease_id, execute)

@app.route("/step", methods=["POST"])
def step():
    payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    action = payload.get("action", "")
    def execute() -> dict[str, Any]:
        client = _get_client()
        obs, reward, done, info = client.step(lease_id, action)
        return {
            "ok": True,
            "observation": _encode_screenshot(obs),
            "reward": reward,
            "done": done,
            "info": info,
        }

    return _execute_mutation("/step", payload, lease_id, execute)

@app.route("/get_obs", methods=["POST", "GET"])
def get_obs():
    if request.method == "GET":
        payload = request.args.to_dict()
    else:
        payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    def execute() -> dict[str, Any]:
        client = _get_client()
        obs = client.get_obs(lease_id)
        return {"ok": True, "observation": _encode_screenshot(obs)}

    return _execute_mutation("/get_obs", payload, lease_id, execute)

@app.route("/prepare_for_probe", methods=["POST"])
def prepare_for_probe():
    payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    def execute() -> dict[str, Any]:
        client = _get_client()
        prepare = getattr(client, "prepare_for_probe", None)
        if callable(prepare):
            prepare(lease_id)
        return {"ok": True}

    return _execute_mutation("/prepare_for_probe", payload, lease_id, execute)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    def execute() -> dict[str, Any]:
        client = _get_client()
        score = client.evaluate(lease_id)
        return {"ok": True, "score": score}

    return _execute_mutation("/evaluate", payload, lease_id, execute)

@app.route("/close", methods=["POST"])
def close():
    payload = request.get_json(force=True) or {}
    lease_id = payload.get("lease_id")
    def execute() -> dict[str, Any]:
        client = _get_client()
        client.close(lease_id)
        with _STATE_LOCK:
            allocation = _LEASE_ALLOCATIONS.get(str(lease_id))
            if allocation is not None:
                allocation["status"] = "CLOSED"
                allocation["closed_at"] = time.time()
        return {"ok": True}

    return _execute_mutation("/close", payload, lease_id, execute)

@app.route("/cancel_allocation", methods=["POST"])
def cancel_allocation():

    payload = request.get_json(force=True) or {}
    allocation_key = str(payload.get("allocation_key") or "")
    if not allocation_key:
        return _structured_error(
            "validation_error",
            "allocation_key is required",
            400,
        )
    supplied_fence = payload.get("fence_token")
    supplied_generation = payload.get("generation")
    with _STATE_LOCK:
        if allocation_key in _PENDING_ALLOCATIONS:
            return _structured_error("allocation_pending", "Allocation is still executing; retry cancellation", 409)
        allocation = _WORKER_ALLOCATIONS.get(allocation_key)
        if allocation is None:
            retired = _closed_record("allocation_key", allocation_key)
            allocation = retired["allocation"] if retired else None
        if allocation is None:
            return jsonify({
                "ok": True,
                "allocation_key": allocation_key,
                "status": "CANCELLED",
                "missing": True,
            })
        if (
            allocation.get("fence_token")
            and str(supplied_fence or "") != str(allocation["fence_token"])
        ):
            return _structured_error(
                "fenced",
                "Allocation fence token does not match",
                409,
                {"allocation_key": allocation_key},
            )
        if (
            supplied_generation is not None
            and int(supplied_generation) != int(allocation.get("generation", 1))
        ):
            return _structured_error(
                "fenced",
                "Allocation generation does not match",
                409,
                {
                    "allocation_key": allocation_key,
                    "generation": allocation.get("generation", 1),
                },
            )
        lease_id = str(allocation["lease_id"])
        was_active = allocation.get("status") == "ACTIVE"
        if was_active:

            allocation["status"] = "CLOSED"
            allocation["closed_at"] = time.time()
            allocation["cleanup_pending"] = True
    if was_active:
        op_token = f"/cancel_allocation:{lease_id}:{uuid.uuid4().hex[:8]}"
        op_record = {
            "token": op_token,
            "endpoint": "/cancel_allocation",
            "lease_id": lease_id,
            "started_at": time.time(),
        }

        def _close_in_background() -> None:

            for attempt in range(3):
                try:
                    with _lease_op_lock(lease_id):
                        _get_client().close(lease_id)
                    with _STATE_LOCK:
                        allocation["cleanup_pending"] = False
                        _INFLIGHT_OPS.pop(op_token, None)
                        _retire_lease(lease_id)
                    return
                except Exception:
                    logger.exception(
                        "background close for allocation %s (lease %s), attempt %s failed",
                        allocation_key, lease_id, attempt + 1,
                    )
                    if attempt < 2:
                        time.sleep(1.0)
            with _STATE_LOCK:
                op_record["started_at"] = time.time() - _ENV_OP_TIMEOUT_S - 1

        with _STATE_LOCK:
            _INFLIGHT_OPS[op_token] = op_record
        _get_op_executor().submit(_close_in_background)
    return jsonify({
        "ok": True,
        "allocation_key": allocation_key,
        "lease_id": lease_id,
        "status": "CANCELLED",
    })

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    return jsonify({
        "ok": True,
        "incarnation": _PROCESS_INCARNATION,
        "active_leases": _active_lease_inventory(),
    })

def _master_worker_payload() -> dict[str, Any]:
    client = _get_client()
    active = len(client._sessions)
    payload = dict(_WORKER_INFO)
    payload["incarnation"] = _PROCESS_INCARNATION
    payload["active_leases"] = _active_lease_inventory()
    payload["status"] = {
        "max_envs": int(payload.get("capacity", 1)),
        "busy_envs": active,
        "leases": active,
    }
    return payload

def _register_with_master(master_url: str) -> bool:
    import requests as req

    response = req.post(
        f"{master_url}/register",
        json=_master_worker_payload(),
        timeout=10,
    )
    if not 200 <= response.status_code < 300:
        logger.warning(
            "Worker registration failed with HTTP %s: %s",
            response.status_code,
            response.text,
        )
        return False
    logger.info("Registered with master at %s", master_url)
    return True

def _heartbeat_loop(master_url: str, interval: float = 5.0):

    import requests as req

    global _HEARTBEAT_LAST_BEAT
    while not _SHUTDOWN:
        _HEARTBEAT_LAST_BEAT = time.time()
        try:
            response = req.post(
                f"{master_url}/heartbeat",
                json=_master_worker_payload(),
                timeout=10,
            )
            body = {}
            try:
                body = response.json()
            except ValueError:
                pass
            if response.status_code == 404 or body.get("code") == "unknown_worker":
                logger.info("Master forgot worker registration; registering again")
                _register_with_master(master_url)
            elif response.status_code == 409 and body.get("code") == "stale_worker_incarnation":
                _register_with_master(master_url)
            elif not 200 <= response.status_code < 300:
                logger.warning(
                    "Worker heartbeat failed with HTTP %s: %s",
                    response.status_code,
                    body or response.text,
                )
        except Exception as exc:
            logger.debug("Worker heartbeat failed: %s", exc)
        time.sleep(interval)

def init_worker(
    *,
    osworld_path: str,
    path_to_vm: str = "",
    port: int = 9801,
    host: str = "0.0.0.0",
    advertise_host: str = "",
    master_url: str = "http://127.0.0.1:9900",
    provider_name: str = "docker",
    headless: bool = True,
    docker_tag: str = "",
):

    global _ENV_CLIENT, _WORKER_INFO, _MASTER_URL, _HEARTBEAT_THREAD

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from eval_runtime.env_adapter import AgentRuntimeEnvClient

    if not advertise_host:
        advertise_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host

    worker_id = f"guirl-worker-{advertise_host}:{port}"
    _WORKER_INFO = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "capacity": 1,
        "base_url": f"http://{advertise_host}:{port}",
        "incarnation": _PROCESS_INCARNATION,
    }
    _MASTER_URL = master_url

    init_attempts = int(os.environ.get("GUI_ENV_WORKER_INIT_ATTEMPTS", "5"))

    extra_kwargs: dict[str, Any] = {}
    cache_dir = os.environ.get("GUI_ENV_CACHE_DIR", "").strip()
    if cache_dir:
        extra_kwargs["cache_dir"] = cache_dir
    for attempt in range(1, init_attempts + 1):
        try:
            _ENV_CLIENT = AgentRuntimeEnvClient(
                osworld_path=osworld_path,
                path_to_vm=path_to_vm or os.environ.get("OSWORLD_VM_PATH", ""),
                provider_name=provider_name,
                headless=headless,
                docker_tag=docker_tag or f"guirl-{worker_id}",
                **extra_kwargs,
            )
            break
        except Exception as exc:
            if attempt >= init_attempts:
                raise
            delay = min(60.0, 10.0 * attempt)
            logger.warning(
                "Env client init attempt %d/%d failed (%s: %s); retrying in %.0fs",
                attempt, init_attempts, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)

    try:
        _register_with_master(master_url)
    except Exception as exc:
        logger.warning("Failed to register with master: %s", exc)

    _HEARTBEAT_THREAD = threading.Thread(target=_heartbeat_loop, args=(master_url,), daemon=True)
    _HEARTBEAT_THREAD.start()

def _shutdown_handler(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    if _ENV_CLIENT:
        _ENV_CLIENT.close_all()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="GUIRL RPC Env Worker Server")
    parser.add_argument("--port", type=int, default=9801)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--advertise-host", type=str, default="")
    parser.add_argument("--master-url", type=str, default="http://127.0.0.1:9900")
    parser.add_argument("--osworld-path", type=str, required=True)
    parser.add_argument("--path-to-vm", type=str, default=os.environ.get("OSWORLD_VM_PATH", ""))
    parser.add_argument("--provider-name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--docker-tag", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    init_worker(
        osworld_path=args.osworld_path,
        path_to_vm=args.path_to_vm,
        port=args.port,
        host=args.host,
        advertise_host=args.advertise_host,
        master_url=args.master_url,
        provider_name=args.provider_name,
        headless=args.headless,
        docker_tag=args.docker_tag,
    )

    logger.info("Starting GUIRL env worker on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
