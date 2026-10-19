from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import logging
import os
import signal
import sys
import threading
from argparse import Namespace
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from .batch_models import ArtifactRef, BatchId, ErrorCode, ProtocolError
from .rpc_env_client import AsyncRpcEnvClient

logger = logging.getLogger("guirl.rollout_master_server")
app = Flask(__name__)

_ARGS: Namespace | None = None
_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None
_MASTER_CLIENT = None
_ROLLOUT_HOST_V2 = None
_ENV_CLIENT: AsyncRpcEnvClient | None = None
_STARTED = False

def _get_loop() -> asyncio.AbstractEventLoop:
    global _LOOP, _LOOP_THREAD
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        _LOOP_THREAD = threading.Thread(
            target=_LOOP.run_forever,
            daemon=True,
            name="guirl-rollout-master-loop",
        )
        _LOOP_THREAD.start()
    return _LOOP

def _run(coro, timeout: float | None = None):
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise

def _json_error(message: str, status_code: int = 500):
    logger.exception(message)
    return jsonify({"ok": False, "error": message}), status_code

def _v2_error(exc: BaseException):
    if isinstance(exc, ProtocolError):
        statuses = {
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.NOT_FOUND: 404,
            ErrorCode.ALREADY_EXISTS: 409,
            ErrorCode.VERSION_CONFLICT: 409,
            ErrorCode.IDEMPOTENCY_CONFLICT: 409,
            ErrorCode.FENCED: 409,
            ErrorCode.INVALID_TRANSITION: 409,
            ErrorCode.BATCH_INCOMPLETE: 422,
        }
        return (
            jsonify({"ok": False, "error": exc.to_dict()}),
            statuses.get(exc.code, 500),
        )
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        error = {
            "code": ErrorCode.VALIDATION_ERROR.value,
            "message": str(exc),
            "details": {},
        }
        return jsonify({"ok": False, "error": error}), 400
    logger.exception("unexpected Rollout Host v2 API error")
    error = {
        "code": ErrorCode.STORAGE_ERROR.value,
        "message": "internal rollout host error",
        "details": {},
    }
    return jsonify({"ok": False, "error": error}), 500

def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value

def _probe_artifact_dir(
    payload: dict[str, Any],
    *,
    lease_id: str,
    sampler_round: int,
) -> Path:

    if _ARGS is None:
        raise RuntimeError("rollout master server is not initialized")

    root = Path(_ARGS.guirl_probe_output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f"probe output root is not writable: {root}")

    client_key = str(
        payload.get("artifact_key")
        or payload.get("example_result_dir")
        or lease_id
    )
    digest = hashlib.sha256(
        f"{client_key}\0{lease_id}".encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:24]
    result_dir = root / digest / f"round_{sampler_round:02d}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir

def _build_args(cli_args: argparse.Namespace) -> Namespace:
    rollout_host_v2 = bool(getattr(cli_args, "rollout_host_v2", False))
    configured_actor_timeout = getattr(cli_args, "actor_timeout_seconds", None)
    actor_timeout_seconds = (
        float(configured_actor_timeout)
        if configured_actor_timeout is not None
        else (1200.0 if rollout_host_v2 else 300.0)
    )

    pull_finished_timeout = getattr(cli_args, "pull_finished_timeout_seconds", None)
    if pull_finished_timeout is None:
        pull_finished_timeout = max(1560.0, actor_timeout_seconds + 360.0)

    actors_cfg = None
    actors_config_path = getattr(cli_args, "actors_config", None)
    if actors_config_path:
        import json

        with open(actors_config_path, "r", encoding="utf-8") as f:
            if actors_config_path.endswith((".yaml", ".yml")):
                import yaml

                loaded = yaml.safe_load(f)
            else:
                loaded = json.load(f)
        actors_cfg = loaded.get("actors") if isinstance(loaded, dict) else loaded
        if not isinstance(actors_cfg, list) or not actors_cfg:
            raise ValueError(
                f"--actors-config {actors_config_path} must contain a non-empty "
                f"'actors' list (got {type(actors_cfg)})"
            )

    return Namespace(
        guirl_env_master_url=cli_args.env_master_url,
        guirl_manifest_path=cli_args.manifest_path,
        guirl_examples_dir=cli_args.examples_dir,
        guirl_batch_size=cli_args.batch_size,
        guirl_env_capacity=cli_args.env_capacity,
        guirl_actor_timeout_seconds=actor_timeout_seconds,
        guirl_actor_max_steps=cli_args.actor_max_steps,
        guirl_actor_harness=cli_args.actor_harness,
        guirl_actors=actors_cfg,
        rollout_host_v2=rollout_host_v2,
        guirl_state_db=getattr(cli_args, "state_db", None),
        guirl_artifact_dir=getattr(cli_args, "artifact_dir", None),
        guirl_probe_output_root=getattr(
            cli_args,
            "probe_output_root",
            "./outputs/rl_artifacts",
        ),
        guirl_actor_concurrency=getattr(cli_args, "actor_concurrency", None),
        guirl_v2_max_attempts=getattr(cli_args, "max_attempts", 2),
        guirl_v2_infra_retry_budget=getattr(
            cli_args, "infra_retry_budget", 20
        ),
        guirl_v2_max_task_replacements=getattr(
            cli_args, "max_task_replacements", 2
        ),
        guirl_v2_failure_policy=getattr(
            cli_args, "v2_failure_policy", "replace_task_group"
        ),
        guirl_v2_claim_ttl_seconds=getattr(
            cli_args, "claim_ttl_seconds", 600.0
        ),
        guirl_v2_close_retry_seconds=getattr(
            cli_args, "close_retry_seconds", 1.0
        ),
        guirl_v2_batch_deadline_seconds=getattr(
            cli_args, "batch_deadline_seconds", 6 * 60 * 60
        ),
        guirl_v2_sqlite_synchronous=getattr(
            cli_args, "sqlite_synchronous", "FULL"
        ),
        guirl_v2_sqlite_busy_timeout_ms=getattr(
            cli_args, "sqlite_busy_timeout_ms", 5_000
        ),
        pull_finished_timeout_seconds=pull_finished_timeout,

        env_restart_command=getattr(cli_args, "env_restart_command", None),
        env_restart_threshold_seconds=cli_args.env_restart_threshold_seconds,
        env_restart_cooldown_seconds=cli_args.env_restart_cooldown_seconds,
        env_restart_command_timeout_seconds=cli_args.env_restart_command_timeout,
        env_restart_healthcheck_wait_seconds=cli_args.env_restart_healthcheck_wait,
        env_restart_min_healthy_workers=cli_args.env_restart_min_healthy_workers,
        env_restart_health_probe_interval_seconds=cli_args.env_restart_health_probe_interval,
        guirl_actor_harness_kwargs={
            "model": cli_args.actor_model,
            "api_key": cli_args.actor_api_key,
            "base_url": cli_args.actor_base_url,
            "max_tokens": cli_args.actor_max_tokens,
            "temperature": cli_args.actor_temperature,
            "max_trajectory_length": cli_args.actor_max_trajectory_length,
            "max_images": cli_args.actor_max_images,
            "client_password": cli_args.client_password,
            "platform": cli_args.platform,
            "enable_thinking": cli_args.actor_enable_thinking,
        },
    )

def _make_env_health_probe_fn(args: Namespace, env_client: AsyncRpcEnvClient):

    min_healthy = int(getattr(args, "env_restart_min_healthy_workers", 1))

    async def _probe() -> bool:
        status = await env_client.status()
        master = status.get("master") or {}
        healthy_workers = int(master.get("healthy_workers") or 0)
        ok = healthy_workers >= min_healthy
        if not ok:
            logger.warning(
                "Env health probe: healthy_workers=%d < required %d",
                healthy_workers, min_healthy,
            )
        return ok

    return _probe

def _make_env_restart_fn(args: Namespace, env_client: AsyncRpcEnvClient):

    restart_command = str(args.env_restart_command)
    command_timeout = float(getattr(args, "env_restart_command_timeout_seconds", 300.0))
    healthcheck_wait = float(getattr(args, "env_restart_healthcheck_wait_seconds", 300.0))

    async def _restart() -> None:
        logger.warning("Env restart: running command: %s", restart_command)
        proc = await asyncio.create_subprocess_shell(
            restart_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=command_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"env restart command timed out after {command_timeout:.0f}s"
            )
        output = (stdout or b"").decode("utf-8", errors="replace").strip()
        if output:
            logger.info("Env restart command output:\n%s", output[-4000:])
        if proc.returncode != 0:
            raise RuntimeError(
                f"env restart command exited with code {proc.returncode}"
            )

        probe = _make_env_health_probe_fn(args, env_client)
        deadline = asyncio.get_running_loop().time() + healthcheck_wait
        while True:
            try:
                if await probe():
                    logger.info("Env stack healthy after restart")
                    return
            except Exception:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"env stack not healthy within {healthcheck_wait:.0f}s after restart"
                )
            await asyncio.sleep(5.0)

    return _restart

def _ensure_started() -> None:
    global _MASTER_CLIENT, _ROLLOUT_HOST_V2, _ENV_CLIENT, _STARTED
    if _STARTED:
        return
    if _ARGS is None:
        raise RuntimeError("rollout master server is not initialized")

    from ..master_datasource.factory import (
        build_master_client,
        build_rollout_host_v2,
    )

    if bool(getattr(_ARGS, "rollout_host_v2", False)):
        _ROLLOUT_HOST_V2 = build_rollout_host_v2(
            _ARGS,
            state_db=_ARGS.guirl_state_db,
            artifact_dir=_ARGS.guirl_artifact_dir,
        )
        _ENV_CLIENT = _ROLLOUT_HOST_V2.env_client
        if getattr(_ARGS, "env_restart_command", None):
            _ROLLOUT_HOST_V2.configure_env_watchdog(
                restart_fn=_make_env_restart_fn(_ARGS, _ENV_CLIENT),
                health_probe_fn=_make_env_health_probe_fn(_ARGS, _ENV_CLIENT),
                threshold_seconds=_ARGS.env_restart_threshold_seconds,
                cooldown_seconds=_ARGS.env_restart_cooldown_seconds,
                probe_interval_seconds=(
                    _ARGS.env_restart_health_probe_interval_seconds
                ),
            )
        _run(_ROLLOUT_HOST_V2.start(), timeout=30)
        _STARTED = True
        logger.info("GUIRL Rollout Host v2 started")
        return

    _ENV_CLIENT = AsyncRpcEnvClient(base_url=_ARGS.guirl_env_master_url)
    env_restart_fn = None
    env_health_probe_fn = None
    if getattr(_ARGS, "env_restart_command", None):
        env_restart_fn = _make_env_restart_fn(_ARGS, _ENV_CLIENT)
        env_health_probe_fn = _make_env_health_probe_fn(_ARGS, _ENV_CLIENT)
        logger.info(
            "Env auto-restart enabled: threshold=%.0fs cooldown=%.0fs "
            "min_healthy_workers=%d command=%r",
            _ARGS.env_restart_threshold_seconds,
            _ARGS.env_restart_cooldown_seconds,
            _ARGS.env_restart_min_healthy_workers,
            _ARGS.env_restart_command,
        )
    _MASTER_CLIENT = build_master_client(
        _ARGS,
        env_restart_fn=env_restart_fn,
        env_health_probe_fn=env_health_probe_fn,
    )
    _run(_MASTER_CLIENT.start(), timeout=30)
    _STARTED = True
    logger.info("GUIRL rollout master started")

def _master_status() -> dict[str, Any]:
    if _ROLLOUT_HOST_V2 is not None:
        return {"mode": "v2", **_ROLLOUT_HOST_V2.status()}
    if _MASTER_CLIENT is None:
        return {"started": False}
    sampler = getattr(_MASTER_CLIENT, "task_sampler", None)
    tasks = getattr(sampler, "tasks", None)
    return {
        "started": _STARTED,
        "batch_size": getattr(_MASTER_CLIENT, "batch_size", None),
        "env_capacity": getattr(_MASTER_CLIENT, "env_capacity", None),
        "finished_queue_size": getattr(_MASTER_CLIENT, "_finished").qsize(),
        "inflight": len(getattr(_MASTER_CLIENT, "_inflight")),
        "task_count": len(tasks) if tasks is not None else None,
    }

@app.get("/healthz")
def healthz():
    try:
        _ensure_started()
        return jsonify({"ok": True, "master": _master_status()})
    except Exception as exc:
        return _json_error(f"healthz failed: {exc}")

@app.get("/status")
def status():
    try:
        _ensure_started()
        return jsonify({"ok": True, "master": _master_status()})
    except Exception as exc:
        return _json_error(f"status failed: {exc}")

@app.post("/pull_finished")
def pull_finished():
    payload = request.get_json(force=True, silent=True) or {}
    num_samples = int(payload.get("num_samples", 1))
    try:
        _ensure_started()
        if _MASTER_CLIENT is None:
            return _v2_error(
                ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "/pull_finished belongs to legacy mode; use /v2/pull_ready",
                )
            )

        pull_timeout = float(getattr(_ARGS, "pull_finished_timeout_seconds", 1560.0))
        entries = _run(_MASTER_CLIENT.pull_finished(num_samples), timeout=pull_timeout)
        return jsonify({"ok": True, "entries": entries})
    except Exception as exc:
        return _json_error(f"pull_finished failed: {exc}")

def _require_v2():
    _ensure_started()
    if _ROLLOUT_HOST_V2 is None:
        raise ProtocolError(
            ErrorCode.INVALID_TRANSITION,
            "Rollout Host v2 is not enabled",
        )
    return _ROLLOUT_HOST_V2

def _batch_id(payload: dict[str, Any]) -> BatchId:
    source = payload.get("batch_id") or payload
    return BatchId(
        str(source["run_id"]),
        int(source["train_step_id"]),
    )

@app.post("/v2/acquire_batch")
def v2_acquire_batch():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        host = _require_v2()
        manifest = _run(
            host.coordinator.acquire_batch(
                run_id=str(payload["run_id"]),
                train_step_id=int(payload["train_step_id"]),
                tasks_per_step=int(payload["tasks_per_step"]),
                request_id=str(payload["request_id"]),
                config_fingerprint=str(payload["config_fingerprint"]),
            ),
            timeout=60,
        )
        return jsonify(
            {
                "ok": True,
                "batch_id": manifest.batch_id.to_dict(),
                "manifest_revision": manifest.revision,
                "payload_hash": manifest.payload_hash,
                "slots": [slot.to_dict() for slot in manifest.slots],
            }
        )
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/pull_ready")
def v2_pull_ready():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        host = _require_v2()
        wait_seconds = max(0.0, float(payload.get("wait_seconds", 0.0)))
        result = _run(
            host.broker.pull_ready(
                _batch_id(payload),
                consumer=str(payload["consumer"]),
                limit=int(payload.get("limit", 1)),
                request_id=str(payload["request_id"]),
                ttl_seconds=float(
                    payload.get(
                        "ttl_seconds",
                        host.config.claim_ttl_seconds,
                    )
                ),
                wait_seconds=wait_seconds,
                include_payload=bool(payload.get("include_payload", True)),
            ),
            timeout=wait_seconds + 30,
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/renew_claim")
def v2_renew_claim():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        host = _require_v2()
        result = _run(
            host.renew(
                claim_id=str(payload["claim_token"]),
                generation=int(payload["generation"]),
                ttl_seconds=float(
                    payload.get(
                        "ttl_seconds",
                        host.config.claim_ttl_seconds,
                    )
                ),
                consumer_id=(
                    str(payload["consumer"])
                    if payload.get("consumer")
                    else None
                ),
                request_id=str(payload["request_id"]),
            ),
            timeout=30,
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/ack")
def v2_ack():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        host = _require_v2()
        artifact = (
            ArtifactRef.from_dict(payload["judge_artifact"])
            if payload.get("judge_artifact")
            else None
        )
        result = _run(
            host.ack(
                claim_id=str(payload["claim_token"]),
                generation=int(payload["generation"]),
                attempt=(
                    int(payload["attempt"])
                    if payload.get("attempt") is not None
                    else None
                ),
                request_id=str(payload["request_id"]),
                judge_payload=payload.get("judge_payload"),
                judge_artifact=artifact,
            ),
            timeout=30,
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/nack")
def v2_nack():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        host = _require_v2()
        result = _run(
            host.nack(
                claim_id=str(payload["claim_token"]),
                generation=int(payload["generation"]),
                request_id=str(payload["request_id"]),
                reason=str(payload.get("reason", "trainer-nack")),
            ),
            timeout=30,
        )
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/seal_batch")
def v2_seal_batch():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        batch = _require_v2().coordinator.seal_batch(_batch_id(payload))
        return jsonify(
            {"ok": True, "state": batch.state.value, "version": batch.version}
        )
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/commit_batch")
def v2_commit_batch():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        batch = _require_v2().coordinator.commit_batch(_batch_id(payload))
        return jsonify(
            {"ok": True, "state": batch.state.value, "version": batch.version}
        )
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/fail_batch")
def v2_fail_batch():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        batch = _run(
            _require_v2().coordinator.fail_batch(
                _batch_id(payload),
                str(payload.get("reason", "operator-failed")),
            ),
            timeout=30,
        )
        return jsonify(
            {"ok": True, "state": batch.state.value, "version": batch.version}
        )
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/mark_in_doubt")
def v2_mark_in_doubt():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        batch = _require_v2().coordinator.mark_in_doubt(
            _batch_id(payload),
            str(payload.get("error", "trainer optimizer outcome unknown")),
        )
        return jsonify(
            {"ok": True, "state": batch.state.value, "version": batch.version}
        )
    except Exception as exc:
        return _v2_error(exc)

@app.post("/v2/recover_batch")
def v2_recover_batch():
    payload = request.get_json(force=True, silent=False) or {}
    try:
        batch = _run(
            _require_v2().coordinator.recover_batch(
                _batch_id(payload),
                request_id=str(payload["request_id"]),
            ),
            timeout=30,
        )
        return jsonify(
            {"ok": True, "state": batch.state.value, "version": batch.version}
        )
    except Exception as exc:
        return _v2_error(exc)

@app.get("/v2/status")
def v2_status():
    try:
        return jsonify({"ok": True, "host": _require_v2().status()})
    except Exception as exc:
        return _v2_error(exc)

@app.post("/close_leases")
def close_leases():
    payload = request.get_json(force=True, silent=True) or {}
    lease_ids = payload.get("lease_ids", [])
    if not isinstance(lease_ids, list):
        return jsonify({"ok": False, "error": "lease_ids must be a list"}), 400
    try:
        _ensure_started()
        if _ROLLOUT_HOST_V2 is not None:
            return _v2_error(
                ProtocolError(
                    ErrorCode.INVALID_TRANSITION,
                    "/close_leases is disabled in v2; use claim ACK/NACK",
                )
            )

        async def _close_all():
            for lease_id in lease_ids:
                await _MASTER_CLIENT._safe_close(lease_id)

        _run(_close_all(), timeout=60)
        return jsonify({"ok": True, "closed": lease_ids})
    except Exception as exc:
        return _json_error(f"close_leases failed: {exc}")

class _SyncEnvClientAdapter:

    def __init__(
        self,
        async_client: AsyncRpcEnvClient,
        *,
        fence_token: str | None = None,
        operation_prefix: str = "probe",
    ):
        self._async_client = async_client
        self._fence_token = fence_token
        self._operation_prefix = operation_prefix
        self._operation_counter = 0

    def _run(self, coro):
        return globals()["_run"](coro, timeout=600)

    def _operation(self, name: str) -> str:
        self._operation_counter += 1
        return (
            f"{self._operation_prefix}:{name}:"
            f"{self._operation_counter}"
        )

    def get_obs(self, lease_id: str) -> dict[str, Any]:
        return self._run(
            self._async_client.get_obs(
                lease_id,
                operation_id=self._operation("get-obs"),
                fence_token=self._fence_token,
            )
        )

    def step(self, lease_id: str, action: str):
        return self._run(
            self._async_client.step(
                lease_id,
                action,
                operation_id=self._operation("step"),
                fence_token=self._fence_token,
            )
        )

    def prepare_for_probe(self, lease_id: str | None = None) -> None:
        if lease_id is None:
            raise ValueError("lease_id is required for prepare_for_probe()")
        return self._run(
            self._async_client.prepare_for_probe(
                lease_id,
                operation_id=self._operation("prepare"),
                fence_token=self._fence_token,
            )
        )

    def evaluate(self, lease_id: str) -> float:
        return self._run(
            self._async_client.evaluate(
                lease_id,
                operation_id=self._operation("evaluate"),
                fence_token=self._fence_token,
            )
        )

    def close(self, lease_id: str) -> None:
        return self._run(
            self._async_client.close(
                lease_id,
                operation_id=self._operation("close"),
                fence_token=self._fence_token,
            )
        )

def _probe_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_agent_type": str(payload.get("probe_agent_type", "Holo3_Probe")),
        "probe_model_name": str(payload.get("probe_model_name", "holo3-1-35b-a3b")),
        "probe_api_key": str(payload.get("probe_api_key", "EMPTY")),
        "probe_base_url": str(payload.get("probe_base_url", "http://127.0.0.1:40001/v1")),
        "probe_max_tokens": int(payload.get("probe_max_tokens", 8192)),
        "probe_temperature": float(payload.get("probe_temperature", 0.0)),
        "probe_history_n": int(payload.get("probe_history_n", 4)),
        "probe_max_steps": int(payload.get("probe_max_steps", 15)),
        "probe_enable_thinking": bool(payload.get("probe_enable_thinking", True)),
        "probe_reasoning_effort": str(payload.get("probe_reasoning_effort", "medium")),
        "probe_max_trajectory_length": int(payload.get("probe_max_trajectory_length", 15)),
        "probe_max_images": int(payload.get("probe_max_images", 3)),
        "probe_client_password": str(payload.get("probe_client_password", "password")),
        "judge_history_n": int(payload.get("history_n", 3)),
    }

@app.post("/run_probe_round")
def run_probe_round():
    payload = request.get_json(force=True, silent=True) or {}
    lease_id = payload.get("lease_id")
    criteria = payload.get("criteria", "")
    instruction = payload.get("instruction", "")
    sampler_round = int(payload.get("sampler_round", 1))
    fence_token = (
        str(payload["fence_token"])
        if payload.get("fence_token") is not None
        else None
    )
    sleep_after_execution = float(payload.get("probe_sleep_after_execution", 2.0))
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400

    try:
        _ensure_started()
        example_result_dir = str(
            _probe_artifact_dir(
                payload,
                lease_id=str(lease_id),
                sampler_round=sampler_round,
            )
        )
        from eval_runtime.guirl.evaluator import (
            get_probe_round_runner,
            make_guirl_probe_agent,
            prepare_env_for_probe,
        )

        env_client = _SyncEnvClientAdapter(
            _ENV_CLIENT,
            fence_token=fence_token,
            operation_prefix=(
                f"probe:{lease_id}:{sampler_round}"
            ),
        )
        guirl_config = _probe_config(payload)
        prepare_env_for_probe(env_client, lease_id)
        probe_agent = make_guirl_probe_agent(criteria, guirl_config)
        probe_agent.reset_for_round(criteria, logger)
        runner = get_probe_round_runner(guirl_config)
        round_record = runner(
            probe_agent=probe_agent,
            env_client=env_client,
            lease_id=lease_id,
            instruction=instruction,
            max_steps=guirl_config["probe_max_steps"],
            example_result_dir=example_result_dir,
            sleep_after_execution=sleep_after_execution,
            sampler_round=sampler_round,
        )
        return jsonify({
            "ok": True,
            "latest_criteria": getattr(probe_agent, "latest_criteria", criteria),
            "criteria_images": _json_safe(getattr(probe_agent, "criteria_images", [])),
            "round_record": _json_safe(round_record),
            "probe_agent_type": guirl_config["probe_agent_type"],
            "artifact_dir": example_result_dir,
        })
    except Exception as exc:
        return _json_error(f"run_probe_round failed: {exc}")

def _shutdown(*_args):
    global _STARTED
    if _ROLLOUT_HOST_V2 is not None and _STARTED:
        try:
            _run(_ROLLOUT_HOST_V2.stop(), timeout=30)
        except Exception:
            logger.exception("Failed to stop Rollout Host v2 cleanly")
    elif _MASTER_CLIENT is not None and _STARTED:
        try:
            _run(_MASTER_CLIENT.stop(), timeout=10)
        except Exception:
            logger.exception("Failed to stop MasterClient cleanly")
    _STARTED = False
    if _LOOP is not None:
        _LOOP.call_soon_threadsafe(_LOOP.stop)

    raise SystemExit(0)

def main(argv: list[str] | None = None) -> int:
    global _ARGS
    parser = argparse.ArgumentParser(description="GUIRL rollout master service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9910)
    parser.add_argument("--env-master-url", default="http://127.0.0.1:9900")
    parser.add_argument("--manifest-path", default="evaluation_examples/service_tasks.json")
    parser.add_argument("--examples-dir", default="evaluation_examples/examples")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--env-capacity", type=int, default=1)
    parser.add_argument(
        "--rollout-host-v2",
        action="store_true",
        help="Enable the durable event-driven v2 Host. Legacy mode is default.",
    )
    parser.add_argument(
        "--state-db",
        default=None,
        help="SQLite state database path (required with --rollout-host-v2).",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifact spool directory (required with --rollout-host-v2).",
    )
    parser.add_argument(
        "--probe-output-root",
        default="./outputs/rl_artifacts",
        help="Server-owned root for /run_probe_round artifacts. Client paths "
        "are treated only as opaque keys and are never written directly.",
    )
    parser.add_argument(
        "--actor-concurrency",
        default=None,
        help="Per-actor concurrency integer or JSON object. In v2, unset "
        "defaults to each actor's fair share ceil(env_capacity/actor_count).",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--infra-retry-budget",
        type=int,
        default=20,
        help="Extra per-slot retries for env-infrastructure failures "
        "(lease lost, env restarting) that do not consume --max-attempts.",
    )
    parser.add_argument("--max-task-replacements", type=int, default=2)
    parser.add_argument(
        "--v2-failure-policy",
        choices=("replace_task_group", "fail_batch"),
        default="replace_task_group",
    )
    parser.add_argument("--claim-ttl-seconds", type=float, default=600.0)
    parser.add_argument("--close-retry-seconds", type=float, default=1.0)
    parser.add_argument(
        "--batch-deadline-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Fail and drain an active v2 batch after this wall-clock deadline.",
    )
    parser.add_argument(
        "--sqlite-synchronous",
        choices=("OFF", "NORMAL", "FULL", "EXTRA"),
        default="FULL",
    )
    parser.add_argument("--sqlite-busy-timeout-ms", type=int, default=5_000)
    parser.add_argument("--actor-harness", default="evocua")
    parser.add_argument(
        "--actors-config",
        default=None,
        help="Path to a JSON/YAML file with an 'actors' list for multi-actor "
        "fan-out (each task runs once per actor). Overrides the single "
        "--actor-* flags when set.",
    )
    parser.add_argument("--actor-base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--actor-model", default="evocua-32b")
    parser.add_argument("--actor-api-key", default="EMPTY")
    parser.add_argument("--actor-max-tokens", type=int, default=1800)
    parser.add_argument("--actor-temperature", type=float, default=0.8)
    parser.add_argument("--actor-max-steps", type=int, default=10)
    parser.add_argument(
        "--actor-timeout-seconds",
        type=float,
        default=None,
        help="Per-actor task timeout. Defaults to 1200s in v2 and preserves "
        "the legacy 300s default otherwise.",
    )
    parser.add_argument(
        "--pull-finished-timeout-seconds",
        type=float,
        default=None,
        help="Server-side guard timeout for /pull_finished. Must exceed the "
        "actor timeout; defaults to actor_timeout + 360s (>= 1560s).",
    )
    parser.add_argument("--actor-max-trajectory-length", type=int, default=12)
    parser.add_argument("--actor-max-images", type=int, default=3)
    parser.add_argument(
        "--env-restart-command",
        default=None,
        help="Shell command that restarts the env stack (workers + env master). "
        "When set, MasterClient auto-runs it after allocate has been failing "
        "continuously for --env-restart-threshold-seconds. Unset = disabled.",
    )
    parser.add_argument(
        "--env-restart-threshold-seconds",
        type=float,
        default=300.0,
        help="Length of the continuous allocate-failure streak that triggers "
        "the env restart command.",
    )
    parser.add_argument(
        "--env-restart-cooldown-seconds",
        type=float,
        default=300.0,
        help="Minimum time between two env restart attempts.",
    )
    parser.add_argument(
        "--env-restart-command-timeout",
        type=float,
        default=300.0,
        help="Kill the env restart command if it runs longer than this.",
    )
    parser.add_argument(
        "--env-restart-healthcheck-wait",
        type=float,
        default=300.0,
        help="After the restart command succeeds, wait up to this long for the "
        "env master healthcheck to pass.",
    )
    parser.add_argument(
        "--env-restart-min-healthy-workers",
        type=int,
        default=1,
        help="Env master /status must report at least this many healthy "
        "workers to count as healthy (watchdog probe + post-restart wait).",
    )
    parser.add_argument(
        "--env-restart-health-probe-interval",
        type=float,
        default=60.0,
        help="Interval between watchdog health probes of the env master.",
    )
    parser.add_argument("--actor-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--client-password", default="password")
    parser.add_argument("--platform", default="ubuntu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _ARGS = _build_args(args)
    probe_output_root = Path(_ARGS.guirl_probe_output_root).expanduser().resolve()
    probe_output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not probe_output_root.is_dir() or not os.access(
        probe_output_root, os.W_OK | os.X_OK
    ):
        raise PermissionError(
            f"--probe-output-root is not writable: {probe_output_root}"
        )
    logger.info("Probe artifacts: %s", probe_output_root)

    for signame in ("SIGTERM", "SIGINT"):
        signal.signal(getattr(signal, signame), _shutdown)

    _ensure_started()
    app.run(host=args.host, port=args.port, threaded=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
