from __future__ import annotations

import argparse
import atexit
import base64
import concurrent.futures
from http.client import HTTPException
import logging
import os
import signal
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from flask import Flask, jsonify, request
import requests

from desktop_env.desktop_env import DesktopEnv
logger = logging.getLogger("gui.env_pool_server")
app = Flask(__name__)

def _json_exception(endpoint: str, lease_id: str | None, exc: Exception) -> tuple[dict[str, Any], int]:
    error_type = type(exc).__name__
    error_message = str(exc)
    logger.exception("%s failed for lease_id=%s: %s", endpoint, lease_id, exc)
    return jsonify(
        {
            "ok": False,
            "error": f"{error_type}: {error_message}",
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc(),
        }
    ), 500

def _encode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    screenshot = obs.get("screenshot")
    out = dict(obs)
    if isinstance(screenshot, (bytes, bytearray)):
        out["screenshot_b64"] = base64.b64encode(screenshot).decode("utf-8")
    else:
        out["screenshot_b64"] = None
    out.pop("screenshot", None)
    return out

@dataclass
class EnvSlot:
    env_id: str
    env: DesktopEnv
    busy: bool = False
    lease_id: str | None = None
    episode_id: str | None = None
    last_used_ts: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

class EnvPool:
    def __init__(
        self,
        *,
        max_envs: int,
        idle_ttl_seconds: int,
        env_kwargs: dict[str, Any],
        prewarm_envs: int = 0,
        prewarm_concurrency: int = 1,
        reset_on_close: bool = True,
        host: str = "127.0.0.1",
        advertise_host: str | None = None,
        port: int = 8000,
        master_url: str = "http://127.0.0.1:9900",
    ) -> None:
        self.max_envs = max_envs
        self.idle_ttl_seconds = idle_ttl_seconds
        self.env_kwargs = env_kwargs
        self.min_envs = max(0, min(prewarm_envs, max_envs))
        self.prewarm_concurrency = max(1, int(prewarm_concurrency))
        self.reset_on_close = reset_on_close
        self._slots: dict[str, EnvSlot] = {}
        self._lease_to_env: dict[str, str] = {}
        self._lock = threading.RLock()
        self._is_shutdown = False
        if not advertise_host:
            advertise_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        worker_id = f"{advertise_host}:{port}:{uuid.uuid4().hex[:8]}"
        self._worker_info = {
            "worker_id": worker_id,
            "pid": os.getpid(),
            "capacity": self.max_envs,
            "base_url": f"http://{advertise_host}:{port}",
            "master_url": master_url,
        }
        if self.min_envs > 0:
            try:
                self._prewarm(self.min_envs)
            except BaseException:
                logger.exception("Prewarm interrupted; cleaning up all managed envs")
                self.shutdown()
                raise

    def _new_slot(self) -> EnvSlot:
        env_id = f"env-{uuid.uuid4().hex[:12]}"
        env = DesktopEnv(**self.env_kwargs)
        return EnvSlot(env_id=env_id, env=env)

    def _create_and_store_slot(self, idx: int, count: int) -> None:
        slot = self._new_slot()
        with self._lock:
            self._slots[slot.env_id] = slot
        logger.info("Prewarmed env %s (%d/%d)", slot.env_id, idx + 1, count)

    def _prewarm(self, count: int) -> None:
        if self.prewarm_concurrency <= 1:
            for idx in range(count):
                self._create_and_store_slot(idx, count)
            return
        logger.info("Prewarming %d envs with concurrency=%d", count, self.prewarm_concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.prewarm_concurrency) as executor:
            futures = [
                executor.submit(self._create_and_store_slot, idx, count)
                for idx in range(count)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _reap_idle_locked(self) -> None:
        now = time.time()
        to_delete: list[str] = []
        removable_budget = max(0, len(self._slots) - self.min_envs)
        for env_id, slot in self._slots.items():
            if slot.busy:
                continue
            if now - slot.last_used_ts < self.idle_ttl_seconds:
                continue
            if removable_budget <= 0:
                break
            try:
                slot.env.close()
            except Exception:
                logger.exception("Failed to close idle env %s", env_id)
            to_delete.append(env_id)
            removable_budget -= 1

        for env_id in to_delete:
            self._slots.pop(env_id, None)

    def allocate(self, episode_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Pool is shutting down")
            self._reap_idle_locked()

            for slot in self._slots.values():
                if not slot.busy:
                    lease_id = f"lease-{uuid.uuid4().hex[:16]}"
                    slot.busy = True
                    slot.lease_id = lease_id
                    slot.episode_id = episode_id
                    slot.last_used_ts = time.time()
                    self._lease_to_env[lease_id] = slot.env_id
                    return {"lease_id": lease_id, "env_id": slot.env_id, "reused": True}

            if len(self._slots) >= self.max_envs:
                raise RuntimeError(f"Pool exhausted: {len(self._slots)}/{self.max_envs}")

            slot = self._new_slot()
            self._slots[slot.env_id] = slot
            lease_id = f"lease-{uuid.uuid4().hex[:16]}"
            slot.busy = True
            slot.lease_id = lease_id
            slot.episode_id = episode_id
            slot.last_used_ts = time.time()
            self._lease_to_env[lease_id] = slot.env_id
            return {"lease_id": lease_id, "env_id": slot.env_id, "reused": False}

    def _get_slot_by_lease_locked(self, lease_id: str) -> EnvSlot:
        env_id = self._lease_to_env.get(lease_id)
        if env_id is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        slot = self._slots.get(env_id)
        if slot is None:
            raise KeyError(f"Lease {lease_id} points to a missing env slot")
        return slot

    def heartbeat(self, lease_id: str) -> None:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            slot.last_used_ts = time.time()

    def reset(self, lease_id: str, task_config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            obs = slot.env.reset(task_config=task_config)
            slot.last_used_ts = time.time()
            return _encode_obs(obs)

    def get_obs(self, lease_id: str) -> dict[str, Any]:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            obs = slot.env._get_obs()
            slot.last_used_ts = time.time()
            return _encode_obs(obs)

    def step(self, lease_id: str, action: Any, sleep_after_execution: float = 0.0) -> dict[str, Any]:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            obs, reward, done, info = slot.env.step(action, sleep_after_execution)
            slot.last_used_ts = time.time()
            return {
                "observation": _encode_obs(obs),
                "reward": reward,
                "done": bool(done),
                "info": info or {},
            }

    def evaluate(self, lease_id: str) -> float:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            slot.last_used_ts = time.time()
            return float(slot.env.evaluate())

    def evaluate_inline(self, lease_id: str, inline_evaluator: str | None = None) -> float:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            slot.last_used_ts = time.time()
            return float(slot.env.evaluate_inline(inline_evaluator))

    def start_recording(self, lease_id: str) -> None:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            slot.env.controller.start_recording()
            slot.last_used_ts = time.time()

    def end_recording(self, lease_id: str, out_path: str) -> None:
        with self._lock:
            slot = self._get_slot_by_lease_locked(lease_id)
        with slot.lock:
            slot.env.controller.end_recording(out_path)
            slot.last_used_ts = time.time()

    def close_lease(self, lease_id: str) -> None:
        with self._lock:
            env_id = self._lease_to_env.pop(lease_id, None)
            if env_id is None:
                return
            slot = self._slots.get(env_id)
            if slot is None:
                return
            slot.lease_id = None
            slot.episode_id = None
            slot.busy = True

        if self.reset_on_close:
            try:
                with slot.lock:
                    slot.env.reset(task_config=None)
            except Exception:
                logger.exception("Failed to reset env %s on close; destroying slot", env_id)
                with self._lock:
                    if self._slots.get(env_id) is slot:
                        self._slots.pop(env_id, None)
                try:
                    with slot.lock:
                        slot.env.close()
                except Exception:
                    logger.exception("Failed to close broken env slot %s", env_id)
                return

        with self._lock:
            if self._slots.get(env_id) is slot:
                slot.busy = False
                slot.last_used_ts = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            busy = sum(1 for slot in self._slots.values() if slot.busy)
            return {
                "total_envs": len(self._slots),
                "busy_envs": busy,
                "idle_envs": len(self._slots) - busy,
                "leases": len(self._lease_to_env),
                "max_envs": self.max_envs,
                "min_envs": self.min_envs,
                "prewarm_concurrency": self.prewarm_concurrency,
                "reset_on_close": self.reset_on_close,
                "is_shutdown": self._is_shutdown,
            }

    def shutdown(self) -> None:
        with self._lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            slots = list(self._slots.values())
            self._slots.clear()
            self._lease_to_env.clear()

        for slot in slots:
            acquired = False
            try:
                acquired = slot.lock.acquire(timeout=1.0)
                if not acquired:
                    logger.warning(
                        "Env %s is still busy during shutdown; forcing close",
                        slot.env_id,
                    )
                slot.env.close()
                logger.info("Closed managed env %s during shutdown", slot.env_id)
            except Exception:
                logger.exception(
                    "Failed to close managed env %s during shutdown",
                    slot.env_id,
                )
            finally:
                if acquired:
                    slot.lock.release()

POOL: EnvPool | None = None
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_DONE = False

def _cleanup_pool(reason: str) -> None:
    global POOL, _CLEANUP_DONE

    with _CLEANUP_LOCK:
        if _CLEANUP_DONE:
            return
        _CLEANUP_DONE = True
        pool = POOL
        POOL = None

    if pool is None:
        logger.info("Env cleanup skipped; no active pool (%s)", reason)
        return

    logger.warning("Cleaning up all managed envs due to %s", reason)
    pool.shutdown()

def _handle_termination_signal(signum: int, _frame: Any) -> None:
    signame = signal.Signals(signum).name
    logger.warning("Received %s; cleaning up all managed envs", signame)
    _cleanup_pool(f"signal {signame}")
    raise SystemExit(128 + signum)

def _schedule_process_shutdown(reason: str) -> None:
    def _shutdown() -> None:
        time.sleep(0.1)
        _cleanup_pool(reason)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_shutdown, daemon=True).start()

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.get("/status")
def status():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    return jsonify({"ok": True, "pool": POOL.status()})

@app.post("/allocate")
def allocate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    try:
        episode_id = data.get("episode_id")
        lease = POOL.allocate(episode_id=episode_id)
        return jsonify({"ok": True, **lease})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/heartbeat")
def heartbeat():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        POOL.heartbeat(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/reset")
def reset():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    task_config = data.get("task_config")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        obs = POOL.reset(lease_id, task_config=task_config)
        return jsonify({"ok": True, "observation": obs})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/get_obs")
def get_obs():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        obs = POOL.get_obs(lease_id)
        return jsonify({"ok": True, "observation": obs})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/step")
def step():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    action = data.get("action")
    sleep_after_execution = float(data.get("sleep_after_execution", 0.0))
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        out = POOL.step(lease_id, action=action, sleep_after_execution=sleep_after_execution)
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/evaluate")
def evaluate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        score = POOL.evaluate(lease_id)
        return jsonify({"ok": True, "score": score})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/evaluate_inline")
def evaluate_inline():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    inline_evaluator = data.get("inline_evaluator")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        score = POOL.evaluate_inline(lease_id, inline_evaluator=inline_evaluator)
        return jsonify({"ok": True, "score": score})
    except Exception as e:
        return _json_exception("evaluate_inline", lease_id, e)

@app.post("/start_recording")
def start_recording():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        POOL.start_recording(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/end_recording")
def end_recording():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    out_path = data.get("out_path")
    if not lease_id or not out_path:
        return jsonify({"ok": False, "error": "lease_id and out_path are required"}), 400
    try:
        POOL.end_recording(lease_id, out_path=out_path)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/close")
def close():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id is required"}), 400
    try:
        POOL.close_lease(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/shutdown")
def shutdown():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool is not initialized"}), 500
    data = request.get_json(force=True, silent=True) or {}
    reason = str(data.get("reason", "remote shutdown"))
    worker_id = POOL._worker_info.get("worker_id")
    pid = POOL._worker_info.get("pid")
    logger.warning("Received remote shutdown request for worker=%s pid=%s reason=%s", worker_id, pid, reason)
    _schedule_process_shutdown(reason)
    return jsonify({"ok": True, "worker_id": worker_id, "pid": pid, "scheduled": True})

def _register():
    if POOL is None:
        raise RuntimeError("Pool is not initialized")
    worker_info = dict(POOL._worker_info)
    master_url = worker_info.pop("master_url").rstrip("/")
    worker_info["status"] = POOL.status()
    response = requests.post(
        f"{master_url}/register",
        json=worker_info,
        timeout=5
    )
    response.raise_for_status()
    logger.info(
        "Worker %s (pid=%s) registered with master at %s",
        worker_info["worker_id"],
        worker_info["pid"],
        master_url,
    )

def _heartbeat():
    while True:
        try:
            if POOL is None:
                return
            response = requests.post(
                f"{POOL._worker_info['master_url'].rstrip('/')}/heartbeat",
                json={
                    "worker_id": POOL._worker_info["worker_id"],
                    "pid": POOL._worker_info["pid"],
                    "base_url": POOL._worker_info["base_url"],
                    "status": POOL.status(),
                },
                timeout=3
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning("Failed to send heartbeat to master: %s", e)
        time.sleep(5)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI environment lease/pool server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-envs", type=int, default=int(os.getenv("GUI_POOL_MAX_ENVS", "64")))
    parser.add_argument("--idle-ttl-seconds", type=int, default=int(os.getenv("GUI_POOL_IDLE_TTL_SECONDS", "600")))
    parser.add_argument("--provider-name", type=str, default=os.getenv("GUI_PROVIDER_NAME", "volcengine"))
    parser.add_argument("--region", type=str, default=os.getenv("GUI_REGION", "us-east-1"))
    parser.add_argument("--path-to-vm", type=str, default=os.getenv("GUI_PATH_TO_VM"))
    parser.add_argument("--action-space", type=str, default=os.getenv("GUI_ACTION_SPACE", "pyautogui"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--observation-type", type=str, default=os.getenv("GUI_OBSERVATION_TYPE", "screenshot"))
    parser.add_argument("--client-password", type=str, default=os.getenv("GUI_CLIENT_PASSWORD", ""))
    parser.add_argument("--screen-width", type=int, default=int(os.getenv("GUI_SCREEN_WIDTH", "1920")))
    parser.add_argument("--screen-height", type=int, default=int(os.getenv("GUI_SCREEN_HEIGHT", "1080")))
    parser.add_argument("--prewarm-envs", type=int, default=int(os.getenv("GUI_PREWARM_ENVS", "0")))
    parser.add_argument(
        "--advertise-host",
        type=str,
        default=os.getenv("GUI_ADVERTISE_HOST"),
        help="Host or IP advertised to master; set this when binding to 0.0.0.0",
    )
    parser.add_argument(
        "--master-url",
        type=str,
        default="http://127.0.0.1:9900",
        help="master_server URL"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run without registering to a master server",
    )

    parser.add_argument(
        "--prewarm-concurrency",
        type=int,
        default=int(os.getenv("GUI_PREWARM_CONCURRENCY", "8")),
        help="Parallel workers used during startup prewarm",
    )
    parser.add_argument(
        "--reset-on-close",
        type=int,
        default=int(os.getenv("GUI_RESET_ON_CLOSE", "1")),
        help="1: reset env when lease closes; 0: only release lease",
    )
    return parser.parse_args()

def main() -> None:
    global POOL
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    atexit.register(_cleanup_pool, "process exit")
    env_kwargs = {
        "path_to_vm": args.path_to_vm,
        "action_space": args.action_space,
        "provider_name": args.provider_name,
        "region": args.region,
        "screen_size": (args.screen_width, args.screen_height),
        "headless": args.headless,
        "os_type": "Ubuntu",
        "require_a11y_tree": args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
        "enable_proxy": True,
        "client_password": args.client_password,
    }
    POOL = EnvPool(
        max_envs=args.max_envs,
        idle_ttl_seconds=args.idle_ttl_seconds,
        env_kwargs=env_kwargs,
        prewarm_envs=args.prewarm_envs,
        prewarm_concurrency=args.prewarm_concurrency,
        reset_on_close=bool(args.reset_on_close),
        host =args.host,
        advertise_host=args.advertise_host,
        port=args.port,
        master_url=args.master_url,
    )
    if not args.standalone:
        _register()
        _heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        _heartbeat_thread.start()
    logger.info(
        "Starting GUI env pool server on %s:%s with max_envs=%s prewarm_envs=%s",
        args.host,
        args.port,
        args.max_envs,
        args.prewarm_envs,
    )
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        _cleanup_pool("server shutdown")

if __name__ == "__main__":
    main()
