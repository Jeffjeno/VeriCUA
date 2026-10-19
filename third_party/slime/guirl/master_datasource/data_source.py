from __future__ import annotations

import atexit
import fcntl
import json
import hashlib
import logging
import math
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

from slime.rollout.data_source import DataSource
from slime.utils.types import Sample

from .step_accumulator import StepAccumulator

logger = logging.getLogger("guirl.data_source")

_REMOTE_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.ceph",
    "fuse.glusterfs",
    "fuse.juicefs",
    "fuse.sshfs",
    "glusterfs",
    "lustre",
    "nfs",
    "nfs4",
    "smbfs",
}

class _TrainerStateLock:

    def __init__(self, root: Path, owner: Mapping[str, Any]) -> None:
        self.path = root / "trainer.lock"
        self._owner = dict(owner)
        self._fd: int | None = None
        self._released = False
        self._atexit_registered = False

    @staticmethod
    def _read_owner(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_owner(self, value: Mapping[str, Any]) -> None:
        if self._fd is None:
            raise RuntimeError("trainer state lock is not held")
        encoded = (
            json.dumps(dict(value), sort_keys=True, default=str) + "\n"
        ).encode("utf-8")
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(self._fd, remaining)
            if written <= 0:
                raise OSError("short write while updating trainer lock metadata")
            remaining = remaining[written:]
        os.fsync(self._fd)

    def acquire(self) -> None:
        if self._fd is not None:
            return
        previous = self._read_owner(self.path)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            current = self._read_owner(self.path) or previous or {}
            raise RuntimeError(
                "GUIRL_V2_TRAINER_STATE_LOCKED "
                + json.dumps(
                    {
                        "message": (
                            "another trainer process owns this v2 state directory"
                        ),
                        "lock_path": str(self.path),
                        "owner": current,
                    },
                    sort_keys=True,
                    default=str,
                )
            ) from exc
        self._fd = fd
        self._released = False
        acquired = {
            **self._owner,
            "lock_path": str(self.path),
            "acquired_at": time.time(),
            "status": "ACTIVE",
        }
        if previous:
            acquired["previous_owner"] = {
                key: previous[key]
                for key in (
                    "instance_id",
                    "pid",
                    "hostname",
                    "run_id",
                    "client_id",
                    "node_id",
                    "acquired_at",
                    "released_at",
                    "status",
                )
                if key in previous
            }
        self._owner = acquired
        self._write_owner(acquired)
        atexit.register(self.release)
        self._atexit_registered = True

    def release(self) -> None:
        fd = self._fd
        if fd is None or self._released:
            return
        self._released = True
        try:
            self._write_owner(
                {
                    **self._owner,
                    "status": "RELEASED",
                    "released_at": time.time(),
                }
            )
        except Exception:
            logger.exception("Failed to update trainer state lock metadata")
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
                self._fd = None
                if self._atexit_registered:
                    try:
                        atexit.unregister(self.release)
                    except Exception:
                        pass
                    self._atexit_registered = False

    def update_owner(self, values: Mapping[str, Any]) -> None:
        if self._fd is None or self._released:
            raise RuntimeError("trainer state lock is not held")
        self._owner = {**self._owner, **dict(values), "status": "ACTIVE"}
        self._write_owner(self._owner)

def _filesystem_type(path: Path) -> str | None:
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    resolved = str(path.resolve())
    matches: list[tuple[int, str]] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        if resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/"):
            matches.append((len(mountpoint), fields[2].lower()))
    return max(matches)[1] if matches else None

class RolloutHostHTTPError(RuntimeError):

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.details = details or {}

class RolloutDataSource(DataSource):

    def __init__(self, args):
        self.args = args
        self.v2_enabled = _as_bool(
            _get_arg(
                args,
                "guirl_rollout_host_v2",
                _get_arg(args, "rollout_host_v2", False),
            )
        )
        self.rollout_master_url = str(
            _get_arg(args, "guirl_rollout_master_url", "http://127.0.0.1:9910")
        ).rstrip("/")

        self.http_timeout_seconds = float(
            _get_arg(args, "guirl_rollout_master_timeout_seconds", 1620.0)
        )

        self.pull_total_deadline_seconds = float(
            _get_arg(args, "guirl_pull_total_deadline_seconds", 7200.0)
        )
        self.pull_retry_backoff_seconds = float(
            _get_arg(args, "guirl_pull_retry_backoff_seconds", 10.0)
        )
        self.pull_retry_backoff_max_seconds = float(
            _get_arg(args, "guirl_pull_retry_backoff_max_seconds", 60.0)
        )

        self.stream_pull = bool(_get_arg(args, "guirl_stream_pull", False))
        self._buffer: list[list[Sample]] = []

        self._index_counter: int = 0
        self.accumulator: StepAccumulator | None = None
        self.client_id: str | None = None
        self.run_id: str | None = None
        self._v2_state_root: Path | None = None
        self._v2_state_lock: _TrainerStateLock | None = None
        self._v2_actor_order: list[str] = []
        self.trainer_node_id: str | None = None
        try:
            if self.v2_enabled:
                self._init_v2_state()
            self._healthcheck()
        except BaseException:
            self.close()
            raise
        logger.info(
            "Using remote GUIRL rollout master at %s (%s)",
            self.rollout_master_url,
            "v2" if self.v2_enabled else "legacy",
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:

        if self.v2_enabled:
            raise RuntimeError(
                "Rollout Host v2 batches must be consumed by "
                "guirl.master_datasource.streaming_rollout; refusing legacy "
                "/pull_finished for a v2 run"
            )

        if self._buffer:
            from_buffer = self._buffer[:num_samples]
            self._buffer = self._buffer[num_samples:]
            if len(from_buffer) >= num_samples:
                return from_buffer[:num_samples]
            if self.stream_pull:

                return from_buffer

            remaining = num_samples - len(from_buffer)
        else:
            from_buffer = []
            remaining = num_samples

        collected: list[list[Sample]] = list(from_buffer)
        deadline = time.monotonic() + self.pull_total_deadline_seconds
        backoff = self.pull_retry_backoff_seconds
        attempt = 0

        while len(collected) < num_samples:
            need = num_samples - len(collected)
            try:
                new_samples = self._pull_samples(need)
                collected.extend(new_samples)

                backoff = self.pull_retry_backoff_seconds
                if self.stream_pull and collected:

                    break
            except Exception as exc:
                attempt += 1
                remaining_budget = deadline - time.monotonic()
                if remaining_budget <= 0:
                    raise RuntimeError(
                        f"rollout master failed to deliver {num_samples} sample(s) "
                        f"within {self.pull_total_deadline_seconds:.0f}s "
                        f"({len(collected)} collected, {attempt} attempt(s)); "
                        f"last error: {exc}"
                    ) from exc
                sleep_s = min(backoff, max(0.0, remaining_budget))
                logger.warning(
                    "pull_finished attempt %d failed (%s); have %d/%d, retrying in %.1fs "
                    "(%.0fs budget left)",
                    attempt, exc, len(collected), num_samples, sleep_s, remaining_budget,
                )
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, self.pull_retry_backoff_max_seconds)

        if len(collected) > num_samples:
            self._buffer = collected[num_samples:] + self._buffer
            collected = collected[:num_samples]
        return collected

    def _pull_samples(self, num_needed: int) -> list[list[Sample]]:

        result = self._post_json("/pull_finished", {"num_samples": num_needed})
        entries = result.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError(f"rollout master returned invalid entries: {result}")
        return [[self._entry_to_sample(entry)] for entry in entries]

    def _entry_to_sample(self, entry: dict[str, Any]) -> Sample:

        prompt = entry.get("prompt", {})
        instruction = ""
        if isinstance(prompt, dict):
            instruction = prompt.get("instruction", "")
        elif isinstance(prompt, str):
            instruction = prompt

        index = self._index_counter
        self._index_counter += 1

        return Sample(
            index=index,
            prompt=instruction,
            metadata={
                "guirl": {
                    "task_id": entry.get("task_id"),
                    "batch_id": entry.get("batch_id"),
                    "lease_id": entry.get("lease_id"),
                    "trajectory": entry.get("trajectory", []),
                    "screenshots": entry.get("screenshots", []),
                    "S": entry.get("S"),
                    "task_config": entry.get("task_config", {}),

                    "actor": (entry.get("metadata") or {}).get("actor"),
                },
            },
        )

    def acquire_batch(self, rollout_id: int, tasks_per_step: int) -> dict[str, Any]:

        run_id, client_id, accumulator = self._v2_context()
        request_id = f"acquire:{self.run_id}:{int(rollout_id)}"
        fingerprint_payload = {
            "rollout_batch_size": int(_get_arg(self.args, "rollout_batch_size", 0)),
            "tasks_per_step": int(tasks_per_step),
            "actor_order": list(self._v2_actor_order),
            "n_samples_per_prompt": int(
                _get_arg(self.args, "n_samples_per_prompt", 1)
            ),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        result = self._v2_post(
            "/v2/acquire_batch",
            {
                "run_id": run_id,
                "train_step_id": int(rollout_id),
                "tasks_per_step": int(tasks_per_step),
                "request_id": request_id,
                "config_fingerprint": fingerprint,
            },
        )
        slots = result.get("slots")
        if not isinstance(slots, list):
            raise RuntimeError(f"v2 acquire returned invalid manifest: {result}")
        self._v2_actor_order = list(
            dict.fromkeys(str(slot["actor_id"]) for slot in slots)
        )
        accumulator.refresh_manifest(
            run_id=run_id,
            train_step_id=int(rollout_id),
            manifest_revision=int(result["manifest_revision"]),
            payload_hash=str(result["payload_hash"]),
            slots=slots,
        )
        return result

    def pull_ready(
        self,
        rollout_id: int,
        *,
        limit: int,
        wait_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:

        run_id, client_id, accumulator = self._v2_context()
        requested_payload = {
            "batch_id": {
                "run_id": run_id,
                "train_step_id": int(rollout_id),
            },
            "consumer": client_id,
            "limit": max(1, int(limit)),
            "ttl_seconds": self._claim_ttl_seconds,
            "wait_seconds": max(0.0, float(wait_seconds)),
            "include_payload": True,
        }
        request_id, persisted_payload = accumulator.reserve_request(
            run_id, int(rollout_id), "pull", requested_payload
        )
        wire_payload = dict(persisted_payload)
        wire_payload["request_id"] = request_id
        result = self._v2_post(
            "/v2/pull_ready",
            wire_payload,
            timeout=max(
                self.http_timeout_seconds,
                float(persisted_payload["wait_seconds"]) + 30.0,
            ),
        )
        entries = result.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError(f"v2 pull returned invalid entries: {result}")
        accumulator.record_claims_and_complete_request(
            run_id,
            int(rollout_id),
            entries,
            request_id,
        )
        return entries

    def renew_claim(self, rollout_id: int, entry: dict[str, Any]) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        expires_at = float(
            entry.get("expires_at")
            or (time.time() + self._claim_ttl_seconds)
        )

        operation = (
            f"renew:{entry['claim_token']}:{int(entry['generation'])}:"
            f"{expires_at:.6f}"
        )
        requested_payload = {
            "claim_token": str(entry["claim_token"]),
            "generation": int(entry["generation"]),
            "consumer": client_id,
            "ttl_seconds": self._claim_ttl_seconds,
        }
        request_id, persisted_payload = accumulator.reserve_request(
            run_id,
            int(rollout_id),
            operation,
            requested_payload,
        )
        wire_payload = dict(persisted_payload)
        wire_payload["request_id"] = request_id
        safety_margin = max(
            0.1,
            min(
                self._renew_http_timeout_seconds,
                self._claim_ttl_seconds / 6.0,
            ),
        )
        retry_budget = expires_at - time.time() - safety_margin
        if retry_budget <= 0:
            raise RuntimeError(
                "claim renewal deadline elapsed before the HTTP attempt"
            )
        result = self._v2_post(
            "/v2/renew_claim",
            wire_payload,
            timeout=min(self._renew_http_timeout_seconds, retry_budget),
            retry_deadline_seconds=retry_budget,
        )
        renewed_expiry = result.get("expires_at")
        if renewed_expiry is None:
            raise RuntimeError(
                "Rollout Host renewal did not return a claim expiry"
            )
        accumulator.record_renewal_and_complete_request(
            run_id,
            int(rollout_id),
            entry,
            request_id,
            expires_at=float(renewed_expiry),
        )
        entry["expires_at"] = float(renewed_expiry)
        return result

    def ack(
        self,
        rollout_id: int,
        entry: Mapping[str, Any],
        judge_artifact: Mapping[str, Any],
    ) -> dict[str, Any]:

        run_id, client_id, accumulator = self._v2_context()
        task_index, actor_id = accumulator.slot_identity(entry)
        judge_payload = accumulator.load_judge_payload(
            run_id,
            int(rollout_id),
            task_index,
            actor_id,
            judge_artifact,
        )
        request_id = (
            f"ack:{run_id}:{int(rollout_id)}:{task_index}:{actor_id}:"
            f"{entry['claim_token']}:{int(entry['generation'])}:"
            f"{judge_artifact['digest']}"
        )
        result = self._v2_post(
            "/v2/ack",
            {
                "claim_token": str(entry["claim_token"]),
                "generation": int(entry["generation"]),
                "attempt": int(entry.get("attempt", 1)),
                "request_id": request_id,
                "judge_payload": judge_payload,
            },
            timeout=min(
                self.http_timeout_seconds,
                self._ack_retry_deadline_seconds,
            ),
            retry_deadline_seconds=self._ack_retry_deadline_seconds,
        )
        host_artifact = result.get("judge_artifact")
        if not isinstance(host_artifact, dict):
            raise RuntimeError(
                "Rollout Host ack did not return its durable Judge artifact"
            )
        accumulator.mark_acked(
            run_id,
            int(rollout_id),
            task_index,
            actor_id,
            claim_token=str(entry["claim_token"]),
            generation=int(entry["generation"]),
            host_judge_artifact=host_artifact,
        )
        return result

    def nack(
        self,
        rollout_id: int,
        entry: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        task_index, actor_id = StepAccumulator.slot_identity(entry)
        request_id = (
            f"nack:{run_id}:{int(rollout_id)}:{task_index}:{actor_id}:"
            f"{entry['claim_token']}:{int(entry['generation'])}"
        )
        return self._v2_post(
            "/v2/nack",
            {
                "claim_token": str(entry["claim_token"]),
                "generation": int(entry["generation"]),
                "request_id": request_id,
                "reason": str(reason),
            },
            timeout=min(
                self.http_timeout_seconds,
                self._ack_retry_deadline_seconds,
            ),
            retry_deadline_seconds=self._ack_retry_deadline_seconds,
        )

    def seal(self, rollout_id: int) -> dict[str, Any]:
        result = self._batch_operation("/v2/seal_batch", rollout_id, "seal")
        if result.get("state") == "SEALED":
            run_id, client_id, accumulator = self._v2_context()
            accumulator.mark_state(run_id, int(rollout_id), "SEALED")
        return result

    def begin_update(self, rollout_id: int) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        accumulator.begin_update(run_id, int(rollout_id))
        return accumulator.journal(run_id, int(rollout_id)) or {}

    def optimizer_succeeded(self, rollout_id: int) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        accumulator.optimizer_succeeded(run_id, int(rollout_id))
        return accumulator.journal(run_id, int(rollout_id)) or {}

    def checkpoint_succeeded(
        self,
        rollout_id: int,
        checkpoint_marker: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        accumulator.checkpoint_succeeded(
            run_id,
            int(rollout_id),
            checkpoint_marker,
        )
        return accumulator.journal(run_id, int(rollout_id)) or {}

    def commit(self, rollout_id: int) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        accumulator.begin_commit(run_id, int(rollout_id))
        result = self._batch_operation("/v2/commit_batch", rollout_id, "commit")
        if result.get("state") == "COMMITTED":
            accumulator.mark_committed(run_id, int(rollout_id))
        else:
            raise RuntimeError(
                f"Rollout Host commit returned unexpected state: {result}"
            )
        return result

    def fail(self, rollout_id: int, reason: str) -> dict[str, Any]:
        result = self._batch_operation(
            "/v2/fail_batch", rollout_id, "fail", reason=reason
        )
        if self.accumulator is not None and self.run_id is not None:
            self.accumulator.clear_revision(
                self.run_id, int(rollout_id), terminal_state="FAILED"
            )
        return result

    def mark_in_doubt(self, rollout_id: int, error: str) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()

        accumulator.mark_optimizer_in_doubt(
            run_id, int(rollout_id), str(error)
        )
        return self._batch_operation(
            "/v2/mark_in_doubt", rollout_id, "in-doubt", error=error
        )

    def batch_status(self, rollout_id: int) -> dict[str, Any] | None:
        run_id, client_id, accumulator = self._v2_context()
        result = self._get_json("/v2/status", timeout=30.0)
        host = result.get("host") or {}
        for batch in host.get("batches") or []:
            identity = batch.get("batch_id") or {}
            if (
                str(identity.get("run_id")) == run_id
                and int(identity.get("train_step_id", -1)) == int(rollout_id)
            ):
                return batch
        return None

    def v2_status(self) -> dict[str, Any]:
        self._require_v2()
        return self._get_json("/v2/status", timeout=30.0)

    def entry_to_v2_sample(
        self,
        rollout_id: int,
        entry: Mapping[str, Any],
    ) -> Sample:

        payload = entry.get("actor_payload") or {}
        slot = entry.get("slot") or {}
        task_index = int(slot["task_index"])
        actor_id = str(entry.get("actor_id") or slot["actor_id"])
        try:
            actor_rank = self._v2_actor_order.index(actor_id)
        except ValueError:
            actor_rank = len(self._v2_actor_order)
        actor_count = max(1, len(self._v2_actor_order))
        rollout_batch_size = int(_get_arg(self.args, "rollout_batch_size", actor_count))
        source_group_id = (
            int(rollout_id) * rollout_batch_size
            + task_index * actor_count
            + actor_rank
        )
        prompt = payload.get("prompt", {})
        instruction = (
            prompt.get("instruction", "")
            if isinstance(prompt, dict)
            else str(prompt or "")
        )
        lease = entry.get("lease") or payload.get("lease") or {}
        metadata = {
            "guirl": {
                "v2": True,
                "run_id": self.run_id,
                "rollout_id": int(rollout_id),
                "task_id": entry.get("task_id"),
                "task_group_id": entry.get("task_group_id"),
                "task_group_revision": entry.get("task_group_revision"),
                "manifest_revision": entry.get("manifest_revision"),
                "slot": dict(slot),
                "actor": actor_id,
                "actor_id": actor_id,
                "attempt": entry.get("attempt"),
                "claim_token": entry.get("claim_token"),
                "claim_generation": entry.get("generation"),
                "lease_id": lease.get("lease_id"),
                "lease": dict(lease),
                "trajectory": payload.get("trajectory", []),
                "screenshots": payload.get("screenshots", []),
                "S": payload.get("native_score", payload.get("S")),
                "task_config": payload.get("task_config", {}),
                "source_group_id": source_group_id,
            }
        }
        kwargs = {
            "index": source_group_id,
            "group_id": source_group_id,
            "prompt": instruction,
            "metadata": metadata,
        }
        try:
            return Sample(**kwargs)
        except TypeError:

            kwargs.pop("group_id")
            sample = Sample(**kwargs)
            sample.group_id = source_group_id
            return sample

    def add_samples(self, samples: list[list[Sample]]) -> None:

        if not samples:
            return

        if self.v2_enabled:

            for group in samples:
                for sample in group:
                    guirl = ((sample.metadata or {}).get("guirl") or {})
                    if not guirl.get("v2") or not guirl.get("claim_token"):
                        continue
                    entry = {
                        "slot": guirl.get("slot") or {},
                        "task_id": guirl.get("task_id"),
                        "task_group_id": guirl.get("task_group_id"),
                        "claim_token": guirl["claim_token"],
                        "generation": guirl.get("claim_generation"),
                    }
                    try:
                        self.nack(
                            int(guirl.get("rollout_id", 0)),
                            entry,
                            "slime-abort",
                        )
                    except Exception:
                        logger.exception("Failed to nack aborted v2 claim")
            return

        lease_ids: list[str] = []
        seen: set[str] = set()
        for group in samples:
            for sample in group:
                guirl = ((sample.metadata or {}).get("guirl") or {})
                if not isinstance(guirl, dict):
                    continue
                lease_id = guirl.get("lease_id")
                if isinstance(lease_id, str) and lease_id and lease_id not in seen:
                    seen.add(lease_id)
                    lease_ids.append(lease_id)

        if not lease_ids:
            logger.warning("Dropping %d aborted GUIRL sample group(s) without leases", len(samples))
            return

        try:
            self._post_json("/close_leases", {"lease_ids": lease_ids}, timeout=60.0)
        except Exception:

            logger.exception("Failed to close aborted GUIRL lease(s): %s — leases may linger", lease_ids)
            return
        logger.info("Dropped %d aborted GUIRL group(s) and closed %d lease(s)", len(samples), len(lease_ids))

    def save(self, rollout_id) -> None:

        pass

    def load(self, rollout_id=None) -> None:

        pass

    def close(self) -> None:

        accumulator = self.accumulator
        self.accumulator = None
        if accumulator is not None:
            try:
                accumulator.close()
            except Exception:
                logger.exception("Failed to close GUIRL v2 accumulator")
        state_lock = self._v2_state_lock
        self._v2_state_lock = None
        if state_lock is not None:
            state_lock.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:

        if self.v2_enabled:
            status = self.v2_status()
            host = status.get("host") or {}
            task_count = host.get("task_count")
            actor_count = host.get("actor_count")
            if task_count is None or actor_count is None:
                raise RuntimeError(
                    "Cannot determine v2 dataset size: Host status must report "
                    "task_count and actor_count"
                )
            task_count = int(task_count)
            actor_count = int(actor_count)
            slots_per_actor = int(
                _get_arg(self.args, "guirl_v2_slots_per_actor", 0)
            )
            if task_count <= 0 or actor_count <= 0 or slots_per_actor <= 0:
                raise RuntimeError(
                    "Cannot determine v2 padded epoch size: task_count, fixed "
                    "actor count N, and slots-per-actor M must all be positive"
                )
            if self._v2_actor_order and actor_count != len(self._v2_actor_order):
                raise RuntimeError(
                    "Rollout Host actor count changed after v2 validation"
                )
            padded_tasks = math.ceil(task_count / slots_per_actor) * slots_per_actor
            return padded_tasks * actor_count

        status = self._get_json("/status", timeout=30.0)
        task_count = ((status.get("master") or {}).get("task_count"))
        if task_count is None:
            raise RuntimeError(
                "Cannot determine dataset size: rollout master did not report task_count"
            )
        return int(task_count)

    def _healthcheck(self) -> None:
        result = self._get_json("/healthz", timeout=30.0)
        if not result.get("ok", False):
            raise RuntimeError(f"rollout master healthcheck failed: {result}")
        master = result.get("master") or {}
        mode = str(master.get("mode") or "legacy").lower()
        if self.v2_enabled:
            if mode != "v2":
                raise self._v2_configuration_error(
                    "trainer enabled v2 but Rollout Host is in legacy mode",
                    trainer_mode="v2",
                    host_mode=mode,
                )
            status = self._get_json("/v2/status", timeout=30.0)
            self._validate_v2_trainer_config(status.get("host") or {})
        elif mode == "v2":
            raise self._v2_configuration_error(
                "legacy trainer cannot consume a v2 Rollout Host run",
                trainer_mode="legacy",
                host_mode="v2",
                action="enable guirl_rollout_host_v2 or use a legacy host",
            )

    def _init_v2_state(self) -> None:
        configured_root = _get_arg(self.args, "guirl_v2_trainer_state_dir", None)
        if not isinstance(configured_root, (str, os.PathLike)) or not str(
            configured_root
        ).strip():
            configured_root = os.environ.get("GUIRL_TRAINER_STATE_DIR")
        if not configured_root:
            raise self._v2_configuration_error(
                "v2 requires an explicit persistent trainer state directory",
                setting="guirl_v2_trainer_state_dir",
                recommendation="/var/tmp/guirl-${USER}/trainer/<run>",
            )
        root = Path(configured_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise self._v2_configuration_error(
                "trainer state path is not a directory",
                path=root,
            )
        filesystem = _filesystem_type(root)
        if filesystem in _REMOTE_FILESYSTEMS:
            raise self._v2_configuration_error(
                "trainer state directory must be on local storage",
                path=root,
                filesystem=filesystem,
            )
        try:
            fd, probe_name = tempfile.mkstemp(
                prefix=".guirl-write-probe.", dir=str(root)
            )
            os.close(fd)
            os.unlink(probe_name)
        except OSError as exc:
            raise self._v2_configuration_error(
                "trainer state directory is not writable",
                path=root,
                error=str(exc),
            ) from exc
        self._v2_state_root = root
        state_lock = _TrainerStateLock(
            root,
            {
                "instance_id": f"trainer-instance-{uuid.uuid4().hex}",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "state_dir": str(root),
                "configured_run_id": _get_arg(
                    self.args, "guirl_v2_run_id", None
                ),
                "configured_client_id": _get_arg(
                    self.args, "guirl_v2_client_id", None
                ),
            },
        )
        state_lock.acquire()
        self._v2_state_lock = state_lock
        identity_path = root / "identity.json"
        configured_client = _get_arg(self.args, "guirl_v2_client_id", None)
        configured_run = _get_arg(self.args, "guirl_v2_run_id", None)
        configured_client = (
            str(configured_client)
            if isinstance(configured_client, (str, int)) and str(configured_client)
            else None
        )
        configured_run = (
            str(configured_run)
            if isinstance(configured_run, (str, int)) and str(configured_run)
            else None
        )
        identity: dict[str, str] = {}
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError):
            logger.warning("Replacing invalid v2 trainer identity file %s", identity_path)
        if identity and configured_run and identity.get("run_id") not in {
            None,
            configured_run,
        }:
            raise self._v2_configuration_error(
                "trainer state directory belongs to a different run_id",
                state_dir=root,
                stored_run_id=identity.get("run_id"),
                configured_run_id=configured_run,
                action="use a new run_id and state directory",
            )
        configured_node = _get_arg(
            self.args, "guirl_v2_trainer_node_id", None
        )
        configured_node = (
            str(configured_node)
            if configured_node is not None and str(configured_node)
            else None
        )
        actual_node = _current_ray_node_id()
        if configured_node and actual_node and configured_node != actual_node:
            raise self._v2_configuration_error(
                "RolloutManager is running on a different Ray node than the "
                "hard-affinity driver node",
                expected_node_id=configured_node,
                actual_node_id=actual_node,
                action=(
                    "stop the run and restore the manager on the original node, "
                    "or explicitly migrate state to operator-approved shared "
                    "supported storage"
                ),
            )
        self.trainer_node_id = (
            actual_node
            or configured_node
            or f"host:{socket.gethostname()}"
        )
        stored_node = identity.get("node_id")
        if stored_node and stored_node != self.trainer_node_id:
            raise self._v2_configuration_error(
                "trainer state directory belongs to a different Ray node; "
                "node-local state cannot be reused silently",
                state_dir=root,
                stored_node_id=stored_node,
                current_node_id=self.trainer_node_id,
                action=(
                    "restore the RolloutManager on the stored node or perform "
                    "an operator-controlled migration to shared supported storage"
                ),
            )
        default_client = "trainer-" + hashlib.sha256(
            f"{socket.gethostname()}:{os.getuid()}:{root}".encode("utf-8")
        ).hexdigest()[:16]
        self.client_id = configured_client or identity.get("client_id") or default_client
        self.run_id = configured_run or identity.get("run_id") or f"run-{uuid.uuid4().hex}"
        desired = {
            "client_id": self.client_id,
            "run_id": self.run_id,
            "node_id": self.trainer_node_id,
        }
        state_lock.update_owner(desired)
        try:
            if identity != desired:
                fd, temp_name = tempfile.mkstemp(
                    prefix=".identity.", suffix=".tmp", dir=str(root)
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(desired, handle, sort_keys=True)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_name, identity_path)
                finally:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
            self.accumulator = StepAccumulator(
                root,
                artifact_root=os.environ.get("GUIRL_V2_TRAINER_ARTIFACT_DIR") or None,
            )
            self._claim_ttl_seconds = float(
                _get_arg(self.args, "guirl_v2_claim_ttl_seconds", 600.0)
            )
            if self._claim_ttl_seconds <= 0:
                raise ValueError("guirl_v2_claim_ttl_seconds must be positive")
            default_renew_timeout = max(
                1.0, min(30.0, self._claim_ttl_seconds / 3.0)
            )
            self._renew_http_timeout_seconds = float(
                _get_arg(
                    self.args,
                    "guirl_v2_renew_http_timeout_seconds",
                    default_renew_timeout,
                )
            )
            if (
                self._renew_http_timeout_seconds <= 0
                or self._renew_http_timeout_seconds
                >= self._claim_ttl_seconds
            ):
                raise self._v2_configuration_error(
                    "renew HTTP timeout must be positive and shorter than claim TTL",
                    renew_http_timeout_seconds=self._renew_http_timeout_seconds,
                    claim_ttl_seconds=self._claim_ttl_seconds,
                )
            default_ack_deadline = max(
                1.0, min(30.0, self._claim_ttl_seconds / 3.0)
            )
            self._ack_retry_deadline_seconds = float(
                _get_arg(
                    self.args,
                    "guirl_v2_ack_retry_deadline_seconds",
                    default_ack_deadline,
                )
            )
            if (
                self._ack_retry_deadline_seconds <= 0
                or self._ack_retry_deadline_seconds
                >= self._claim_ttl_seconds
            ):
                raise self._v2_configuration_error(
                    "ack retry deadline must be positive and shorter than claim TTL",
                    ack_retry_deadline_seconds=self._ack_retry_deadline_seconds,
                    claim_ttl_seconds=self._claim_ttl_seconds,
                )
            default_ack_total = max(
                self._ack_retry_deadline_seconds,
                min(300.0, self._claim_ttl_seconds),
            )
            self._ack_total_deadline_seconds = float(
                _get_arg(
                    self.args,
                    "guirl_v2_ack_total_deadline_seconds",
                    default_ack_total,
                )
            )
            if (
                self._ack_total_deadline_seconds <= 0
                or self._ack_total_deadline_seconds
                < self._ack_retry_deadline_seconds
            ):
                raise self._v2_configuration_error(
                    "ack total deadline must be positive and no shorter than "
                    "one ack retry window",
                    ack_total_deadline_seconds=self._ack_total_deadline_seconds,
                    ack_retry_deadline_seconds=self._ack_retry_deadline_seconds,
                )
        except BaseException:
            self.close()
            raise

    def _require_v2(self) -> None:
        if not self.v2_enabled:
            raise RuntimeError("Rollout Host v2 client called while v2 is disabled")

    def _v2_context(self) -> tuple[str, str, StepAccumulator]:
        self._require_v2()
        if self.run_id is None or self.client_id is None or self.accumulator is None:
            raise RuntimeError(
                "GUIRL_V2_STATE_ERROR: v2 client identity/state is not initialized"
            )
        return self.run_id, self.client_id, self.accumulator

    @staticmethod
    def _v2_configuration_error(message: str, **details: Any) -> ValueError:
        return ValueError(
            "GUIRL_V2_CONFIG_VALIDATION_FAILED "
            + json.dumps(
                {"message": message, "details": details},
                sort_keys=True,
                default=str,
            )
        )

    def _validate_v2_trainer_config(self, host: Mapping[str, Any]) -> None:
        actor_order = [str(actor) for actor in host.get("actor_order") or []]
        actor_count = int(host.get("actor_count") or len(actor_order))
        expected = _get_arg(self.args, "guirl_v2_expected_actors", None)
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except json.JSONDecodeError:
                expected = [
                    item.strip() for item in expected.split(",") if item.strip()
                ]
        expected_order = (
            [str(actor) for actor in expected]
            if isinstance(expected, (list, tuple))
            else []
        )
        rollout_batch = int(_get_arg(self.args, "rollout_batch_size", 0))
        global_batch = int(_get_arg(self.args, "global_batch_size", 0))
        n_samples = int(_get_arg(self.args, "n_samples_per_prompt", 0))
        configured_slots = _get_arg(
            self.args, "guirl_v2_slots_per_actor", None
        )
        slots_per_actor = (
            int(configured_slots) if configured_slots is not None else None
        )
        compatibility_override = _as_bool(
            _get_arg(
                self.args,
                "guirl_v2_allow_batch_size_compatibility",
                False,
            )
        )
        errors: list[dict[str, Any]] = []
        if (
            actor_count <= 0
            or len(actor_order) != actor_count
            or len(set(actor_order)) != len(actor_order)
            or any(not actor for actor in actor_order)
        ):
            errors.append(
                {
                    "field": "host.actor_order",
                    "message": "host actor manifest must be non-empty and unique",
                    "value": actor_order,
                    "actor_count": actor_count,
                }
            )
        if not expected_order:
            errors.append(
                {
                    "field": "guirl_v2_expected_actors",
                    "message": "fixed actor order must be configured explicitly",
                }
            )
        elif expected_order != actor_order:
            errors.append(
                {
                    "field": "guirl_v2_expected_actors",
                    "message": "configured actor order does not match the host",
                    "expected": expected_order,
                    "actual": actor_order,
                }
            )
        if n_samples != 1:
            errors.append(
                {
                    "field": "n_samples_per_prompt",
                    "message": "Rollout Host v2 requires exactly one sample per prompt",
                    "value": n_samples,
                }
            )
        if not compatibility_override:
            if slots_per_actor is None or slots_per_actor <= 0:
                errors.append(
                    {
                        "field": "guirl_v2_slots_per_actor",
                        "message": "positive per-actor slot count M is required",
                    }
                )
            else:
                expected_batch = actor_count * slots_per_actor
                if rollout_batch != expected_batch:
                    errors.append(
                        {
                            "field": "rollout_batch_size",
                            "message": "must equal N actors × M slots",
                            "expected": expected_batch,
                            "actual": rollout_batch,
                            "N": actor_count,
                            "M": slots_per_actor,
                        }
                    )
            if global_batch != rollout_batch:
                errors.append(
                    {
                        "field": "global_batch_size",
                        "message": "must equal rollout_batch_size",
                        "rollout_batch_size": rollout_batch,
                        "global_batch_size": global_batch,
                    }
                )
        if errors:
            raise self._v2_configuration_error(
                "trainer v2 configuration is incompatible with the host",
                errors=errors,
                compatibility_override=(
                    "set guirl_v2_allow_batch_size_compatibility=true only "
                    "for an intentional legacy migration"
                ),
            )
        self._v2_actor_order = actor_order
        self._persist_v2_config_identity(
            {
                "run_id": self.run_id,
                "trainer_node_id": self.trainer_node_id,
                "rollout_master_url": self.rollout_master_url,
                "actor_order": actor_order,
                "slots_per_actor": slots_per_actor,
                "rollout_batch_size": rollout_batch,
                "global_batch_size": global_batch,
                "n_samples_per_prompt": n_samples,
                "compatibility_override": compatibility_override,
                "reward_mode": _get_arg(self.args, "guirl_reward_mode", "shaped"),
            }
        )

    def _persist_v2_config_identity(self, desired: Mapping[str, Any]) -> None:
        if self._v2_state_root is None:
            raise RuntimeError(
                "GUIRL_V2_STATE_ERROR: trainer state directory is not initialized"
            )
        config_path = self._v2_state_root / "config.json"
        existing: dict[str, Any] | None = None
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            raise self._v2_configuration_error(
                "trainer config fingerprint file is unreadable",
                path=config_path,
                error=str(exc),
            ) from exc
        desired_dict = dict(desired)

        if existing is not None:
            existing.setdefault("reward_mode", "shaped")
        if existing is not None and existing != desired_dict:
            raise self._v2_configuration_error(
                "actor manifest or config fingerprint changed for existing state",
                state_dir=self._v2_state_root,
                stored=existing,
                requested=desired_dict,
                action="choose a new guirl_v2_run_id and trainer state directory",
            )
        if existing is None:
            fd, temp_name = tempfile.mkstemp(
                prefix=".config.", suffix=".tmp", dir=str(self._v2_state_root)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(desired_dict, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, config_path)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

    def _batch_operation(
        self,
        endpoint: str,
        rollout_id: int,
        operation: str,
        **extra: Any,
    ) -> dict[str, Any]:
        run_id, client_id, accumulator = self._v2_context()
        payload = {
            "batch_id": {
                "run_id": run_id,
                "train_step_id": int(rollout_id),
            },
            "request_id": f"{operation}:{run_id}:{int(rollout_id)}",
            **extra,
        }
        return self._v2_post(endpoint, payload)

    def _v2_post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        retry_deadline_seconds: float | None = None,
    ) -> dict[str, Any]:

        retry_window = (
            self.pull_total_deadline_seconds
            if retry_deadline_seconds is None
            else max(0.0, float(retry_deadline_seconds))
        )
        deadline = time.monotonic() + retry_window
        base_backoff = self.pull_retry_backoff_seconds
        max_backoff = self.pull_retry_backoff_max_seconds
        if retry_deadline_seconds is not None:
            base_backoff = min(
                base_backoff,
                max(0.05, retry_window / 10.0),
            )
            max_backoff = min(
                max_backoff,
                max(base_backoff, retry_window / 3.0),
            )
        backoff = base_backoff
        while True:
            remaining_before_attempt = deadline - time.monotonic()
            if remaining_before_attempt <= 0:
                raise RuntimeError(
                    f"POST {endpoint} retry deadline elapsed"
                )
            attempt_timeout = timeout
            if timeout is not None:
                attempt_timeout = min(
                    float(timeout), remaining_before_attempt
                )
            try:
                return self._post_json(
                    endpoint,
                    payload,
                    timeout=attempt_timeout,
                )
            except RolloutHostHTTPError as exc:
                if exc.status is not None and exc.status < 500:
                    raise
                if time.monotonic() >= deadline:
                    raise
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"POST {endpoint} retry deadline elapsed"
                )
            time.sleep(min(max(0.0, backoff), remaining))
            backoff = min(
                max(backoff * 2.0, base_backoff),
                max_backoff,
            )

    def _get_json(self, endpoint: str, timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.rollout_master_url}{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=timeout or self.http_timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GET {url} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"GET {url} returned non-object JSON: {data!r}")
        return data

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.rollout_master_url}{endpoint}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self.http_timeout_seconds
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                body = {}
            error = body.get("error") if isinstance(body, dict) else {}
            if not isinstance(error, dict):
                error = {"message": str(error)}
            raise RolloutHostHTTPError(
                str(error.get("message") or f"HTTP {exc.code}"),
                code=error.get("code"),
                status=exc.code,
                details=dict(error.get("details") or {}),
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"POST {url} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"POST {url} returned non-object JSON: {data!r}")
        if not data.get("ok", False):
            error = data.get("error")
            if isinstance(error, dict):
                raise RolloutHostHTTPError(
                    str(error.get("message") or error),
                    code=error.get("code"),
                    details=dict(error.get("details") or {}),
                )
            raise RuntimeError(f"POST {url} returned not-ok: {data}")
        return data

def _get_arg(args, key: str, default: Any = None) -> Any:
    if hasattr(args, key):
        val = getattr(args, key)
        if val is not None:
            return val
    return default

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False

def _current_ray_node_id() -> str | None:

    try:
        import ray

        if hasattr(ray, "is_initialized") and not ray.is_initialized():
            return None
        value = ray.get_runtime_context().get_node_id()
        return str(value) if value is not None and str(value) else None
    except Exception:
        return None
