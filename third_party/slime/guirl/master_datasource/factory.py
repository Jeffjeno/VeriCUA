from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

logger = logging.getLogger("guirl.factory")

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

def _get_arg(args: Namespace, key: str, default: Any = None) -> Any:

    if hasattr(args, key):
        val = getattr(args, key)
        if val is not None:
            return val
    return default

def _configuration_error(message: str, **details: Any) -> ValueError:
    return ValueError(
        "GUIRL_V2_CONFIG_VALIDATION_FAILED "
        + json.dumps(
            {"message": message, "details": details},
            sort_keys=True,
            default=str,
        )
    )

def _filesystem_type(path: Path) -> str | None:

    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    resolved = str(path.resolve())
    best: tuple[int, str] | None = None
    for line in mounts:
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        if resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/"):
            candidate = (len(mountpoint), fields[2].lower())
            if best is None or candidate[0] > best[0]:
                best = candidate
    return None if best is None else best[1]

def _ensure_local_writable_directory(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise _configuration_error(f"{label} must be a directory", path=path)
    filesystem = _filesystem_type(path)
    if filesystem in _REMOTE_FILESYSTEMS:
        raise _configuration_error(
            f"{label} must be on local storage",
            path=path,
            filesystem=filesystem,
        )
    try:
        fd, probe = tempfile.mkstemp(prefix=".guirl-write-probe.", dir=str(path))
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        raise _configuration_error(
            f"{label} is not writable",
            path=path,
            error=str(exc),
        ) from exc

def _ensure_writable_artifact_directory(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise _configuration_error("artifact_dir must be a directory", path=path)
    try:
        fd, probe = tempfile.mkstemp(prefix=".guirl-artifact-probe.", dir=str(path))
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        raise _configuration_error(
            "artifact_dir is not writable",
            path=path,
            error=str(exc),
        ) from exc

def _validate_v2_paths(
    state_db: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    if str(state_db) == ":memory:":
        raise _configuration_error(
            "Rollout Host v2 requires a persistent state database",
            state_db=state_db,
        )
    database = Path(state_db).expanduser().resolve()
    artifacts = Path(artifact_dir).expanduser().resolve()
    if database.exists() and database.is_dir():
        raise _configuration_error("state_db must be a file path", state_db=database)
    if artifacts.exists() and not artifacts.is_dir():
        raise _configuration_error(
            "artifact_dir must be a directory path",
            artifact_dir=artifacts,
        )
    if database == artifacts or artifacts in database.parents or database in artifacts.parents:
        raise _configuration_error(
            "state_db and artifact_dir must be distinct sibling paths",
            state_db=database,
            artifact_dir=artifacts,
        )
    _ensure_local_writable_directory(database.parent, "state_db parent")
    _ensure_writable_artifact_directory(artifacts)
    return database, artifacts

def _normalize_actor_concurrency(
    value: Any,
    *,
    actor_ids: list[str],
    env_capacity: int,
) -> dict[str, int]:
    fair_share = max(1, math.ceil(env_capacity / len(actor_ids)))
    if value is None or value == "":
        return {actor_id: fair_share for actor_id in actor_ids}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = int(value)
            except ValueError as exc:
                raise _configuration_error(
                    "actor_concurrency must be a positive integer or JSON object",
                    actor_concurrency=value,
                ) from exc
    if isinstance(value, dict):
        unknown = sorted(set(map(str, value)) - set(actor_ids))
        if unknown:
            raise _configuration_error(
                "actor_concurrency contains actors outside the fixed manifest",
                unknown_actors=unknown,
                actor_order=actor_ids,
            )
        normalized = {}
        for actor_id in actor_ids:
            raw_limit = value.get(actor_id, fair_share)
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError) as exc:
                raise _configuration_error(
                    "actor concurrency limits must be positive integers",
                    actor_id=actor_id,
                    value=raw_limit,
                ) from exc
            if limit <= 0:
                raise _configuration_error(
                    "actor concurrency limits must be positive",
                    actor_id=actor_id,
                    value=limit,
                )
            normalized[actor_id] = limit
        return normalized
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise _configuration_error(
            "actor_concurrency must be a positive integer or JSON object",
            actor_concurrency=value,
        ) from exc
    if limit <= 0:
        raise _configuration_error(
            "actor_concurrency must be positive",
            actor_concurrency=limit,
        )
    return {actor_id: limit for actor_id in actor_ids}

def _build_actor_clients(args: Namespace, env_client) -> dict[str, Any]:

    from ..env_service.actor_client import ActorClient
    from ..env_service.harness_factory import create_guirl_harness

    default_max_steps = int(_get_arg(args, "guirl_actor_max_steps", 15))

    actors_cfg = _get_arg(args, "guirl_actors", None)
    if isinstance(actors_cfg, str):
        actors_cfg = json.loads(actors_cfg)

    actor_clients: dict[str, Any] = {}
    if actors_cfg:
        if not isinstance(actors_cfg, list):
            raise ValueError(f"guirl_actors must be a list, got {type(actors_cfg)}")
        for index, entry in enumerate(actors_cfg):
            if not isinstance(entry, dict):
                raise ValueError(f"guirl_actors entries must be dicts, got {entry!r}")
            raw_harness = entry.get("harness") or entry.get("name")
            raw_name = entry.get("name") or raw_harness
            if not isinstance(raw_harness, str) or not raw_harness.strip():
                raise ValueError(
                    f"guirl_actors[{index}] requires a non-empty harness"
                )
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"guirl_actors[{index}] requires a non-empty name")
            harness_type = raw_harness.strip()
            name = raw_name.strip()
            if name in actor_clients:
                raise ValueError(f"duplicate actor name in guirl_actors: {name!r}")
            harness_kwargs = dict(entry.get("harness_kwargs") or {})

            def harness_factory(
                _harness_type=harness_type,
                _harness_kwargs=dict(harness_kwargs),
            ):
                return create_guirl_harness(
                    _harness_type, dict(_harness_kwargs)
                )

            actor_clients[name] = ActorClient(
                env_client=env_client,
                harness_factory=harness_factory,
                max_steps=int(entry.get("max_steps", default_max_steps)),
            )
        return actor_clients

    harness_type = str(_get_arg(args, "guirl_actor_harness", "holo31"))
    harness_kwargs = _get_arg(args, "guirl_actor_harness_kwargs", {}) or {}
    if isinstance(harness_kwargs, str):
        harness_kwargs = json.loads(harness_kwargs)

    def harness_factory(
        _harness_type=harness_type,
        _harness_kwargs=dict(harness_kwargs),
    ):
        return create_guirl_harness(_harness_type, dict(_harness_kwargs))

    actor_clients[harness_type] = ActorClient(
        env_client=env_client,
        harness_factory=harness_factory,
        max_steps=default_max_steps,
    )
    return actor_clients

def build_master_client(args: Namespace, env_restart_fn=None, env_health_probe_fn=None):

    from ..env_service.rpc_env_client import AsyncRpcEnvClient
    from ..env_service.master_client import MasterClient
    from ..env_service.task_sampler import OSWorldTaskSampler

    env_master_url = str(_get_arg(args, "guirl_env_master_url", "http://127.0.0.1:9900"))
    env_client = AsyncRpcEnvClient(base_url=env_master_url)

    manifest_path = str(_get_arg(args, "guirl_manifest_path", "evaluation_examples/sft_tasks.json"))
    examples_dir = str(_get_arg(args, "guirl_examples_dir", "evaluation_examples/examples"))
    task_sampler = OSWorldTaskSampler(
        examples_dir=examples_dir,
        manifest_path=manifest_path,
        cycle=True,
        shuffle=True,
    )

    actor_clients = _build_actor_clients(args, env_client)

    batch_size = int(_get_arg(args, "guirl_batch_size", 8))

    env_capacity = int(_get_arg(args, "guirl_env_capacity", batch_size))
    agent_timeout = float(_get_arg(args, "guirl_actor_timeout_seconds", 600.0))

    pull_first_item_timeout = _get_arg(args, "guirl_pull_first_item_timeout_seconds", None)
    if pull_first_item_timeout is not None:
        pull_first_item_timeout = float(pull_first_item_timeout)
    env_restart_threshold = float(_get_arg(args, "env_restart_threshold_seconds", 300.0))
    env_restart_cooldown = float(_get_arg(args, "env_restart_cooldown_seconds", 300.0))
    env_health_probe_interval = float(
        _get_arg(args, "env_restart_health_probe_interval_seconds", 60.0)
    )
    master_client = MasterClient(
        env_client=env_client,
        task_sampler=task_sampler,
        actor_clients=actor_clients,
        batch_size=batch_size,
        env_capacity=env_capacity,
        agent_timeout_seconds=agent_timeout,
        pull_first_item_timeout_seconds=pull_first_item_timeout,
        env_restart_fn=env_restart_fn,
        env_restart_threshold_seconds=env_restart_threshold,
        env_restart_cooldown_seconds=env_restart_cooldown,
        env_health_probe_fn=env_health_probe_fn,
        env_health_probe_interval_seconds=env_health_probe_interval,
    )

    logger.info(
        "Built MasterClient: env=%s, actors=%s, tasks=%s, batch=%d, env_capacity=%d",
        env_master_url, list(actor_clients.keys()),
        manifest_path, batch_size, env_capacity,
    )
    return master_client

def build_rollout_host_v2(
    args: Namespace,
    *,
    state_db: str | None = None,
    artifact_dir: str | None = None,
):

    from ..env_service.artifact_store import ArtifactStore
    from ..env_service.rollout_host_v2 import RolloutHostV2, RolloutHostV2Config
    from ..env_service.rpc_env_client import AsyncRpcEnvClient
    from ..env_service.state_store import SQLiteStateStore
    from ..env_service.task_sampler import OSWorldTaskSampler

    database = state_db or _get_arg(args, "guirl_state_db")
    artifacts_path = artifact_dir or _get_arg(args, "guirl_artifact_dir")
    if not database or not artifacts_path:
        raise _configuration_error(
            "Rollout Host v2 requires explicit state_db and artifact_dir",
            state_db=database,
            artifact_dir=artifacts_path,
        )
    database_path, artifacts = _validate_v2_paths(database, artifacts_path)

    env_capacity = int(_get_arg(args, "guirl_env_capacity", 1))
    if env_capacity <= 0:
        raise _configuration_error(
            "env_capacity must be positive",
            env_capacity=env_capacity,
        )

    env_client = AsyncRpcEnvClient(
        base_url=str(
            _get_arg(args, "guirl_env_master_url", "http://127.0.0.1:9900")
        )
    )
    sampler = OSWorldTaskSampler(
        examples_dir=str(
            _get_arg(args, "guirl_examples_dir", "evaluation_examples/examples")
        ),
        manifest_path=str(
            _get_arg(
                args,
                "guirl_manifest_path",
                "evaluation_examples/sft_tasks.json",
            )
        ),
        cycle=True,
        shuffle=True,
    )
    tasks = getattr(sampler, "tasks", None)
    if tasks is not None and not tasks:
        raise _configuration_error(
            "task manifest must contain at least one task",
            manifest_path=_get_arg(args, "guirl_manifest_path"),
        )
    actors = _build_actor_clients(args, env_client)
    actor_ids = list(actors)
    if not actor_ids or any(not actor_id.strip() for actor_id in actor_ids):
        raise _configuration_error(
            "actor manifest must contain non-empty actor names",
            actor_order=actor_ids,
        )
    if len(set(actor_ids)) != len(actor_ids):
        raise _configuration_error(
            "actor manifest names must be unique",
            actor_order=actor_ids,
        )
    concurrency = _normalize_actor_concurrency(
        _get_arg(args, "guirl_actor_concurrency", None),
        actor_ids=actor_ids,
        env_capacity=env_capacity,
    )
    config = RolloutHostV2Config(
        env_capacity=env_capacity,
        actor_concurrency=concurrency,
        max_attempts=int(_get_arg(args, "guirl_v2_max_attempts", 2)),
        infra_retry_budget=int(
            _get_arg(args, "guirl_v2_infra_retry_budget", 20)
        ),
        max_task_replacements=int(
            _get_arg(args, "guirl_v2_max_task_replacements", 2)
        ),
        failure_policy=str(
            _get_arg(args, "guirl_v2_failure_policy", "replace_task_group")
        ),
        claim_ttl_seconds=float(
            _get_arg(args, "guirl_v2_claim_ttl_seconds", 600.0)
        ),
        close_retry_seconds=float(
            _get_arg(args, "guirl_v2_close_retry_seconds", 1.0)
        ),
        actor_timeout_seconds=float(
            _get_arg(args, "guirl_actor_timeout_seconds", 1200.0)
        ),
        batch_deadline_seconds=float(
            _get_arg(args, "guirl_v2_batch_deadline_seconds", 6 * 60 * 60)
        ),
    )
    return RolloutHostV2(
        store=SQLiteStateStore(
            str(database_path),
            synchronous=str(_get_arg(args, "guirl_v2_sqlite_synchronous", "FULL")),
            busy_timeout_ms=int(
                _get_arg(args, "guirl_v2_sqlite_busy_timeout_ms", 5_000)
            ),
        ),
        artifacts=ArtifactStore(str(artifacts)),
        task_sampler=sampler,
        env_client=env_client,
        actor_clients=actors,
        config=config,
    )
