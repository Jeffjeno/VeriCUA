from __future__ import annotations

import importlib
from typing import Any

from .artifact_store import ArtifactStore
from .batch_models import (
    BatchId,
    BatchState,
    ManifestSlot,
    ProtocolError,
    SlotKey,
    SlotState,
)
from .master_client import BatchSet, MasterClient, TaskRecord
from .state_store import SQLiteStateStore
from .task_sampler import OSWorldTaskSampler

__all__ = [
    "AsyncRpcEnvClient",
    "ActorClient",
    "ArtifactStore",
    "BatchId",
    "BatchSet",
    "BatchState",
    "ManifestSlot",
    "MasterClient",
    "OSWorldTaskSampler",
    "ProtocolError",
    "RolloutHostV2",
    "RolloutHostV2Config",
    "SQLiteStateStore",
    "SlotKey",
    "SlotState",
    "TaskRecord",
]

_LAZY: dict[str, tuple[str, str]] = {
    "AsyncRpcEnvClient": (".rpc_env_client", "AsyncRpcEnvClient"),
    "ActorClient": (".actor_client", "ActorClient"),
    "RolloutHostV2": (".rollout_host_v2", "RolloutHostV2"),
    "RolloutHostV2Config": (".rollout_host_v2", "RolloutHostV2Config"),
}

def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
