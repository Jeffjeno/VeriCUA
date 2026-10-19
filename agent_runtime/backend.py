from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from agent_runtime.types import Task

@dataclass
class ProxyConfig:

    host_proxy: str | None = None
    vm_proxy: str | None = None
    no_proxy: str = "127.0.0.1,localhost,::1"

class BackendSession:

    def __init__(self, task: Task, **kwargs: Any) -> None:
        self.task = task
        self._metadata = kwargs
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:

        self._closed = True

    def __enter__(self) -> BackendSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

class BackendProvider(ABC):

    @abstractmethod
    def create_session(self, task: Task, **kwargs: Any) -> BackendSession:
        pass

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def capabilities(self) -> dict[str, Any]:

        return {}

    def task_resources(self) -> dict[str, float]:

        return {}

    def create_snapshot(self, session: BackendSession, name: str) -> str:

        raise NotImplementedError(
            f"{type(self).__name__} does not support create_snapshot(). "
            "Use a pre-built image/qcow2 instead."
        )

class LocalBackendProvider(BackendProvider):

    def create_session(self, task: Task, **kwargs: Any) -> BackendSession:
        return BackendSession(task=task, **kwargs)

    def capabilities(self) -> dict[str, Any]:
        return {"type": "local", "managed": False}

class UploadIncompleteError(RuntimeError):

    def __init__(
        self,
        path: str,
        expected_size: int,
        actual_size: int | None,
        *,
        elapsed: float | None = None,
    ) -> None:
        self.path = path
        self.expected_size = expected_size
        self.actual_size = actual_size
        self.elapsed = elapsed
        size_part = "missing" if actual_size is None else f"{actual_size} bytes"
        elapsed_part = "" if elapsed is None else f" after {elapsed:.1f}s"
        super().__init__(
            f"Upload incomplete: {path} expected={expected_size} bytes "
            f"actual={size_part}{elapsed_part}"
        )

__all__ = [
    "BackendProvider",
    "BackendSession",
    "LocalBackendProvider",
    "ProxyConfig",
    "UploadIncompleteError",
]
