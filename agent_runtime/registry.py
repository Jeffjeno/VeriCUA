from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")

@dataclass
class _Entry(Generic[T]):
    name: str
    factory: Callable[..., T] | None = None
    module_path: str | None = None
    class_name: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Callable[..., T]:
        if self.factory is not None:
            return self.factory
        if self.module_path is None or self.class_name is None:
            raise RuntimeError(f"Registry entry '{self.name}' has no factory or import path.")
        module = importlib.import_module(self.module_path)
        cls = getattr(module, self.class_name)
        self.factory = cls
        return cls

class _Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, _Entry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T] | None = None,
        *,
        module_path: str | None = None,
        class_name: str | None = None,
        description: str = "",
        tags: Iterable[str] = (),
        overwrite: bool = False,
        **metadata: Any,
    ) -> Callable[..., T] | None:
        if name in self._entries and not overwrite:
            raise ValueError(
                f"{self.kind} '{name}' is already registered. "
                f"Pass overwrite=True to replace it."
            )
        self._entries[name] = _Entry(
            name=name,
            factory=factory,
            module_path=module_path,
            class_name=class_name,
            description=description,
            tags=tuple(tags),
            metadata=metadata,
        )
        return factory

    def decorator(
        self,
        name: str,
        *,
        description: str = "",
        tags: Iterable[str] = (),
        overwrite: bool = False,
        **metadata: Any,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def _wrap(cls: Callable[..., T]) -> Callable[..., T]:
            self.register(
                name,
                factory=cls,
                description=description,
                tags=tags,
                overwrite=overwrite,
                **metadata,
            )
            return cls

        return _wrap

    def resolve(self, name: str) -> Callable[..., T]:
        try:
            entry = self._entries[name]
        except KeyError as e:
            raise KeyError(
                f"No {self.kind} registered under name '{name}'. "
                f"Available: {sorted(self._entries)}"
            ) from e
        return entry.resolve()

    def get_entry(self, name: str) -> _Entry[T]:
        return self._entries[name]

    def list(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> list[tuple[str, _Entry[T]]]:
        return sorted(self._entries.items())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

_benchmarks: _Registry[Any] = _Registry("benchmark")
_harnesses: _Registry[Any] = _Registry("harness")

def register_benchmark(
    name: str,
    *,
    description: str = "",
    tags: Iterable[str] = (),
    overwrite: bool = False,
    **metadata: Any,
):

    return _benchmarks.decorator(
        name, description=description, tags=tags, overwrite=overwrite, **metadata
    )

def register_benchmark_lazy(
    name: str, module_path: str, class_name: str, **kwargs: Any
) -> None:
    _benchmarks.register(name, module_path=module_path, class_name=class_name, **kwargs)

def load_benchmark(name: str, **kwargs: Any) -> Any:

    cls = _benchmarks.resolve(name)
    return cls(**kwargs)

def list_benchmarks() -> list[str]:
    return _benchmarks.list()

def benchmark_entries() -> list[tuple[str, _Entry[Any]]]:
    return _benchmarks.items()

def register_harness(
    name: str,
    *,
    description: str = "",
    tags: Iterable[str] = (),
    overwrite: bool = False,
    **metadata: Any,
):

    return _harnesses.decorator(
        name, description=description, tags=tags, overwrite=overwrite, **metadata
    )

def register_harness_lazy(
    name: str, module_path: str, class_name: str, **kwargs: Any
) -> None:
    _harnesses.register(name, module_path=module_path, class_name=class_name, **kwargs)

def resolve_harness(name: str) -> Any:
    return _harnesses.resolve(name)

def list_harnesses() -> list[str]:
    return _harnesses.list()

def harness_entries() -> list[tuple[str, _Entry[Any]]]:
    return _harnesses.items()

_BUILTIN_BENCHMARKS: list[tuple[str, str, str, str]] = [
    ("mobile_world", "agent_runtime.benchmarks.mobile_world", "MobileWorldBenchmark", "MobileWorld Android GUI benchmark (HTTP server adapter)."),
    ("android_world", "agent_runtime.benchmarks.android_world", "AndroidWorldBenchmark", "AndroidWorld Android GUI benchmark (Docker server adapter)."),
    ("osworld", "agent_runtime.benchmarks.osworld", "OSWorldBenchmark", "OSWorld desktop GUI benchmark (QEMU/VMware/Docker VM adapter)."),
]

for _name, _mod, _cls, _desc in _BUILTIN_BENCHMARKS:
    register_benchmark_lazy(_name, _mod, _cls, description=_desc)

_BUILTIN_HARNESSES: list[tuple[str, str, str, str]] = [
    (
        "claude-cua",
        "agent_runtime.harnesses.claude_cua",
        "ClaudeCUAHarness",
        "Claude beta Messages API computer-use harness for OSWorld/OpenComputer-style desktops.",
    ),
    (
        "gpt54-cua",
        "agent_runtime.harnesses.gpt54_cua",
        "GPT54CUAHarness",
        "GPT-5.4 Responses API computer-use harness for OSWorld/OpenComputer-style desktops.",
    ),
    (
        "holo3",
        "agent_runtime.harnesses.holo3",
        "Holo3Harness",
        "Holo3 chat-completions structured-output GUI harness for OSWorld/OpenComputer-style desktops.",
    ),
    (
        "qwen35vl",
        "agent_runtime.harnesses.qwen35vl",
        "Qwen35VLHarness",
        "Qwen3.5-VL XML tool-call GUI harness for OSWorld-style desktops.",
    ),
    (
        "qwen3vl",
        "agent_runtime.harnesses.qwen3vl",
        "Qwen3VLHarness",
        "Qwen3-VL JSON tool-call GUI harness for OSWorld-style desktops.",
    ),
    (
        "qwen3vl-probe",
        "agent_runtime.harnesses.qwen3vl_probe",
        "Qwen3VLProbeHarness",
        "Qwen3-VL probe agent for GUIRL criteria-based evaluation.",
    ),
    (
        "holo31-probe",
        "agent_runtime.harnesses.holo31_probe",
        "Holo31ProbeHarness",
        "Holo3.1 native function-calling probe agent for GUIRL criteria-based evaluation.",
    ),
    (
        "evocua",
        "agent_runtime.harnesses.evocua",
        "EvoCUAHarness",
        "EvoCUA native GUI agent harness for OSWorld-style desktops.",
    ),
    (
        "owl15",
        "agent_runtime.harnesses.owl15",
        "Owl15Harness",
        "Owl1.5 GUI agent harness for OSWorld-style desktops.",
    ),
]

for _name, _mod, _cls, _desc in _BUILTIN_HARNESSES:
    register_harness_lazy(_name, _mod, _cls, description=_desc)

__all__ = [
    "benchmark_entries",
    "harness_entries",
    "list_benchmarks",
    "list_harnesses",
    "load_benchmark",
    "register_benchmark",
    "register_benchmark_lazy",
    "register_harness",
    "register_harness_lazy",
    "resolve_harness",
]
