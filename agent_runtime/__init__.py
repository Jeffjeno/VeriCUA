from __future__ import annotations

from agent_runtime.backend import BackendProvider, BackendSession, LocalBackendProvider, ProxyConfig
from agent_runtime.benchmark import Benchmark
from agent_runtime.env import Env
from agent_runtime.evaluator import Evaluator
from agent_runtime.harness import Harness
from agent_runtime.model_client import ModelCapabilities, ModelClient, ModelResponse
from agent_runtime.registry import (
    list_benchmarks,
    list_harnesses,
    load_benchmark,
    register_benchmark,
    register_harness,
    resolve_harness,
)
from agent_runtime.runner import Runner
from agent_runtime.types import (
    Action,
    ActionType,
    EnvCapabilities,
    EvaluationEvent,
    EvaluationResult,
    HarnessContext,
    JudgeConfig,
    Modality,
    ModalityKind,
    Observation,
    RunSummary,
    RunTaskResult,
    Task,
    Trace,
    TraceEvent,
)

class _BackendsNamespace:

    @staticmethod
    def local(**kwargs):
        return LocalBackendProvider(**kwargs)

    @staticmethod
    def mobile_world(**kwargs):
        from agent_runtime.benchmarks.mobile_world import MobileWorldBackendProvider

        return MobileWorldBackendProvider(**kwargs)

    @staticmethod
    def android_world(**kwargs):
        from agent_runtime.benchmarks.android_world import AndroidWorldBackendProvider

        return AndroidWorldBackendProvider(**kwargs)

    @staticmethod
    def osworld(**kwargs):
        from agent_runtime.benchmarks.osworld import OSWorldBackendProvider

        return OSWorldBackendProvider(**kwargs)

backends = _BackendsNamespace()

class _RuntimesNamespace:

    @staticmethod
    def local(**kwargs):
        from agent_runtime.runtime.local import LocalRuntime

        return LocalRuntime(**kwargs)

    @staticmethod
    def ray(**kwargs):
        from agent_runtime.runtime.ray_runtime import RayRuntime

        return RayRuntime(**kwargs)

runtimes = _RuntimesNamespace()

class _AnalysisNamespace:

    @staticmethod
    def open_store(path=None):

        from agent_runtime.storage.result_store import DuckDBResultStore

        return DuckDBResultStore(path)

analysis = _AnalysisNamespace()

__all__ = [
    "Action",
    "ActionType",
    "BackendProvider",
    "BackendSession",
    "Benchmark",
    "Env",
    "EnvCapabilities",
    "EvaluationEvent",
    "EvaluationResult",
    "Evaluator",
    "Harness",
    "HarnessContext",
    "JudgeConfig",
    "LocalBackendProvider",
    "ProxyConfig",
    "ModelCapabilities",
    "ModelClient",
    "ModelResponse",
    "Modality",
    "ModalityKind",
    "Observation",
    "RunSummary",
    "RunTaskResult",
    "Runner",
    "Task",
    "Trace",
    "TraceEvent",
    "backends",
    "runtimes",
    "analysis",
    "list_benchmarks",
    "list_harnesses",
    "load_benchmark",
    "register_benchmark",
    "register_harness",
    "resolve_harness",
]

__version__ = "0.1.0"
