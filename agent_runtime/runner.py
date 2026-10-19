from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_runtime.backend import BackendProvider, LocalBackendProvider
from agent_runtime.benchmark import Benchmark
from agent_runtime.evaluator import Evaluator
from agent_runtime.harness import Harness
from agent_runtime.runtime.base import Runtime
from agent_runtime.runtime.local import LocalRuntime
from agent_runtime.types import (
    EvaluationResult,
    RunSummary,
    RunTaskResult,
    Task,
)

@dataclass
class RunSpec:

    name: str
    benchmark: Benchmark
    tasks: list[Task]
    harness: Harness | None = None
    harness_factory: Callable[[], Harness] | None = None
    backend: BackendProvider = field(default_factory=LocalBackendProvider)

    evaluator_factory: Callable[[Task], Evaluator] | None = None

    concurrency: int | None = None
    max_retries: int = 0
    task_timeout: float | None = None
    max_steps: int | None = None
    output_dir: str | Path | None = None
    env_options: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_resources: dict[str, float] = field(default_factory=dict)

class RunResult:

    def __init__(
        self,
        results: list[RunTaskResult],
        runs: list[RunSpec],
        store: Any = None,
    ) -> None:
        self.results = results
        self.runs = runs
        self._store = store

    def summary(
        self, by: list[str] | None = None
    ) -> RunSummary | list[dict[str, Any]]:

        if by is None:
            s = RunSummary()
            for r in self.results:
                s.update(r)
            return s

        if self._store is not None:
            run_ids = [r.name for r in self.runs]
            if len(run_ids) == 1:
                return self._store.summary(run_id=run_ids[0], by=by)
            return self._store.summary(by=by)

        return _in_memory_group_by(self.results, self.runs, by)

    def succeeded(self) -> list[RunTaskResult]:
        return [r for r in self.results if r.evaluation and r.evaluation.success]

    def failed(self) -> list[RunTaskResult]:
        return [r for r in self.results if r.evaluation and not r.evaluation.success]

    def __len__(self) -> int:
        return len(self.results)

def _in_memory_group_by(
    results: list[RunTaskResult],
    runs: list[RunSpec],
    by: list[str],
) -> list[dict[str, Any]]:

    from collections import defaultdict

    run_tags: dict[str, dict[str, str]] = {}
    for run_spec in runs:
        for task in run_spec.tasks:
            run_tags[task.task_id] = run_spec.tags

    groups: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "succeeded": 0, "failed": 0, "scores": []}
    )
    for r in results:
        task_id = r.trace.task_id if r.trace else ""
        tags = run_tags.get(task_id, {})
        key = tuple(tags.get(k, "") for k in by)
        g = groups[key]
        g["total"] += 1
        if r.evaluation and r.evaluation.success:
            g["succeeded"] += 1
        else:
            g["failed"] += 1
        if r.evaluation and r.evaluation.score is not None:
            g["scores"].append(r.evaluation.score)

    out = []
    for key, g in sorted(groups.items()):
        row = dict(zip(by, key))
        row["total"] = g["total"]
        row["succeeded"] = g["succeeded"]
        row["failed"] = g["failed"]
        row["avg_score"] = (
            sum(g["scores"]) / len(g["scores"]) if g["scores"] else None
        )
        out.append(row)
    return out

class Runner:

    def __init__(
        self,
        runtime: Runtime | None = None,
        global_concurrency: int | None = None,
        store: Any = None,
    ) -> None:
        self.runtime = runtime or LocalRuntime(
            num_workers=global_concurrency or 1,
        )
        self.global_concurrency = global_concurrency
        self.store = store
        self._runs: list[RunSpec] = []

    def submit(
        self,
        name: str,
        benchmark: Benchmark,
        tasks: list[Task],
        harness: Harness | None = None,
        harness_factory: Callable[[], Harness] | None = None,
        evaluator: Evaluator | Callable[[Task], Evaluator] | None = None,
        backend: BackendProvider | None = None,
        concurrency: int | None = None,
        max_retries: int = 0,
        task_timeout: float | None = None,
        max_steps: int | None = None,
        output_dir: str | Path | None = None,
        env_options: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        task_resources: dict[str, float] | None = None,
    ) -> Runner:

        if harness is None and harness_factory is None:
            raise ValueError("Must provide either harness or harness_factory")

        if evaluator is None:
            evaluator_factory = None
        elif isinstance(evaluator, Evaluator):
            evaluator_factory = lambda task, _ev=evaluator: _ev
        elif callable(evaluator):
            evaluator_factory = evaluator
        else:
            raise TypeError(f"evaluator must be an Evaluator, callable, or None, got {type(evaluator)}")

        self._runs.append(RunSpec(
            name=name,
            benchmark=benchmark,
            tasks=tasks,
            harness=harness,
            harness_factory=harness_factory,
            backend=backend or LocalBackendProvider(),
            evaluator_factory=evaluator_factory,
            concurrency=concurrency,
            max_retries=max_retries,
            task_timeout=task_timeout,
            max_steps=max_steps,
            output_dir=output_dir,
            env_options=env_options or {},
            tags=tags or {},
            metadata=metadata or {},
            task_resources=task_resources or {},
        ))
        return self

    def run(self) -> RunResult:

        store = self.store or self._auto_init_store()
        results = self.runtime.run(
            self._runs,
            global_concurrency=self.global_concurrency,
            store=store,
        )
        if store is not None:
            for run_spec in self._runs:
                store.finish_run(run_spec.name)
        return RunResult(results=results, runs=self._runs, store=store)

    def _auto_init_store(self) -> Any:

        output_dir = next(
            (r.output_dir for r in self._runs if r.output_dir), None,
        )
        if output_dir is None:
            return None
        try:
            from agent_runtime.storage.result_store import DuckDBResultStore

            store_path = Path(output_dir) / "store.duckdb"
            return DuckDBResultStore(store_path)
        except Exception:
            return None

__all__ = ["RunResult", "RunSpec", "Runner"]
