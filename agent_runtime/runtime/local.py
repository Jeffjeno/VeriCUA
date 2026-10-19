from __future__ import annotations

import contextlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from agent_runtime.runtime.base import Runtime
from agent_runtime.types import RunTaskResult

if TYPE_CHECKING:
    from agent_runtime.runner import RunSpec

def run_task_attempt(
    run: RunSpec,
    task: Any,
    writer: Any | None = None,
) -> RunTaskResult:

    session = None
    env = None
    harness = run.harness_factory() if run.harness_factory else run.harness

    live_attached = False
    began_task = False

    try:
        session = run.backend.create_session(task)
        env = run.benchmark.make_env(task=task, session=session, **run.env_options)

        if writer is not None:
            try:
                writer.begin_task(task.task_id, task=task)
                began_task = True
            except Exception:
                import logging as _lg
                _lg.getLogger(__name__).exception(
                    "TraceWriter.begin_task failed for %s (continuing without live writes)",
                    task.task_id,
                )
            if began_task:
                def _on_event(ev, _w=writer, _tid=task.task_id):
                    _w.record_event(_tid, ev)
                env.trace.add_event_listener(_on_event)
                live_attached = True

        env.reset()

        evaluator = run.evaluator_factory(task) if run.evaluator_factory else None
        result = env.run(
            harness=harness,
            evaluator=evaluator,
            max_steps=run.max_steps,
            timeout=run.task_timeout,
        )

        if writer is not None and began_task:
            try:
                writer.end_task(
                    task.task_id,
                    result.trace if result.trace is not None else env.trace,
                    result.evaluation,
                )
                began_task = False
            except Exception:
                import logging as _lg
                _lg.getLogger(__name__).exception(
                    "TraceWriter.end_task failed for %s", task.task_id,
                )

        return result

    except Exception as error:
        import logging
        logging.getLogger(__name__).error(
            "Task %s failed: %s", task.task_id, error, exc_info=True,
        )
        failed_result = RunTaskResult.failed(task_id=task.task_id, error=error)
        if writer is not None and began_task:
            try:
                writer.end_task(
                    task.task_id,
                    failed_result.trace,
                    failed_result.evaluation,
                )
            except Exception:
                import logging as _lg
                _lg.getLogger(__name__).exception(
                    "TraceWriter.end_task on failure path failed for %s",
                    task.task_id,
                )
        return failed_result

    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
        if hasattr(harness, "reset") and callable(harness.reset):
            with contextlib.suppress(Exception):
                harness.reset()

        if live_attached and env is not None:
            with contextlib.suppress(Exception):
                env.trace.remove_event_listener(_on_event)

def _run_single_run_spec(
    run_spec: RunSpec,
    writer: Any,
    progress_callback: Any | None,
) -> list[RunTaskResult]:

    concurrency = run_spec.concurrency or 1
    if concurrency <= 1:
        return _run_single_run_spec_serial(run_spec, writer, progress_callback)
    return _run_single_run_spec_concurrent(
        run_spec, writer, progress_callback, concurrency,
    )

def _run_single_run_spec_serial(
    run_spec: RunSpec,
    writer: Any,
    progress_callback: Any | None,
) -> list[RunTaskResult]:

    results: list[RunTaskResult] = []
    for task in run_spec.tasks:
        result = run_task_attempt(run_spec, task, writer=writer)
        results.append(result)
        if progress_callback is not None:
            progress_callback()
    return results

def _run_single_run_spec_concurrent(
    run_spec: RunSpec,
    writer: Any,
    progress_callback: Any | None,
    concurrency: int,
) -> list[RunTaskResult]:

    import threading

    lock = threading.Lock()

    def safe_progress() -> None:
        if progress_callback is not None:
            with lock:
                progress_callback()

    def task_worker(task: Any) -> RunTaskResult:
        result = run_task_attempt(run_spec, task, writer=writer)
        safe_progress()
        return result

    results: list[RunTaskResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(task_worker, t): t for t in run_spec.tasks
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

class LocalRuntime(Runtime):

    name = "local"

    def __init__(
        self,
        *,
        num_workers: int = 1,
        progress: bool = True,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.progress = progress

    def run(
        self,
        runs: list[RunSpec],
        global_concurrency: int | None = None,
        store: Any = None,
    ) -> list[RunTaskResult]:
        if not runs:
            return []

        for r in runs:
            r.backend.setup()

        if store is not None:
            for run_spec in runs:
                store.record_run(
                    run_id=run_spec.name,
                    run_name=run_spec.name,
                    benchmark=run_spec.tasks[0].benchmark if run_spec.tasks else None,
                    tags=run_spec.tags,
                    total_tasks=len(run_spec.tasks),
                )

        total_tasks = sum(len(r.tasks) for r in runs)
        max_workers = global_concurrency or self.num_workers
        writer = None

        output_dir = runs[0].output_dir if runs else None
        if output_dir:
            from agent_runtime.storage.writer import TraceWriter

            run_id = runs[0].name or uuid.uuid4().hex[:12]
            writer = TraceWriter(
                output_dir,
                run_id=run_id,
                run_config={"runs": [r.name for r in runs]},
            )

        bar = _maybe_progress_bar(total_tasks, enabled=self.progress)
        progress_callback = bar.update if bar is not None else None

        try:
            if len(runs) <= 1 or max_workers <= 1:
                all_results = self._run_serial(runs, writer, progress_callback)
            else:
                all_results = self._run_parallel(runs, writer, progress_callback, max_workers)
        finally:
            if writer is not None:
                writer.close()
            if bar is not None:
                bar.close()
            for r in runs:
                with contextlib.suppress(Exception):
                    r.backend.teardown()

        if store is not None:
            idx = 0
            for run_spec in runs:
                n = len(run_spec.tasks)
                for result in all_results[idx : idx + n]:
                    attempt_id = uuid.uuid4().hex
                    store.record_attempt_result(
                        attempt_id=attempt_id,
                        result=result,
                        run_id=run_spec.name,
                        tags=run_spec.tags,
                    )
                idx += n

        return all_results

    def _run_serial(
        self,
        runs: list[RunSpec],
        writer: Any,
        progress_callback: Any | None,
    ) -> list[RunTaskResult]:

        all_results: list[RunTaskResult] = []
        for run_spec in runs:
            results = _run_single_run_spec(run_spec, writer, progress_callback)
            all_results.extend(results)
        return all_results

    def _run_parallel(
        self,
        runs: list[RunSpec],
        writer: Any,
        progress_callback: Any | None,
        max_workers: int,
    ) -> list[RunTaskResult]:

        import threading

        lock = threading.Lock()

        def safe_progress():
            if progress_callback is not None:
                with lock:
                    progress_callback()

        def run_spec_worker(run_spec: RunSpec) -> list[RunTaskResult]:
            results: list[RunTaskResult] = []
            for task in run_spec.tasks:
                result = run_task_attempt(run_spec, task, writer=writer)
                results.append(result)
                safe_progress()
            return results

        all_results: list[RunTaskResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(run_spec_worker, run_spec): i
                for i, run_spec in enumerate(runs)
            }
            indexed: dict[int, list[RunTaskResult]] = {}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    indexed[idx] = fut.result()
                except Exception as error:
                    run_spec = runs[idx]
                    indexed[idx] = [
                        RunTaskResult.failed(task_id=t.task_id, error=error)
                        for t in run_spec.tasks
                    ]

        for i in range(len(runs)):
            all_results.extend(indexed.get(i, []))

        return all_results

def _maybe_progress_bar(total: int, *, enabled: bool):
    if not enabled or os.environ.get("AGENT_RUNTIME_NO_PROGRESS"):
        return None
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="tasks")
    except ImportError:
        return None

__all__ = ["LocalRuntime", "run_task_attempt"]
