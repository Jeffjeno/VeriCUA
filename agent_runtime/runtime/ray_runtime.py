from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_runtime.logging import get_logger
from agent_runtime.runtime.base import Runtime
from agent_runtime.types import RunTaskResult

if TYPE_CHECKING:
    from agent_runtime.runner import RunSpec

log = get_logger("runtime.ray")

class RayRuntime(Runtime):

    name = "ray"

    def __init__(
        self,
        *,
        address: str | None = None,
        default_task_resources: dict[str, float] | None = None,
        runtime_env: dict[str, Any] | None = None,
        progress: bool = True,
    ) -> None:
        self.address = address
        self.default_task_resources = default_task_resources or {}
        self.runtime_env = runtime_env or {}
        self.progress = progress

    def _ensure_ray(self):
        try:
            import ray
        except ImportError as e:
            raise RuntimeError(
                "RayRuntime requires the optional `ray` extra: "
                "`pip install 'agent_runtime[ray]'`."
            ) from e

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
            if self.address:
                init_kwargs["address"] = self.address
            if self.runtime_env:
                init_kwargs["runtime_env"] = self.runtime_env
            ray.init(**init_kwargs)
        return ray

    def _resolve_task_resources(self, run_spec: RunSpec) -> dict[str, float]:

        if run_spec.task_resources:
            return run_spec.task_resources
        backend_res = run_spec.backend.task_resources()
        if backend_res:
            return backend_res
        return self.default_task_resources

    def run(
        self,
        runs: list[RunSpec],
        global_concurrency: int | None = None,
        store: Any = None,
    ) -> list[RunTaskResult]:
        if not runs:
            return []

        ray = self._ensure_ray()

        @ray.remote
        def _run_task_on_worker(run_spec, task, output_dir):

            from agent_runtime.runtime.local import run_task_attempt
            from agent_runtime.storage.writer import TraceWriter

            max_vm_retries = int(run_spec.metadata.get("max_vm_retries", 2))
            last_exc = None

            writer = None
            if output_dir:
                try:
                    writer = TraceWriter(output_dir, run_id=run_spec.name)
                except Exception:
                    log.exception(
                        "Failed to construct TraceWriter for %s; "
                        "falling back to end-of-task write",
                        task.task_id,
                    )
                    writer = None

            try:
                for attempt in range(1 + max_vm_retries):
                    result = None
                    try:
                        run_spec.backend.setup()
                        result = run_task_attempt(run_spec, task, writer=writer)
                        return result
                    except (TimeoutError, ConnectionError, OSError) as exc:
                        last_exc = exc
                        diag = _collect_host_diagnostics()
                        log.warning(
                            "Task %s VM startup failed (attempt %d/%d): %s | host: %s",
                            task.task_id, attempt + 1, 1 + max_vm_retries,
                            exc, diag,
                        )
                        if attempt < max_vm_retries:
                            import time as _time
                            backoff = 10 * (2 ** attempt)
                            log.info("Retrying task %s in %ds...", task.task_id, backoff)
                            try:
                                run_spec.backend.teardown()
                            except Exception:
                                pass
                            _time.sleep(backoff)
                            continue

                        result = RunTaskResult.failed(task_id=task.task_id, error=exc)
                        if result.evaluation:
                            result.evaluation.error_type = "vm_resource_exhaustion"
                            result.evaluation.reason = f"{exc} | {diag}"
                        return result
                    except Exception as exc:
                        log.error("Remote task %s failed: %s", task.task_id, exc)
                        result = RunTaskResult.failed(task_id=task.task_id, error=exc)
                        return result
                    finally:

                        if (
                            writer is None
                            and result is not None
                            and output_dir
                        ):
                            _write_trace(output_dir, run_spec.name, result)
                        try:
                            run_spec.backend.teardown()
                        except Exception:
                            pass

                result = RunTaskResult.failed(
                    task_id=task.task_id,
                    error=last_exc or RuntimeError("max retries exhausted"),
                )
                return result
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        log.exception(
                            "Failed to close TraceWriter for %s", task.task_id,
                        )

        total_tasks = sum(len(r.tasks) for r in runs)
        log.info(
            "RayRuntime: %d run(s), %d total tasks, address=%s",
            len(runs),
            total_tasks,
            self.address or "<local>",
        )

        if store is not None:
            for run_spec in runs:
                store.record_run(
                    run_id=run_spec.name,
                    run_name=run_spec.name,
                    benchmark=run_spec.tasks[0].benchmark if run_spec.tasks else None,
                    tags=run_spec.tags,
                    total_tasks=len(run_spec.tasks),
                )

        max_in_flight = global_concurrency or total_tasks
        pending: list = []
        all_refs: list = []
        ref_meta: dict = {}

        for run_spec in runs:
            resources = self._resolve_task_resources(run_spec)
            remote_opts: dict[str, Any] = {}
            if resources:
                remote_opts["resources"] = resources

            output_dir = str(Path(run_spec.output_dir).resolve()) if run_spec.output_dir else None

            task_ids = [t.task_id for t in run_spec.tasks]
            if store is not None:
                remaining_ids = set(store.pending_tasks(run_spec.name, task_ids))
                tasks_to_run = [t for t in run_spec.tasks if t.task_id in remaining_ids]
                skipped = len(run_spec.tasks) - len(tasks_to_run)
                if skipped > 0:
                    log.info("Checkpoint resume: skipping %d already-succeeded tasks for %s",
                             skipped, run_spec.name)
            else:
                tasks_to_run = run_spec.tasks

            for task in tasks_to_run:
                attempt_id = uuid.uuid4().hex

                if store is not None:
                    store.record_attempt_start(
                        run_id=run_spec.name,
                        task_id=task.task_id,
                        attempt_id=attempt_id,
                        benchmark=task.benchmark,
                    )

                if len(pending) >= max_in_flight:
                    ready, pending = ray.wait(pending, num_returns=1)
                    all_refs.extend(ready)

                ref = _run_task_on_worker.options(**remote_opts).remote(
                    run_spec, task, output_dir,
                )
                ref_meta[ref] = (attempt_id, run_spec.name, run_spec.tags)
                pending.append(ref)

        all_refs.extend(pending)

        results: list[RunTaskResult] = []
        bar = _maybe_progress_bar(len(all_refs), enabled=self.progress)
        remaining = list(all_refs)

        recent_vm_failures = 0
        total_collected = 0
        concurrency_reduced = False

        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            ref = done[0]
            try:
                result = ray.get(ref)
            except Exception as exc:
                log.error("ray.get() failed for a task: %s", exc)
                task_id = "unknown"
                if ref in ref_meta:
                    task_id = ref_meta[ref][0]
                result = RunTaskResult.failed(task_id=task_id, error=exc)
            results.append(result)
            total_collected += 1

            if store is not None and ref in ref_meta:
                attempt_id, run_id, tags = ref_meta[ref]
                store.record_attempt_result(
                    attempt_id=attempt_id,
                    result=result,
                    run_id=run_id,
                    tags=tags,
                )

            is_vm_failure = (
                result.evaluation
                and result.evaluation.error_type == "vm_resource_exhaustion"
            )
            if is_vm_failure:
                recent_vm_failures += 1
                if (
                    not concurrency_reduced
                    and recent_vm_failures >= 3
                    and recent_vm_failures > total_collected * 0.3
                ):
                    old_max = max_in_flight
                    max_in_flight = max(1, max_in_flight // 2)
                    concurrency_reduced = True
                    log.warning(
                        "Adaptive throttle: %d/%d tasks failed with VM resource "
                        "exhaustion. Reducing max_in_flight %d → %d.",
                        recent_vm_failures, total_collected,
                        old_max, max_in_flight,
                    )

            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        for run_spec in runs:
            if run_spec.output_dir:
                abs_output = str(Path(run_spec.output_dir).resolve())
                _write_summary(abs_output, run_spec.name, results, runs)

        return results

def _collect_host_diagnostics() -> str:

    parts = []
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                mb = int(line.split()[1]) // 1024
                parts.append(f"mem_avail={mb}MB")
                break
    except Exception:
        pass

    try:
        import subprocess
        n = subprocess.check_output(
            "docker ps -q | wc -l", shell=True, timeout=5,
        ).decode().strip()
        parts.append(f"containers={n}")
    except Exception:
        pass

    try:
        load = open("/proc/loadavg").read().split()[0]
        parts.append(f"load={load}")
    except Exception:
        pass

    try:
        inst_max = int(open("/proc/sys/fs/inotify/max_user_instances").read().strip())
        parts.append(f"inotify_inst_max={inst_max}")
    except Exception:
        pass

    import socket
    parts.append(f"host={socket.gethostname()}")

    return " ".join(parts) if parts else "diagnostics unavailable"

def _write_trace(output_dir: str | Path, run_id: str, result: RunTaskResult) -> None:

    try:
        from agent_runtime.storage.writer import TraceWriter

        writer = TraceWriter(output_dir, run_id=run_id)
        if result.trace is not None:
            writer.write(result.trace)
        writer.close()
    except Exception as exc:
        log.warning("Failed to write trace for %s: %s", run_id, exc)

def _write_summary(
    output_dir: str | Path,
    run_id: str,
    results: list[RunTaskResult],
    runs: list,
) -> None:

    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    succeeded = sum(
        1 for r in results if r.evaluation and r.evaluation.success
    )
    failed = sum(
        1 for r in results if r.evaluation and not r.evaluation.success
    )
    import time as _time

    scores = [r.evaluation.score for r in results if r.evaluation and r.evaluation.score is not None]
    error_count = len(results) - succeeded - failed

    summary = {
        "run_id": run_id,
        "total_tasks": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "error_count": error_count,
        "avg_score": sum(scores) / len(scores) if scores else None,
        "success_rate": succeeded / len(results) if results else 0.0,
        "runs": [r.name for r in runs],
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

def _maybe_progress_bar(total: int, *, enabled: bool):
    if not enabled or os.environ.get("AGENT_RUNTIME_NO_PROGRESS"):
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc="tasks (ray)")
    except ImportError:
        return None

__all__ = ["RayRuntime"]
