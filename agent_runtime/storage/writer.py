from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import IO, Any

from agent_runtime.types import (
    EvaluationResult,
    ModalityKind,
    RunSummary,
    Task,
    Trace,
    TraceEvent,
)

logger = logging.getLogger(__name__)

def extract_screenshot_from_event(
    event: TraceEvent, task_dir: Path, *, step_index: int | None = None
) -> int:

    if event.observation is None:
        return 0
    idx = event.step_index if step_index is None else step_index
    art_dir = task_dir / "artifacts" / "screenshots"
    new_parts: list | None = None
    count = 0
    for j, part in enumerate(event.observation.parts):
        if (
            part.kind is not ModalityKind.IMAGE
            or not part.content
            or not isinstance(part.content, str)
        ):
            continue
        try:
            png_bytes = base64.b64decode(part.content)
        except Exception:
            continue
        art_dir.mkdir(parents=True, exist_ok=True)
        fname = f"step_{idx:03d}.png"
        (art_dir / fname).write_bytes(png_bytes)
        uri = f"artifacts/screenshots/{fname}"
        if new_parts is None:
            new_parts = list(event.observation.parts)
        new_parts[j] = part.model_copy(update={"content": None, "uri": uri})
        count += 1
    if new_parts is not None:
        event.observation = event.observation.model_copy(update={"parts": new_parts})
    return count

def extract_a11y_tree_from_event(
    event: TraceEvent, task_dir: Path, *, step_index: int | None = None
) -> int:

    if event.observation is None:
        return 0
    idx = event.step_index if step_index is None else step_index
    art_dir = task_dir / "artifacts" / "a11y_trees"
    new_parts: list | None = None
    count = 0
    for j, part in enumerate(event.observation.parts):
        if part.kind is not ModalityKind.A11Y_TREE or not part.content:
            continue
        art_dir.mkdir(parents=True, exist_ok=True)
        fname = f"step_{idx:03d}.txt"
        content_str = str(part.content)
        (art_dir / fname).write_text(content_str, encoding="utf-8")
        uri = f"artifacts/a11y_trees/{fname}"
        if new_parts is None:
            new_parts = list(event.observation.parts)
        new_parts[j] = part.model_copy(update={"content": None, "uri": uri})
        count += 1
    if new_parts is not None:
        event.observation = event.observation.model_copy(update={"parts": new_parts})
    return count

def extract_screenshots(trace: Trace, task_dir: Path) -> int:

    total = 0
    for i, ev in enumerate(trace.events):
        total += extract_screenshot_from_event(ev, task_dir, step_index=i)
    return total

def extract_a11y_trees(trace: Trace, task_dir: Path) -> int:

    total = 0
    for i, ev in enumerate(trace.events):
        total += extract_a11y_tree_from_event(ev, task_dir, step_index=i)
    return total

class _TaskState:

    __slots__ = ("events_fh", "lock", "started_at", "step_count", "task_dir", "task_id")

    def __init__(self, task_id: str, task_dir: Path, events_fh: IO[str]) -> None:
        self.task_id = task_id
        self.task_dir = task_dir
        self.events_fh = events_fh
        self.started_at = time.time()
        self.lock = threading.Lock()
        self.step_count = 0

class TraceWriter:

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        run_config: dict[str, Any] | None = None,
        extract_artifacts: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.extract_artifacts = extract_artifacts
        self._summary = RunSummary(run_id=run_id)
        self._started_at = time.time()
        self._task_count = 0
        self._tasks: dict[str, _TaskState] = {}
        self._summary_lock = threading.Lock()

        manifest = {
            "run_id": run_id,
            "started_at": self._started_at,
            "config": run_config or {},
        }
        _atomic_write(self.output_dir / "manifest.json", json.dumps(manifest, indent=2))

    def begin_task(self, task_id: str, *, task: Task | None = None) -> Path:

        resolved = task_id or f"task_{self._task_count:04d}"
        task_dir = self.output_dir / "tasks" / resolved
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "artifacts").mkdir(exist_ok=True)

        existing = self._tasks.pop(resolved, None)
        if existing is not None:
            with contextlib.suppress(Exception):
                existing.events_fh.close()

        events_path = task_dir / "events.jsonl"
        fh = open(events_path, "w", encoding="utf-8")
        state = _TaskState(resolved, task_dir, fh)
        self._tasks[resolved] = state

        if task is not None:
            _atomic_write(
                task_dir / "task.json",
                task.model_dump_json(indent=2),
            )

        _atomic_write(
            task_dir / "status.json",
            json.dumps(
                {
                    "task_id": resolved,
                    "state": "running",
                    "started_at": state.started_at,
                    "last_step": -1,
                    "last_ts": state.started_at,
                },
                indent=2,
            ),
        )
        return task_dir

    def record_event(self, task_id: str, event: TraceEvent) -> None:

        state = self._tasks.get(task_id)
        if state is None:
            state = _TaskState(
                task_id,
                self.output_dir / "tasks" / task_id,
                None,
            )
            state.task_dir.mkdir(parents=True, exist_ok=True)
            (state.task_dir / "artifacts").mkdir(exist_ok=True)
            state.events_fh = open(
                state.task_dir / "events.jsonl", "a", encoding="utf-8"
            )
            self._tasks[task_id] = state

        with state.lock:
            if self.extract_artifacts:
                extract_screenshot_from_event(event, state.task_dir)
                extract_a11y_tree_from_event(event, state.task_dir)

            line = event.model_dump_json()
            try:
                state.events_fh.write(line)
                state.events_fh.write("\n")
                state.events_fh.flush()
                os.fsync(state.events_fh.fileno())
            except Exception:
                logger.exception(
                    "Failed to append event to events.jsonl for task %s", task_id,
                )

            state.step_count += 1

            _atomic_write(
                state.task_dir / "status.json",
                json.dumps(
                    {
                        "task_id": task_id,
                        "state": "running",
                        "started_at": state.started_at,
                        "last_step": event.step_index,
                        "last_ts": event.timestamp,
                    },
                    indent=2,
                ),
            )

    def end_task(
        self,
        task_id: str,
        trace: Trace,
        evaluation: EvaluationResult | None = None,
    ) -> None:

        state = self._tasks.pop(task_id, None)
        if state is None:

            state = _TaskState(
                task_id,
                self.output_dir / "tasks" / task_id,
                None,
            )
            state.task_dir.mkdir(parents=True, exist_ok=True)

        if state.events_fh is not None:
            with contextlib.suppress(Exception):
                state.events_fh.flush()
                os.fsync(state.events_fh.fileno())
                state.events_fh.close()

        if self.extract_artifacts:
            extract_screenshots(trace, state.task_dir)
            extract_a11y_trees(trace, state.task_dir)

        eval_obj = evaluation if evaluation is not None else trace.evaluation
        if eval_obj is not None and trace.evaluation is None:
            trace.evaluation = eval_obj

        _atomic_write(state.task_dir / "trace.json", trace.model_dump_json(indent=2))

        if eval_obj is not None:
            eval_data = {
                "task_id": trace.task_id,
                "benchmark": trace.benchmark,
                **eval_obj.model_dump(),
            }
            _atomic_write(
                state.task_dir / "evaluation.json",
                json.dumps(eval_data, indent=2, ensure_ascii=False),
            )

        ended_at = time.time()
        _atomic_write(
            state.task_dir / "status.json",
            json.dumps(
                {
                    "task_id": task_id,
                    "state": "done",
                    "started_at": state.started_at,
                    "ended_at": ended_at,
                    "last_step": (
                        trace.events[-1].step_index if trace.events else -1
                    ),
                    "last_ts": (
                        trace.events[-1].timestamp if trace.events else ended_at
                    ),
                    "num_events": len(trace.events),
                    "success": (
                        eval_obj.success if eval_obj is not None else None
                    ),
                },
                indent=2,
            ),
        )

        from agent_runtime.types import RunTaskResult

        with self._summary_lock:
            self._summary.update(RunTaskResult(trace=trace, evaluation=eval_obj))
            self._task_count += 1

    def write(self, trace: Trace) -> None:

        task_id = trace.task_id or f"task_{self._task_count:04d}"
        self.begin_task(task_id)
        for ev in trace.events:
            self.record_event(task_id, ev)
        self.end_task(task_id, trace, trace.evaluation)

    def close(self) -> None:

        for task_id, state in list(self._tasks.items()):
            if state.events_fh is not None:
                with contextlib.suppress(Exception):
                    state.events_fh.flush()
                    state.events_fh.close()
            with contextlib.suppress(Exception):
                _atomic_write(
                    state.task_dir / "status.json",
                    json.dumps(
                        {
                            "task_id": task_id,
                            "state": "aborted",
                            "started_at": state.started_at,
                            "ended_at": time.time(),
                            "last_step": state.step_count - 1,
                        },
                        indent=2,
                    ),
                )
            self._tasks.pop(task_id, None)

        summary_data = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "ended_at": time.time(),
            **self._summary.model_dump(exclude={"run_id"}),
        }
        _atomic_write(self.output_dir / "summary.json", json.dumps(summary_data, indent=2))

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    )
    try:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)
        raise

__all__ = [
    "TraceWriter",
    "extract_a11y_tree_from_event",
    "extract_a11y_trees",
    "extract_screenshot_from_event",
    "extract_screenshots",
]
