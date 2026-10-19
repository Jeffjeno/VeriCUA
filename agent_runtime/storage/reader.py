from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_runtime.types import Trace, TraceEvent

def iter_runs(root: str | Path) -> Iterator[Path]:

    root = Path(root)
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            yield child

def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    return json.loads(Path(run_dir, "manifest.json").read_text())

def iter_task_dirs(run_dir: str | Path) -> Iterator[Path]:

    tasks_dir = Path(run_dir) / "tasks"
    if not tasks_dir.exists():
        return
    for child in sorted(tasks_dir.iterdir()):
        if child.is_dir() and (child / "trace.json").exists():
            yield child

def load_trace(task_dir: str | Path) -> Trace:

    path = Path(task_dir) / "trace.json"
    return Trace.model_validate_json(path.read_text())

def load_status(task_dir: str | Path) -> dict[str, Any] | None:

    path = Path(task_dir) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

def load_partial_trace(task_dir: str | Path) -> Trace | None:

    task_path = Path(task_dir)
    final = task_path / "trace.json"
    if final.exists():
        return load_trace(task_path)

    events_path = task_path / "events.jsonl"
    task_path_meta = task_path / "task.json"
    if not events_path.exists() and not task_path_meta.exists():
        return None

    task_id = task_path.name
    benchmark = ""
    metadata: dict[str, Any] = {"partial": True}

    if task_path_meta.exists():
        try:
            meta_json = json.loads(task_path_meta.read_text())
        except json.JSONDecodeError:
            meta_json = {}
        task_id = meta_json.get("task_id", task_id) or task_id
        benchmark = meta_json.get("benchmark", "") or ""

        for key in ("instruction", "tags", "suite"):
            if key in meta_json:
                metadata[f"task_{key}"] = meta_json[key]

    events: list = []
    warnings: list[str] = []
    if events_path.exists():
        for line_no, line in enumerate(
            events_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TraceEvent.model_validate_json(line))
            except Exception as exc:
                warnings.append(f"line {line_no}: {exc!s}")

    if warnings:
        metadata["partial_load_warnings"] = warnings

    return Trace(
        task_id=task_id,
        benchmark=benchmark,
        events=events,
        evaluation=None,
        ended_at=None,
        metadata=metadata,
    )

def iter_traces(
    run_dir: str | Path, *, include_partial: bool = False
) -> Iterator[Trace]:

    if include_partial:
        tasks_dir = Path(run_dir) / "tasks"
        if not tasks_dir.exists():
            return
        for child in sorted(tasks_dir.iterdir()):
            if not child.is_dir():
                continue
            trace = load_partial_trace(child)
            if trace is not None:
                yield trace
        return

    for task_dir in iter_task_dirs(run_dir):
        yield load_trace(task_dir)

def load_traces(
    run_dir: str | Path, *, include_partial: bool = False
) -> list[Trace]:
    return list(iter_traces(run_dir, include_partial=include_partial))

def load_evaluation(task_dir: str | Path) -> dict[str, Any] | None:

    path = Path(task_dir) / "evaluation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def iter_evaluations(run_dir: str | Path) -> Iterator[dict[str, Any]]:

    for task_dir in iter_task_dirs(run_dir):
        ev = load_evaluation(task_dir)
        if ev is not None:
            yield ev

def load_summary(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir, "summary.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())

__all__ = [
    "iter_evaluations",
    "iter_runs",
    "iter_task_dirs",
    "iter_traces",
    "load_evaluation",
    "load_manifest",
    "load_partial_trace",
    "load_status",
    "load_summary",
    "load_trace",
    "load_traces",
]
