from __future__ import annotations

from agent_runtime.storage.reader import (
    iter_evaluations,
    iter_runs,
    iter_task_dirs,
    iter_traces,
    load_evaluation,
    load_manifest,
    load_partial_trace,
    load_status,
    load_summary,
    load_trace,
    load_traces,
)
from agent_runtime.storage.writer import TraceWriter

try:
    from agent_runtime.storage.duckdb_index import DuckDBIndex

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

try:
    from agent_runtime.storage.result_store import (
        DuckDBResultStore,
        ResultStore,
    )

    _HAS_RESULT_STORE = True
except ImportError:
    _HAS_RESULT_STORE = False

__all__ = [
    "TraceWriter",
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
if _HAS_DUCKDB:
    __all__.append("DuckDBIndex")
if _HAS_RESULT_STORE:
    __all__.extend(["DuckDBResultStore", "ResultStore"])
