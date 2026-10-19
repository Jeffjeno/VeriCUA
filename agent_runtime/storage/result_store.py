from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_runtime.types import RunTaskResult

logger = logging.getLogger(__name__)

class ResultStore(ABC):

    @abstractmethod
    def record_run(
        self,
        run_id: str,
        run_name: str,
        benchmark: str | None = None,
        tags: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        total_tasks: int = 0,
    ) -> None: ...

    @abstractmethod
    def record_attempt_start(
        self,
        run_id: str,
        task_id: str,
        attempt_id: str,
        benchmark: str | None = None,
        worker_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    def record_attempt_result(
        self,
        attempt_id: str,
        result: RunTaskResult,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        trace_path: str | None = None,
    ) -> None: ...

    @abstractmethod
    def finish_run(self, run_id: str) -> None: ...

    @abstractmethod
    def summary(
        self,
        run_id: str | None = None,
        by: list[str] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def pending_tasks(
        self,
        run_id: str,
        all_task_ids: list[str],
    ) -> list[str]:
        pass

    @abstractmethod
    def query(self, sql: str, **params: Any) -> Any: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    run_name     VARCHAR,
    benchmark    VARCHAR,
    started_at   DOUBLE,
    ended_at     DOUBLE,
    total_tasks  INTEGER DEFAULT 0,
    tags         JSON,
    config       JSON
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id   VARCHAR PRIMARY KEY,
    run_id       VARCHAR NOT NULL,
    run_name     VARCHAR,
    task_id      VARCHAR NOT NULL,
    benchmark    VARCHAR,
    status       VARCHAR NOT NULL DEFAULT 'queued',
    worker_id    VARCHAR,
    started_at   DOUBLE,
    ended_at     DOUBLE,
    duration_s   DOUBLE,
    error_type   VARCHAR,
    error_msg    VARCHAR
);

CREATE TABLE IF NOT EXISTS evaluations (
    attempt_id   VARCHAR PRIMARY KEY,
    run_id       VARCHAR NOT NULL,
    task_id      VARCHAR NOT NULL,
    benchmark    VARCHAR,
    success      BOOLEAN,
    score        DOUBLE,
    reason       VARCHAR,
    error_type   VARCHAR,
    num_steps    INTEGER,
    tags         JSON,
    trace_path   VARCHAR,
    artifacts_dir VARCHAR
);
"""

class DuckDBResultStore(ResultStore):

    def __init__(self, path: str | Path | None = None) -> None:
        try:
            import duckdb
        except ImportError as e:
            raise RuntimeError(
                "DuckDBResultStore requires the optional `duckdb` extra: "
                "`pip install 'agent_runtime[duckdb]'`."
            ) from e

        self.path = Path(path).resolve() if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        db_path = str(self.path) if self.path else ":memory:"
        self._con = duckdb.connect(database=db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._con.execute(stmt)

    def record_run(
        self,
        run_id: str,
        run_name: str,
        benchmark: str | None = None,
        tags: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        total_tasks: int = 0,
    ) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, run_name, benchmark, started_at, total_tasks, tags, config)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                run_name,
                benchmark,
                time.time(),
                total_tasks,
                json.dumps(tags or {}),
                json.dumps(config or {}),
            ],
        )

    def record_attempt_start(
        self,
        run_id: str,
        task_id: str,
        attempt_id: str,
        benchmark: str | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO attempts
                (attempt_id, run_id, task_id, benchmark, status, worker_id, started_at)
            VALUES (?, ?, ?, ?, 'running', ?, ?)
            """,
            [attempt_id, run_id, task_id, benchmark, worker_id, time.time()],
        )

    def record_attempt_result(
        self,
        attempt_id: str,
        result: RunTaskResult,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        trace_path: str | None = None,
    ) -> None:
        now = time.time()
        ev = result.evaluation
        task_id = result.trace.task_id if result.trace else ""
        benchmark = result.trace.benchmark if result.trace else ""
        num_steps = result.trace.num_events if result.trace else 0

        success = ev.success if ev else None
        status = "succeeded" if (ev and ev.success) else "failed"

        self._con.execute(
            """
            UPDATE attempts
            SET status = ?, ended_at = ?,
                duration_s = ? - COALESCE(started_at, ?),
                error_type = ?, error_msg = ?
            WHERE attempt_id = ?
            """,
            [
                status,
                now,
                now,
                now,
                ev.error_type if ev else None,
                ev.reason if (ev and not ev.success) else None,
                attempt_id,
            ],
        )

        self._con.execute(
            """
            INSERT OR IGNORE INTO attempts
                (attempt_id, run_id, task_id, benchmark, status, ended_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [attempt_id, run_id or "", task_id, benchmark, status, now],
        )

        self._con.execute(
            """
            INSERT OR REPLACE INTO evaluations
                (attempt_id, run_id, task_id, benchmark, success, score,
                 reason, error_type, num_steps, tags, trace_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                attempt_id,
                run_id or "",
                task_id,
                benchmark,
                success,
                ev.score if ev else None,
                ev.reason if ev else None,
                ev.error_type if ev else None,
                num_steps,
                json.dumps(tags or {}),
                trace_path,
            ],
        )

    def finish_run(self, run_id: str) -> None:
        self._con.execute(
            "UPDATE runs SET ended_at = ? WHERE run_id = ?",
            [time.time(), run_id],
        )

    def summary(
        self,
        run_id: str | None = None,
        by: list[str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:

        if by:
            return self._summary_by(run_id, by)

        where = "WHERE run_id = ?" if run_id else ""
        params = [run_id] if run_id else []

        row = self._con.execute(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) AS succeeded,
                   SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failed,
                   AVG(score) AS avg_score
            FROM evaluations {where}
            """,
            params,
        ).fetchone()

        total, succeeded, failed, avg_score = row
        return {
            "total_tasks": total or 0,
            "succeeded": succeeded or 0,
            "failed": failed or 0,
            "avg_score": avg_score,
            "success_rate": (succeeded or 0) / total if total else None,
        }

    def _summary_by(
        self, run_id: str | None, by: list[str]
    ) -> list[dict[str, Any]]:
        select_cols = []
        group_cols = []
        for i, key in enumerate(by):
            col_alias = key
            col_expr = f"tags->>'{key}'"
            select_cols.append(f"{col_expr} AS {col_alias}")
            group_cols.append(str(i + 1))

        select_str = ", ".join(select_cols)
        group_str = ", ".join(group_cols)

        where = "WHERE run_id = ?" if run_id else ""
        params = [run_id] if run_id else []

        sql = f"""
            SELECT {select_str},
                   COUNT(*) AS total,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) AS succeeded,
                   SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) AS failed,
                   AVG(score) AS avg_score
            FROM evaluations {where}
            GROUP BY {group_str}
            ORDER BY {group_str}
        """
        rows = self._con.execute(sql, params).fetchall()
        col_names = by + ["total", "succeeded", "failed", "avg_score"]
        return [dict(zip(col_names, row)) for row in rows]

    def pending_tasks(
        self,
        run_id: str,
        all_task_ids: list[str],
    ) -> list[str]:
        succeeded = set()
        rows = self._con.execute(
            "SELECT DISTINCT task_id FROM attempts WHERE run_id = ? AND status = 'succeeded'",
            [run_id],
        ).fetchall()
        for (tid,) in rows:
            succeeded.add(tid)
        return [t for t in all_task_ids if t not in succeeded]

    def query(self, sql: str, **params: Any) -> Any:

        return self._con.execute(sql, list(params.values()) if params else [])

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None

__all__ = ["DuckDBResultStore", "ResultStore"]
