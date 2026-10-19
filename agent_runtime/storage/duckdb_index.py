from __future__ import annotations

from pathlib import Path
from typing import Any

class DuckDBIndex:

    def __init__(
        self,
        root: str | Path = "./runs",
        *,
        persist_path: str | Path | None = None,
    ) -> None:
        try:
            import duckdb
        except ImportError as e:
            raise RuntimeError(
                "DuckDBIndex requires the optional `duckdb` extra: "
                "`pip install 'agent_runtime[duckdb]'`."
            ) from e

        self.root = Path(root).resolve()
        self.persist_path = Path(persist_path).resolve() if persist_path else None
        self._con = duckdb.connect(
            database=str(self.persist_path) if self.persist_path else ":memory:"
        )
        self.refresh()

    def refresh(self) -> None:

        glob_traj = str(self.root / "**" / "trajectories.jsonl")
        glob_manifest = str(self.root / "**" / "manifest.json")
        glob_summary = str(self.root / "**" / "summary.json")

        self._con.execute(
            f"""
            CREATE OR REPLACE VIEW trajectories AS
              SELECT *
              FROM read_json_auto('{glob_traj}', filename=true, format='nd', union_by_name=true);
            """
        )
        self._con.execute(
            f"""
            CREATE OR REPLACE VIEW runs AS
              SELECT *
              FROM read_json_auto('{glob_manifest}', filename=true, union_by_name=true);
            """
        )
        self._con.execute(
            f"""
            CREATE OR REPLACE VIEW summaries AS
              SELECT *
              FROM read_json_auto('{glob_summary}', filename=true, union_by_name=true);
            """
        )

    def query(self, sql: str, **params: Any):

        return self._con.execute(sql, params).df()

    def query_arrow(self, sql: str, **params: Any):
        return self._con.execute(sql, params).arrow()

    def execute(self, sql: str, **params: Any):
        return self._con.execute(sql, params)

    def close(self) -> None:
        self._con.close()

__all__ = ["DuckDBIndex"]
