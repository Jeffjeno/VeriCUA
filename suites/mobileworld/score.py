#!/usr/bin/env python3
"""Aggregate mobileworld evaluator summaries and policy/judge agreement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def build_report(result_root: Path) -> dict[str, Any]:
    summary_paths = sorted(result_root.rglob("replay_eval/eval_summary.json"))
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    for path in summary_paths:
        try:
            summary = _load(path)
        except Exception as exc:
            invalid.append(f"{path}: {exc}")
            continue
        episode_path = path.parents[1] / "episode.json"
        episode = _load(episode_path) if episode_path.is_file() else {}
        policy_result = summary.get("policy_result")
        policy_success = (
            float(policy_result) >= 1.0 if policy_result is not None else None
        )
        label = str(summary.get("final_label") or "error").lower()
        rows.append(
            {
                "task_name": str(summary.get("example_id") or episode.get("task_name") or ""),
                "candidate_id": episode.get("candidate_id"),
                "label": label,
                "policy_result": policy_result,
                "policy_success": policy_success,
                "original_score": episode.get("original_score"),
                "error_message": summary.get("error_message"),
                "path": str(path),
            }
        )

    labels = Counter(row["label"] for row in rows)
    policy_known = [row for row in rows if row["policy_success"] is not None]
    decided = [
        row for row in policy_known
        if row["label"] in {"success", "failure"}
    ]
    matches = sum(
        (row["label"] == "success") == row["policy_success"]
        for row in decided
    )
    replay_drift = [
        row["task_name"]
        for row in rows
        if row["policy_result"] is not None
        and row["original_score"] is not None
        and (float(row["policy_result"]) >= 1.0)
        != (float(row["original_score"]) >= 1.0)
    ]
    task_counts = Counter(row["task_name"] for row in rows if row["task_name"])
    return {
        "result_root": str(result_root),
        "total_summaries": len(rows),
        "unique_tasks": len(task_counts),
        "coverage_of_mobileworld": len(task_counts) / 100.0,
        "labels": dict(sorted(labels.items())),
        "judge_success_rate": (
            labels.get("success", 0) / len(rows) if rows else None
        ),
        "policy_success_rate": (
            sum(bool(row["policy_success"]) for row in policy_known)
            / len(policy_known)
            if policy_known
            else None
        ),
        "judge_policy_agreement": {
            "decided": len(decided),
            "matches": matches,
            "rate": matches / len(decided) if decided else None,
        },
        "replay_label_drift": {
            "count": len(replay_drift),
            "tasks": sorted(replay_drift),
        },
        "duplicate_tasks": sorted(
            task for task, count in task_counts.items() if count > 1
        ),
        "error_tasks": sorted(
            row["task_name"]
            for row in rows
            if row["label"] == "error"
        ),
        "uncertain_tasks": sorted(
            row["task_name"]
            for row in rows
            if row["label"] == "uncertain"
        ),
        "invalid_summaries": invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--error-tasks", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    report = build_report(result_root)
    if args.error_tasks:
        for task_name in report["error_tasks"]:
            print(task_name)
        return 0
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else result_root / "score_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["invalid_summaries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
