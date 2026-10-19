#!/usr/bin/env python3
"""Aggregate webarena eval summaries and judge/policy agreement."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_runtime.web_arena.dataset import load_webarena_dataset


def build_report(result_root: Path) -> dict[str, Any]:
    rows = []
    invalid = []
    manifest_path = result_root / "suite_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append(f"{manifest_path}: {exc}")
    dataset_name = str(manifest.get("dataset") or "webarena")
    rollout_repeats = max(int(manifest.get("rollout_repeats") or 1), 1)
    for path in sorted(result_root.rglob("replay_eval/eval_summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "candidate_id": str(summary.get("example_id") or path.parents[1].name),
                "rollout_id": str(summary.get("rollout_id") or "0"),
                "label": str(summary.get("final_label") or "error").lower(),
                "policy_result": summary.get("policy_result"),
                "error_message": summary.get("error_message"),
            })
        except Exception as exc:
            invalid.append(f"{path}: {exc}")
    labels = Counter(row["label"] for row in rows)
    decided = [
        row for row in rows
        if row["policy_result"] is not None and row["label"] in {"success", "failure"}
    ]
    matches = sum(
        (row["label"] == "success") == (float(row["policy_result"]) >= 1.0)
        for row in decided
    )
    expected_tasks = len(load_webarena_dataset(dataset_name, project_root=PROJECT_ROOT).records)
    expected = expected_tasks * rollout_repeats
    return {
        "result_root": str(result_root),
        "dataset": dataset_name,
        "rollout_repeats": rollout_repeats,
        "expected_tasks": expected_tasks,
        "expected_summaries": expected,
        "total_summaries": len(rows),
        "expected_dataset_tasks": expected_tasks,
        "coverage": len(rows) / expected if expected else None,
        "labels": dict(sorted(labels.items())),
        "judge_success_rate": labels.get("success", 0) / len(rows) if rows else None,
        "judge_policy_agreement": {
            "decided": len(decided), "matches": matches,
            "rate": matches / len(decided) if decided else None,
        },
        "error_tasks": sorted(row["candidate_id"] for row in rows if row["label"] == "error"),
        "uncertain_tasks": sorted(row["candidate_id"] for row in rows if row["label"] == "uncertain"),
        "invalid_summaries": invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--error-tasks", action="store_true")
    args = parser.parse_args()
    report = build_report(args.result_root.expanduser().resolve())
    if args.error_tasks:
        print("\n".join(report["error_tasks"]))
        return 0
    output = args.output or args.result_root / "score_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report["invalid_summaries"]))


if __name__ == "__main__":
    raise SystemExit(main())
