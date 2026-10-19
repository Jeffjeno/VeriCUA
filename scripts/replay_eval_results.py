"""Validate replay outputs against the expected evaluation inventory."""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def gt_label(value) -> str | None:
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return "success" if number >= 1 else "failure"
    except (TypeError, ValueError, OverflowError):
        return None


def collect_results(result_dir: Path, expected_manifest: Path | None = None, rollouts: int = 4):
    if rollouts < 1:
        raise ValueError("expected rollouts must be positive")
    paths = sorted(result_dir.rglob("replay_eval/eval_summary.json"))
    records = []
    issues = []
    for path in paths:
        summary = load_json(path)
        if summary is None:
            issues.append(f"unreadable summary: {path}")
            continue
        records.append((path, summary))
    inventory_path = result_dir / "eval_manifest.json"
    inventory = load_json(inventory_path)
    expected = []
    if inventory_path.exists():
        if inventory is None or not isinstance(inventory.get("tasks"), list):
            raise ValueError(f"invalid evaluation inventory: {inventory_path}")
        expected = inventory["tasks"]
    elif expected_manifest is not None:
        manifest = load_json(expected_manifest)
        if manifest is None or any(not isinstance(ids, list) for ids in manifest.values()):
            raise ValueError(f"invalid domain-to-example manifest: {expected_manifest}")
        expected = [
            {"domain": str(domain), "example_id": str(example), "rollout_id": str(i)}
            for domain, examples in manifest.items() for example in examples
            if str(example) != "onboard" for i in range(rollouts)
        ]
    else:
        issues.append("expected manifest is required; coverage cannot be verified")
    if expected_manifest is not None and inventory_path.exists():
        manifest = load_json(expected_manifest)
        if manifest is None or any(not isinstance(ids, list) for ids in manifest.values()):
            raise ValueError(f"invalid domain-to-example manifest: {expected_manifest}")
        specified = {(str(domain), str(example), str(i))
                     for domain, examples in manifest.items() for example in examples
                     if str(example) != "onboard" for i in range(rollouts)}
        inventory_keys = {tuple(str(item[field]) for field in ("domain", "example_id", "rollout_id"))
                          for item in expected}
        if specified != inventory_keys:
            raise ValueError("expected manifest does not match the saved evaluation inventory")
    by_task = {}
    for path, summary in records:
        key = tuple(str(summary.get(field, "")) for field in ("domain", "example_id", "rollout_id"))
        by_task.setdefault(key, []).append((path, summary))
    expected_keys = set()
    resolved = []
    completed = failed = missing = 0
    for item in expected:
        key = tuple(str(item[field]) for field in ("domain", "example_id", "rollout_id"))
        if key in expected_keys:
            raise ValueError(f"duplicate task in expected manifest: {key}")
        expected_keys.add(key)
        matches = by_task.get(key, [])
        if len(matches) > 1:
            issues.append(f"duplicate task results: {key}")
        summary = dict(matches[0][1]) if len(matches) == 1 else dict(item)
        if not matches:
            missing += 1
            summary.update(final_label="missing", final_status="missing")
        elif len(matches) != 1:
            failed += 1
            summary.update(final_label="error", final_status="error")
        else:
            mismatch = any(
                field in item and summary.get(field) != item[field]
                for field in ("run_id", "policy_model", "eval_model", "source_trajectory")
            )
            if mismatch:
                issues.append(f"result identity does not match inventory: {key}")
                summary.update(final_label="error", final_status="error")
            if (
                mismatch or summary.get("error") or summary.get("final_status") == "error"
                or summary.get("final_label") not in {"success", "failure", "uncertain", "yes", "no"}
                or gt_label(summary.get("policy_result")) is None
            ):
                failed += 1
            else:
                completed += 1
        resolved.append(summary)
    for key in by_task.keys() - expected_keys:
        if expected:
            issues.append(f"unexpected task result: {key}")
    if not expected:
        resolved = [summary for _, summary in records]
    identities = {tuple(str(s.get(field, "")) for field in ("policy_model", "eval_model", "run_id"))
                  for _, s in records}
    if len(identities) > 1:
        issues.append("mixed policy models, judge models or evaluation runs")
    return resolved, {
        "expected": len(expected), "completed": completed, "failed": failed, "missing": missing,
        "issues": issues, "coverage_verified": bool(expected), "eval_summary_count": len(paths),
    }
