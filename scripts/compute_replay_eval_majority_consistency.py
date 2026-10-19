#!/usr/bin/env python3
"""Validate and score repeated judgements of a fixed source trajectory."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.replay_eval_results import collect_results, gt_label
except ModuleNotFoundError:
    from replay_eval_results import collect_results, gt_label


def compute(result_dir: Path, expected_rollouts: int, expected_manifest: Path | None = None):
    if expected_rollouts < 1:
        raise ValueError("expected rollouts must be positive")
    summaries, accounting = collect_results(result_dir, expected_manifest, expected_rollouts)
    groups = defaultdict(list)
    identities = set()
    for summary in summaries:
        identity = tuple(str(summary.get(field, "")) for field in ("policy_model", "eval_model", "run_id"))
        if summary.get("final_label") != "missing":
            identities.add(identity)
        groups[(str(summary.get("domain", "")), str(summary.get("example_id", "")))].append(summary)
    issues = list(accounting["issues"])
    if len(identities) > 1:
        issues.append("mixed policy models, judge models or evaluation runs")
    incomplete, invalid, mismatches = [], [], []
    complete = matches = known = unanimous = ties = errors = 0
    by_domain = defaultdict(lambda: {
        "tasks": 0, "complete": 0, "gt_known": 0, "strict_match": 0, "unanimous": 0, "ties": 0,
    })
    expected_ids = {str(i) for i in range(expected_rollouts)}
    for (domain, example), records in sorted(groups.items()):
        bucket = by_domain[domain]
        bucket["tasks"] += 1
        detail = {"domain": domain, "example_id": example}
        ids = [str(s.get("rollout_id", "")) for s in records]
        labels = [str(s.get("final_label", "missing")) for s in records]
        if "error" in labels or any(s.get("error") for s in records):
            errors += 1
        if len(ids) != expected_rollouts or set(ids) != expected_ids or "missing" in labels:
            incomplete.append(dict(detail, rollout_ids=ids, have=sum(label != "missing" for label in labels),
                                   expected=expected_rollouts))
            continue
        sources = {s.get("source_trajectory") for s in records}
        if len(sources) > 1 or None in sources or "" in sources:
            invalid.append(dict(detail, reason="mixed or unspecified source trajectories"))
            continue
        group_identities = {
            tuple(str(s.get(field, "")) for field in ("policy_model", "eval_model", "run_id"))
            for s in records
        }
        if len(group_identities) != 1:
            invalid.append(dict(detail, reason="mixed model or run identities"))
            continue
        if any(not field for field in next(iter(group_identities))):
            invalid.append(dict(detail, reason="unspecified model or run identity"))
            continue
        gt_values = [s.get("policy_result") for s in records]
        gt_labels = {gt_label(value) for value in gt_values}
        if None in gt_labels or len(gt_labels) != 1 or len({float(v) for v in gt_values}) != 1:
            invalid.append(dict(detail, reason="GT conflict or missing GT", policy_results=gt_values))
            continue
        known += 1
        bucket["gt_known"] += 1
        if any(label not in {"success", "failure", "uncertain"} for label in labels):
            invalid.append(dict(detail, reason="error or invalid prediction", labels=labels))
            continue
        complete += 1
        bucket["complete"] += 1
        counts = Counter(labels)
        pred, count = counts.most_common(1)[0]
        if count * 2 <= expected_rollouts:
            pred = "tie"
            ties += 1
            bucket["ties"] += 1
        if len(counts) == 1:
            unanimous += 1
            bucket["unanimous"] += 1
        gt = next(iter(gt_labels))
        if pred == gt:
            matches += 1
            bucket["strict_match"] += 1
        else:
            mismatches.append(dict(detail, gt=gt, majority_pred=pred, vote_counts=dict(counts)))
    total = len(groups)
    valid = bool(total and accounting["coverage_verified"] and not (
        incomplete or invalid or issues or accounting["failed"] or accounting["missing"]
    ))
    return {
        "result_dir": str(result_dir), "expected_rollouts": expected_rollouts,
        "accounting": accounting, "valid": valid, "issues": issues,
        "task_groups": total, "task_groups_complete": complete,
        "task_groups_incomplete": len(incomplete), "invalid_groups": len(invalid),
        "gt_known": known, "strict_match": matches,
        "strict_denominator": total,
        "strict_rate": matches / total if total and len(identities) <= 1 else None,
        "unanimous_tasks": unanimous, "unanimous_rate": unanimous / total if total else None,
        "tie_tasks": ties, "error_tasks": errors, "by_domain": dict(by_domain),
        "incomplete": incomplete, "invalid": invalid, "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = compute(args.result_dir, args.expected_rollouts, args.expected_manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Task groups: {result['task_groups']} (complete {result['task_groups_complete']}, "
              f"incomplete {result['task_groups_incomplete']}, invalid {result['invalid_groups']})")
        rate = result["strict_rate"]
        print(f"Majority reward consistency: {result['strict_match']}/{result['strict_denominator']} "
              + (f"= {rate:.2%}" if rate is not None else "= invalid"))
        print("Accounting: " + json.dumps(result["accounting"], ensure_ascii=False))
        for issue in result["issues"]:
            print("ERROR: " + issue)
        for item in result["invalid"][:20]:
            print("INVALID: " + json.dumps(item, ensure_ascii=False))
        if not result["valid"]:
            print("Evaluation is incomplete or invalid; this is not a final score.")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
