#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.replay_eval_results import collect_results, gt_label
except ModuleNotFoundError:
    from replay_eval_results import collect_results, gt_label


GUIRL_REWARD_LABELS = {"success", "failure", "uncertain"}


def _gt_reward_label(policy_result: Any) -> str | None:
    return gt_label(policy_result)


def _pred_reward_label(raw: Any) -> str:
    pred = str(raw or "missing").strip().lower()
    if pred in GUIRL_REWARD_LABELS:
        return pred
    if pred == "error":
        return "error"
    return "missing"


def _reward_matches_gt(pred: str, gt: str | None) -> bool | None:
    if gt is None:
        return None
    # uncertain / error / missing cannot match replay GT (success/failure).
    return pred == gt


def _new_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "gt_known": 0,
        "strict_match": 0,
        "labels": Counter(),
        "statuses": Counter(),
        "gt_labels": Counter(),
        "sampler_used": 0,
        "sampler_rounds": Counter(),
        "errors": 0,
    }


def _add(bucket: dict[str, Any], summary: dict[str, Any]) -> tuple[str | None, str, bool | None]:
    pred = _pred_reward_label(summary.get("final_label"))
    if summary.get("error") or summary.get("final_status") == "error":
        pred = "error"
    gt = _gt_reward_label(summary.get("policy_result"))
    sampler_rounds = int(summary.get("sampler_rounds") or 0)
    is_match = _reward_matches_gt(pred, gt)

    bucket["total"] += 1
    bucket["labels"][pred] += 1
    bucket["statuses"][str(summary.get("final_status", "missing"))] += 1
    bucket["sampler_rounds"][sampler_rounds] += 1
    if sampler_rounds > 0:
        bucket["sampler_used"] += 1
    if pred == "error" or summary.get("final_status") == "error":
        bucket["errors"] += 1

    if gt is not None:
        bucket["gt_known"] += 1
        bucket["gt_labels"][gt] += 1
        if is_match:
            bucket["strict_match"] += 1

    return gt, pred, is_match


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _format_rate(num: int, den: int) -> str:
    rate = _rate(num, den)
    if rate is None:
        return "n/a"
    return f"{num}/{den} = {rate:.2%}"


def _serializable_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "total": bucket["total"],
        "gt_known": bucket["gt_known"],
        "strict_match": bucket["strict_match"],
        "strict_rate": _rate(bucket["strict_match"], bucket["total"]),
        "labels": dict(bucket["labels"]),
        "statuses": dict(bucket["statuses"]),
        "gt_labels": dict(bucket["gt_labels"]),
        "sampler_used": bucket["sampler_used"],
        "sampler_rounds": dict(bucket["sampler_rounds"]),
        "errors": bucket["errors"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute reward consistency from replay_eval/eval_summary.json files."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Number of mismatches to print in text mode.",
    )
    args = parser.parse_args()

    records, accounting = collect_results(args.result_dir, args.expected_manifest, args.expected_rollouts)
    overall = _new_bucket()
    by_domain: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    mismatches: list[dict[str, Any]] = []

    for summary in records:
        gt, pred, is_match = _add(overall, summary)
        domain = str(summary.get("domain") or "unknown")
        _add(by_domain[domain], summary)
        if is_match is False:
            mismatches.append(
                {
                    "domain": domain,
                    "example_id": summary.get("example_id"),
                    "rollout_id": summary.get("rollout_id"),
                    "policy_result": summary.get("policy_result"),
                    "gt": gt,
                    "pred": pred,
                    "final_status": summary.get("final_status"),
                    "sampler_rounds": summary.get("sampler_rounds", 0),
                }
            )

    output = {
        "accounting": accounting,
        "result_dir": str(args.result_dir),
        "eval_summary_count": accounting["eval_summary_count"],
        "overall": _serializable_bucket(overall),
        "by_domain": {
            domain: _serializable_bucket(bucket)
            for domain, bucket in sorted(by_domain.items())
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[: args.max_mismatches],
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if accounting["coverage_verified"] and not (accounting["failed"] or accounting["missing"] or accounting["issues"]) else 2

    print(f"Result dir: {args.result_dir}")
    print(f"Eval summaries: {accounting['eval_summary_count']}")
    print()
    print("Overall")
    print(
        "  reward consistency:  "
        f"{_format_rate(overall['strict_match'], overall['total'])} "
        "(uncertain/error/missing count as mismatch)"
    )
    print("  accounting: " + json.dumps(accounting))
    print(f"  labels:             {dict(overall['labels'])}")
    print(f"  final_status:       {dict(overall['statuses'])}")
    print(f"  gt labels:          {dict(overall['gt_labels'])}")
    print(
        "  sampler used:       "
        f"{overall['sampler_used']}/{overall['total']} "
        f"({overall['sampler_used'] / overall['total']:.2%} if total>0)"
        if overall["total"]
        else "  sampler used:       n/a"
    )
    print(f"  sampler rounds:     {dict(overall['sampler_rounds'])}")
    print()
    print("By domain")
    for domain, bucket in sorted(by_domain.items()):
        print(
            f"  {domain}: reward {_format_rate(bucket['strict_match'], bucket['total'])}; "
            f"labels {dict(bucket['labels'])}"
        )
    print()
    print(f"Mismatches: {len(mismatches)}")
    for item in mismatches[: args.max_mismatches]:
        print(
            "  "
            f"{item['domain']}/{item['example_id']}/{item['rollout_id']}: "
            f"gt={item['gt']} pred={item['pred']} "
            f"policy_result={item['policy_result']} "
            f"status={item['final_status']} "
            f"sampler_rounds={item['sampler_rounds']}"
        )

    return 0 if accounting["coverage_verified"] and not (accounting["failed"] or accounting["missing"] or accounting["issues"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
