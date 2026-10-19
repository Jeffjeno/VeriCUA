#!/usr/bin/env python3
"""Validate the mobileworld dataset, evaluator configs, and service endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_runtime.mobile_world.dataset import load_mobile_world_dataset

SUITE_DIR = Path(__file__).resolve().parent
CONFIGS = {
    "guirl_qwen4b_gui_owl": SUITE_DIR / "configs" / "guirl_qwen4b_gui_owl.yaml",
}


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _probe_url(url: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
        return response.ok, f"HTTP {response.status_code} {url}"
    except requests.RequestException as exc:
        return False, f"{url}: {exc}"


def _backends(args: argparse.Namespace) -> list[str]:
    if args.backend:
        return args.backend
    raw = os.getenv("MOBILE_WORLD_BACKENDS", "http://127.0.0.1:6800")
    return [value for value in raw.replace(",", " ").split() if value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["all", *CONFIGS],
        default="all",
    )
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checks: list[dict[str, Any]] = []
    try:
        dataset = load_mobile_world_dataset("mobileworld", project_root=PROJECT_ROOT)
        tasks = {record.task_name for record in dataset.records}
        checks.append(
            _check(
                "dataset",
                len(tasks) == 100,
                f"{len(dataset.records)} trajectories, {len(tasks)} tasks",
            )
        )
        dataset_ok = True
    except Exception as exc:
        dataset = None
        tasks = set()
        dataset_ok = False
        checks.append(_check("dataset", False, repr(exc)))

    methods = list(CONFIGS) if args.method == "all" else [args.method]
    active_requested = bool(methods)
    for method in methods:
        path = CONFIGS[method]
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            evaluator_type = str(config.get("evaluator_type") or "")
            expected = "guirl"
            valid = evaluator_type == expected
            if method.startswith("guirl"):
                valid = valid and str(config.get("probe_agent_type") or "").startswith(
                    "MobileWorld"
                )
            checks.append(
                _check(
                    f"config:{method}",
                    valid,
                    f"{path.relative_to(PROJECT_ROOT)} ({evaluator_type})",
                )
            )
            if args.skip_llm:
                continue
            base_urls = [str(config.get("base_url") or "")]
            if method.startswith("guirl"):
                base_urls.append(str(config.get("probe_base_url") or ""))
            for index, base_url in enumerate(base_urls):
                ok, detail = _probe_url(_models_url(base_url))
                checks.append(
                    _check(
                        f"llm:{method}:{'judge' if index == 0 else 'probe'}",
                        ok,
                        detail,
                    )
                )
        except Exception as exc:
            checks.append(_check(f"config:{method}", False, repr(exc)))

    if active_requested:
        for backend in _backends(args):
            try:
                health = requests.get(f"{backend.rstrip('/')}/health", timeout=60)
                task_list = requests.get(
                    f"{backend.rstrip('/')}/task/list",
                    timeout=10,
                )
                health_payload = health.json() if health.ok else {}
                listed = task_list.json() if task_list.ok else []
                listed_names = {
                    str(item.get("name"))
                    for item in listed
                    if isinstance(item, dict)
                }
                missing = tasks - listed_names if dataset_ok else set()
                checks.append(
                    _check(
                        f"backend:{backend}",
                        health.ok
                        and bool(health_payload.get("ok"))
                        and task_list.ok
                        and not missing,
                        (
                            f"healthy={health_payload.get('ok')} "
                            f"tasks={len(listed_names)} missing_mobileworld={len(missing)}"
                        ),
                    )
                )
            except Exception as exc:
                checks.append(_check(f"backend:{backend}", False, repr(exc)))

    hard_failures = [
        check
        for check in checks
        if not check["ok"]
        and (
            not check["name"].startswith("llm:")
            or args.strict_llm
        )
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "ok": not hard_failures,
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for check in checks:
            marker = "OK" if check["ok"] else "FAIL"
            print(f"[{marker}] {check['name']}: {check['detail']}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
