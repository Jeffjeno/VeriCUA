#!/usr/bin/env python3
"""Validate webarena data, evaluator configs, runtimes, and optional endpoints."""

from __future__ import annotations

import argparse
import importlib.util
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

from eval_runtime.web_arena.dataset import load_webarena_dataset

SUITE_DIR = Path(__file__).resolve().parent
CONFIGS = {
    "guirl_qwen4b_holo": SUITE_DIR / "configs" / "guirl_qwen4b_holo.yaml",
}


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _probe(base_url: str) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/models"
    try:
        response = requests.get(url, timeout=5)
        return response.ok, f"HTTP {response.status_code} {url}"
    except requests.RequestException as exc:
        return False, f"{url}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []
    try:
        dataset = load_webarena_dataset(project_root=PROJECT_ROOT)
        records = dataset.records
        candidate_ids = {str(record.get("candidate_id")) for record in records}
        sources_ok = all(
            Path(record["actions_file"]).is_file()
            and len(record.get("source_screenshots") or []) == int(record["action_count"]) + 1
            and all(Path(path).is_file() for path in record.get("source_screenshots") or [])
            for record in records
        )
        checks.append(_check(
            "dataset", len(records) == 122 and len(candidate_ids) == 122 and sources_ok,
            f"{len(records)} trajectories, {len(candidate_ids)} unique, sources_ok={sources_ok}",
        ))
    except Exception as exc:
        checks.append(_check("dataset", False, repr(exc)))

    methods = list(CONFIGS) if args.method == "all" else [args.method]
    for method in methods:
        path = CONFIGS[method]
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            evaluator_type = str(config.get("evaluator_type") or "")
            valid = evaluator_type == "guirl"
            if evaluator_type == "guirl":
                valid = valid and config.get("probe_agent_type") == "WebArenaHolo_Probe"
            checks.append(_check(f"config:{method}", valid, evaluator_type))
            if not args.skip_llm:
                for role, base_url in [("judge", config.get("base_url"))] + (
                    [("probe", config.get("probe_base_url"))] if evaluator_type == "guirl" else []
                ):
                    ok, detail = _probe(str(base_url or ""))
                    checks.append(_check(f"llm:{method}:{role}", ok, detail))
        except Exception as exc:
            checks.append(_check(f"config:{method}", False, repr(exc)))

    web_python = sys.executable
    browsergym_ok = importlib.util.find_spec("browsergym") is not None
    checks.append(
        _check(
            "runtime:webarena",
            browsergym_ok,
            f"python={web_python} browsergym={'available' if browsergym_ok else 'missing'}",
        )
    )
    hard_failures = [
        item for item in checks if not item["ok"]
        and (not item["name"].startswith("llm:") or args.strict_llm)
    ]
    if args.json:
        print(json.dumps({"ok": not hard_failures, "checks": checks}, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"[{'OK' if item['ok'] else 'FAIL'}] {item['name']}: {item['detail']}")
    return int(bool(hard_failures))


if __name__ == "__main__":
    raise SystemExit(main())
