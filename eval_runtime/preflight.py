"""Prepare and validate the OSWorld suite's effective configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

import yaml

from eval_runtime.config import load_eval_dataset_config


def prepare(args):
    root = Path(args.project_dir)
    dataset = load_eval_dataset_config(args.dataset, str(root))
    manifest_path = Path(dataset["test_all_meta_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(not isinstance(v, list) for v in manifest.values()):
        raise ValueError("dataset manifest must map domains to example IDs")
    if args.domain != "all":
        manifest = {args.domain: manifest[args.domain]}
    policy = dataset["policy"]
    source_root = Path(policy["source_result_root"])
    if not source_root.is_dir():
        raise ValueError(f"trajectory directory not found: {source_root}")
    screenshot_root = source_root / "pyautogui" / "screenshot"
    if screenshot_root.is_dir():
        model_root = screenshot_root / str(policy.get("source_model_name", ""))
        candidates = [p for p in screenshot_root.iterdir() if p.is_dir()]
        source_root = model_root if model_root.is_dir() else candidates[0] if len(candidates) == 1 else source_root
    count = dataset["task_n_combinations"]
    if count != args.rollouts or count < 1:
        raise ValueError("dataset rollout count must match EXPECTED_ROLLOUTS")
    repeat = max(1, int(policy.get("source_rollout_repeat", 1)))
    tasks = 0
    for domain, examples in manifest.items():
        for example in examples:
            if example == "onboard":
                continue
            tasks += 1
            task_path = Path(dataset["test_config_base_dir"]) / dataset["examples_subdir"] / domain / f"{example}.json"
            json.loads(task_path.read_text(encoding="utf-8"))
            for source_id in {str(i // repeat) for i in range(count)}:
                candidates = [source_root / domain / example / source_id / "traj.jsonl",
                              source_root / domain / example / "traj.jsonl"]
                if not any(p.is_file() for p in candidates):
                    raise ValueError(f"missing trajectory: {domain}/{example}/{source_id}")
    if not tasks:
        raise ValueError("empty evaluation manifest")
    if not Path(args.vm).is_file():
        raise ValueError(f"VM image not found: {args.vm}")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("judge config must be a mapping")
    if getattr(args, "checkpoint", ""):
        cfg["checkpoint_path"] = str(Path(args.checkpoint).resolve())
    for key, value in (("base_url", args.judge_url), ("model_name", args.model)):
        cfg[key] = value
        if isinstance(cfg.get("model"), dict):
            cfg["model"][key] = value
    probe = args.probe_url or cfg.get("probe_base_url", "")
    cfg["probe_base_url"] = probe.removesuffix("/models").rstrip("/")
    for key in ("base_url", "probe_base_url"):
        url = str(cfg[key])
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or "<" in url or ">" in url:
            raise ValueError(f"invalid {key}: {url}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def check_endpoints(config, include_judge=True):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    pairs = [(cfg["probe_base_url"], cfg["probe_model_name"], cfg.get("probe_api_key", ""))]
    if include_judge:
        pairs.append((cfg["base_url"], cfg["model_name"], cfg.get("api_key", "")))
    for base, model, key in pairs:
        url = base.rstrip("/") + "/models"
        request = Request(url, headers={"Authorization": f"Bearer {key}"})
        with urlopen(request, timeout=10) as response:
            data = json.load(response)
        if model not in {entry.get("id") for entry in data.get("data", [])}:
            raise ValueError(f"endpoint {url} is not serving {model}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-endpoints")
    parser.add_argument("--probe-only", action="store_true")
    for key in ("project-dir", "dataset", "domain", "vm", "config", "output", "judge-url", "model", "probe-url", "checkpoint"):
        parser.add_argument("--" + key, default="all" if key == "domain" else "")
    parser.add_argument("--rollouts", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.check_endpoints:
            check_endpoints(args.check_endpoints, not args.probe_only)
        else:
            prepare(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")


if __name__ == "__main__":
    main()
