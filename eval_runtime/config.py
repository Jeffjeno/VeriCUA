from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class EvalConfig:
    osworld_path: str = ""
    provider_name: str = "docker"
    path_to_vm: str = ""
    headless: bool = True
    screen_width: int = 1920
    screen_height: int = 1080
    docker_tag: str = ""
    policy_config_path: str = ""
    test_all_meta_path: str = ""
    test_config_base_dir: str = ""
    examples_subdir: str = "examples"
    result_dir: str = "./results/replay_eval"
    domain: str = "all"
    task_n_combinations: int = 1
    max_steps: int = 50
    sleep_after_execution: float = 5.0
    num_envs: int = 4
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    replay_eval_config_path: str = ""
    replay_eval_mode: str = "memory"
    max_eval_rounds: int = 3
    sampler_max_steps: int = 10
    log_level: str = "INFO"
    extra: dict[str, Any] = field(default_factory=dict)

def load_eval_dataset_config(config_path: str, project_dir: str = "") -> dict[str, Any]:
    config_path = os.path.expandvars(os.path.expanduser(config_path))
    if not os.path.isabs(config_path):
        if project_dir:
            joined = os.path.join(project_dir, config_path)
            if os.path.isfile(joined):
                config_path = joined
            elif os.path.isfile(config_path):
                config_path = os.path.abspath(config_path)
            else:
                config_path = joined
        else:
            config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid eval dataset config: {config_path}")
    def resolve_path(value: str) -> str:
        if not value:
            return ""
        value = os.path.expandvars(os.path.expanduser(str(value)))
        path = Path(value)
        if not path.is_absolute():
            path = Path(project_dir) / path
        return str(path.resolve())
    name = str(cfg.get("name") or Path(config_path).stem)
    paths = cfg.get("paths") or {}
    policy = dict(cfg.get("policy") or {})
    rollout = cfg.get("rollout") or {}
    source_result_root = resolve_path(paths.get("source_result_root", ""))
    if not source_result_root:
        raise ValueError(f"paths.source_result_root is required in {config_path}")
    policy["source_result_root"] = source_result_root
    policy.setdefault("agent_type", "Replay")
    return {
        "name": name,
        "policy": policy,
        "test_all_meta_path": resolve_path(paths.get("test_all_meta_path", "")),
        "test_config_base_dir": resolve_path(paths.get("test_config_base_dir", "")),
        "examples_subdir": str(paths.get("examples_subdir", "examples")),
        "task_n_combinations": int(rollout.get("task_n_combinations", 1)),
        "max_steps": int(rollout.get("max_steps", policy.get("max_steps", 50))),
        "result_dir": resolve_path(rollout["result_dir"]) if rollout.get("result_dir") else "",
    }

def load_replay_eval_config(config_path: str) -> dict[str, Any]:
    config_path = os.path.expandvars(os.path.expanduser(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid replay eval config: {config_path}")
    return cfg

def resolve_replay_eval_model_config(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    model_cfg = cfg.get("model")
    inherited = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    for key in ("model_name", "model", "api_key", "base_url", "max_retry",
                "retry_waiting_seconds", "max_tokens", "temperature", "top_p",
                "top_k", "min_p", "presence_penalty", "repetition_penalty",
                "max_pixels", "enable_thinking"):
        if key in cfg and key not in inherited:
            inherited[key] = cfg[key]
    if "model" in inherited and not inherited.get("model_name"):
        inherited["model_name"] = inherited.pop("model")
    else:
        inherited.pop("model", None)
    missing = [k for k in ("model_name", "api_key", "base_url")
               if not (inherited.get(k) or "").strip()]
    if missing:
        raise ValueError(f"missing {', '.join(missing)} in replay eval config: {config_path}")
    inherited.setdefault("max_tokens", 16384)
    inherited.setdefault("temperature", 0.0)
    return inherited
