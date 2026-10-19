from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger("eval_runtime.passive_baseline")

PASSIVE_BASELINE_TYPES = frozenset({"zerogui", "distrl", "digirl"})

REPLAY_EVAL_DIR = "replay_eval"
REPLAY_EVAL_SUMMARY_FILE = "eval_summary.json"

_LABEL_TO_SUITE = {
    "yes": "success",
    "no": "failure",
    "success": "success",
    "failure": "failure",
    "uncertain": "uncertain",
    "error": "error",
}

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid passive baseline config: {path}")
    return payload

def peek_evaluator_type(replay_eval_config_path: str) -> str:
    cfg = _load_yaml(replay_eval_config_path)
    return str(
        cfg.get("evaluator_type")
        or cfg.get("judge_evaluator")
        or cfg.get("agent_judge_type")
        or ""
    ).strip().lower()

def build_passive_baseline_config(replay_eval_config_path: str) -> dict[str, Any]:

    cfg = _load_yaml(replay_eval_config_path)
    evaluator_type = peek_evaluator_type(replay_eval_config_path)
    if evaluator_type not in PASSIVE_BASELINE_TYPES:
        raise ValueError(
            f"passive baseline expects evaluator_type in {sorted(PASSIVE_BASELINE_TYPES)}, "
            f"got: {evaluator_type!r}"
        )
    model_name = str(cfg.get("model_name") or "").strip()
    model_block = cfg.get("model")
    if not model_name and isinstance(model_block, dict):
        model_name = str(model_block.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(f"passive baseline config missing model_name: {replay_eval_config_path}")
    return {
        "evaluator_type": evaluator_type,
        "config_path": os.path.abspath(replay_eval_config_path),
        "model_name": model_name,
        "raw": cfg,
    }

def _suite_label(raw_label: Any) -> str:
    key = str(raw_label or "error").strip().lower()
    return _LABEL_TO_SUITE.get(key, "error" if key == "error" else "uncertain")

def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def eval_summary_path(example_result_dir: str) -> str:
    return os.path.join(example_result_dir, REPLAY_EVAL_DIR, REPLAY_EVAL_SUMMARY_FILE)

def write_eval_summary(example_result_dir: str, payload: dict[str, Any]) -> None:
    _write_json(eval_summary_path(example_result_dir), payload)

def run_passive_baseline_eval_after_replay(
    *,
    example: dict[str, Any],
    example_result_dir: str,
    domain: str,
    example_id: str,
    rollout_id: str,
    policy_model: str,
    policy_result: float | None,
    args: Any,
) -> dict[str, Any]:

    from mm_agents.replay_eval_agent import ReplayEvalAgent

    cfg = getattr(args, "passive_baseline_config", None) or {}
    config_path = cfg.get("config_path") or args.replay_eval_config_path
    evaluator_type = cfg.get("evaluator_type") or peek_evaluator_type(config_path)

    agent = ReplayEvalAgent(config_path=config_path)
    round_summary = agent.evaluate_rollout(
        example=example,
        example_result_dir=example_result_dir,
        policy_model=policy_model,
        domain=domain,
        example_id=example_id,
        rollout_id=str(rollout_id),
        ground_truth=(
            None
            if policy_result is None
            else bool(policy_result >= 1.0)
        ),
        include_sampler=False,
    )

    raw_label = round_summary.get("label", "error")
    suite_label = _suite_label(raw_label)
    policy_env_success = policy_result is not None and policy_result >= 1.0
    final_judge_success = suite_label == "success"

    rounds = [
        {
            "round": 1,
            "label": suite_label,
            "status": suite_label,
            "content": round_summary.get("content"),
            "decision_raw": round_summary.get("decision_raw"),
            "raw_agent_label": raw_label,
        }
    ]
    summary: dict[str, Any] = {
        "domain": domain,
        "example_id": example_id,
        "rollout_id": str(rollout_id),
        "policy_model": policy_model,
        "eval_model": cfg.get("model_name") or agent.model_name,
        "policy_result": policy_result,
        "final_label": suite_label,
        "final_status": suite_label,
        "sampler_rounds": 0,
        "rounds": rounds,
        "created_at": datetime.datetime.now().isoformat(),
        "evaluator_type": evaluator_type,
        "passive_baseline": True,
        "judge_records": [
            {
                "round": 1,
                "status": suite_label,
                "criteria": round_summary.get("content") or "",
            }
        ],
        "final_judge_status": suite_label,
        "final_judge_success": final_judge_success,
        "policy_env_score": policy_result,
        "policy_env_success": policy_env_success,
        "judge_matches_policy": final_judge_success == policy_env_success,
        "probe_steps": 0,
        "probe_rounds": 0,
        "decision_raw": round_summary.get("decision_raw"),
        "content": round_summary.get("content"),
        "error_message": round_summary.get("error_message"),
        "duration": round_summary.get("duration"),
        "reward_model": round_summary.get("reward_model"),
    }
    write_eval_summary(example_result_dir, summary)
    logger.info(
        "Passive baseline (%s) %s/%s/%s: label=%s policy=%s",
        evaluator_type,
        domain,
        example_id,
        rollout_id,
        suite_label,
        policy_result,
    )
    return summary
