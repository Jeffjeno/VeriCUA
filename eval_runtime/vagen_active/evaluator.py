from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from eval_runtime.vagen_active.agent_factory import (
    VAGEN_QWEN_EVALUATOR_TYPE,
    VAGEN_EVALUATOR_TYPE,
    build_vagen_eval_agent,
)
from eval_runtime.vagen_active.env_adapter import VagenEnvAdapter

logger = logging.getLogger("eval_runtime.vagen_active.evaluator")

REPLAY_EVAL_DIR = "replay_eval"
REPLAY_EVAL_SUMMARY_FILE = "eval_summary.json"
VAGEN_EVAL_SUBDIR = "vagen_eval"

def eval_summary_path(example_result_dir: str) -> str:
    return os.path.join(example_result_dir, REPLAY_EVAL_DIR, REPLAY_EVAL_SUMMARY_FILE)

def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_trajectory_from_result_dir(example_result_dir: str) -> dict[str, Any]:

    traj_path = os.path.join(example_result_dir, "traj.jsonl")
    screenshots: list[bytes] = []
    actions: list[dict[str, Any]] = []
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"traj.jsonl not found in {example_result_dir}")

    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = entry.get("action")
            response = entry.get("response") or ""

            parts = []
            resp_text = str(response).strip()
            if resp_text:
                parts.append(f"[TEXT] {resp_text}")
            if action:
                action_tool = str(entry.get("action_tool") or "computer")
                parts.append(f"[TOOL_USE] {action_tool}: {action}")
            raw_response = "\n".join(parts)
            screenshot_file = entry.get("screenshot_file") or ""
            screenshot_path = os.path.join(example_result_dir, screenshot_file)
            if not screenshot_file or not os.path.exists(screenshot_path):
                continue
            with open(screenshot_path, "rb") as sf:
                data = sf.read()
            if not data:
                continue
            screenshots.append(data)
            actions.append({"raw_response": raw_response})

    if not screenshots:
        raise ValueError(f"no replay screenshots found in {example_result_dir}")
    return {"screenshots": screenshots, "actions": actions}

def _map_final_label(result: dict[str, Any], cfg: dict[str, Any]) -> str:
    parse_method = str(result.get("parse_method") or "").strip().lower()
    if parse_method in ("empty", ""):

        return "error"
    if parse_method == "failed":

        return "error"
    if (
        bool(cfg.get("low_confidence_as_uncertain", False))
        and str(result.get("confidence", "")).upper() == "LOW"
    ):
        return "uncertain"
    return "success" if result.get("success") else "failure"

def run_vagen_eval_after_replay(
    *,
    env_client: Any,
    lease_id: str,
    example: dict[str, Any],
    example_result_dir: str,
    domain: str,
    example_id: str,
    rollout_id: str,
    policy_model: str,
    policy_result: float | None,
    args: Any,
) -> dict[str, Any]:

    cfg = getattr(args, "vagen_eval_config", None) or {}
    max_round = int(cfg.get("max_round", 35))
    save_dir = os.path.join(example_result_dir, REPLAY_EVAL_DIR, VAGEN_EVAL_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)

    os.environ["TOOL_CALL_IMG_TEMP"] = os.path.join(save_dir, "tmp_imgs")

    error_message: str | None = None
    result: dict[str, Any] = {}
    try:
        trajectory = load_trajectory_from_result_dir(example_result_dir)

        env_client.prepare_for_probe(lease_id)

        agent = build_vagen_eval_agent(cfg)
        platform = str(cfg.get("platform") or "").strip().lower()
        if platform in {"mobileworld", "webarena", "web_arena"}:
            eval_env = env_client
        else:
            eval_env = VagenEnvAdapter(
                env_client, lease_id, shell_timeout=int(cfg.get("shell_timeout", 120))
            )
        result = agent.eval_task(
            env=eval_env,
            task_config=example,
            trajectory=trajectory,
            save_dir=save_dir,
            max_round=max_round,
        )
    except Exception as exc:
        logger.error(
            "VAGEN eval failed for %s/%s/%s: %s",
            domain, example_id, rollout_id, exc, exc_info=True,
        )
        error_message = str(exc)
        result = {}

    if error_message is not None:
        final_label = "error"
    else:
        final_label = _map_final_label(result, cfg)

    content = result.get("reason") or ""
    decision_raw = result.get("raw_output") or ""
    eval_steps = int(result.get("eval_steps") or 0)
    policy_env_success = policy_result is not None and policy_result >= 1.0
    final_judge_success = final_label == "success"
    judge_matches_policy = (
        final_label in ("success", "failure")
        and final_judge_success == policy_env_success
    )

    summary: dict[str, Any] = {
        "domain": domain,
        "example_id": example_id,
        "rollout_id": str(rollout_id),
        "policy_model": policy_model,
        "eval_model": cfg.get("model") or cfg.get("evaluator_type") or VAGEN_EVALUATOR_TYPE,
        "policy_result": policy_result,
        "final_label": final_label,
        "final_status": final_label,
        "sampler_rounds": 0,
        "rounds": [
            {
                "round": 1,
                "label": final_label,
                "status": final_label,
                "content": content,
                "decision_raw": decision_raw,
                "vagen_confidence": result.get("confidence"),
                "vagen_parse_method": result.get("parse_method"),
            }
        ],
        "created_at": datetime.datetime.now().isoformat(),
        "evaluator_type": cfg.get("evaluator_type") or VAGEN_EVALUATOR_TYPE,
        "vagen_active_baseline": True,
        "judge_records": [
            {"round": 1, "status": final_label, "criteria": content}
        ],
        "final_judge_status": final_label,
        "final_judge_success": final_judge_success,
        "policy_env_score": policy_result,
        "policy_env_success": policy_env_success,
        "judge_matches_policy": judge_matches_policy,
        "probe_steps": eval_steps,
        "probe_rounds": eval_steps,
        "decision_raw": decision_raw,
        "content": content,
        "error_message": error_message,
        "vagen_confidence": result.get("confidence"),
        "vagen_parse_method": result.get("parse_method"),
        "vagen_score": result.get("score"),
        "format_retry_attempts": int(result.get("format_retry_attempts") or 0),
        "vagen_config": {
            "model": cfg.get("model"),
            "max_round": max_round,
            "max_tokens": cfg.get("max_tokens"),
            "only_n_most_recent_images_eval": cfg.get("only_n_most_recent_images_eval"),
            "temperature": cfg.get("temperature"),
            "top_p": cfg.get("top_p"),
            "low_confidence_as_uncertain": cfg.get("low_confidence_as_uncertain"),
            "platform": cfg.get("platform"),
            "format_retry_rounds": cfg.get("format_retry_rounds", 2),
            "active_tool": {
                "mobileworld": "mobile_use",
                "webarena": "browser_use",
                "web_arena": "browser_use",
            }.get(str(cfg.get("platform") or "").strip().lower(), "computer"),
        },
    }
    _write_json(eval_summary_path(example_result_dir), summary)
    logger.info(
        "VAGEN active judge %s/%s/%s: label=%s policy=%s steps=%d",
        domain, example_id, rollout_id, final_label, policy_result, eval_steps,
    )
    return summary
