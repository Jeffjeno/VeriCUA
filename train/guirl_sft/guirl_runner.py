from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Any

logger = logging.getLogger("train.guirl_sft.guirl_runner")

def run_guirl_for_task(
    *,
    env_client: Any,
    lease_id: str,
    instruction: str,
    traj_dir: str,
    policy_score: float,
    guirl_config: dict[str, Any],
    max_eval_rounds: int = 3,
    sleep_after_execution: float = 2.0,
) -> dict[str, Any]:

    from eval_runtime.guirl.judge import (
        JUDGE_SYSTEM_PROMPT,
        build_judge_user_content,
        parse_judge_output,
    )
    from eval_runtime.guirl.evaluator import (
        call_guirl_judge,
        load_policy_replay_for_guirl,
        make_guirl_probe_agent,
        get_probe_round_runner,
        prepare_env_for_probe,
    )

    replay_actions, replay_screenshots = load_policy_replay_for_guirl(traj_dir)
    if not replay_screenshots:
        try:
            import base64

            obs = env_client.get_obs(lease_id)
            replay_screenshots = [base64.b64encode(obs["screenshot"]).decode("utf-8")]
        except Exception:
            return _empty_result(policy_score, "no_screenshots")

    history_n = guirl_config.get("judge_history_n", 3)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]}
    ]

    user_content_r0 = build_judge_user_content(
        task=instruction,
        task_traj_imgs=replay_screenshots,
        prev_criteria=None,
        action_history=replay_actions,
        history_n=history_n,
    )
    messages.append({"role": "user", "content": user_content_r0})

    raw_response = _call_judge_with_retry(messages, guirl_config, traj_dir=traj_dir, round_idx=0)
    if raw_response is None:
        return _empty_result(policy_score, "judge_call_failed_r0")
    messages.append(_make_assistant_msg(raw_response, guirl_config))

    try:
        parsed = parse_judge_output(raw_response)
        criteria = parsed.criteria_text
        status = parsed.status
    except ValueError as exc:
        logger.warning("Judge parse failed (round 0): %s", exc)
        return _build_result(messages, "parse_failure", policy_score, 1)

    logger.info(
        "Judge round 0: status=%s criteria_len=%d traj_dir=%s",
        status, len(criteria), traj_dir,
    )

    probe_errors: list[dict[str, Any]] = []
    judge_round = 1
    while status == "uncertain" and judge_round < max_eval_rounds:
        prepare_env_for_probe(env_client, lease_id)
        logger.info(
            "Starting probe round %d (judge status=%s, max_eval_rounds=%d)",
            judge_round, status, max_eval_rounds,
        )
        try:
            probe_agent = make_guirl_probe_agent(criteria, guirl_config)
            run_probe_round = get_probe_round_runner(guirl_config)
            probe_agent.reset_for_round(criteria, logger)

            probe_result = run_probe_round(
                probe_agent=probe_agent,
                env_client=env_client,
                lease_id=lease_id,
                instruction=instruction,
                max_steps=guirl_config.get("probe_max_steps", 15),
                example_result_dir=traj_dir,
                sleep_after_execution=sleep_after_execution,
                sampler_round=judge_round,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            err_record = {
                "round": judge_round,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": tb,
            }
            probe_errors.append(err_record)
            _write_probe_error(traj_dir, err_record)
            logger.error(
                "Probe round %d failed (%s): %s\n%s",
                judge_round, type(exc).__name__, exc, tb,
            )
            break

        logger.info(
            "Probe round %d complete: steps=%s done=%s",
            judge_round,
            probe_result.get("steps"),
            probe_result.get("done"),
        )

        probe_latest = probe_agent.latest_criteria
        judge_round += 1

        user_content_rN = build_judge_user_content(
            task=instruction,
            task_traj_imgs=replay_screenshots,
            prev_criteria=probe_latest,
            action_history=None,
            history_n=history_n,
            criteria_option_screenshot=probe_agent.criteria_images,
        )
        messages.append({"role": "user", "content": user_content_rN})

        raw_response = _call_judge_with_retry(messages, guirl_config, traj_dir=traj_dir, round_idx=judge_round)
        if raw_response is None:
            break
        messages.append(_make_assistant_msg(raw_response, guirl_config))

        try:
            parsed = parse_judge_output(raw_response)
            criteria = parsed.criteria_text
            status = parsed.status
        except ValueError as exc:
            logger.warning("Judge parse failed (round %d): %s", judge_round, exc)
            status = "parse_failure"
            break

        if probe_result.get("done"):
            break

    S_prime: float | None = None
    try:
        S_prime = env_client.evaluate(lease_id)
    except Exception as exc:
        logger.warning("Post-probe evaluate() failed: %s", exc)

    return _build_result(
        messages, status, policy_score, judge_round,
        S_prime=S_prime, probe_errors=probe_errors,
    )

def _call_judge_with_retry(
    messages: list[dict[str, Any]],
    guirl_config: dict[str, Any],
    max_retries: int = 3,
    traj_dir: str | None = None,
    round_idx: int = 0,
) -> str | None:

    from eval_runtime.guirl.evaluator import call_guirl_judge
    from eval_runtime.guirl.judge import parse_judge_output

    last_raw: str | None = None
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = call_guirl_judge(messages, guirl_config)
            last_raw = raw
            parse_judge_output(raw)
            return raw
        except Exception as exc:
            last_exc = exc
            logger.warning("Judge call/parse attempt %d/%d: %s", attempt + 1, max_retries, exc)

    if traj_dir and last_raw is not None:
        try:
            debug_dir = os.path.join(traj_dir, "replay_eval", "guirl_judge")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"round_{round_idx:02d}_failed_response.txt"), "w") as f:
                f.write(last_raw)
            with open(os.path.join(debug_dir, f"round_{round_idx:02d}_failed_error.txt"), "w") as f:
                f.write(repr(last_exc))
        except Exception:
            pass
    return None

def _make_assistant_msg(content: str, guirl_config: dict[str, Any]) -> dict[str, Any]:

    return {"role": "assistant", "content": content}

def _write_probe_error(traj_dir: str, err_record: dict[str, Any]) -> None:

    probe_dir = os.path.join(traj_dir, "replay_eval", "guirl_probe")
    os.makedirs(probe_dir, exist_ok=True)
    error_path = os.path.join(probe_dir, "probe_error.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(err_record, f, ensure_ascii=False, indent=2)

def _build_result(
    messages: list[dict[str, Any]],
    judge_final: str,
    S: float,
    num_rounds: int,
    S_prime: float | None = None,
    probe_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reward = _compute_reward(judge_final, S, S_prime)
    result: dict[str, Any] = {
        "messages": messages,
        "reward": reward,
        "judge_final": judge_final,
        "num_rounds": num_rounds,
        "S": S,
        "S_prime": S_prime,
    }
    if probe_errors:
        result["probe_errors"] = probe_errors
    return result

def _empty_result(S: float, reason: str) -> dict[str, Any]:
    return {
        "messages": [], "reward": 0.0, "judge_final": "error",
        "num_rounds": 0, "S": S, "S_prime": None, "error": reason,
    }

def _compute_reward(judge_final: str, S: float | None, S_prime: float | None = None) -> float:

    if judge_final in ("parse_failure", "uncertain", "error"):
        return 0.0
    if S is None:
        return 0.0
    s_label = "success" if float(S) >= 1.0 else "failure"
    reward = 1.0 if judge_final == s_label else 0.0
    if S_prime is not None and float(S) != float(S_prime):
        reward *= 0.5
    return reward
