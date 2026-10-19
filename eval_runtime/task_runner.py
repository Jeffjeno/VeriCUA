from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Any

from eval_runtime.markers import step_or_handle_marker

logger = logging.getLogger("eval_runtime.task_runner")

def setup_logger(example: dict[str, Any], example_result_dir: str) -> logging.Logger:
    runtime_logger = logging.getLogger(f"eval_runtime.task.{example.get('id', 'unknown')}")
    runtime_logger.setLevel(logging.DEBUG)
    os.makedirs(example_result_dir, exist_ok=True)

    log_path = os.path.join(example_result_dir, "runtime.log")

    for h in list(runtime_logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        runtime_logger.removeHandler(h)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    runtime_logger.addHandler(handler)
    return runtime_logger

def _extract_sampler_messages(agent: Any) -> Any:
    candidates = [agent, getattr(agent, "base_agent", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        messages = getattr(candidate, "last_messages", None)
        if messages is not None:
            return messages
    return None

def run_single_example_client(
    agent,
    env_client,
    lease: dict[str, Any],
    example: dict[str, Any],
    max_steps: int,
    instruction: str,
    args: Any,
    example_result_dir: str,
    scores: list,
    time_list: list | None = None,
) -> float:

    runtime_logger = setup_logger(example, example_result_dir)

    if time_list is not None:
        time_list.append(("run_single_example_start", time.time()))

    env_client.reset(lease["lease_id"], task_config=example)
    if time_list is not None:
        time_list.append(("run_single_example_env_reset", time.time()))

    try:
        agent.reset(runtime_logger)
    except Exception:
        agent.reset()
    if time_list is not None:
        time_list.append(("run_single_example_agent_reset", time.time()))

    time.sleep(getattr(args, "sleep_after_reset", 30.0))

    obs = env_client.get_obs(lease["lease_id"])
    if time_list is not None:
        time_list.append(("run_single_example_get_obs_0", time.time()))

    done = False
    step_idx = 0

    while not done and step_idx < max_steps:
        response, actions = agent.predict(instruction, obs, time_list=time_list)
        if time_list is not None:
            time_list.append((f"agent_predict_step_{step_idx + 1}", time.time()))

        if not actions:
            logger.info("Agent returned no actions at step %d; stopping.", step_idx + 1)
            break

        for action_idx, action in enumerate(actions, start=1):
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)

            sleep = getattr(args, "sleep_after_execution", 2.0)
            obs, reward, done, _ = step_or_handle_marker(
                env_client, lease["lease_id"], action, sleep
            )

            if time_list is not None:
                time_list.append((f"env_step_{step_idx + 1}_{action_idx}", time.time()))

            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"), "wb") as f:
                f.write(obs.get("screenshot") or b"")

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                }, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time.sleep(getattr(args, "sleep_before_eval", 20.0))
    if time_list is not None:
        time_list.append(("run_single_example_before_evaluate", time.time()))

    result = env_client.evaluate(lease["lease_id"])
    if time_list is not None:
        time_list.append(("run_single_example_after_evaluate", time.time()))

    logger.info("Result: %.2f", result)
    scores.append(result)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    return result

def run_single_sampler_client(
    sampler_agent,
    env_client,
    lease: dict[str, Any],
    example: dict[str, Any],
    max_steps: int,
    instruction: str,
    args: Any,
    example_result_dir: str,
    scores: list,
    sampler_round: int = 1,
    sampler_context: dict[str, Any] | None = None,
) -> float:

    runtime_logger = setup_logger(example, example_result_dir)

    messages_dir = os.path.join(example_result_dir, "messages")
    os.makedirs(messages_dir, exist_ok=True)
    message_index_path = os.path.join(example_result_dir, "message_traj.jsonl")
    rounds_index_path = os.path.join(example_result_dir, "rounds.jsonl")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")

    try:
        sampler_agent.reset(runtime_logger)
    except Exception:
        sampler_agent.reset()

    obs = env_client.get_obs(lease["lease_id"])

    if hasattr(sampler_agent, "prepare_round"):
        try:
            sampler_agent.prepare_round(sampler_context or {}, instruction=instruction)
        except TypeError:
            sampler_agent.prepare_round(sampler_context or {})

    done = False
    step_idx = 0

    with open(rounds_index_path, "a", encoding="utf-8") as rf:
        rf.write(json.dumps({
            "event": "round_start",
            "round": sampler_round,
            "timestamp": run_timestamp,
            "instruction": instruction,
        }, ensure_ascii=False))
        rf.write("\n")

    while not done and step_idx < max_steps:
        prompt_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
        response, actions = sampler_agent.predict(instruction, obs)

        sampler_messages = _extract_sampler_messages(sampler_agent)
        if sampler_messages is not None:
            message_file_name = f"messages_step_{step_idx + 1}_{prompt_timestamp}.json"
            message_file_path = os.path.join(messages_dir, message_file_name)
            try:
                with open(message_file_path, "w", encoding="utf-8") as mf:
                    json.dump(sampler_messages, mf, ensure_ascii=False)
                with open(message_index_path, "a", encoding="utf-8") as mif:
                    mif.write(json.dumps({
                        "step_num": step_idx + 1,
                        "timestamp": prompt_timestamp,
                        "messages_file": os.path.join("messages", message_file_name),
                    }, ensure_ascii=False))
                    mif.write("\n")
            except Exception as msg_e:
                logger.warning("Failed to save sampler messages at step %d: %s", step_idx + 1, msg_e)

        if not actions:
            logger.info("Sampler returned no actions at step %d; stopping.", step_idx + 1)
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Sampler step %d: %s", step_idx + 1, action)

            sleep = getattr(args, "sleep_after_execution", 2.0)
            obs, _, done, _ = step_or_handle_marker(
                env_client, lease["lease_id"], action, sleep
            )

            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"), "wb") as f:
                f.write(obs.get("screenshot") or b"")

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "done": done,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                }, ensure_ascii=False))
                f.write("\n")

            if done:
                logger.info("Sampler episode done.")
                break
        step_idx += 1

    time.sleep(getattr(args, "sleep_before_eval", 20.0))
    result = env_client.evaluate(lease["lease_id"])
    logger.info("Sampler round %d result: %.2f", sampler_round, result)
    scores.append(result)

    round_end_payload = {
        "event": "round_end",
        "round": sampler_round,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f"),
        "result": result,
    }
    if hasattr(sampler_agent, "get_round_metadata"):
        try:
            round_end_payload["sampler_metadata"] = sampler_agent.get_round_metadata()
        except Exception as metadata_exc:
            round_end_payload["sampler_metadata_error"] = str(metadata_exc)

    with open(rounds_index_path, "a", encoding="utf-8") as rf:
        rf.write(json.dumps(round_end_payload, ensure_ascii=False))
        rf.write("\n")

    return result
