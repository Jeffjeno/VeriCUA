from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
from typing import Any, Callable

import yaml

from eval_runtime.guirl.judge import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_messages,
    build_judge_user_content,
    parse_judge_output,
)
from eval_runtime.guirl.sft_export import maybe_export_sft_from_eval_summary
from eval_runtime.llm_rate_limit import throttle_shared_llm_request

logger = logging.getLogger("eval_runtime.guirl")

def prepare_env_for_probe(env_client: Any, lease_id: str | None = None) -> None:

    prepare = getattr(env_client, "prepare_for_probe", None)
    if callable(prepare):
        prepare(lease_id)

REPLAY_EVAL_DIR = "replay_eval"
REPLAY_EVAL_SUMMARY_FILE = "eval_summary.json"
GUIRL_JUDGE_PARSE_MAX_RETRIES = 3
THINKING_BLOCK_RE = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
CONTEXT_CRITERIA_VERDICT_RE = re.compile(
    r"(^\s*Status\s*:|</?criteria>|:\s*(?:success|failure|uncertain)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

ProbeEntry = tuple[type, Callable[..., dict[str, Any]]]
_PROBE_REGISTRY: dict[str, ProbeEntry] = {}

def _load_yaml_config_strict(config_path: str, config_label: str) -> dict[str, Any]:
    if not config_path:
        raise ValueError(f"{config_label} config path is empty")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_label} config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid {config_label} config format: {config_path}")
    return cfg

def _extract_replay_eval_model_config(
    replay_eval_cfg: dict[str, Any],
    replay_eval_config_path: str,
) -> dict[str, Any]:
    model_cfg = replay_eval_cfg.get("model")
    inherited = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    for key in (
        "model_name",
        "model",
        "api_key",
        "base_url",
        "max_retry",
        "retry_waiting_seconds",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "max_pixels",
        "enable_thinking",
    ):
        if key in replay_eval_cfg and key not in inherited:
            inherited[key] = replay_eval_cfg[key]

    if "model" in inherited and not inherited.get("model_name"):
        inherited["model_name"] = inherited.pop("model")
    else:
        inherited.pop("model", None)

    missing = []
    for key in ("model_name", "api_key", "base_url"):
        value = inherited.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
    if missing:
        raise ValueError(
            f"missing {', '.join(missing)} in replay eval config: "
            f"{replay_eval_config_path}"
        )
    inherited.setdefault("max_tokens", 16384)
    inherited.setdefault("temperature", 0.0)
    return inherited

def build_guirl_config(replay_eval_config_path: str) -> dict[str, Any]:

    cfg = _load_yaml_config_strict(replay_eval_config_path, "replay eval")
    model_cfg = _extract_replay_eval_model_config(cfg, replay_eval_config_path)
    include_context_examples = cfg.get("judge_include_context_criteria_examples", False)
    if not isinstance(include_context_examples, bool):
        raise ValueError("judge_include_context_criteria_examples must be true or false")

    raw_context_examples = cfg.get("judge_context_criteria_examples", [])
    if raw_context_examples is None:
        raw_context_examples = []
    if not isinstance(raw_context_examples, list):
        raise ValueError("judge_context_criteria_examples must be a list")
    context_examples: list[dict[str, str]] = []
    config_dir = os.path.dirname(os.path.abspath(replay_eval_config_path))
    for idx, raw_example in enumerate(raw_context_examples, start=1):
        if not isinstance(raw_example, dict):
            raise ValueError(
                f"judge_context_criteria_examples[{idx}] must be a mapping with "
                "task and criteria, with an optional screenshot_path"
            )
        allowed_keys = {"task", "criteria", "screenshot_path"}
        required_keys = {"task", "criteria"}
        if not required_keys.issubset(raw_example) or not set(raw_example).issubset(
            allowed_keys
        ):
            raise ValueError(
                f"judge_context_criteria_examples[{idx}] must contain task and "
                "criteria, and may contain only screenshot_path as an optional key"
            )
        task = raw_example["task"]
        criteria = raw_example["criteria"]
        screenshot_path = raw_example.get("screenshot_path")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                f"judge_context_criteria_examples[{idx}].task must be a non-empty string"
            )
        if not isinstance(criteria, str) or not criteria.strip():
            raise ValueError(
                f"judge_context_criteria_examples[{idx}].criteria must be a non-empty string"
            )
        if CONTEXT_CRITERIA_VERDICT_RE.search(criteria):
            raise ValueError(
                f"judge_context_criteria_examples[{idx}].criteria must contain criteria text only, "
                "without verdicts or criteria/status output tags"
            )
        example = {
            "task": task.strip(),
            "criteria": criteria.strip(),
        }
        if screenshot_path is not None:
            if not isinstance(screenshot_path, str) or not screenshot_path.strip():
                raise ValueError(
                    f"judge_context_criteria_examples[{idx}].screenshot_path must be "
                    "a non-empty string when provided"
                )
            resolved_screenshot_path = os.path.abspath(
                os.path.join(config_dir, screenshot_path)
            )
            if not os.path.isfile(resolved_screenshot_path):
                raise FileNotFoundError(
                    f"judge_context_criteria_examples[{idx}].screenshot_path not found: "
                    f"{resolved_screenshot_path}"
                )
            example["screenshot_path"] = resolved_screenshot_path
        context_examples.append(example)
    if include_context_examples and not 2 <= len(context_examples) <= 3:
        raise ValueError(
            "judge_context_criteria_examples must contain 2 or 3 examples when enabled"
        )

    return {
        "evaluator_type": str(cfg.get("evaluator_type") or "").strip().lower(),
        "probe_agent_type": str(cfg.get("probe_agent_type", "Qwen3VL_Probe")).strip(),
        "judge_model_name": model_cfg["model_name"],
        "judge_api_key": model_cfg.get("api_key", "EMPTY"),
        "judge_base_url": model_cfg["base_url"],
        "judge_max_tokens": int(model_cfg.get("max_tokens", cfg.get("judge_max_tokens", 120000))),
        "judge_temperature": float(model_cfg.get("temperature", cfg.get("judge_temperature", 0.0))),
        "judge_top_k": (
            int(model_cfg["top_k"]) if model_cfg.get("top_k") is not None else None
        ),
        "judge_enable_thinking": bool(model_cfg.get("enable_thinking", cfg.get("judge_enable_thinking", False))),
        "judge_strip_history_thinking": bool(
            cfg.get("judge_strip_history_thinking", cfg.get("strip_history_thinking", True))
        ),
        "judge_history_n": int(cfg.get("history_n", cfg.get("judge_history_n", 3))),
        "judge_include_context_criteria_examples": include_context_examples,
        "judge_context_criteria_examples": context_examples,
        "probe_model_name": str(cfg.get("probe_model_name") or model_cfg["model_name"]),
        "probe_api_key": str(cfg.get("probe_api_key") or model_cfg.get("api_key", "EMPTY")),
        "probe_base_url": str(cfg.get("probe_base_url") or model_cfg["base_url"]),
        "probe_max_tokens": int(cfg.get("probe_max_tokens", 32768)),
        "probe_temperature": float(cfg.get("probe_temperature", 0.0)),
        "probe_history_n": int(cfg.get("probe_history_n", 4)),
        "probe_max_steps": int(cfg.get("probe_max_steps", 15)),
        "probe_enable_thinking": bool(cfg.get("probe_enable_thinking", False)),
        "probe_reasoning_effort": str(cfg.get("probe_reasoning_effort", "medium")),
        "probe_max_trajectory_length": int(cfg.get("probe_max_trajectory_length", 20)),
        "probe_max_images": int(cfg.get("probe_max_images", 3)),
        "probe_client_password": str(cfg.get("probe_client_password", "password")),
    }

def _sanitize_messages_for_disk(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    def _sanitize_part(part: Any) -> Any:
        if not isinstance(part, dict):
            return part
        out = dict(part)
        if out.get("type") == "image_url":
            image_url = out.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "")
                out["image_url"] = {
                    **image_url,
                    "url": (
                        f"<omitted data-url bytes≈{len(url)}>"
                        if url.startswith("data:")
                        else (url if len(url) < 512 else url[:64] + f"...<{len(url)}>")
                    ),
                }
            else:
                out["image_url"] = "<omitted>"
        elif out.get("type") == "image":

            for key in ("source", "data", "image"):
                if key in out and isinstance(out[key], (str, dict)):
                    out[key] = "<omitted>"
        return out

    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        item = dict(msg)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_sanitize_part(p) for p in content]
        elif isinstance(content, str) and len(content) > 200_000:
            item["content"] = content[:2000] + f"...<omitted {len(content)} chars>"
        sanitized.append(item)
    return sanitized

def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _write_json_best_effort(path: str, payload: dict[str, Any]) -> None:

    try:
        _write_json(path, payload)
    except OSError as exc:
        logger.warning("Failed to write debug JSON %s: %s", path, exc)

def _write_bytes(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)

def _b64_png(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def _load_context_criteria_examples(
    configured_examples: list[dict[str, str]],
) -> list[dict[str, str]]:

    examples: list[dict[str, str]] = []
    for example in configured_examples:
        loaded_example = {
            "task": example["task"],
            "criteria": example["criteria"],
        }
        screenshot_path = example.get("screenshot_path")
        if screenshot_path:
            with open(screenshot_path, "rb") as f:
                loaded_example["screenshot_b64"] = _b64_png(f.read())
        examples.append(loaded_example)
    return examples

def _data_image_debug_ref(url: str) -> str:
    if not url.startswith("data:image/"):
        return url
    base64_part = url.split(",", 1)[1] if "," in url else url
    return f"<data_image_base64_prefix:{base64_part[:10]} len={len(url)}>"

def _sanitize_eval_summary_payload(obj: Any) -> Any:
    if isinstance(obj, str):
        return _data_image_debug_ref(obj)
    if isinstance(obj, list):
        return [_sanitize_eval_summary_payload(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _sanitize_eval_summary_payload(value) for key, value in obj.items()}
    return obj

def eval_summary_path(example_result_dir: str) -> str:
    return os.path.join(example_result_dir, REPLAY_EVAL_DIR, REPLAY_EVAL_SUMMARY_FILE)

def write_eval_summary(example_result_dir: str, payload: dict[str, Any]) -> None:
    _write_json(eval_summary_path(example_result_dir), _sanitize_eval_summary_payload(payload))

def parse_policy_result(policy_result: Any) -> float | None:
    try:
        return float(policy_result)
    except Exception:
        return None

def load_policy_replay_for_guirl(example_result_dir: str) -> tuple[list[str], list[str]]:
    traj_path = os.path.join(example_result_dir, "traj.jsonl")
    replay_actions: list[str] = []
    replay_screenshots: list[str] = []
    if not os.path.exists(traj_path):
        return replay_actions, replay_screenshots
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "step_num" not in record:
                continue
            action = record.get("action")
            if action is not None:
                replay_actions.append(str(action))
            screenshot_file = record.get("screenshot_file")
            if screenshot_file:
                screenshot_path = os.path.join(example_result_dir, screenshot_file)
                if os.path.exists(screenshot_path):
                    with open(screenshot_path, "rb") as img_f:
                        replay_screenshots.append(_b64_png(img_f.read()))
    return replay_actions, replay_screenshots

def _normalize_judge_response_for_history(content: str, reasoning_content: str = "") -> str:

    content = content or ""
    reasoning_content = (reasoning_content or "").strip()
    if reasoning_content:
        if "<think>" in content and "</think>" in content:
            return content
        return f"<think>\n{reasoning_content}\n</think>\n\n{content.lstrip()}"

    stripped = content.lstrip()
    if "</think>" in stripped and "<think>" not in stripped.split("</think>", 1)[0]:
        return f"<think>\n{stripped}"
    return content

def strip_thinking_for_history(content: str) -> str:

    stripped = THINKING_BLOCK_RE.sub("", content or "")
    return stripped.lstrip("\n")

def _strip_latest_assistant_thinking(messages: list[dict[str, Any]]) -> None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = strip_thinking_for_history(str(msg.get("content") or ""))
            return

def _strip_historical_user_images(messages: list[dict[str, Any]]) -> int:

    removed = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept: list[Any] = []
        message_removed = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image"}:
                removed += 1
                message_removed += 1
                continue
            kept.append(part)
        if message_removed:
            kept.append(
                {
                    "type": "text",
                    "text": "Historical screenshots omitted; use the current round evidence.",
                }
            )
        message["content"] = kept
    return removed

def call_guirl_judge(messages: list[dict[str, Any]], guirl_config: dict[str, Any]) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=guirl_config["judge_api_key"],
        base_url=guirl_config["judge_base_url"],
    )
    params: dict[str, Any] = {
        "model": guirl_config["judge_model_name"],
        "messages": messages,
        "max_tokens": guirl_config["judge_max_tokens"],
        "temperature": guirl_config["judge_temperature"],
    }
    params["extra_body"] = {
        "chat_template_kwargs": {
            "enable_thinking": guirl_config["judge_enable_thinking"],
        }
    }
    if guirl_config.get("judge_top_k") is not None:
        params["extra_body"]["top_k"] = guirl_config["judge_top_k"]
    throttle_shared_llm_request()
    response = client.chat.completions.create(**params)
    choice = response.choices[0].message
    content = choice.content or ""
    reasoning_content = getattr(choice, "reasoning_content", None) or ""
    return _normalize_judge_response_for_history(content, reasoning_content)

def run_guirl_judge_round(
    *,
    round_idx: int,
    instruction: str,
    replay_screenshots: list[str],
    prev_criteria: str | None,
    action_history: list[str] | str | None,
    example_result_dir: str,
    guirl_config: dict[str, Any],
    criteria_option_screenshot: Any | None = None,
) -> tuple[str, str]:
    messages = build_judge_messages(
        task=instruction,
        task_traj_imgs=replay_screenshots,
        prev_criteria=prev_criteria,
        action_history=action_history,
        history_n=guirl_config["judge_history_n"],
        criteria_option_screenshot=criteria_option_screenshot,
    )
    judge_dir = os.path.join(example_result_dir, REPLAY_EVAL_DIR, "guirl_judge")
    msg_path = os.path.join(judge_dir, f"round_{round_idx:02d}_messages.json")
    _write_json_best_effort(
        msg_path, {"messages": _sanitize_messages_for_disk(messages)}
    )

    last_exc: Exception | None = None
    for attempt in range(1, GUIRL_JUDGE_PARSE_MAX_RETRIES + 1):
        raw = call_guirl_judge(messages, guirl_config)
        suffix = "" if attempt == 1 else f"_attempt{attempt}"
        out_path = os.path.join(judge_dir, f"round_{round_idx:02d}{suffix}_response.txt")
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(raw)
        except OSError as exc:
            logger.warning("Failed to write judge response %s: %s", out_path, exc)
        try:
            parsed = parse_judge_output(raw)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "GUIRL judge parse failed (round %d, attempt %d/%d): %s",
                round_idx,
                attempt,
                GUIRL_JUDGE_PARSE_MAX_RETRIES,
                exc,
            )
            if attempt >= GUIRL_JUDGE_PARSE_MAX_RETRIES:
                _write_json_best_effort(
                    os.path.join(judge_dir, f"round_{round_idx:02d}_parse_error.json"),
                    {
                        "error": repr(exc),
                        "raw_response_path": os.path.relpath(out_path, example_result_dir),
                        "messages_path": os.path.relpath(msg_path, example_result_dir),
                        "raw_response": raw,
                        "attempts": attempt,
                    },
                )
                raise
            continue
        _write_json_best_effort(
            os.path.join(judge_dir, f"round_{round_idx:02d}_parsed.json"),
            {
                "criteria_text": parsed.criteria_text,
                "status": parsed.status,
                "raw_response_path": os.path.relpath(out_path, example_result_dir),
                "attempts": attempt,
            },
        )
        return parsed.criteria_text, parsed.status

    raise last_exc or RuntimeError(f"Judge parse retry exhausted for round {round_idx}")

def _resolve_probe_entry(probe_agent_type: str) -> ProbeEntry:
    if probe_agent_type in _PROBE_REGISTRY:
        return _PROBE_REGISTRY[probe_agent_type]

    if probe_agent_type == "Holo31_Probe":
        from mm_agents.holo31_probe_module import (
            Holo31ProbeAgent,
            run_guirl_probe_round_client,
        )

        entry: ProbeEntry = (Holo31ProbeAgent, run_guirl_probe_round_client)
    elif probe_agent_type == "Holo3_Probe":
        from mm_agents.holo3_probe_module import (
            Holo3ProbeAgent,
            run_guirl_probe_round_client,
        )

        entry = (Holo3ProbeAgent, run_guirl_probe_round_client)
    elif probe_agent_type == "Qwen3VL_Probe":
        from mm_agents.qwen3vl_agent_probe_module import (
            Qwen3VLAgent,
            run_guirl_probe_round_client,
        )

        entry = (Qwen3VLAgent, run_guirl_probe_round_client)
    elif probe_agent_type in (
        "MobileWorldQwen3VL_Probe",
        "MobileWorldGUIOwl_Probe",
    ):
        from eval_runtime.mobile_world.native_probe import (
            MobileWorldGUIOwlProbeAgent,
            MobileWorldQwen3VLProbeAgent,
            run_mobile_world_guirl_probe_round,
        )

        probe_cls = (
            MobileWorldQwen3VLProbeAgent
            if probe_agent_type == "MobileWorldQwen3VL_Probe"
            else MobileWorldGUIOwlProbeAgent
        )
        entry = (probe_cls, run_mobile_world_guirl_probe_round)
    elif probe_agent_type == "WebArenaHolo_Probe":
        from eval_runtime.web_arena.holo_probe import (
            WebArenaHoloProbeAgent,
            run_webarena_guirl_probe_round,
        )

        entry = (WebArenaHoloProbeAgent, run_webarena_guirl_probe_round)
    else:
        raise ValueError(
            f"Unknown probe_agent_type '{probe_agent_type}'. "
            f"Available: {sorted([*list(_PROBE_REGISTRY.keys()), 'Holo3_Probe', 'Holo31_Probe', 'Qwen3VL_Probe', 'MobileWorldQwen3VL_Probe', 'MobileWorldGUIOwl_Probe', 'WebArenaHolo_Probe'])}"
        )

    _PROBE_REGISTRY[probe_agent_type] = entry
    return entry

def make_guirl_probe_agent(criteria: str, guirl_config: dict[str, Any]):
    probe_agent_type = guirl_config.get("probe_agent_type", "Qwen3VL_Probe")
    probe_cls, _ = _resolve_probe_entry(probe_agent_type)

    model_kwargs = {
        "model_name": guirl_config["probe_model_name"],
        "api_key": guirl_config["probe_api_key"],
        "base_url": guirl_config["probe_base_url"],
        "max_tokens": guirl_config["probe_max_tokens"],
        "temperature": guirl_config["probe_temperature"],
    }

    common_kwargs: dict[str, Any] = {
        "model": model_kwargs,
        "platform": "ubuntu",
        "action_space": "pyautogui",
        "observation_type": "screenshot",
        "enable_thinking": guirl_config["probe_enable_thinking"],
        "criteria": criteria,
        "enable_log": False,
    }

    if probe_agent_type in ("Holo31_Probe", "Holo3_Probe", "WebArenaHolo_Probe"):
        common_kwargs["client_password"] = guirl_config.get("probe_client_password", "password")
        common_kwargs["reasoning_effort"] = guirl_config.get("probe_reasoning_effort", "medium")
        common_kwargs["max_trajectory_length"] = guirl_config.get("probe_max_trajectory_length", 20)
        common_kwargs["max_images"] = guirl_config.get("probe_max_images", 3)
    elif probe_agent_type in (
        "MobileWorldQwen3VL_Probe",
        "MobileWorldGUIOwl_Probe",
    ):
        common_kwargs["history_n"] = guirl_config["probe_history_n"]
        common_kwargs["max_evidence_images"] = guirl_config.get("probe_max_images", 3)
    else:
        common_kwargs["coordinate_type"] = "relative"
        common_kwargs["api_backend"] = "openai"
        common_kwargs["history_n"] = guirl_config["probe_history_n"]

    agent = probe_cls(**common_kwargs)
    agent.reset(logger)
    return agent

def get_probe_round_runner(guirl_config: dict[str, Any]):
    probe_agent_type = guirl_config.get("probe_agent_type", "Qwen3VL_Probe")
    _, runner_fn = _resolve_probe_entry(probe_agent_type)
    return runner_fn

def make_eval_summary(
    *,
    domain: str,
    example_id: str,
    rollout_id: str,
    policy_model: str,
    eval_model: str,
    policy_result: float | None,
    rounds: list[dict[str, Any]],
    sampler_rounds: int,
) -> dict[str, Any]:
    final_round = rounds[-1] if rounds else {}
    return {
        "domain": domain,
        "example_id": example_id,
        "rollout_id": str(rollout_id),
        "policy_model": policy_model,
        "eval_model": eval_model,
        "policy_result": policy_result,
        "final_label": final_round.get("label", "error"),
        "final_status": final_round.get("status"),
        "sampler_rounds": sampler_rounds,
        "rounds": rounds,
        "created_at": datetime.datetime.now().isoformat(),
    }

def _call_judge_with_retry(
    messages: list[dict[str, Any]],
    guirl_config: dict[str, Any],
    round_idx: int,
    example_result_dir: str,
) -> tuple[str, str, str]:

    judge_dir = os.path.join(example_result_dir, REPLAY_EVAL_DIR, "guirl_judge")
    msg_path = os.path.join(judge_dir, f"round_{round_idx:02d}_messages.json")

    _write_json_best_effort(
        msg_path, {"messages": _sanitize_messages_for_disk(messages)}
    )

    last_exc: Exception | None = None
    for attempt in range(1, GUIRL_JUDGE_PARSE_MAX_RETRIES + 1):
        raw = call_guirl_judge(messages, guirl_config)
        suffix = "" if attempt == 1 else f"_attempt{attempt}"
        out_path = os.path.join(judge_dir, f"round_{round_idx:02d}{suffix}_response.txt")
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(raw)
        except OSError as exc:
            logger.warning("Failed to write judge response %s: %s", out_path, exc)
        try:
            parsed = parse_judge_output(raw)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "GUIRL judge parse failed (round %d, attempt %d/%d): %s",
                round_idx, attempt, GUIRL_JUDGE_PARSE_MAX_RETRIES, exc,
            )
            if attempt >= GUIRL_JUDGE_PARSE_MAX_RETRIES:
                _write_json_best_effort(
                    os.path.join(judge_dir, f"round_{round_idx:02d}_parse_error.json"),
                    {
                        "error": repr(exc),
                        "raw_response_path": os.path.relpath(out_path, example_result_dir),
                        "messages_path": os.path.relpath(msg_path, example_result_dir),
                        "raw_response": raw,
                        "attempts": attempt,
                    },
                )
                raise
            continue
        _write_json_best_effort(
            os.path.join(judge_dir, f"round_{round_idx:02d}_parsed.json"),
            {
                "criteria_text": parsed.criteria_text,
                "status": parsed.status,
                "raw_response_path": os.path.relpath(out_path, example_result_dir),
                "attempts": attempt,
            },
        )
        return parsed.criteria_text, parsed.status, raw

    raise last_exc or RuntimeError(f"Judge parse retry exhausted for round {round_idx}")

def run_guirl_eval_after_replay(
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
    guirl_config = args.guirl_config
    replay_actions, replay_screenshots = load_policy_replay_for_guirl(example_result_dir)
    current_obs = env_client.get_obs(lease_id)
    if not replay_screenshots:
        replay_screenshots = [_b64_png(current_obs["screenshot"])]
    _write_bytes(
        os.path.join(example_result_dir, REPLAY_EVAL_DIR, "guirl_after_replay.png"),
        current_obs["screenshot"],
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]}
    ]

    user_content_r0 = build_judge_user_content(
        task=example["instruction"],
        task_traj_imgs=replay_screenshots,
        prev_criteria=None,
        action_history=replay_actions,
        history_n=guirl_config["judge_history_n"],
        context_criteria_examples=(
            _load_context_criteria_examples(
                guirl_config["judge_context_criteria_examples"]
            )
            if guirl_config.get("judge_include_context_criteria_examples", False)
            else None
        ),
    )
    messages.append({"role": "user", "content": user_content_r0})
    criteria, status, raw_response = _call_judge_with_retry(
        messages, guirl_config, round_idx=1, example_result_dir=example_result_dir,
    )
    messages.append({"role": "assistant", "content": raw_response})

    judge_records = [{"round": 1, "status": status, "criteria": criteria}]
    probe_agent = make_guirl_probe_agent(criteria, guirl_config)
    run_probe_round = get_probe_round_runner(guirl_config)

    probe_round = 0
    probe_step = 0
    judge_round = 1
    probe_round_records: list[dict[str, Any]] = []
    judge_criteria_history: list[str] = [criteria]
    probe_updated_criteria_per_round: list[str] = []
    all_probe_evidence: list[dict[str, Any]] = []

    while status == "uncertain" and judge_round < args.max_eval_rounds:
        if guirl_config.get("judge_strip_history_thinking", True):
            _strip_latest_assistant_thinking(messages)

        removed_history_images = _strip_historical_user_images(messages)
        if removed_history_images:
            logger.debug(
                "Removed %d historical judge image(s) before round %d",
                removed_history_images,
                judge_round + 1,
            )

        prepare_env_for_probe(env_client, lease_id)
        probe_round += 1
        probe_agent.reset_for_round(criteria, logger)

        probe_result = run_probe_round(
            probe_agent=probe_agent,
            env_client=env_client,
            lease_id=lease_id,
            instruction=example["instruction"],
            max_steps=guirl_config["probe_max_steps"],
            example_result_dir=example_result_dir,
            sleep_after_execution=args.sleep_after_execution,
            sampler_round=probe_round,
        )
        current_obs = probe_result["obs"]
        probe_step += int(probe_result.get("steps") or 0)
        probe_latest = probe_agent.latest_criteria
        probe_updated_criteria_per_round.append(probe_latest)
        probe_round_records.append({key: value for key, value in probe_result.items() if key != "obs"})
        all_probe_evidence.extend(probe_agent.criteria_images)

        _write_bytes(
            os.path.join(
                example_result_dir,
                REPLAY_EVAL_DIR,
                f"guirl_after_probe_round_{probe_round:02d}.png",
            ),
            current_obs["screenshot"],
        )
        judge_round += 1

        user_content_rN = build_judge_user_content(
            task=example["instruction"],
            task_traj_imgs=replay_screenshots,
            prev_criteria=probe_latest,
            action_history=None,
            history_n=guirl_config["judge_history_n"],
            criteria_option_screenshot=probe_agent.criteria_images,
        )
        messages.append({"role": "user", "content": user_content_rN})
        criteria, status, raw_response = _call_judge_with_retry(
            messages, guirl_config, round_idx=judge_round,
            example_result_dir=example_result_dir,
        )
        messages.append({"role": "assistant", "content": raw_response})

        judge_criteria_history.append(criteria)
        judge_records.append({"round": judge_round, "status": status, "criteria": criteria})
        if probe_result["done"]:
            break

    post_probe_env_score: float | None = None
    try:
        post_probe_env_score = env_client.evaluate(lease_id)
    except Exception as exc:
        logger.warning("post-probe env.evaluate() failed: %s", exc)

    final_judge_success = status == "success"
    policy_env_success = policy_result is not None and policy_result >= 1.0
    post_probe_env_success = (
        post_probe_env_score is not None and post_probe_env_score >= 1.0
    )
    rounds = [
        {
            "round": record["round"],
            "label": record["status"],
            "status": record["status"],
            "content": record["criteria"],
            "decision_raw": record["status"],
        }
        for record in judge_records
    ]
    summary = make_eval_summary(
        domain=domain,
        example_id=example_id,
        rollout_id=rollout_id,
        policy_model=policy_model,
        eval_model=guirl_config["judge_model_name"],
        policy_result=policy_result,
        rounds=rounds,
        sampler_rounds=probe_step,
    )
    summary.update(
        {
            "evaluator_type": "guirl",
            "judge_records": judge_records,
            "final_judge_status": status,
            "final_judge_success": final_judge_success,
            "final_criteria": criteria,
            "judge_criteria_history": judge_criteria_history,
            "probe_updated_criteria_per_round": probe_updated_criteria_per_round,
            "policy_env_score": policy_result,
            "policy_env_success": policy_env_success,
            "post_probe_env_score": post_probe_env_score,
            "post_probe_env_success": post_probe_env_success,
            "judge_matches_policy": final_judge_success == policy_env_success,
            "probe_steps": probe_step,
            "probe_rounds": probe_round,
            "probe_round_records": probe_round_records,
            "probe_final_criteria_images": all_probe_evidence,
            "guirl_config": {
                key: value for key, value in guirl_config.items() if "api_key" not in key
            },
            "messages": messages,
        }
    )
    write_eval_summary(example_result_dir, summary)

    if getattr(args, "export_guirl_sft_data", False):
        sft_output_dir = os.path.join(os.path.dirname(example_result_dir), "sft_data")
        try:
            maybe_export_sft_from_eval_summary(
                summary,
                output_dir=sft_output_dir,
                reward_threshold=1.0,
                strip_images=bool(getattr(args, "export_guirl_sft_strip_images", True)),
            )
        except Exception as exc:

            logger.warning(
                "GUIRL SFT export failed for %s/%s/%s: %s",
                domain,
                example_id,
                rollout_id,
                exc,
            )

    return summary
