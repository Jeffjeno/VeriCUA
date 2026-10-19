from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .judge_prompts import (
    JudgeStatus,
    build_judge_messages,
    format_action_history,
    parse_judge_output,
)
from .probe_runner import ProbeResult, ProbeRunner

logger = logging.getLogger("guirl.judge_evaluator")

JUDGE_PARSE_MAX_RETRIES = 3

@dataclass
class JudgeRoundRecord:

    round_idx: int
    messages: list[dict[str, Any]]
    response_text: str
    criteria_text: str
    status: JudgeStatus
    timing: float = 0.0

@dataclass
class JudgeEvalResult:

    final_status: JudgeStatus
    rounds: list[JudgeRoundRecord]
    probe_results: list[ProbeResult] = field(default_factory=list)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    full_response_text: str = ""
    timing: float = 0.0

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

async def run_judge_eval_loop(
    *,
    judge_fn,
    probe_runner: ProbeRunner | None,
    lease_id: str | None,
    task_instruction: str,
    trajectory_screenshots: list[str],
    action_history: list[str] | None = None,
    max_rounds: int = 2,
    history_n: int = 3,
    parse_error_status: JudgeStatus = "parse_failure",
) -> JudgeEvalResult:

    t_start = time.monotonic()
    rounds: list[JudgeRoundRecord] = []
    probe_results: list[ProbeResult] = []
    all_messages: list[dict[str, Any]] = []
    all_response_texts: list[str] = []

    current_criteria: str | None = None
    current_status: JudgeStatus = "uncertain"
    criteria_option_screenshot: list[dict[str, Any]] | None = None

    for round_idx in range(max_rounds):
        t_round = time.monotonic()

        messages = build_judge_messages(
            task=task_instruction,
            task_traj_imgs=trajectory_screenshots,
            prev_criteria=current_criteria,
            action_history=action_history if round_idx == 0 else None,
            history_n=history_n,
            criteria_option_screenshot=criteria_option_screenshot,
        )

        response_text = ""
        last_exc: Exception | None = None
        for attempt in range(JUDGE_PARSE_MAX_RETRIES):
            try:
                judge_output = await judge_fn(messages)
                if isinstance(judge_output, tuple):
                    response_text = judge_output[0]
                else:
                    response_text = judge_output

                parsed = parse_judge_output(response_text)
                current_criteria = parsed.criteria_text
                current_status = parsed.status
                break
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "Judge parse failed (round %d, attempt %d/%d): %s",
                    round_idx, attempt + 1, JUDGE_PARSE_MAX_RETRIES, exc,
                )
                if attempt >= JUDGE_PARSE_MAX_RETRIES - 1:
                    current_status = parse_error_status
                    current_criteria = current_criteria or ""
        else:
            if last_exc:
                logger.error("Judge parse exhausted retries: %s", last_exc)

        round_record = JudgeRoundRecord(
            round_idx=round_idx,
            messages=messages,
            response_text=response_text,
            criteria_text=current_criteria or "",
            status=current_status,
            timing=time.monotonic() - t_round,
        )
        rounds.append(round_record)
        all_messages.extend(messages)
        all_messages.append({"role": "assistant", "content": response_text})
        all_response_texts.append(response_text)

        if current_status != "uncertain":
            break
        if round_idx >= max_rounds - 1:
            break
        if probe_runner is None or lease_id is None:
            break

        try:
            probe_result = await probe_runner.run_probe_round(
                lease_id=lease_id,
                criteria_text=current_criteria or "",
                task_instruction=task_instruction,
            )
            probe_results.append(probe_result)

            if probe_result.criteria_states:
                updated_lines = []
                for state in probe_result.criteria_states:
                    updated_lines.append(f"{state.index}. {state.text}:{state.status}")
                current_criteria = "\n".join(updated_lines)

            criteria_option_screenshot = probe_result.criteria_images or None

        except Exception as exc:
            logger.error("Probe round failed: %s", exc)
            criteria_option_screenshot = None

    final_status = current_status

    return JudgeEvalResult(
        final_status=final_status,
        rounds=rounds,
        probe_results=probe_results,
        all_messages=all_messages,
        full_response_text="\n".join(all_response_texts),
        timing=time.monotonic() - t_start,
    )

def validate_reward_mode(mode: str) -> str:
    if mode not in {"shaped", "binary"}:
        raise ValueError(f"Unknown GUIRL reward mode: {mode!r}; expected shaped or binary")
    return mode

def compute_reward(
    judge_final: JudgeStatus,
    S: float | None,
    *,
    used_probe: bool = False,
    reward_mode: str = "shaped",
) -> float:

    validate_reward_mode(reward_mode)
    if reward_mode == "binary":
        if S is None or judge_final not in {"success", "failure"}:
            return 0.0
        label = "success" if float(S) >= 1.0 else "failure"
        return float(judge_final == label)

    if judge_final == "parse_failure":
        return -1.0

    if S is None:
        return -1.0

    if judge_final == "uncertain":
        return -1.0

    s_label = "success" if float(S) >= 1.0 else "failure"
    correct = judge_final == s_label

    if correct:
        return 1.0 if not used_probe else 0.7
    return -0.7 if not used_probe else -0.3
