from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from mm_agents.reward_model.agent_judge import (
    AgentJudgeEvaluator as BaseAgentJudgeEvaluator,
)


QUICK_JUDGE_SYSTEM_PROMPT = """
You are an expert in evaluating the performance of a GUI agent. The agent is
designed to help a human user complete tasks across desktop applications and
web pages. Given the user's intent, the agent's action history, and the final
state shown in the relevant screenshots, decide whether the agent's execution
is success, failure, or uncertain according to the given information.

There are several types of tasks (not limited to the list below):
1. Information seeking: The user wants to obtain certain information from a
webpage, document, or application (e.g., product info, map routes, file properties, system info). The agent’s response must contain the requested information
or explicitly state that it is not available. If the agent encounters an exception
and returns error content, the task is a failure.
2. Navigation: The user wants to navigate to a specific page, view, folder, or
screen within a website or application. Carefully examine the action history and
final state to determine success. No need to consider the agent’s response.
3. Content modification: The user wants to modify content in a webpage, document, or application (e.g., edit text, change formatting, update settings). Verify
the final state reflects the requested changes. No need to consider the agent’s
response.
4. File operations: The user wants to manage files or folders (create, rename,
move, copy, delete, compress/extract). Confirm the filesystem state matches the
request.
5. Application setup/installation: The user wants to install, uninstall, or configure software, extensions, or plugins. Confirm the final state indicates the
requested installation/configuration is complete.
6. Media actions: The user wants to open, play, pause, record, or manipulate
images/audio/video. Confirm the media state reflects the requested action.
7. Communication: The user wants to read/send emails or messages. Confirm
the content is correctly composed/sent or retrieved.
8. Terminal/command execution: The user wants to run shell commands. Verify
commands were executed correctly and outputs match the intent.
Be careful about the sufficiency of the agent’s actions. For example, when asked
to list the top-searched items, the agent should order items by search count and
return the top entries. If required actions (sorting, filtering, saving) are missing,
the task is likely a failure.

Important rules:
1. The user intent is the only source of task requirements; do not add extra
   requirements beyond it.
2. Use visible screenshot evidence and action history. Do not treat labels in
   the action history such as Done, Success, Completed, or saved as evidence by
   themselves.
3. Do not infer success from clicking Save, Submit, Apply, OK, or Done alone;
   require a visible or otherwise provided post-condition when verification is
   needed.
4. If required actions such as sorting, filtering, saving, sending, or running
   a command are missing, the task is likely failure.

Judge criterion:
- success: clear evidence shows the requested task is completed.
- failure: clear evidence shows the task is incomplete, incorrect, or failed.
- uncertain: evidence is insufficient or ambiguous after the observed execution.

Output exactly one line and nothing else:
Status: "success" or "failure" or "uncertain"
""".strip()


QUICK_JUDGE_USR_TEMPLATE = """
User Intent:
{instruction}

Action History:
{action_history}

The following {history_n} screenshots are the latest observations, shown in chronological order.
""".strip()


class AgentJudgeEvaluator(BaseAgentJudgeEvaluator):
    """Quick single-pass GUI task-completion judge.

    This class keeps the same public evaluator interface as AgentJudgeEvaluator
    but skips memory construction and directly judges from action history plus
    the latest screenshots. The raw model prompt uses the AER three-way
    Status format, while results are mapped to the existing three-way
    reward interface: success -> True, failure -> False, uncertain -> None.
    """

    def _build_rollout_result_extra(
        self,
        meta: Dict[str, Any],
        history_len: int,
        rollout_idx: int,
        rollout_num: int,
        judge_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = super()._build_rollout_result_extra(
            meta=meta,
            history_len=history_len,
            rollout_idx=rollout_idx,
            rollout_num=rollout_num,
            judge_output=judge_output,
        )
        extra["method"] = "quick_judge"
        extra["history_n"] = self.history_n
        return extra

    async def _run_agent_judge(
        self, instruction: str, history: List[Tuple[Dict[str, Any], Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "mr_rounds": 0,
            "mr_calls_gen_mem": 0,
            "mr_calls_judge": 0,
            "mr_trace": [],
            "judge_msg": [],
            "memory_msgs": [],
            "decision_raw": None,
        }

        judge_msg = self._build_quick_judge_messages(
            instruction=instruction,
            history=history,
        )

        async def _judge_once() -> Dict[str, Any]:
            resp = await self._chat_completion_async_with_sem(judge_msg)
            message = resp.choices[0].message
            content = message.content or ""
            decision, decision_raw = self._status_to_bool(content)
            raw_response = resp.model_dump() if hasattr(resp, "model_dump") else None
            return {
                "content": content,
                "decision": decision,
                "decision_raw": decision_raw,
                "message": message.model_dump(),
                "raw_response": raw_response,
            }

        rollout_num = max(1, int(self.rollout_num))
        judge_outputs = await asyncio.gather(
            *[_judge_once() for _ in range(rollout_num)]
        )

        meta["mr_calls_judge"] = rollout_num
        meta["judge_msg"].append(judge_msg)
        meta["judge_msg"].append(deepcopy(judge_outputs))
        if judge_outputs:
            meta["decision_raw"] = judge_outputs[0].get("decision_raw")

        return judge_outputs, meta

    def _build_quick_judge_messages(
        self,
        instruction: str,
        history: List[Tuple[Dict[str, Any], Any]],
    ) -> List[Dict[str, Any]]:
        recent = history[-self.history_n :]
        action_history = self._format_action_history(history)
        user_prompt = QUICK_JUDGE_USR_TEMPLATE.format(
            instruction=instruction,
            action_history=action_history,
            history_n=len(recent),
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for obs, _ in recent:
            content.append(obs)

        return [
            {"role": "system", "content": QUICK_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _format_action_history(history: List[Tuple[Dict[str, Any], Any]]) -> str:
        lines = []
        for idx, (_, act) in enumerate(history, start=1):
            lines.append(f"[STEP {idx}] {act}")
        return "\n".join(lines) if lines else "No action history provided."

    @classmethod
    def _status_to_bool(cls, content: str) -> Tuple[bool | None, str]:
        normalized = cls._strip_think_content(content).strip().lower()

        status = ""
        for line in normalized.splitlines():
            if line.strip().startswith("status:"):
                status = line.split(":", 1)[1].strip().strip('"\' ')
                break

        if not status:
            decision = cls._parse_decision(content)
            if decision == "yes":
                return True, "success"
            if decision == "no":
                return False, "failure"
            if decision == "uncertain":
                return None, "uncertain"
            status = normalized

        if "success" in status:
            return True, "success"
        if "failure" in status or "fail" in status:
            return False, "failure"
        if "uncertain" in status or "unknown" in status:
            return None, "uncertain"
        return None, "parse_error"


class IndependentRolloutAgentJudgeEvaluator(AgentJudgeEvaluator):
    """Compatibility alias for callers that expect independent rollouts."""


__all__ = ["AgentJudgeEvaluator", "IndependentRolloutAgentJudgeEvaluator"]
