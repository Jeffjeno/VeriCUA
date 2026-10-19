from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

JudgeStatus = Literal["success", "failure", "uncertain"]
JUDGE_STATUS_VALUES: tuple[JudgeStatus, ...] = ("success", "failure", "uncertain")

CRITERIA_RE = re.compile(r"<criteria>\s*(.*?)\s*</criteria>", re.DOTALL | re.IGNORECASE)
STATUS_RE = re.compile(r"Status:\s*(success|failure|uncertain)\b", re.IGNORECASE)
THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

@dataclass(frozen=True)
class ParsedJudgeOutput:
    criteria_text: str
    status: JudgeStatus

JUDGE_SYSTEM_PROMPT = """You are an expert in evaluating the performance of a GUI agent. The agent is designed to help a human user complete tasks across desktop applications and web pages.
Your job is to:
1. write concise, concrete, and verifiable task-success criteria;
2. judge whether each criterion is currently success, failure, or uncertain based on the provided evidence;
3. decide the overall task status: success, failure, or uncertain.

There are several types of tasks (not limited to the list below):
1. Information seeking: The user wants to obtain information from a webpage, document, or application. The final response must contain the requested information or explicitly state that it is unavailable. If the agent returns an exception or irrelevant error content, the task is a failure.
2. Navigation: The user wants to navigate to a specific page, view, folder, screen, or application state. Examine the action history and final visual state. The agent's response is usually not required.
3. Content modification: The user wants to modify content in a webpage, document, file, setting, or application. Verify that the final state reflects the requested change.
4. File operations: The user wants to create, rename, move, copy, delete, compress, extract, or inspect files or folders. Verify that the filesystem state matches the request.
5. Application setup or configuration: The user wants to install, uninstall, enable, disable, or configure software, extensions, plugins, or settings. Verify that the final applied state indicates the requested setup/configuration is complete.
6. Media actions: The user wants to open, play, pause, record, view, or manipulate images, audio, or video. Verify that the media state reflects the requested action.
7. Communication: The user wants to read, compose, send, or retrieve emails/messages. Verify the intended content, recipient, and sent/retrieved state when applicable.
8. Terminal or command execution: The user wants to run shell commands. Verify that the commands were executed correctly and that outputs or side effects match the intent.

==Overall Status(check in this order)==
1. success: all criteria are success.
2. failure: otherwise, if any criterion is failure (or contradicted / done incorrectly).
3. uncertain: otherwise (no failure, but at least one uncertain).

==Output exactly this format==
<criteria>
1. <concrete task-success standard>:<success|failure|uncertain>
2. <concrete task-success standard>:<success|failure|uncertain>
3. <concrete task-success standard>:<success|failure|uncertain>

</criteria>
Status: <success|failure|uncertain>

==Example:Open report.docx in LibreOffice Writer, change the document title to "2026 Q1 Sales Review", and save the file.==
<criteria>
1. report.docx is opened in LibreOffice Writer:success
2. The document title text reads exactly "2026 Q1 Sales Review":uncertain
3. The change is persisted to report.docx (file saved, no unsaved-changes indicator):failure
</criteria>
Status: failure

==Keep it in mind==
1. If previous criteria are not provided, produce 3 to 5 criteria.
2. If previous criteria are provided, keep the criteria text unchanged. The probe agent has already assessed the status of each criterion — treat the probe's assessment as your default baseline. Only override a status (e.g. downgrade 'success' back to 'uncertain') when you observe clear, direct visual contradiction: for example, a file that is visibly unmodified, a setting that is clearly still at its default value, or an error message that proves the task was not completed. Do not downgrade to 'uncertain' merely because the evidence is limited, indirect, or ambiguous.
3. Use only the user task as requirements; do not add unrelated or overly strict criteria.
4. Do not treat action-history labels like Done, Success, Completed, saved, clicked Save/Submit/OK/Apply as evidence by themselves.
"""

def image_content(screenshot_b64: str, mime_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{screenshot_b64}"}}

def format_action_history(action_history: list[str] | str | None) -> str:
    if action_history is None:
        return "None"
    if isinstance(action_history, str):
        return action_history.strip() or "None"

    if not action_history:
        return "None"
    return "\n".join(f"Step {idx + 1}: {action}" for idx, action in enumerate(action_history))

def _history_screenshot_content(task_traj_imgs: Any, history_n: int | None = None) -> list[dict[str, Any]]:

    screenshots = task_traj_imgs if isinstance(task_traj_imgs, list) else [task_traj_imgs]
    if history_n and history_n > 0:
        screenshots = screenshots[-history_n:]
    return [image_content(screenshot) for screenshot in screenshots if screenshot]

def _criterion_line(criteria: str, option: int) -> str:
    pattern = re.compile(rf"^\s*{option}\.\s*.*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(criteria)
    return match.group(0).strip() if match else f"{option}. <criterion not found>"

def _evidence_content_with_criteria(criteria_option_screenshot: Any) -> list[dict[str, Any]]:

    if criteria_option_screenshot is None:
        return []
    evidence_items = (
        criteria_option_screenshot
        if isinstance(criteria_option_screenshot, list)
        else [criteria_option_screenshot]
    )
    content: list[dict[str, Any]] = []
    for item in evidence_items:
        opt = item.get("option")
        image = item.get("images")
        if image:
            content.append({"type": "text", "text": f"Evidence image {opt} for the current criteria."})
            content.append(image_content(image))
    return content

def _context_criteria_example_content(
    examples: list[dict[str, str]],
) -> list[dict[str, Any]]:

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Reference examples below demonstrate how to derive concise, concrete, "
                "and verifiable criteria from a user task and its final observation. "
                "Their criteria contain no verdicts. The examples are unrelated to the "
                "current task and are not evidence for it; do not infer their outcomes."
            ),
        }
    ]
    for idx, example in enumerate(examples, start=1):
        screenshot_b64 = example.get("screenshot_b64")
        reference_intro = (
            "The following screenshot is the final observation for this reference example."
            if screenshot_b64
            else "This is a text-only reference example; use its task and criteria as guidance."
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"==Reference example {idx}==\n"
                    f"User Intent:\n{example['task']}\n\n"
                    f"{reference_intro}"
                ),
            }
        )
        if screenshot_b64:
            content.append(image_content(screenshot_b64))
        content.append(
            {
                "type": "text",
                "text": f"Reference Criteria:\n{example['criteria']}",
            }
        )
    return content

def build_judge_user_content(
    *,
    task: str,
    task_traj_imgs: Any,
    prev_criteria: str | None = None,
    action_history: list[str] | str | None = None,
    history_n: int | None = None,
    criteria_option_screenshot: Any | None = None,
    context_criteria_examples: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:

    if prev_criteria:
        prev_criteria = prev_criteria.strip()
        has_evidence = bool(
            criteria_option_screenshot
            and (isinstance(criteria_option_screenshot, list) and len(criteria_option_screenshot) > 0)
        )

        if has_evidence:
            user_text = (
                f"User Intent:\n{task}\n\n"
                f"Criteria:\n{prev_criteria}\n\n"
                "The attached screenshot(s) are evidence collected by the probe "
                "for the criteria, shown in chronological order."
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            content.extend(_evidence_content_with_criteria(criteria_option_screenshot))
        else:
            screenshot_count = len(task_traj_imgs) if isinstance(task_traj_imgs, list) else 1
            actual_count = min(screenshot_count, history_n) if history_n and history_n > 0 else screenshot_count
            user_text = (
                f"User Intent:\n{task}\n\n"
                f"Criteria:\n{prev_criteria}\n\n"
                f"The probe agent did not find corresponding evidence for the criteria. "
                f"Please judge based on the following {actual_count} latest screenshots "
                "from the original agent trajectory, shown in chronological order."
            )
            content = [{"type": "text", "text": user_text}]
            content.extend(_history_screenshot_content(task_traj_imgs, history_n))

        return content
    else:
        screenshot_count = len(task_traj_imgs) if isinstance(task_traj_imgs, list) else 1
        actual_count = min(screenshot_count, history_n) if history_n and history_n > 0 else screenshot_count
        current_task_text = (
            f"User Intent:\n{task}\n\n"
            "Criteria:\nNo criteria available.\n\n"
            f"Action History:\n{format_action_history(action_history)}\n\n"
            f"The following {actual_count} screenshots are the latest observations, "
            "shown in chronological order."
        )
        content = [{"type": "text", "text": current_task_text}]
        if context_criteria_examples:
            content = _context_criteria_example_content(context_criteria_examples)
            content.append(
                {
                    "type": "text",
                    "text": f"==Current task==\n{current_task_text}",
                }
            )
        content.extend(_history_screenshot_content(task_traj_imgs, history_n))
        return content

def build_judge_messages(
    *,
    task: str,
    task_traj_imgs: Any,
    prev_criteria: str | None = None,
    action_history: list[str] | str | None = None,
    history_n: int | None = None,
    criteria_option_screenshot: Any | None = None,
    context_criteria_examples: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:

    user_content = build_judge_user_content(
        task=task,
        task_traj_imgs=task_traj_imgs,
        prev_criteria=prev_criteria,
        action_history=action_history,
        history_n=history_n,
        criteria_option_screenshot=criteria_option_screenshot,
        context_criteria_examples=context_criteria_examples,
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

def parse_judge_output(text: str) -> ParsedJudgeOutput:
    visible_text = THINKING_RE.sub("", text)
    criteria_matches = list(CRITERIA_RE.finditer(visible_text))
    status_matches = list(STATUS_RE.finditer(visible_text))
    if not criteria_matches:
        raise ValueError("Judge output is missing <criteria>...</criteria> block.")
    if not status_matches:
        raise ValueError("Judge output is missing `Status: success|failure|uncertain`.")

    criteria_match = criteria_matches[-1]
    status_match = status_matches[-1]
    status = status_match.group(1).lower()
    if status not in JUDGE_STATUS_VALUES:
        raise ValueError(f"Invalid judge status: {status}")
    return ParsedJudgeOutput(criteria_text=criteria_match.group(1).strip(), status=status)

__all__ = [
    "CRITERIA_RE",
    "JUDGE_STATUS_VALUES",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeStatus",
    "ParsedJudgeOutput",
    "STATUS_RE",
    "THINKING_RE",
    "build_judge_messages",
    "build_judge_user_content",
    "format_action_history",
    "image_content",
    "parse_judge_output",
]
