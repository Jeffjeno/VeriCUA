from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from mm_agents.reward_model.base_evaluator import BaseEvaluator
from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample


SYSTEM_PROMPT = (
    "You are an expert at analyzing computer usage task completion from screenshots."
)

USER_PROMPT_TEMPLATE = """You will be given a task instruction and a series of screenshots of the task execution. 
Please analyze the screenshots and provide a detailed analysis of the task completion by following the steps below:
1. First, analyze and understand the task instruction. Describe what should the screenshots look like if the task is completed successfully.
2. Describe what you observe in each screenshot, analysis what actions were taken and what changes were made to the UI to achieve the task (or mistakes made).
3. When you analyze the screenshots, please pay attention to the very detailed elements and changes in the UI. Every small detail may affect the final result.
4. After all screenshots are analyzed, provide a overall reasoning about how the task was completed or failed at **the final state**. Make sure you have considered all demands of the task instruction.
5. Determine if the task was completed at **the final state** (the last screenshot) successfully (score 1 for success, 0 for failure). If the task is completed during the process but not at the final state, it should be considered as failure (0 score).

Provide your response strictly in the following format:
TASK REQUIREMENT:
[Your understanding of the task instruction]

SCREENSHOT ANALYSIS:
Screenshot 1:
[Analysis of first screenshot]
Screenshot 2:
[Analysis of second screenshot]
...

REASONING:
[Your reasoning]

FINAL ANSWER:
[Your final answer]

SCORE: [0/1]

Here is an example:
(Task Instruction: Please help me backup my emails in "Bills" folder in Thunderbird and store the .eml files with only subject names to my Google Drive folder called "emails".)

TASK REQUIREMENT:
- Backup the emails in "Bills" folder in Thunderbird.
- Store the backup .eml files with only subject names, and the emails should be saved in the Google Drive folder called "emails".
- Once succeed, the emails should be visible in the Google Drive folder "emails". Or at least there should be a saving action performed.

SCREENSHOT ANALYSIS:

Screenshot 1:
- Thunderbird email client is open.
- The "Bills" folder is visible under "Local Folders."
- There is no observable action performed yet in this screenshot.

Screenshot 2:
- The "Bills" folder has been selected, and the folder content is displayed.
- Two emails are visible: "Amazon Web Services Invoice Available" and "Your receipt from X (formerly Twitter)."
- No further actions are taken on the emails.

Screenshot 3:
- Both emails in the "Bills" folder are selected.
- Content previews of both emails are displayed on the right-hand side.
- No observable attempt to export or save the emails is visible.

Screenshot 4:
- The right-click context menu is accessed for the selected emails.
- The "Save As..." option is hovered over, indicating intent to save the selected emails.

Screenshot 5:
- The file navigation window opens, allowing the user to choose a save destination.
- No specific Google Drive folder (e.g., "emails") is accessed or visible in this screenshot.

Screenshot 6:
- The "Desktop" option in the file picker is hovered over.
- Still no indication of Google Drive folder ("emails") selection.

Screenshot 7:
- The "Show other locations" option is hovered over in the file picker.
- No confirmation that the user is navigating to Google Drive or saving the files with subject names only.

Screenshot 8:
- The "Software Updates Available" notification appears. The file picker is still open without any observable confirmation of file saving or destination selection.
- It remains unclear where or if the emails have been saved.

REASONING:
Based on the screenshots provided:
1. While there was some intent to save the emails (as shown by the selection and access of the "Save As..." function), there is no confirmation that the .eml files were saved with subject names only and placed in the required Google Drive folder ("emails").
2. The screenshots lack evidence of the completion of the task as per the instructions.

FINAL ANSWER:
The task was not completed successfully due to the lack of observable saving action.

SCORE: 0

Now, please **strictly follow the format** and analyze the following screenshots (The last line should only be SCORE: [0/1], no other text):
Task Instruction: {instruction}
Screenshots (by order):
"""


def parse_output(text: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "task_requirement": None,
        "screenshot_analysis": None,
        "reasoning": None,
        "final_answer": None,
        "score": None,
    }
    if not text:
        return result

    text = text.strip()
    sections = {
        "task_requirement": r"TASK REQUIREMENT:\s*(.*?)(?=\n\s*SCREENSHOT ANALYSIS:|\Z)",
        "screenshot_analysis": r"SCREENSHOT ANALYSIS:\s*(.*?)(?=\n\s*REASONING:|\Z)",
        "reasoning": r"REASONING:\s*(.*?)(?=\n\s*FINAL ANSWER:|\Z)",
        "final_answer": r"FINAL ANSWER:\s*(.*?)(?=\n\s*SCORE:|\Z)",
    }
    for key, pattern in sections.items():
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    score_matches = re.findall(r"SCORE:\s*\[?\s*([01])\s*\]?", text, flags=re.IGNORECASE)
    if score_matches:
        result["score"] = score_matches[-1]
    return result


class ZeroGuiEvaluator(BaseEvaluator):
    def __init__(self, llm, max_screenshots: int = 15) -> None:
        super().__init__(llm)
        # 0 = no truncation (official ZeroGUI feeds the full trajectory).
        # >0 = keep the last N frames after prepending step_0 when present.
        self.max_screenshots = max(int(max_screenshots), 0)

    def _evaluate_sync(self, task: TaskSample) -> EvaluationResult:
        messages = self._build_messages(task)
        response = self.llm.chat_completion_sync(messages)
        content = self._extract_response_text(response)
        parsed = parse_output(content)
        decision, decision_raw = self._score_to_bool(parsed.get("score"), content)
        return self._result(
            task,
            decision,
            content=content,
            reasoning=parsed.get("reasoning") or "",
            method="zerogui",
            decision_raw=decision_raw,
            max_screenshots=self.max_screenshots,
            judge_msg=messages,
            memory_msgs=[],
        )

    async def _evaluate_async(self, task: TaskSample) -> EvaluationResult:
        messages = self._build_messages(task)
        response = await self.llm.chat_completion_async(messages)
        content = self._extract_response_text(response)
        parsed = parse_output(content)
        decision, decision_raw = self._score_to_bool(parsed.get("score"), content)
        return self._result(
            task,
            decision,
            content=content,
            reasoning=parsed.get("reasoning") or "",
            method="zerogui",
            decision_raw=decision_raw,
            max_screenshots=self.max_screenshots,
            judge_msg=messages,
            memory_msgs=[],
        )

    @staticmethod
    def _find_step0_path(result_dir: str | None) -> Path | None:
        """Official ZeroGUI always includes the pre-action initial screenshot (step_0)."""
        if not result_dir:
            return None
        root = Path(result_dir)
        exact = root / "step_0.png"
        if exact.is_file():
            return exact.resolve()
        matches = sorted(root.glob("step_0*.png"))
        return matches[0].resolve() if matches else None

    @staticmethod
    def _path_already_listed(paths: list[str], candidate: Path) -> bool:
        cand = str(candidate)
        cand_name = candidate.name
        for item in paths:
            s = str(item)
            if s == cand or s.endswith("/" + cand_name) or Path(s).name == cand_name:
                if "step_0" in Path(s).name:
                    return True
        return False

    def _prepare(self, task: TaskSample) -> TaskSample:
        prepared = self._clone_task(task)
        history_images = list(prepared.history_images or [])
        history_actions = list(prepared.history_actions or [])
        extra = prepared.extra if isinstance(prepared.extra, dict) else {}
        raw_history_images = list(extra.get("_reward_input_history_images") or [])
        raw_current = extra.get("_reward_input_current_image")

        # Align with official: prepend initial frame (step_0) when on disk.
        step0 = self._find_step0_path(extra.get("example_result_dir"))
        if step0 is not None:
            step0_s = str(step0)
            if not self._path_already_listed(raw_history_images, step0):
                raw_history_images = [step0_s] + raw_history_images
            if not self._path_already_listed(history_images, step0):
                history_images = [step0_s] + history_images
                # No matching action for the initial observation frame.
                history_actions = [""] + list(history_actions)

        if self.max_screenshots > 0:
            # Keep last N frames total = (N-1) history + current.
            keep_history = max(self.max_screenshots - 1, 0)
            history_images = history_images[-keep_history:] if keep_history else []
            history_actions = history_actions[-keep_history:] if keep_history else []
            raw_history_images = raw_history_images[-keep_history:] if keep_history else []

        prepared.history_images = history_images
        prepared.history_actions = history_actions
        if isinstance(prepared.extra, dict):
            prepared.extra["_reward_input_history_images"] = raw_history_images
            if raw_current is not None:
                prepared.extra["_reward_input_current_image"] = raw_current
        return super()._prepare(prepared)

    def _validate(self, task: TaskSample) -> EvaluationResult | None:
        missing_fields = []
        if not task.instruction:
            missing_fields.append("instruction")
        if not task.current_image:
            missing_fields.append("current_image")
        if missing_fields:
            return self._error(
                task,
                "missing_required_fields",
                method="zerogui",
                missing_fields=missing_fields,
            )
        return None

    def _build_messages(self, task: TaskSample) -> list[dict[str, Any]]:
        image_refs = self._raw_image_refs(task)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": USER_PROMPT_TEMPLATE.format(instruction=task.instruction)}
        ]
        for image_ref in image_refs:
            # Official ZeroGUI sets detail=high for OpenAI-compatible VLMs.
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_ref, "detail": "high"},
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _raw_image_refs(self, task: TaskSample) -> list[str]:
        extra = task.extra or {}
        raw_history_images = list(extra.get("_reward_input_history_images") or [])
        raw_current_image = extra.get("_reward_input_current_image")
        refs = raw_history_images + ([raw_current_image] if raw_current_image else [])
        if refs:
            return [self._encode_official_image_ref(ref) for ref in refs]
        return [self._encode_official_image_ref(ref) for ref in list(task.history_images or []) + [task.current_image]]

    @staticmethod
    def _encode_official_image_ref(image_ref: str) -> str:
        if not image_ref:
            raise ValueError("image reference is empty")
        if str(image_ref).startswith("data:image/") or str(image_ref).startswith("http"):
            return str(image_ref)

        image_path = BaseEvaluator._resolve_image_path(image_ref)
        with image_path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if hasattr(response, "choices") and response.choices:
            message = getattr(response.choices[0], "message", None)
            content = getattr(message, "content", "")
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            return str(content or "")
        return str(response)

    @staticmethod
    def _score_to_bool(score: str | None, raw_text: str) -> tuple[bool | None, str]:
        if score == "1":
            return True, "score_1"
        if score == "0":
            return False, "score_0"
        if re.search(r"SCORE:\s*\[?\s*[01]\s*\]?", raw_text, flags=re.IGNORECASE):
            return None, "parse_error"
        return None, "parse_error"


__all__ = ["ZeroGuiEvaluator", "parse_output"]
