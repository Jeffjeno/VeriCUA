from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("guirl.probe_runner")

PROBE_SYSTEM_PROMPT = """You are a GUI evidence probe running inside a desktop GUI agent loop.

A separate GUI agent Jeffery has already attempted the user's task in this environment. \
Your task is not to redo the whole task; your task is to inspect the current GUI state, \
collect just enough evidence, and update the judge criteria using the `criteria` tool.

At each step, choose exactly one action from two modes:
1. Exploration: use pyautogui actions (click, type, hotkey, scroll, wait) to navigate and reveal evidence.
2. Criteria update: call the criteria tool to formally update a criterion's state.

IMPORTANT RULES about the `criteria` tool:
- You MUST call the criteria tool to update each criterion when you observe evidence. \
Reasoning in plain text is NOT sufficient — the judge only sees criteria updates made \
through the criteria tool call.
- Update criteria one at a time. Each criteria call updates exactly one criterion number.
- Only output DONE AFTER you have called criteria for every uncertain criterion that you \
have evidence for. If you have not called criteria at all, do NOT output DONE.

Follow the screenshot shown for the current step. Coordinates must be relative to the \
exact screenshot you see.

Use type/click to focus inputs before typing. Prefer hotkeys and Enter for shortcuts. \
Scroll when content may be off-screen.

Do not perform actions that modify the environment (e.g. saving files, deleting, editing). \
Your role is to observe and verify, not to change the state.

Output format for each step (choose one):
- A pyautogui action to navigate/verify, OR
- criteria(option=N, status='success'|'failure'|'uncertain') to update one criterion, OR
- DONE when finished collecting evidence for all uncertain criteria you could verify"""

CRITERIA_TOOL_RE = re.compile(
    r"<tool_use>\s*criteria\s*\(\s*option\s*=\s*(\d+)\s*,\s*status\s*=\s*['\"]?(success|failure|uncertain)['\"]?\s*\)",
    re.IGNORECASE,
)

CRITERIA_BARE_RE = re.compile(
    r"criteria\s*\(\s*option\s*=\s*(\d+)\s*,\s*status\s*=\s*['\"]?(success|failure|uncertain)['\"]?\s*\)",
    re.IGNORECASE,
)

@dataclass
class CriterionState:
    index: int
    text: str
    status: str = "uncertain"
    evidence_screenshot: str | None = None

@dataclass
class ProbeResult:
    criteria_states: list[CriterionState]
    criteria_images: list[dict[str, Any]]
    num_steps: int
    timing: float

class ProbeRunner:

    def __init__(
        self,
        *,
        env_client,
        base_url: str = "http://127.0.0.1:40001/v1",
        model_name: str = "Holo3.1-35B-A3B",
        api_key: str = "EMPTY",
        max_steps: int = 15,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        self.env_client = env_client
        self.model_name = model_name
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._openai = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def run_probe_round(
        self,
        *,
        lease_id: str,
        criteria_text: str,
        task_instruction: str,
    ) -> ProbeResult:

        t_start = time.monotonic()

        criteria_states = self._parse_criteria(criteria_text)

        num_steps = 0
        for step_idx in range(self.max_steps):
            obs = await self.env_client.get_obs(lease_id)
            screenshot_bytes = obs.get("screenshot", b"")
            if isinstance(screenshot_bytes, bytes) and screenshot_bytes:
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            else:
                screenshot_b64 = ""

            messages = self._build_messages(
                task_instruction, criteria_states, screenshot_b64
            )

            response_text = await self._call_llm(messages)
            num_steps += 1

            self._process_criteria_updates(response_text, criteria_states, screenshot_b64)

            if "DONE" in response_text.upper():
                break

            action_code = self._extract_action(response_text)
            if not action_code:
                continue

            try:
                await self.env_client.step(lease_id, action_code)
            except Exception as exc:
                logger.warning("Probe step %d failed: %s", step_idx, exc)
                break

        criteria_images = []
        for state in criteria_states:
            if state.evidence_screenshot:
                criteria_images.append({
                    "option": state.index,
                    "images": state.evidence_screenshot,
                })

        return ProbeResult(
            criteria_states=criteria_states,
            criteria_images=criteria_images,
            num_steps=num_steps,
            timing=time.monotonic() - t_start,
        )

    def _parse_criteria(self, criteria_text: str) -> list[CriterionState]:

        states = []
        lines = criteria_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\d+)\.\s*(.+?):(success|failure|uncertain)\s*$", line, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                text = match.group(2).strip()
                status = match.group(3).lower()
                states.append(CriterionState(index=idx, text=text, status=status))
            else:
                match2 = re.match(r"(\d+)\.\s*(.+)$", line)
                if match2:
                    idx = int(match2.group(1))
                    text = match2.group(2).strip()
                    states.append(CriterionState(index=idx, text=text, status="uncertain"))
        return states

    def _build_messages(
        self,
        task_instruction: str,
        criteria_states: list[CriterionState],
        screenshot_b64: str,
    ) -> list[dict[str, Any]]:

        criteria_text = "\n".join(
            f"{s.index}. {s.text}:{s.status}" for s in criteria_states
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
        ]

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"The task Jeffery attempted is:\n{task_instruction}\n\n"
                    f"Current Criteria:\n{criteria_text}\n\n"
                    "Generate the next step: either explore with a pyautogui action, "
                    "or update exactly one criterion with criteria(option=N, status=...) "
                    "when the current screenshot provides evidence.\n\n"
                    "<observation>\nCurrent screenshot:"
                ),
            },
        ]

        if screenshot_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            })
        user_content.append({"type": "text", "text": "\n</observation>"})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _process_criteria_updates(
        self,
        response_text: str,
        criteria_states: list[CriterionState],
        current_screenshot_b64: str,
    ) -> None:

        for pattern in (CRITERIA_TOOL_RE, CRITERIA_BARE_RE):
            for match in pattern.finditer(response_text):
                option = int(match.group(1))
                new_status = match.group(2).lower()
                for state in criteria_states:
                    if state.index == option:
                        state.status = new_status
                        if current_screenshot_b64:
                            state.evidence_screenshot = current_screenshot_b64
                        break

    @staticmethod
    def _extract_action(text: str) -> str:

        text = text.strip()
        if text.upper() in ("DONE", "FAIL", "WAIT"):
            return ""
        if "criteria(" in text.lower():
            return ""
        if "```python" in text:
            start = text.index("```python") + len("```python")
            end_idx = text.find("```", start)
            end = end_idx if end_idx != -1 else len(text)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            newline = text.find("\n", start)
            if newline != -1:
                start = newline + 1
            end_idx = text.find("```", start)
            end = end_idx if end_idx != -1 else len(text)
            return text[start:end].strip()
        if "pyautogui" in text or "import" in text:
            return text
        return ""

    async def _call_llm(self, messages: list[dict[str, Any]]) -> str:

        try:
            response = await self._openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Probe LLM call failed: %s", exc)
            raise
