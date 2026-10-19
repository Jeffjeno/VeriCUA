from __future__ import annotations

import base64
import io
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from mm_agents.reward_model.base_evaluator import BaseEvaluator
from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample


GENERAL_SYSTEM_PROMPT = """You're an expert in evaluating whether the Screenshot successfully completes the Task.""".strip()


def build_prompt_general(intent: str) -> tuple[str, list[str], list[str]]:
    prompts = [
        """Task: Open the settings.
Q: What should I expect to see on the screenshot if I've opened the settings?
A: I should expect to see I'm in the settings app. The screenshot shows the home screen of a mobile device, with various app icons displayed, including the settings app icon, but the settings app is not opened.
Status: failure""",
        """Task: Find hotels in washington dc
Q: What should I expect to see on the screenshot if I've searched for hotels in Washington, DC?
A: I should expect to see I'm in a search results page for hotels in Washington, DC. The screenshot shows a Google search page with the search field populated with the query "hotels in washington dc" and a list of suggested searches related to hotels in Washington, DC, but it does not show any search results for hotels in Washington, DC.
Status: failure""",
        """Task: What's a good restaurant in Portland?
Q: What should I expect to see on the screenshot if I've searched for a good restaurant in Portland?
A: I should expect to see I'm in a search results page for a good restaurant in Portland. The screenshot shows a Google search page with a search input field for "good restaurant in portland" and a map results preview showing business locations near Portland, like "Li Pigeon", "Portland City Grill", and "Higgins",
Status: success""",
        """Task: What's on the menu at In-N-Out?
Q: What should I expect to see on the screenshot if I've searched for the menu at In-N-Out?
A: I should expect to see a menu page for In-N-Out, including product names, thumbnails and prices. The screenshot shows a Google search page with a search input field for "In-N-Out menu" and some page snippets of In-N-Out indicating potential menu items, but does not actually show the actual menu.
Status: failure""",
        """Task: What's the news in Suriname?
Q: What should I expect to see on the screenshot if I've searched for the news in Suriname?
A: I should expect to see some news in Suriname, such as someone did something or some accident happens in Suriname. The screenshot shows a Google search page with a search input field for "Suriname news today" and some page snippets indicating potential news items, but does not actually show the news.
Status: failure""",
        """Task: What's the weather like in Chicago?
Q: What should I expect to see on the screenshot if I've searched for the weather in Chicago?
A: I should expect to see some exact values like temperature, humidity, wind speed, and weather condition in Chicago. The screenshot shows a Google search page with a search input field for "weather in Chicago" and some page snippets indicating potential weather information. Although one page snippet contains some weather information, the information is not comprehensive enough to determine the weather in Chicago.
Status: failure""",
        """Task: Set an alarm for 6pm.
Q: What should I expect to see on the screenshot if I've set an alarm for 6pm?
A: I should expect to see some alarms including a 6pm alarm activated in the clock app. The screenshot shows an attempt to set an alarm for 6pm in the clock app, but the alarm is not set yet.
Status: failure""",
        """Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com but blocked with a cookie consent banner.
Status: failure""",
        """Task: What's the news in French today?
Q: What should I expect to see on the screenshot if I've searched for the news in French today?
A: I should expect to see some news in French today, such as someone did something or some accident happens in French today. The screenshot shows I'm in the website france24.com and can see the news, like something about the Olympic flame.
Status: success""",
        f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've <repeat the task>?
A: I should expect to see <first expectation, then what's in the given screenshot.>
Status: success or failure (don't return anything else)
Start with "Q:".""",
    ]

    image_dir = Path(__file__).resolve().parent / "assets" / "images"
    few_shot_images = [
        image_dir / "screenshot_menu.png",
        image_dir / "screenshot_hotel.png",
        image_dir / "screenshot_restaurant.png",
        image_dir / "screenshot_foodmenu.png",
        image_dir / "screenshot_news.png",
        image_dir / "screenshot_weather.png",
        image_dir / "screenshot_alarm.png",
        image_dir / "screenshot_frenchnews_blocked.png",
        image_dir / "screenshot_frenchnews_okay.png",
    ]
    return GENERAL_SYSTEM_PROMPT, prompts, [str(path) for path in few_shot_images]


def parse_output(text: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "think": None,
        "question": None,
        "answer": None,
        "status": None,
    }
    if not text:
        return result

    text = text.strip()
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    body = text
    if think_match:
        result["think"] = think_match.group(1).strip()
        body = text[: think_match.start()] + text[think_match.end() :]

    q_match = re.search(r"Q:\s*(.*?)(?=\nA:|\Z)", body, flags=re.DOTALL | re.IGNORECASE)
    a_match = re.search(r"A:\s*(.*?)(?=\nStatus:|\Z)", body, flags=re.DOTALL | re.IGNORECASE)
    status_match = re.findall(r"Status:\s*(success|failure|uncertain)", body, flags=re.IGNORECASE)

    if q_match:
        result["question"] = q_match.group(1).strip()
    if a_match:
        result["answer"] = a_match.group(1).strip()
    if status_match:
        result["status"] = status_match[-1].lower()
    return result


class DigirlEvaluator(BaseEvaluator):
    def __init__(self, llm, history_n: int = 3) -> None:
        super().__init__(llm)
        self.history_n = max(int(history_n), 1)

    def _evaluate_sync(self, task: TaskSample) -> EvaluationResult:
        messages = self._build_messages(task)
        response = self.llm.chat_completion_sync(messages)
        content = self._extract_response_text(response)
        parsed = parse_output(content)
        decision, decision_raw = self._status_to_bool(parsed.get("status"), content)
        qa_lines = []
        if parsed.get("question"):
            qa_lines.append(f"Q: {parsed['question']}")
        if parsed.get("answer"):
            qa_lines.append(f"A: {parsed['answer']}")
        qa_text = "\n".join(qa_lines) if qa_lines else content
        return self._result(
            task,
            decision,
            content=qa_text,
            reasoning=parsed.get("think") or "",
            method="digirl",
            decision_raw=decision_raw,
            history_n=self.history_n,
            judge_msg=messages,
            memory_msgs=[],
        )

    async def _evaluate_async(self, task: TaskSample) -> EvaluationResult:
        response = await self.llm.chat_completion_async(self._build_messages(task))
        content = self._extract_response_text(response)
        parsed = parse_output(content)
        decision, decision_raw = self._status_to_bool(parsed.get("status"), content)
        qa_lines = []
        if parsed.get("question"):
            qa_lines.append(f"Q: {parsed['question']}")
        if parsed.get("answer"):
            qa_lines.append(f"A: {parsed['answer']}")
        qa_text = "\n".join(qa_lines) if qa_lines else content
        return self._result(
            task,
            decision,
            content=qa_text,
            reasoning=parsed.get("think") or "",
            method="digirl",
            decision_raw=decision_raw,
            history_n=self.history_n,
            judge_msg=self._build_messages(task),
            memory_msgs=[],
        )

    def _prepare(self, task: TaskSample) -> TaskSample:
        prepared = self._clone_task(task)
        history_images = list(prepared.history_images or [])
        history_actions = list(prepared.history_actions or [])
        if history_images:
            paired = list(zip(history_images, history_actions))
            paired = paired[-self.history_n :]
            prepared.history_images = [img for img, _ in paired]
            prepared.history_actions = [action for _, action in paired]
        else:
            prepared.history_actions = []
        prepared = super()._prepare(prepared)
        prepared.current_image = self._prepare_digirl_image_field(prepared.current_image)
        prepared.history_images = [
            self._prepare_digirl_image_field(image_ref)
            for image_ref in prepared.history_images
        ]
        return prepared

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
                method="digirl",
                missing_fields=missing_fields,
            )
        return None

    def _build_messages(self, task: TaskSample) -> list[dict[str, Any]]:
        system_msg, prompts, few_shot_image_paths = build_prompt_general(task.instruction)
        content: list[dict[str, Any]] = [{"type": "text", "text": system_msg + "\n=====Examples====="}]

        for image_path, prompt in zip(few_shot_image_paths, prompts[:-1]):
            if not Path(image_path).exists():
                continue
            few_shot_url = self._prepare_digirl_image_field(image_path)
            content.append({"type": "text", "text": "\nScreenshot:"})
            content.append({"type": "image_url", "image_url": {"url": few_shot_url}})
            content.append({"type": "text", "text": prompt})

        content.append({"type": "text", "text": "=====Your Turn====="})
        content.append({"type": "text", "text": "\nScreenshot: "})
        content.append({"type": "image_url", "image_url": {"url": task.current_image}})
        content.append({"type": "text", "text": prompts[-1]})

        return [{"role": "user", "content": content}]

    @classmethod
    def _prepare_digirl_image_field(cls, image_ref: str) -> str:
        if not image_ref:
            raise ValueError("image reference is empty")
        if cls._is_remote_url(image_ref):
            return image_ref
        image = cls._load_image(image_ref)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def _load_image(cls, image_ref: str) -> Image.Image:
        if cls._is_data_url(image_ref):
            header, _, encoded = image_ref.partition(",")
            if not encoded:
                raise ValueError("malformed data image url")
            image_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_bytes)).copy()

        image_path = cls._resolve_image_path(image_ref)
        with Image.open(image_path) as image:
            return image.copy()

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
    def _status_to_bool(status: str | None, raw_text: str) -> tuple[bool | None, str]:
        if status == "success":
            return True, "success"
        if status == "failure":
            return False, "failure"
        normalized = raw_text.strip().lower()
        if "status:" in normalized and "uncertain" in normalized:
            return None, "uncertain"
        return None, "parse_error"


__all__ = ["DigirlEvaluator", "build_prompt_general", "parse_output"]
