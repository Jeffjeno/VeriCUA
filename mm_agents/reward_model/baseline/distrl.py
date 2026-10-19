from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from PIL import Image

from mm_agents.reward_model.base_evaluator import BaseEvaluator
from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample


SYSTEM_PROMPT = """You're an expert in evaluating whether the Screenshot successfully completes the Task.""".strip()


def _image_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "images"


def build_prompt_webshop(intent: str) -> tuple[str, list[str], list[str]]:
    prompts = [
        """Task: Go to bestbuy.com
Q: What should I expect to see on the screenshot if I've gone to bestbuy.com?
A: I should expect to see I'm on the Best Buy website, which usually shows the Best Buy logo with some featured products and categories. The screenshot shows I'm searching for "bestbuy.com" in Google search (with some search suggestions) instead of being on the Best Buy website.
Status: failure""",
        f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've {intent}?
A: I should expect to see <describe expected outcome>. The screenshot shows <describe actual screenshot>.
Status: success or failure (don't return anything else).
Start with "Q:".""",
    ]
    image_paths = [
        _image_dir() / "step1_bestbuy.png",
    ]
    return SYSTEM_PROMPT, prompts, [str(path) for path in image_paths]


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
A: I should expect to see I'm in a search results page for a good restaurant in Portland. The screenshot shows a Google search page with a search input field for "good restaurant in portland" and a map results preview showing business locations near Portland, like "Li Pigeon", "Portland City Grill", and "Higgins".
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
Q: What should I expect to see on the screenshot if I've {intent}?
A: I should expect to see <describe expected outcome>. The screenshot shows <describe actual screenshot>.
Status: success or failure (don't return anything else).
Start with "Q:".""",
    ]
    image_paths = [
        _image_dir() / "screenshot_menu.png",
        _image_dir() / "screenshot_hotel.png",
        _image_dir() / "screenshot_restaurant.png",
        _image_dir() / "screenshot_foodmenu.png",
        _image_dir() / "screenshot_news.png",
        _image_dir() / "screenshot_weather.png",
        _image_dir() / "screenshot_alarm.png",
        _image_dir() / "screenshot_frenchnews_blocked.png",
        _image_dir() / "screenshot_frenchnews_okay.png",
    ]
    return SYSTEM_PROMPT, prompts, [str(path) for path in image_paths]


def build_prompt_app_install(intent: str) -> tuple[str, list[str], list[str]]:
    prompts = [
        """Task: Install a browser extension.
Q: What should I expect to see on the screenshot if I've installed the browser extension?
A: I should expect to see the extension visible in the browser toolbar or a clear success state in the extension page. The screenshot shows the extension store page before installation is completed.
Status: failure""",
        f"""Task: {intent}
Respond in this format:
Q: What should I expect to see on the screenshot if I've {intent}?
A: I should expect to see <describe expected outcome>. The screenshot shows <describe actual screenshot>.
Status: success or failure (don't return anything else).
Start with "Q:".""",
    ]
    image_paths = [
        _image_dir() / "app_install_extension.png",
    ]
    return SYSTEM_PROMPT, prompts, [str(path) for path in image_paths]


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


class DistrlEvaluator(BaseEvaluator):
    def __init__(self, llm, task_set: str = "general") -> None:
        super().__init__(llm)
        self.task_set = (task_set or "general").strip().lower()

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
            method="distrl",
            task_set=self.task_set,
            decision_raw=decision_raw,
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
            method="distrl",
            task_set=self.task_set,
            decision_raw=decision_raw,
            judge_msg=self._build_messages(task),
            memory_msgs=[],
        )

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
                method="distrl",
                task_set=self.task_set,
                missing_fields=missing_fields,
            )
        return None

    def _prepare(self, task: TaskSample) -> TaskSample:
        prepared = super()._prepare(task)
        prepared.current_image = self._prepare_distrl_image_field(prepared.current_image)
        prepared.history_images = [
            self._prepare_distrl_image_field(image_ref)
            for image_ref in prepared.history_images
        ]
        return prepared

    def _build_messages(self, task: TaskSample) -> list[dict[str, Any]]:
        system_msg, prompts, few_shot_image_paths = self._select_prompt_bundle(task.instruction)
        content: list[dict[str, Any]] = [{"type": "text", "text": system_msg + "\n=====Examples====="}]

        for image_path, prompt in zip(few_shot_image_paths, prompts[:-1]):
            if not Path(image_path).exists():
                continue
            few_shot_url = self._prepare_distrl_image_field(image_path)
            content.append({"type": "text", "text": "\nScreenshot:"})
            content.append({"type": "image_url", "image_url": {"url": few_shot_url}})
            content.append({"type": "text", "text": prompt})

        content.append({"type": "text", "text": "=====Your Turn====="})
        content.append({"type": "text", "text": "\nScreenshot: "})
        content.append({"type": "image_url", "image_url": {"url": task.current_image}})
        content.append({"type": "text", "text": prompts[-1]})
        return [{"role": "user", "content": content}]

    @classmethod
    def _prepare_distrl_image_field(cls, image_ref: str) -> str:
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
            _, _, encoded = image_ref.partition(",")
            if not encoded:
                raise ValueError("malformed data image url")
            return Image.open(io.BytesIO(base64.b64decode(encoded))).copy()

        image_path = cls._resolve_image_path(image_ref)
        with Image.open(image_path) as image:
            return image.copy()

    def _select_prompt_bundle(self, intent: str) -> tuple[str, list[str], list[str]]:
        if self.task_set == "webshop":
            return build_prompt_webshop(intent)
        if self.task_set == "app_install":
            return build_prompt_app_install(intent)
        return build_prompt_general(intent)

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


__all__ = [
    "DistrlEvaluator",
    "build_prompt_general",
    "build_prompt_webshop",
    "build_prompt_app_install",
    "parse_output",
]
