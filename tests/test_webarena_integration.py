from __future__ import annotations

import copy
import io
import os
from types import SimpleNamespace
import unittest
from pathlib import Path

from PIL import Image

from eval_runtime.guirl.evaluator import build_guirl_config
from eval_runtime.vagen_active.agent_factory import (
    build_vagen_eval_agent,
)
from eval_runtime.web_arena.vagen_agent import (
    BROWSER_USE_TOOL,
    CHECK_SCREENSHOT_TOOL,
    WebArenaQwenVagenEvalAgent,
    browser_action_from_qwen,
)

class WebArenaQwenProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("WA_EXTERNAL_HOST")
        os.environ["WA_EXTERNAL_HOST"] = "wa-test"
        os.environ["WA_INSTANCE"] = "0"
        os.environ["WA_PORT_STRIDE"] = "1000"
        os.environ["WA_SHARE_READONLY"] = "1"

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("WA_EXTERNAL_HOST", None)
        else:
            os.environ["WA_EXTERNAL_HOST"] = self._old

    def test_browser_actions_use_browsergym_surface(self):
        action, error = browser_action_from_qwen(
            {"action": "click", "coordinate": [12, 34]},
            width=1280,
            height=720,
            current_url="http://wa-test:8023/users/sign_in",
        )
        self.assertIsNone(error)
        self.assertEqual(action, "mouse_click(12, 34, button='left')")

        action, error = browser_action_from_qwen(
            {"action": "scroll", "direction": "down", "amount": 120},
            width=1280,
            height=720,
            current_url="http://wa-test:8023/",
        )
        self.assertIsNone(error)
        self.assertEqual(action, "scroll(0, 120)")

    def test_browser_probe_rejects_mutating_or_external_actions(self):
        action, error = browser_action_from_qwen(
            {"action": "type", "text": "submit", "press_enter": True},
            width=1280,
            height=720,
            current_url="http://wa-test:8023/",
        )
        self.assertIsNone(action)
        self.assertIn("submission", error or "")

        action, error = browser_action_from_qwen(
            {"action": "goto", "url": "https://example.com/"},
            width=1280,
            height=720,
            current_url="http://wa-test:8023/",
        )
        self.assertIsNone(action)
        self.assertIn("outside", error or "")

    def test_factory_selects_webarena_qwen_agent(self):
        cfg = {
            "evaluator_type": "vagen_qwen",
            "platform": "WebArena",
            "model": "test-model",
            "api_url": "http://model.test/v1",
            "api_key": "EMPTY",
            "max_tokens": 128,
            "screen_width": 1280,
            "screen_height": 720,
            "only_n_most_recent_images_eval": 3,
        }
        agent = build_vagen_eval_agent(cfg)
        self.assertIsInstance(agent, WebArenaQwenVagenEvalAgent)

    def test_suite_config_builds_qwen4b_judge_and_native_probe(self):
        path = Path(__file__).parents[1] / "suites" / "webarena" / "configs" / "guirl_qwen4b_holo.yaml"
        cfg = build_guirl_config(str(path))
        self.assertEqual(cfg["judge_model_name"], "Qwen3-VL-4B-Thinking")
        self.assertEqual(cfg["probe_agent_type"], "WebArenaHolo_Probe")

    def test_webarena_declares_native_qwen_tools(self):
        agent = WebArenaQwenVagenEvalAgent(
            model="test-model",
            api_url="http://model.test/v1",
            api_key="EMPTY",
            max_tokens=128,
        )
        tools = agent._request_tools()
        self.assertEqual([tool["function"]["name"] for tool in tools], [
            "browser_use", "check_screenshot",
        ])
        browser_parameters = BROWSER_USE_TOOL["function"]["parameters"]
        self.assertIn("action", browser_parameters["required"])
        self.assertFalse(browser_parameters["additionalProperties"])
        self.assertFalse(CHECK_SCREENSHOT_TOOL["function"]["parameters"]["additionalProperties"])

    def test_native_tool_call_uses_openai_tool_result_history(self):
        agent = WebArenaQwenVagenEvalAgent(
            model="test-model",
            api_url="http://model.test/v1",
            api_key="EMPTY",
            max_tokens=128,
            only_n_most_recent_images_eval=2,
        )
        screenshot_buffer = io.BytesIO()
        Image.new("RGB", (1280, 720), "white").save(screenshot_buffer, format="PNG")
        screenshot = screenshot_buffer.getvalue()

        class FakeEnv:
            def get_obs(self, _action):
                return {
                    "screenshot": screenshot,
                    "url": "http://wa-test:8023/",
                }

            def step_browser(self, _command):
                return self.get_obs("")

        native_message = SimpleNamespace(
            content=None,
            reasoning_content="",
            tool_calls=[SimpleNamespace(
                id="call_browser_1",
                function=SimpleNamespace(
                    name="browser_use",
                    arguments='{"action":"wait","duration":0}',
                ),
            )],
        )
        final_message = SimpleNamespace(
            content=(
                "EVALUATION RESULT:\nStatus: SUCCESS\n"
                "Confidence: HIGH\nReasoning: final state is verified"
            ),
            reasoning_content="",
            tool_calls=None,
        )
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=native_message)]),
            SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
        ]
        captured_messages = []

        def fake_request(messages, tools=None):
            captured_messages.append(copy.deepcopy(messages))
            self.assertEqual([tool["function"]["name"] for tool in tools or []], [
                "browser_use", "check_screenshot",
            ])
            return responses.pop(0)

        agent._request = fake_request
        result = agent.eval_task(
            FakeEnv(),
            {"id": "native-tool-test", "instruction": "Inspect the page."},
            {"actions": [], "screenshots": [screenshot]},
            "/tmp/webarena_qwen_native_tool_test",
            max_round=2,
        )
        self.assertTrue(result["success"])
        history = captured_messages[1]
        tool_messages = [message for message in history if message["role"] == "tool"]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_browser_1")
        self.assertIn("Browser inspection action executed.", tool_messages[0]["content"])
        self.assertTrue(any(
            message["role"] == "user"
            and any(part.get("type") == "image_url" for part in message.get("content") or [])
            for message in history
        ))
        self.assertTrue(Path("/tmp/webarena_qwen_native_tool_test/eval_result/round_1.png").is_file())

if __name__ == "__main__":
    unittest.main()
