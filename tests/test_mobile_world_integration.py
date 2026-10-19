from __future__ import annotations

import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from eval_runtime.guirl.evaluator import _resolve_probe_entry
from eval_runtime.mobile_world.dataset import (
    load_mobile_world_dataset,
    materialize_episode,
)
from eval_runtime.mobile_world.env_client import (
    MobileWorldEvalEnvClient,
    MobileWorldStepResult,
    validate_mobile_action,
)
from eval_runtime.mobile_world.vagen_agent import (
    MOBILE_USE_TOOL,
    MobileWorldClaudeEvalAgent,
    MobileWorldQwenVagenEvalAgent,
)
from eval_runtime.vagen_active.agent_factory import build_vagen_eval_agent

def _png(color: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 8), color=color).save(buffer, format="PNG")
    return buffer.getvalue()

class _Response:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        return self.payload

class _Session:
    def __init__(self):
        self.calls = []
        self.screenshot = _png()

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else "/"
        if path == "/health":
            return _Response({"ok": True})
        if path == "/screenshot":
            return _Response(
                {"b64_png": base64.b64encode(self.screenshot).decode("ascii")}
            )
        if path == "/task/eval":
            return _Response({"score": 1.0, "reason": "Success"})
        return _Response({"result": "OK"})

    def close(self):
        return None

class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parents[1] / "config/mobile_world_datasets/mobileworld.yaml"
        import yaml
        config = yaml.safe_load(config_path.read_text())
        dataset_root = Path(__file__).resolve().parents[1] / config["paths"]["dataset_root"]
        if not dataset_root.is_dir():
            raise unittest.SkipTest("Install the MobileWorld replay dataset to run dataset integration tests")
        cls.dataset = load_mobile_world_dataset("mobileworld")

    def test_registered_dataset_has_one_record_per_task(self):
        self.assertEqual(len(self.dataset.records), 100)
        self.assertEqual(
            len({record.task_name for record in self.dataset.records}),
            100,
        )

    def test_materializer_uses_native_action_tool(self):
        episode = self.dataset.load_episode(self.dataset.records[0])
        with tempfile.TemporaryDirectory() as tmp:
            output = materialize_episode(episode, Path(tmp) / "episode")
            rows = [
                json.loads(line)
                for line in (output / "traj.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), len(episode.steps))
            self.assertTrue(all(row["action_tool"] == "mobile_use" for row in rows))
            self.assertEqual(
                rows[-1]["screenshot_semantics"],
                "fallback_last_pre_action_observation",
            )
            self.assertEqual(
                json.loads(rows[0]["action"]),
                episode.steps[0].action,
            )

    def test_materializer_accepts_exact_replay_frames(self):
        episode = self.dataset.load_episode(self.dataset.records[0])
        frames = [_png("blue") for _ in episode.steps]
        with tempfile.TemporaryDirectory() as tmp:
            output = materialize_episode(
                episode,
                Path(tmp) / "episode",
                initial_screenshot=_png("red"),
                post_action_screenshots=frames,
                native_score=1.0,
            )
            metadata = json.loads(
                (output / "episode.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["final_screenshot_from_live_replay"])
            self.assertEqual((output / "step_1.png").read_bytes(), frames[0])

class EnvClientTest(unittest.TestCase):
    def test_native_action_validation(self):
        self.assertEqual(
            validate_mobile_action({"action_type": "click", "x": 1.2, "y": 3.6}),
            {"action_type": "click", "x": 1, "y": 4},
        )
        with self.assertRaises(ValueError):
            validate_mobile_action({"action_type": "click", "x": 1})
        with self.assertRaises(ValueError):
            validate_mobile_action({"action_type": "computer", "x": 1, "y": 2})

    def test_terminal_marker_is_not_sent_to_step_endpoint(self):
        session = _Session()
        client = MobileWorldEvalEnvClient(
            "http://mobile.test",
            step_wait_time=0,
            health_poll_interval=0,
            session=session,
        )
        client.ensure_initialized()
        client.step({"action_type": "finished"})
        step_calls = [
            call for call in session.calls if call[1].endswith("/step")
        ]
        self.assertEqual(step_calls, [])

    def test_click_uses_native_step_endpoint(self):
        session = _Session()
        client = MobileWorldEvalEnvClient(
            "http://mobile.test",
            step_wait_time=0,
            session=session,
        )
        result = client.step({"action_type": "click", "x": 2, "y": 3})
        self.assertTrue(result.executed)
        step_calls = [
            call for call in session.calls if call[1].endswith("/step")
        ]
        self.assertEqual(len(step_calls), 1)
        self.assertEqual(
            step_calls[0][2]["json"]["action"]["action_type"],
            "click",
        )

class ActiveEvaluatorTest(unittest.TestCase):
    @staticmethod
    def _vagen_agent() -> MobileWorldClaudeEvalAgent:
        return MobileWorldClaudeEvalAgent(
            platform="MobileWorld",
            model="test-model",
            max_tokens=128,
            api_url="http://model.test",
            api_key="test",
            screen_size=(1080, 2400),
        )

    def test_guirl_mobile_probe_registry(self):
        qwen_cls, qwen_runner = _resolve_probe_entry(
            "MobileWorldQwen3VL_Probe"
        )
        owl_cls, owl_runner = _resolve_probe_entry(
            "MobileWorldGUIOwl_Probe"
        )
        self.assertIn("Qwen3VL", qwen_cls.__name__)
        self.assertIn("GUIOwl", owl_cls.__name__)
        self.assertIs(qwen_runner, owl_runner)

    def test_vagen_exposes_mobile_use_not_computer(self):
        function = MOBILE_USE_TOOL["function"]
        self.assertEqual(function["name"], "mobile_use")
        action_types = function["parameters"]["properties"]["action_type"]["enum"]
        self.assertIn("click", action_types)
        self.assertIn("drag", action_types)
        self.assertNotIn("computer", action_types)

    def test_vagen_scales_model_view_coordinates_to_native_device(self):
        action = self._vagen_agent()._model_action_to_native(
            {
                "action_type": "drag",
                "start_x": 324,
                "start_y": 720,
                "end_x": 647,
                "end_y": 1439,
            }
        )
        self.assertEqual(
            action,
            {
                "action_type": "drag",
                "start_x": 540,
                "start_y": 1200,
                "end_x": 1078,
                "end_y": 2398,
            },
        )

    def test_vagen_sends_predictable_model_view_image_size(self):
        encoded = self._vagen_agent()._encode_model_view_image(
            Image.new("RGB", (1080, 2400), "white")
        )
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.size, (648, 1440))

    def test_vagen_retries_noncompliant_forced_final_judgment(self):
        class ScriptedAgent(MobileWorldClaudeEvalAgent):
            def __init__(self):
                super().__init__(
                    platform="MobileWorld",
                    model="test-model",
                    max_tokens=128,
                    api_url="http://model.test",
                    api_key="test",
                )
                self.responses = iter(
                    [
                        [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "mobile_use",
                                "input": {"action_type": "wait"},
                            }
                        ],
                        [
                            {
                                "type": "text",
                                "text": "I should continue checking the screen.",
                            }
                        ],
                        [
                            {
                                "type": "text",
                                "text": (
                                    "EVALUATION RESULT:\n"
                                    "Reasoning: A required step is missing.\n"
                                    "Status: FAILURE\n"
                                    "Confidence: HIGH"
                                ),
                            }
                        ],
                    ]
                )

            def _call_eval_model(self, **_):
                return next(self.responses)

        class FakeEnv:
            def step(self, action):
                return MobileWorldStepResult(
                    action=action,
                    executed=True,
                    terminal=False,
                    screenshot=_png(),
                    response={},
                )

            def screenshot(self, *, wait=False):
                return _png()

        agent = ScriptedAgent()
        with tempfile.TemporaryDirectory() as tmp:
            result = agent.eval_task(
                env=FakeEnv(),
                task_config={"id": "test-task", "instruction": "Do the task"},
                trajectory={
                    "screenshots": [_png()],
                    "actions": [{"raw_response": "No required action was taken"}],
                },
                save_dir=tmp,
                max_round=1,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["confidence"], "HIGH")
        self.assertEqual(result["parse_method"], "full_format")

    def test_vagen_result_parser_matches_native_semantics(self):
        result = MobileWorldClaudeEvalAgent._parse_eval_result(
            "EVALUATION RESULT:\n"
            "Reasoning: The final state is correct.\n"
            "Status: SUCCESS\n"
            "Confidence: HIGH",
            2,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["parse_method"], "full_format")

    def test_qwen_factory_uses_native_mobile_agent(self):
        agent = build_vagen_eval_agent(
            {
                "evaluator_type": "vagen_qwen",
                "platform": "MobileWorld",
                "model": "test-model",
                "api_url": "http://model.test/v1",
                "api_key": "EMPTY",
                "max_tokens": 128,
                "screen_width": 1080,
                "screen_height": 2400,
                "only_n_most_recent_images_eval": 3,
            }
        )
        self.assertIsInstance(agent, MobileWorldQwenVagenEvalAgent)
        self.assertEqual(agent.model_name, "test-model")

if __name__ == "__main__":
    unittest.main()
