from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from mm_agents.base_agent import Agent
from mm_agents.reward_model.agent_judge_independent import (
    IndependentRolloutAgentJudgeEvaluator as DefaultIndependentRolloutAgentJudgeEvaluator,
)
from mm_agents.reward_model.evaluation_result import EvaluationResult
from mm_agents.reward_model.task_sample import TaskSample


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid replay eval config: {path}")
    return payload


def _action_to_text(action: Any) -> str:
    if isinstance(action, str):
        return action
    return json.dumps(action, ensure_ascii=False)


def _label_from_result(result: EvaluationResult) -> str:
    extra = dict(result.extra or {})
    if result.status == "error" or extra.get("decision_raw") == "parse_error":
        return "error"
    value = result.llm_judge
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "uncertain"


def _data_image_debug_ref(url: str) -> str:
    if not url.startswith("data:image/"):
        return url
    base64_part = url.split(",", 1)[1] if "," in url else url
    return f"<data_image_base64_prefix:{base64_part[:10]} len={len(url)}>"


def _image_url_map_for_summary(task: TaskSample) -> dict[str, str]:
    task_extra = dict(task.extra or {})
    raw_current_image = task_extra.get("_reward_input_current_image")
    raw_history_images = task_extra.get("_reward_input_history_images") or []
    if not isinstance(raw_history_images, list):
        raw_history_images = []

    raw_images = [*raw_history_images]
    if raw_current_image:
        raw_images.append(raw_current_image)

    prepared_images = [*list(task.history_images or [])]
    if task.current_image:
        prepared_images.append(task.current_image)

    image_url_map: dict[str, str] = {}
    for prepared, raw in zip(prepared_images, raw_images):
        if prepared and raw:
            image_url_map[str(prepared)] = str(raw)
    return image_url_map


def _replace_image_urls_for_summary(obj: Any, image_url_map: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _data_image_debug_ref(obj)
    if isinstance(obj, list):
        return [_replace_image_urls_for_summary(item, image_url_map) for item in obj]
    if isinstance(obj, dict):
        output: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "image_url" and isinstance(value, dict):
                image_url = dict(value)
                url = image_url.get("url")
                if isinstance(url, str):
                    image_url["url"] = image_url_map.get(url, _data_image_debug_ref(url))
                output[key] = _replace_image_urls_for_summary(image_url, image_url_map)
            else:
                output[key] = _replace_image_urls_for_summary(value, image_url_map)
        return output
    return obj


def _load_traj_items(
    result_dir: str,
    *,
    source: str,
    strict: bool,
) -> list[dict[str, Any]]:
    traj_path = Path(result_dir) / "traj.jsonl"
    if not traj_path.exists():
        if strict:
            raise FileNotFoundError(f"traj.jsonl not found: {traj_path}")
        return []

    items: list[dict[str, Any]] = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "step_num" not in record:
                continue
            screenshot_file = record.get("screenshot_file")
            if not screenshot_file:
                continue
            items.append(
                {
                    "source": source,
                    "timestamp": str(record.get("action_timestamp", "")),
                    "step_num": int(record.get("step_num", 0) or 0),
                    "action_text": _action_to_text(record.get("action", "")),
                    "screenshot_path": str((Path(result_dir) / screenshot_file).resolve()),
                }
            )
    return items


def build_replay_eval_task_sample(
    *,
    example: dict[str, Any],
    example_result_dir: str,
    policy_model: str,
    domain: str,
    example_id: str,
    rollout_id: str,
    ground_truth: bool | None = None,
    sampler_result_dir: str | None = None,
    include_sampler: bool = False,
) -> TaskSample:
    policy_items = _load_traj_items(example_result_dir, source="policy", strict=True)
    sampler_items: list[dict[str, Any]] = []
    if include_sampler and sampler_result_dir:
        sampler_items = _load_traj_items(
            sampler_result_dir,
            source="sampler",
            strict=False,
        )

    traj_items = policy_items + sampler_items
    traj_items.sort(
        key=lambda item: (
            item["timestamp"],
            0 if item["source"] == "policy" else 1,
            item["step_num"],
        )
    )
    if not traj_items:
        raise ValueError(f"empty trajectory for replay eval: {domain}/{example_id}")

    screenshot_paths = [item["screenshot_path"] for item in traj_items]
    action_texts = [item["action_text"] for item in traj_items]

    env_result = None
    result_path = Path(example_result_dir) / "result.txt"
    if result_path.exists():
        try:
            env_result = float(result_path.read_text(encoding="utf-8").strip())
        except Exception:
            env_result = None

    if ground_truth is None and env_result is not None:
        ground_truth = env_result >= 1.0

    return TaskSample(
        task_id=f"{example_id}-{rollout_id}",
        instruction=example["instruction"],
        policy_model=policy_model,
        ground_truth=ground_truth,
        current_image=screenshot_paths[-1],
        history_images=screenshot_paths[:-1],
        last_action=action_texts[-1],
        history_actions=action_texts[:-1],
        extra={
            "domain": domain,
            "example_id": example_id,
            "rollout_id": str(rollout_id),
            "example_result_dir": str(Path(example_result_dir).resolve()),
            "env_result": env_result,
            "include_sampler": include_sampler,
            "policy_steps": len(policy_items),
            "sampler_steps": len(sampler_items),
            "_reward_input_current_image": screenshot_paths[-1],
            "_reward_input_history_images": screenshot_paths[:-1],
        },
    )


@Agent.register("ReplayEval")
class ReplayEvalAgent(Agent):
    """Three-way replay evaluator using the same judge flow as SFT data making."""

    LOGGER_NAME = "ReplayEvalAgent"

    def __init__(self, config_path: str | None = None, **kwargs: Any):
        super().__init__(config_path=None)
        self.config_path = config_path
        cfg = _load_yaml(config_path) if config_path else dict(kwargs)
        model_cfg = dict(cfg.get("model") or cfg)

        model_name = model_cfg.get("model_name") or model_cfg.get("model")
        if not model_name:
            raise ValueError("replay eval config requires model_name")
        for required in ("base_url", "api_key"):
            if not model_cfg.get(required):
                raise ValueError(f"replay eval config requires {required}")

        llm = SimpleNamespace(
            model=model_name,
            base_url=model_cfg["base_url"],
            api_key=model_cfg["api_key"],
            temperature=model_cfg.get("temperature", 0),
            max_tokens=model_cfg.get("max_tokens", 8192),
            top_p=model_cfg.get("top_p"),
            top_k=model_cfg.get("top_k"),
            min_p=model_cfg.get("min_p"),
            presence_penalty=model_cfg.get("presence_penalty"),
            repetition_penalty=model_cfg.get("repetition_penalty"),
            enable_thinking=model_cfg.get("enable_thinking", True),
            request_timeout=model_cfg.get("request_timeout"),
        )
        self.model_name = str(model_name)
        self.window = int(cfg.get("window", 6))
        self.stride = int(cfg.get("stride", 5))
        self.history_n = int(cfg.get("history_n", 3))
        self.evaluator_type = str(
            cfg.get("evaluator_type")
            or cfg.get("judge_evaluator")
            or cfg.get("agent_judge_type")
            or "default"
        ).lower()
        if self.evaluator_type in {"memory", "agentjudge_memory", "compact_memory"}:
            from mm_agents.reward_model.agent_judge_memory import (
                IndependentRolloutAgentJudgeEvaluator,
            )

            self.evaluator = IndependentRolloutAgentJudgeEvaluator(
                llm,
                window=self.window,
                stride=self.stride,
                history_n=self.history_n,
                rollout_num=1,
                memory_max_tokens=cfg.get("memory_max_tokens", 8192),
                store_raw_messages=bool(cfg.get("store_raw_messages", True)),
            )
        elif self.evaluator_type in {"quick", "quick_judge", "agent_quick_judge"}:
            from mm_agents.reward_model.agent_quick_judge import (
                IndependentRolloutAgentJudgeEvaluator,
            )

            self.evaluator = IndependentRolloutAgentJudgeEvaluator(
                llm,
                window=self.window,
                stride=self.stride,
                history_n=self.history_n,
                rollout_num=1,
            )
        elif self.evaluator_type in {"digirl"}:
            from mm_agents.reward_model.baseline.digirl import DigirlEvaluator

            self.evaluator = DigirlEvaluator(
                llm,
                history_n=self.history_n,
            )
        elif self.evaluator_type in {"distrl"}:
            from mm_agents.reward_model.baseline.distrl import DistrlEvaluator

            self.evaluator = DistrlEvaluator(
                llm,
                task_set=str(cfg.get("task_set", "general")),
            )
        elif self.evaluator_type in {"zerogui"}:
            from mm_agents.reward_model.baseline.zerogui import ZeroGuiEvaluator

            self.evaluator = ZeroGuiEvaluator(
                llm,
                max_screenshots=int(cfg.get("max_screenshots", 0)),
            )
        elif self.evaluator_type in {"default", "agentjudge", "independent"}:
            self.evaluator = DefaultIndependentRolloutAgentJudgeEvaluator(
                llm,
                window=self.window,
                stride=self.stride,
                history_n=self.history_n,
                rollout_num=1,
            )
        else:
            raise ValueError(f"unsupported replay eval evaluator_type: {self.evaluator_type}")

    def reset(self, **kwargs: Any) -> None:
        return None

    def predict(self, instruction: str, obs: dict[str, Any], **kwargs: Any) -> Any:
        raise NotImplementedError("ReplayEvalAgent is a post-rollout evaluator.")

    def evaluate_rollout(
        self,
        *,
        example: dict[str, Any],
        example_result_dir: str,
        policy_model: str,
        domain: str,
        example_id: str,
        rollout_id: str,
        ground_truth: bool | None = None,
        sampler_result_dir: str | None = None,
        include_sampler: bool = False,
        sampler_checklist: list[str] | None = None,
        sampler_checklist_report: str | None = None,
        sampler_generated_task: str | None = None,
    ) -> dict[str, Any]:
        del sampler_checklist, sampler_checklist_report, sampler_generated_task
        task_sample = build_replay_eval_task_sample(
            example=example,
            example_result_dir=example_result_dir,
            policy_model=policy_model,
            domain=domain,
            example_id=example_id,
            rollout_id=rollout_id,
            ground_truth=ground_truth,
            sampler_result_dir=sampler_result_dir,
            include_sampler=include_sampler,
        )
        result_or_results = self.evaluator.evaluate_sync(task_sample)
        results = (
            result_or_results
            if isinstance(result_or_results, list)
            else [result_or_results]
        )
        if len(results) != 1:
            raise RuntimeError(f"expected one replay eval result, got {len(results)}")
        return self.result_to_summary(results[0])

    @staticmethod
    def result_to_summary(result: EvaluationResult) -> dict[str, Any]:
        task = result.task_sample
        extra = dict(result.extra or {})
        image_url_map = _image_url_map_for_summary(task)
        memory_context = _replace_image_urls_for_summary(
            copy.deepcopy(extra.get("memory_msgs")),
            image_url_map,
        )
        judge_context = _replace_image_urls_for_summary(
            copy.deepcopy(extra.get("judge_msg")),
            image_url_map,
        )
        return {
            "label": _label_from_result(result),
            "llm_judge": result.llm_judge,
            "decision_raw": extra.get("decision_raw"),
            "status": result.status,
            "content": result.content,
            "error_message": result.error_message,
            "duration": result.duration,
            "reward_model": result.reward_model,
            "task_id": task.task_id,
            "policy_model": task.policy_model,
            "ground_truth": task.ground_truth,
            "policy_steps": task.extra.get("policy_steps"),
            "sampler_steps": task.extra.get("sampler_steps"),
            "include_sampler": task.extra.get("include_sampler", False),
            "history_n": extra.get("history_n", 3),
            "mr_window": extra.get("mr_window"),
            "mr_stride": extra.get("mr_stride"),
            "compact_memory": extra.get("compact_memory"),
            "memory_trace": copy.deepcopy(extra.get("memory_trace")),
            "store_raw_messages": extra.get("store_raw_messages", False),
            "memory_context": memory_context,
            "judge_context": judge_context,
        }


__all__ = ["ReplayEvalAgent", "build_replay_eval_task_sample"]
