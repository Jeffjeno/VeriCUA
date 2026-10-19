from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from eval_runtime.agents.base import Agent
from eval_runtime.agents.agent_config import ReplayAgentConfig
from eval_runtime.agents.registry import register

SPECIAL_ACTIONS = {"WAIT", "FAIL", "DONE"}

def _clean_command(command: str) -> str:
    return command.rstrip()

def _normalize_replay_action(record: dict[str, Any]) -> str | None:
    action = record.get("action")
    top_level_command = record.get("command")

    if isinstance(action, str):
        return _clean_command(action)

    if isinstance(action, dict):
        action_type = str(action.get("action_type") or "").upper()
        if action_type in SPECIAL_ACTIONS:
            return action_type

        command = action.get("command")
        if isinstance(command, str) and command.strip():
            return _clean_command(command)

        nested_action = action.get("action")
        if isinstance(nested_action, str):
            nested_action = _clean_command(nested_action)
            if nested_action in SPECIAL_ACTIONS or nested_action.startswith("pyautogui."):
                return nested_action

    if isinstance(top_level_command, str) and top_level_command.strip():
        return _clean_command(top_level_command)

    return None

@register("Replay")
class ReplayAgent(Agent):
    LOGGER_NAME = "ReplayAgent"

    def __init__(self, config_path: str = None, **kwargs: Any):
        super().__init__(config_path=config_path, **kwargs)
        self.config_path = config_path
        if config_path is not None:
            self.config = ReplayAgentConfig.from_yaml(config_path)
        else:
            self.config = ReplayAgentConfig(**kwargs)

        self.logger = logging.getLogger(self.LOGGER_NAME)
        self.source_result_root = self._resolve_source_result_root(
            self.config.source_result_root
        )
        self.source_model_name = (
            self.config.source_model_name
            or os.path.basename(os.path.normpath(self.source_result_root))
        )
        self.stamp = self.config.stamp
        self.default_rollout_id = str(self.config.default_rollout_id)
        self.source_rollout_repeat = max(1, int(self.config.source_rollout_repeat))
        self.strict_missing = bool(self.config.strict_missing)

        self._task_context: Optional[Tuple[str, str, str]] = None
        self._traj_path: Optional[str] = None
        self._actions: List[str] = []
        self._cursor = 0

    def _resolve_source_result_root(self, path: str) -> str:
        root = self._resolve_path(path)
        screenshot_root = os.path.join(root, "pyautogui", "screenshot")
        if not os.path.isdir(screenshot_root):
            return root

        source_model_name = self.config.source_model_name
        if source_model_name:
            candidate = os.path.join(screenshot_root, source_model_name)
            if os.path.isdir(candidate):
                return candidate

        model_dirs = [
            os.path.join(screenshot_root, name)
            for name in sorted(os.listdir(screenshot_root))
            if os.path.isdir(os.path.join(screenshot_root, name))
        ]
        if len(model_dirs) == 1:
            return model_dirs[0]

        raise ValueError(
            "source_result_root contains multiple models under "
            f"{screenshot_root}; set source_model_name explicitly."
        )

    def _resolve_path(self, path: str) -> str:
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        if self.config_path:
            cfg_dir = os.path.dirname(os.path.abspath(self.config_path))
            candidate = os.path.abspath(os.path.join(cfg_dir, expanded))
            if os.path.exists(candidate):
                return candidate
        return os.path.abspath(expanded)

    def set_task_context(self, domain: str, example_id: str, rollout_id: Optional[str] = None) -> None:
        rollout = self.default_rollout_id if rollout_id is None else str(rollout_id)
        self._task_context = (str(domain), str(example_id), rollout)
        self._cursor = 0
        self._load_actions()

    def reset(self, _logger=None, **kwargs: Dict[str, Any]) -> None:
        self.logger = _logger if _logger is not None else logging.getLogger(self.LOGGER_NAME)
        self._cursor = 0
        self._load_actions()

    def predict(self, instruction: str, obs: Dict[str, Any], **kwargs: Any):
        if not self._actions:
            self._load_actions()
        if self._cursor >= len(self._actions):
            return "", []
        action = self._actions[self._cursor]
        self._cursor += 1
        return "", [action]

    def _load_actions(self) -> None:
        if self._task_context is None:
            message = (
                "ReplayAgent task context is not set. Call set_task_context("
                "domain, example_id, rollout_id) before reset/predict."
            )
            if self.strict_missing:
                raise RuntimeError(message)
            self.logger.warning(message)
            self._traj_path = None
            self._actions = ["DONE"]
            return

        domain, example_id, rollout_id = self._task_context
        source_rollout_id = self._source_rollout_id(rollout_id)
        traj_path = self._resolve_traj_path(domain, example_id, source_rollout_id)

        if traj_path == self._traj_path and self._actions:
            return

        if traj_path is None:
            candidates = [
                os.path.join(self.source_result_root, domain, example_id, source_rollout_id, "traj.jsonl"),
                os.path.join(self.source_result_root, domain, example_id, "traj.jsonl"),
            ]
            message = (
                f"Replay trajectory not found for output rollout {rollout_id} "
                f"(source rollout {source_rollout_id}). Tried: " + ", ".join(candidates)
            )
            if self.strict_missing:
                raise FileNotFoundError(message)
            self.logger.warning(message)
            self._traj_path = candidates[0]
            self._actions = ["DONE"]
            self._cursor = 0
            return

        actions: List[str] = []
        with open(traj_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning("Skipping invalid JSON in %s:%s: %s", traj_path, line_no, exc)
                    continue
                if not isinstance(record, dict):
                    continue
                action = _normalize_replay_action(record)
                if action is None:
                    self.logger.warning("Skipping unsupported replay action in %s:%s: %s", traj_path, line_no, record.get("action"))
                    continue
                actions.append(action)

        if not actions:
            message = f"Replay trajectory has no action entries: {traj_path}"
            if self.strict_missing:
                raise ValueError(message)
            self.logger.warning(message)
            actions = ["DONE"]

        self._traj_path = traj_path
        self._actions = actions
        self._cursor = 0
        self.logger.info("Loaded replay trajectory %s with %d actions for output rollout %s", traj_path, len(actions), rollout_id)

    def _source_rollout_id(self, rollout_id: str) -> str:
        if self.source_rollout_repeat <= 1:
            return str(rollout_id)
        try:
            return str(int(rollout_id) // self.source_rollout_repeat)
        except ValueError:
            message = f"source_rollout_repeat requires numeric rollout ids; got {rollout_id!r}"
            if self.strict_missing:
                raise ValueError(message)
            self.logger.warning(message)
            return str(rollout_id)

    def _resolve_traj_path(self, domain: str, example_id: str, rollout_id: str) -> Optional[str]:
        candidates = [
            os.path.join(self.source_result_root, domain, example_id, rollout_id, "traj.jsonl"),
            os.path.join(self.source_result_root, domain, example_id, "traj.jsonl"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
