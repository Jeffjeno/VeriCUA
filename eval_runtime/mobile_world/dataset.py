from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows

def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def render_action(action: dict[str, Any]) -> str:

    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

@dataclass(frozen=True)
class MobileWorldStep:
    index: int
    task_goal: str
    prediction: str | None
    action: dict[str, Any]
    observation_before_path: Path
    ask_user_response: str | None = None
    tool_call: Any = None

    @property
    def action_type(self) -> str | None:
        value = self.action.get("action_type")
        return str(value) if value is not None else None

@dataclass(frozen=True)
class MobileWorldRecord:
    candidate_id: str
    task_name: str
    trajectory_dir: Path
    original_score: float
    original_reason: str | None
    source: str
    apps: tuple[str, ...]
    tags: tuple[str, ...]
    manifest_row: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.original_score >= 1.0

@dataclass(frozen=True)
class MobileWorldEpisode:
    record: MobileWorldRecord
    instruction: str
    steps: tuple[MobileWorldStep, ...]

    @property
    def example(self) -> dict[str, Any]:
        return {
            "id": self.record.task_name,
            "instruction": self.instruction,
            "benchmark": "mobile_world",
            "domain": "mobile_world",
            "task_name": self.record.task_name,
            "related_apps": list(self.record.apps),
            "tags": list(self.record.tags),
            "candidate_id": self.record.candidate_id,
        }

@dataclass(frozen=True)
class MobileWorldDataset:
    name: str
    project_root: Path
    config_path: Path
    config: dict[str, Any]
    dataset_root: Path
    records: tuple[MobileWorldRecord, ...]

    def get(self, task_name: str) -> MobileWorldRecord:
        for record in self.records:
            if record.task_name == task_name:
                return record
        raise KeyError(f"Task not found in {self.name}: {task_name}")

    def select(self, task_names: Iterable[str] | None = None) -> list[MobileWorldRecord]:
        if task_names is None:
            return list(self.records)
        wanted = {str(value) for value in task_names}
        selected = [record for record in self.records if record.task_name in wanted]
        found = {record.task_name for record in selected}
        missing = sorted(wanted - found)
        if missing:
            raise KeyError(f"Tasks not found in {self.name}: {missing}")
        return selected

    def load_episode(self, record: MobileWorldRecord | str) -> MobileWorldEpisode:
        if isinstance(record, str):
            record = self.get(record)
        traj_path = record.trajectory_dir / "traj.json"
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
        task_block = payload.get("0")
        if not isinstance(task_block, dict):
            raise ValueError(f"Missing trajectory key '0': {traj_path}")
        trajectory = task_block.get("traj")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"Empty trajectory: {traj_path}")

        steps: list[MobileWorldStep] = []
        for fallback_index, raw_step in enumerate(trajectory, start=1):
            if not isinstance(raw_step, dict):
                raise ValueError(
                    f"Invalid trajectory step {fallback_index}: {traj_path}"
                )
            action = raw_step.get("action")
            if not isinstance(action, dict):
                raise ValueError(
                    f"Missing action at trajectory step {fallback_index}: {traj_path}"
                )
            step_index = int(raw_step.get("step") or fallback_index)
            screenshot_file = raw_step.get("screenshot_file")
            if screenshot_file:
                screenshot_path = record.trajectory_dir / str(screenshot_file)
            else:
                screenshot_path = (
                    record.trajectory_dir
                    / "screenshots"
                    / f"{record.task_name}-0-{step_index}.png"
                )
            if not screenshot_path.is_file():
                raise FileNotFoundError(
                    f"Missing observation for {record.candidate_id} step "
                    f"{step_index}: {screenshot_path}"
                )
            steps.append(
                MobileWorldStep(
                    index=step_index,
                    task_goal=str(raw_step.get("task_goal") or ""),
                    prediction=(
                        str(raw_step["prediction"])
                        if raw_step.get("prediction") is not None
                        else None
                    ),
                    action=dict(action),
                    observation_before_path=screenshot_path.resolve(),
                    ask_user_response=(
                        str(raw_step["ask_user_response"])
                        if raw_step.get("ask_user_response") is not None
                        else None
                    ),
                    tool_call=raw_step.get("tool_call"),
                )
            )
        instruction = next(
            (step.task_goal for step in steps if step.task_goal),
            record.task_name,
        )
        return MobileWorldEpisode(
            record=record,
            instruction=instruction,
            steps=tuple(steps),
        )

def _resolve(project_root: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()

def load_mobile_world_dataset(
    name: str = "mobileworld",
    *,
    project_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> MobileWorldDataset:
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    registry = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else root / "config" / "mobile_world_datasets" / "registry.yaml"
    )
    registry_value = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    datasets = registry_value.get("datasets")
    if not isinstance(datasets, dict) or name not in datasets:
        raise KeyError(f"MobileWorld dataset is not registered: {name}")
    config_path = Path(str(datasets[name]))
    if not config_path.is_absolute():
        config_path = registry.parent / config_path
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("name") != name:
        raise ValueError(
            f"Dataset config name mismatch: expected {name}, got {config.get('name')}"
        )
    if config.get("benchmark") != "mobile_world":
        raise ValueError(f"Dataset {name} is not a MobileWorld dataset")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Dataset config has no paths mapping: {config_path}")
    dataset_root = _resolve(root, str(paths["dataset_root"]))
    manifest_path = _resolve(root, str(paths["manifest"]))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    records: list[MobileWorldRecord] = []
    seen_tasks: set[str] = set()
    seen_candidates: set[str] = set()
    for row in _read_jsonl(manifest_path):
        candidate_id = str(row["candidate_id"])
        task_name = str(row["task_name"])
        if candidate_id in seen_candidates:
            raise ValueError(f"Duplicate candidate ID in {name}: {candidate_id}")
        if task_name in seen_tasks:
            raise ValueError(f"Duplicate task in one-per-task dataset {name}: {task_name}")
        trajectory_dir = dataset_root / "trajectories" / candidate_id
        if not trajectory_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset trajectory is missing: {trajectory_dir}"
            )
        records.append(
            MobileWorldRecord(
                candidate_id=candidate_id,
                task_name=task_name,
                trajectory_dir=trajectory_dir.resolve(),
                original_score=float(row["original_score"]),
                original_reason=(
                    str(row["original_reason"])
                    if row.get("original_reason") is not None
                    else None
                ),
                source=str(row.get("source") or ""),
                apps=tuple(str(value) for value in row.get("apps", [])),
                tags=tuple(str(value) for value in row.get("tags", [])),
                manifest_row=row,
            )
        )
        seen_candidates.add(candidate_id)
        seen_tasks.add(task_name)

    expected_total = int((config.get("stats") or {}).get("trajectories") or 0)
    if expected_total and len(records) != expected_total:
        raise ValueError(
            f"Dataset {name} expected {expected_total} trajectories, found {len(records)}"
        )
    return MobileWorldDataset(
        name=name,
        project_root=root,
        config_path=config_path,
        config=config,
        dataset_root=dataset_root,
        records=tuple(sorted(records, key=lambda item: item.task_name)),
    )

def materialize_episode(
    episode: MobileWorldEpisode,
    output_dir: str | Path,
    *,
    initial_screenshot: bytes | Path | None = None,
    post_action_screenshots: Iterable[bytes | Path] | None = None,
    final_screenshot: bytes | Path | None = None,
    native_score: float | None = None,
    native_reason: str | None = None,
    overwrite: bool = False,
) -> Path:

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    steps = list(episode.steps)
    if isinstance(initial_screenshot, bytes):
        (destination / "step_0.png").write_bytes(initial_screenshot)
    elif isinstance(initial_screenshot, Path):
        _link_or_copy(
            initial_screenshot.expanduser().resolve(),
            destination / "step_0.png",
        )
    else:
        _link_or_copy(steps[0].observation_before_path, destination / "step_0.png")
    replay_frames = list(post_action_screenshots or [])
    if replay_frames and len(replay_frames) != len(steps):
        raise ValueError(
            "post_action_screenshots must contain one frame per trajectory step: "
            f"expected {len(steps)}, got {len(replay_frames)}"
        )
    final_frame_available = bool(replay_frames) or final_screenshot is not None
    traj_rows: list[dict[str, Any]] = []
    for offset, step in enumerate(steps):
        output_step = offset + 1
        screenshot_name = f"step_{output_step}.png"
        screenshot_path = destination / screenshot_name
        if replay_frames:
            replay_frame = replay_frames[offset]
            if isinstance(replay_frame, bytes):
                screenshot_path.write_bytes(replay_frame)
            else:
                _link_or_copy(replay_frame.expanduser().resolve(), screenshot_path)
            semantics = "post_action_from_live_replay"
        elif offset + 1 < len(steps):
            _link_or_copy(
                steps[offset + 1].observation_before_path,
                screenshot_path,
            )
            semantics = "post_action_from_next_observation"
        elif isinstance(final_screenshot, bytes):
            screenshot_path.write_bytes(final_screenshot)
            semantics = "post_action_from_live_replay"
        elif isinstance(final_screenshot, Path):
            _link_or_copy(final_screenshot.expanduser().resolve(), screenshot_path)
            semantics = "post_action_from_live_replay"
        else:
            _link_or_copy(step.observation_before_path, screenshot_path)
            semantics = "fallback_last_pre_action_observation"

        traj_rows.append(
            {
                "step_num": output_step,
                "action_timestamp": "",
                "action": render_action(step.action),
                "action_tool": "mobile_use",
                "native_action": step.action,
                "response": step.prediction or "",
                "reward": (
                    float(native_score)
                    if native_score is not None and offset == len(steps) - 1
                    else 0.0
                ),
                "done": offset == len(steps) - 1,
                "screenshot_file": screenshot_name,
                "screenshot_semantics": semantics,
                "ask_user_response": step.ask_user_response,
                "tool_call": step.tool_call,
            }
        )

    with (destination / "traj.jsonl").open("w", encoding="utf-8") as handle:
        for row in traj_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    score = (
        float(native_score)
        if native_score is not None
        else episode.record.original_score
    )
    (destination / "result.txt").write_text(f"{score}\n", encoding="utf-8")
    _write_json(destination / "example.json", episode.example)
    _write_json(
        destination / "episode.json",
        {
            "dataset": "mobileworld",
            "benchmark": "mobile_world",
            "candidate_id": episode.record.candidate_id,
            "task_name": episode.record.task_name,
            "instruction": episode.instruction,
            "source": episode.record.source,
            "original_score": episode.record.original_score,
            "original_reason": episode.record.original_reason,
            "native_score": score,
            "native_reason": native_reason,
            "num_steps": len(steps),
            "final_screenshot_from_live_replay": final_frame_available,
            "trajectory_format": "mobile_world_post_action_materialized_v1",
        },
    )
    return destination

def result_episode_dir(
    result_root: str | Path,
    *,
    policy_model: str,
    task_name: str,
    rollout_id: str | int,
) -> Path:
    return (
        Path(result_root).expanduser().resolve()
        / "mobile_world_json"
        / "screenshot"
        / policy_model
        / "mobile_world"
        / task_name
        / str(rollout_id)
    )

__all__ = [
    "MobileWorldDataset",
    "MobileWorldEpisode",
    "MobileWorldRecord",
    "MobileWorldStep",
    "load_mobile_world_dataset",
    "materialize_episode",
    "render_action",
    "result_episode_dir",
]
