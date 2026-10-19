from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, IO

import rollout_runtime

import agent_runtime as ap
from agent_runtime.harnesses.holo3 import Holo3Harness
from agent_runtime.harnesses.holo31 import Holo31Harness
from agent_runtime.runner import RunSpec
from agent_runtime.runtime.local import run_task_attempt
from agent_runtime.types import Action, HarnessContext, ModalityKind, Observation, RunTaskResult, TraceEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rollout_runtime.osworld_batch")

class NoopHarness(ap.Harness):
    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        return Action.terminate(reason="DONE")

def create_harness(args: argparse.Namespace) -> ap.Harness:
    if args.noop:
        return NoopHarness()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "") or "EMPTY"
    base_url = args.api_base or os.environ.get("OPENAI_BASE_URL", "")

    if args.harness == "holo3":
        return Holo3Harness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_trajectory_length=args.max_trajectory_length,
            max_images=args.holo3_max_images,
            client_password=args.client_password,
            proxy_url=args.host_proxy,
            platform=args.platform,
            request_timeout=args.request_timeout,
            enable_thinking=args.holo3_enable_thinking,
            reasoning_effort=args.reasoning_effort,
            logprobs=args.logprobs,
        )

    if args.harness == "holo31":
        return Holo31Harness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_trajectory_length=args.max_trajectory_length,
            max_images=args.holo3_max_images,
            client_password=args.client_password,
            platform=args.platform,
            request_timeout=args.request_timeout,
            enable_thinking=args.holo3_enable_thinking,
            reasoning_effort=args.reasoning_effort,
            logprobs=args.logprobs,
        )

    if args.harness == "qwen35vl":
        from agent_runtime.harnesses.qwen35vl import Qwen35VLHarness
        return Qwen35VLHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            platform=args.platform,
            coordinate_type=args.coordinate_type,
            history_n=args.history_n,
            image_max=args.image_max,
            fold_size=args.fold_size,
            logprobs=args.logprobs,
        )

    if args.harness == "qwen3vl":
        from agent_runtime.harnesses.qwen3vl import Qwen3VLHarness
        return Qwen3VLHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            platform=args.platform,
            coordinate_type=args.coordinate_type,
            history_n=args.history_n,
            image_max=args.image_max,
            fold_size=args.fold_size,
            enable_thinking=args.enable_thinking,
            logprobs=args.logprobs,
        )

    if args.harness == "qwen3vl-ref":
        from agent_runtime.harnesses.qwen3vl_ref import Qwen3VLRefHarness
        return Qwen3VLRefHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            coordinate_type=args.coordinate_type,
            history_n=args.history_n,
            image_max=args.image_max,
            fold_size=args.fold_size,
            enable_thinking=args.enable_thinking,
            logprobs=args.logprobs,
        )

    if args.harness == "qwen3vl-probe":
        from agent_runtime.harnesses.qwen3vl_probe import Qwen3VLProbeHarness
        return Qwen3VLProbeHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            platform=args.platform,
            logprobs=args.logprobs,
        )

    if args.harness == "holo31-probe":
        from agent_runtime.harnesses.holo31_probe import Holo31ProbeHarness
        return Holo31ProbeHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            platform=args.platform,
            client_password=args.client_password,
            logprobs=args.logprobs,
        )

    if args.harness == "holo3-probe":
        from agent_runtime.harnesses.holo3_probe import Holo3ProbeHarness
        return Holo3ProbeHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            platform=args.platform,
            client_password=args.client_password,
            logprobs=args.logprobs,
        )

    if args.harness == "evocua":
        from agent_runtime.harnesses.evocua import EvoCUAHarness
        return EvoCUAHarness(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            logprobs=args.logprobs,
        )

    if args.harness == "owl15":
        from agent_runtime.harnesses.owl15 import Owl15Harness
        return Owl15Harness(
            model=args.model,
            api_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            platform=args.platform,
            logprobs=args.logprobs,
        )

    harness_cls = ap.resolve_harness(args.harness)
    return harness_cls(model=args.model, api_key=api_key, base_url=base_url)

_KNOWN_DOMAINS = (
    "chrome", "gimp", "libreoffice_calc", "libreoffice_impress",
    "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code",
)

def _task_domain(task: Any) -> str:

    native_ref = getattr(task, "native_ref", None) or {}
    domain = native_ref.get("domain")
    if not domain:
        task_config = native_ref.get("task_config") or {}
        domain = task_config.get("_domain")
    if not domain:
        domain = getattr(task, "suite", None)
    if not domain:
        domain = (getattr(task, "metadata", None) or {}).get("domain")
    return domain or "unknown"

def _rollout_dir(traj_root: Path, domain: str, task_id: str, rollout_idx: int) -> Path:

    return Path(traj_root) / domain / task_id / f"r{rollout_idx}"

def plan_rollouts(
    tasks: list[Any],
    *,
    traj_root: str | Path,
    num_rollouts: int,
) -> tuple[list[Any], dict[str, tuple[str, str, int]], int]:

    traj_root = Path(traj_root)
    run_tasks: list[Any] = []
    rollout_map: dict[str, tuple[str, str, int]] = {}
    skipped = 0

    for task in tasks:
        domain = _task_domain(task)
        for r in range(num_rollouts):
            out_dir = _rollout_dir(traj_root, domain, task.task_id, r)
            if (out_dir / "result.txt").is_file():
                skipped += 1
                continue
            if out_dir.exists():
                for child in out_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        with contextlib.suppress(Exception):
                            child.unlink()
            clone_id = f"{task.task_id}__r{r}" if num_rollouts > 1 else task.task_id
            clone = task.model_copy(update={"task_id": clone_id}) if clone_id != task.task_id else task
            run_tasks.append(clone)
            rollout_map[clone_id] = (task.task_id, domain, r)

    return run_tasks, rollout_map, skipped

class _RolloutTaskState:

    __slots__ = ("task_id", "out_dir", "traj_fh", "lock", "action_step_count")

    def __init__(self, task_id: str, out_dir: Path, traj_fh: IO[str]) -> None:
        self.task_id = task_id
        self.out_dir = out_dir
        self.traj_fh = traj_fh
        self.lock = threading.Lock()
        self.action_step_count = 0

class RolloutTraceWriter:

    def __init__(
        self,
        traj_root: Path,
        rollout_map: dict[str, tuple[str, str, int]],
    ) -> None:
        self.traj_root = Path(traj_root)
        self.rollout_map = rollout_map
        self._tasks: dict[str, _RolloutTaskState] = {}
        self._global_lock = threading.Lock()

    def _resolve_out_dir(self, task_id: str) -> Path:
        mapped = self.rollout_map.get(task_id)
        if mapped:
            orig_task_id, domain, rollout_idx = mapped
            return _rollout_dir(self.traj_root, domain, orig_task_id, rollout_idx)
        return self.traj_root / "unknown" / task_id / "r0"

    def begin_task(self, task_id: str, *, task: Any = None) -> Path:
        out_dir = self._resolve_out_dir(task_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._global_lock:
            existing = self._tasks.pop(task_id, None)
            if existing is not None:
                with contextlib.suppress(Exception):
                    existing.traj_fh.close()

        traj_path = out_dir / "traj.jsonl"
        fh = open(traj_path, "a", encoding="utf-8")
        state = _RolloutTaskState(task_id, out_dir, fh)
        with self._global_lock:
            self._tasks[task_id] = state
        return out_dir

    def record_event(self, task_id: str, event: TraceEvent) -> None:
        state = self._tasks.get(task_id)
        if state is None:
            out_dir = self.begin_task(task_id)
            state = self._tasks[task_id]

        with state.lock:
            self._save_screenshot(event, state)

            if event.action is None:
                return

            state.action_step_count += 1
            action_ts = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d@%H%M%S%f")
            record = self._event_to_traj_record(event, state.action_step_count)

            self._save_step_detail(event, state, state.action_step_count, action_ts, record)

            try:
                state.traj_fh.write(json.dumps(record, ensure_ascii=False))
                state.traj_fh.write("\n")
                state.traj_fh.flush()
                os.fsync(state.traj_fh.fileno())
            except Exception:
                logger.exception("Failed to append traj.jsonl for task %s", task_id)

    def end_task(self, task_id: str, trace: Any, evaluation: Any = None) -> None:
        with self._global_lock:
            state = self._tasks.pop(task_id, None)

        if state is None:
            state = _RolloutTaskState(
                task_id, self._resolve_out_dir(task_id), None
            )
            state.out_dir.mkdir(parents=True, exist_ok=True)

        if state.traj_fh is not None:
            with contextlib.suppress(Exception):
                state.traj_fh.flush()
                os.fsync(state.traj_fh.fileno())
                state.traj_fh.close()

        is_genuine_eval = (
            evaluation is not None
            and not getattr(evaluation, "error_type", None)
        )

        if is_genuine_eval:
            score = evaluation.score if evaluation.score is not None else 0.0
            (state.out_dir / "result.txt").write_text(f"{score}\n", encoding="utf-8")
        else:
            score = None
            error_info = ""
            if evaluation is not None:
                error_info = f" error_type={evaluation.error_type} reason={evaluation.reason}"
            logger.warning(
                "Task %s ended without genuine evaluation (crash/error%s); "
                "result.txt NOT written — will be retried on next resume.",
                task_id, error_info,
            )

        mapped = self.rollout_map.get(task_id)
        if mapped:
            orig_task_id, domain, rollout_idx = mapped
            if is_genuine_eval:
                logger.info(
                    "Rollout done: %s/%s/r%d  steps=%d  score=%.2f",
                    domain, orig_task_id, rollout_idx, state.action_step_count, score,
                )
            else:
                logger.warning(
                    "Rollout error: %s/%s/r%d  steps=%d  (will retry on resume)",
                    domain, orig_task_id, rollout_idx, state.action_step_count,
                )

    def close(self) -> None:
        for task_id, state in list(self._tasks.items()):
            if state.traj_fh is not None:
                with contextlib.suppress(Exception):
                    state.traj_fh.flush()
                    state.traj_fh.close()
        self._tasks.clear()

    def __enter__(self) -> RolloutTraceWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _save_screenshot(self, event: TraceEvent, state: _RolloutTaskState) -> None:

        if event.observation is None:
            return
        for part in event.observation.parts:
            if part.kind is not ModalityKind.IMAGE:
                continue
            if not part.content or not isinstance(part.content, str):
                continue
            try:
                png_bytes = base64.b64decode(part.content)
            except Exception:
                continue
            step_num = state.action_step_count + 1 if event.action is not None else 0
            ts = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d@%H%M%S%f")
            if step_num == 0:
                fname = "step_0.png"
            else:
                fname = f"step_{step_num}_{ts}.png"
            (state.out_dir / fname).write_bytes(png_bytes)
            break

    @staticmethod
    def _get_holo_meta(event: TraceEvent) -> dict[str, Any]:

        if event.action is None:
            return {}
        action_data = event.action.model_dump() if event.action else {}
        meta = action_data.get("metadata") or {}
        return (
            meta.get("holo3") or meta.get("holo31") or meta.get("qwen35vl")
            or meta.get("qwen3vl") or meta.get("qwen3vl_ref")
            or meta.get("qwen3vl_probe") or meta.get("holo31_probe")
            or meta.get("holo3_probe") or meta.get("evocua")
            or meta.get("owl15") or {}
        )

    def _save_step_detail(
        self,
        event: TraceEvent,
        state: _RolloutTaskState,
        step_num: int,
        action_ts: str,
        traj_record: dict[str, Any],
    ) -> None:

        holo_meta = self._get_holo_meta(event)
        logprobs_data = holo_meta.get("logprobs_data")
        last_messages = holo_meta.get("last_messages")

        if logprobs_data is None and last_messages is None:
            return

        if logprobs_data is not None:
            traj_record["logprobs_summary"] = logprobs_data["summary"]

        steps_dir = state.out_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        step_detail: dict[str, Any] = {
            "step_num": step_num,
            "action_timestamp": action_ts,
        }
        if last_messages is not None:
            step_detail["messages"] = _sanitize_messages_for_save(last_messages)
        step_detail["completion"] = traj_record.get("response", "")
        if logprobs_data is not None:
            step_detail["logprobs"] = logprobs_data

        step_path = steps_dir / f"step_{step_num}_{action_ts}.json"
        try:
            step_path.write_text(
                json.dumps(step_detail, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to write step detail %s", step_path)

    @staticmethod
    def _event_to_traj_record(event: TraceEvent, step_num: int) -> dict[str, Any]:

        action = event.action
        action_data = action.model_dump() if action else {}
        meta = action_data.get("metadata") or {}
        harness_meta = (
            meta.get("holo3") or meta.get("holo31") or meta.get("qwen35vl")
            or meta.get("qwen3vl") or meta.get("qwen3vl_ref")
            or meta.get("qwen3vl_probe") or meta.get("holo31_probe")
            or meta.get("holo3_probe") or meta.get("evocua")
            or meta.get("owl15") or {}
        )

        action_type = action_data.get("type", "")
        action_args = action_data.get("arguments") or {}
        native_action = (event.info or {}).get("native_action", "")
        if not native_action:
            if action_type == "terminate":
                native_action = "DONE"
            elif action_type == "noop":
                native_action = "WAIT"
            elif action_type == "click":
                native_action = f"pyautogui.click(x={action_args.get('x', 0)}, y={action_args.get('y', 0)})"
            elif action_type == "double_click":
                native_action = f"pyautogui.doubleClick(x={action_args.get('x', 0)}, y={action_args.get('y', 0)})"
            elif action_type == "right_click":
                native_action = f"pyautogui.rightClick(x={action_args.get('x', 0)}, y={action_args.get('y', 0)})"
            elif action_type == "type":
                text = action_args.get("text", "")
                native_action = f"pyautogui.typewrite({repr(text)}, interval=0.03)"
            elif action_type == "key_press":
                keys_list = action_args.get('keys') or []
                key = keys_list[0] if keys_list else action_args.get('key', '')
                native_action = f"pyautogui.press({key!r})"
            elif action_type == "hotkey":
                keys = action_args.get("keys", [])
                native_action = f"pyautogui.hotkey({', '.join(repr(k) for k in keys)})"
            elif action_type == "scroll":
                native_action = f"pyautogui.scroll({action_args.get('dy', action_args.get('amount', 0))})"
            elif action_type == "hover":
                native_action = f"pyautogui.moveTo(x={action_args.get('x', 0)}, y={action_args.get('y', 0)})"
            else:
                native_action = json.dumps(action_data, ensure_ascii=False)

        response_text = harness_meta.get("response_text") or action_data.get("raw") or ""
        action_ts = datetime.fromtimestamp(event.timestamp).strftime("%Y%m%d@%H%M%S%f")
        is_done = event.terminated or native_action == "DONE"
        screenshot_file = f"step_{step_num}_{action_ts}.png"

        return {
            "step_num": step_num,
            "action_timestamp": action_ts,
            "action": native_action,
            "response": response_text,
            "reward": event.reward or 0.0,
            "done": is_done,
            "info": {"done": True} if is_done else {},
            "screenshot_file": screenshot_file,
        }

def _sanitize_messages_for_save(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    sanitized = copy.deepcopy(messages)
    image_index = 0
    for msg in sanitized:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                content[i] = {"type": "image_ref", "image_index": image_index}
                image_index += 1
    return sanitized

def _configure_logging(*, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)
    if debug:
        os.environ["RUN_AGENT_RUNTIME_DEBUG"] = "1"
        for name in ("rollout_runtime", "agent_runtime", "agent_runtime.harnesses.holo3", "agent_runtime.harnesses.holo31", "desktopenv"):
            logging.getLogger(name).setLevel(logging.DEBUG)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run OSWorld splits with Holo via AgentRuntime")
    parser.add_argument("--osworld-path", default=os.environ.get("OSWORLD_PATH", ""),
                        help="OSWorld root (must contain desktop_env/ and evaluation_examples/)")
    parser.add_argument("--vm-path", default=os.environ.get("OSWORLD_VM_PATH", ""),
                        help="Path to Ubuntu.qcow2 VM image")
    parser.add_argument("--provider", default="docker", choices=["docker", "vmware", "virtualbox"])
    parser.add_argument("--num-vms", type=int, default=2, help="Number of concurrent VMs")
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to run (default: all)")
    parser.add_argument("--domain", default=None, help="Filter by domain (default: all domains)")
    parser.add_argument("--split", default="osworld_all",
                        help="evaluation_examples/{split}.json under --osworld-path")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/osworld_{split}_{harness})")
    parser.add_argument("--noop", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--action-space", default="pyautogui")
    parser.add_argument("--settle-time", type=float, default=20.0)
    parser.add_argument("--sleep-after-exec", type=float, default=2.0)
    parser.add_argument("--offline-only", action="store_true",
                        help="Only run tasks that don't require internet access")
    parser.add_argument("--list-only", action="store_true", help="List tasks and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")

    parser.add_argument("--harness", default="holo3", help="holo3, holo31, ...")
    parser.add_argument("--model", default="holo3-1-35b-a3b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-trajectory-length", type=int, default=20)
    parser.add_argument("--client-password", default="password")
    parser.add_argument("--platform", default="ubuntu")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--holo3-max-images", type=int, default=3)
    parser.add_argument("--holo3-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coordinate-type", default="absolute",
                        help="Coordinate type for qwen35vl/qwen3vl harness: 'absolute' (pixel coords in processed resolution) "
                             "or 'relative' (normalized [0,999] grid). Qwen3.6 models require 'relative'.")
    parser.add_argument("--history-n", type=int, default=4,
                        help="Number of recent history turns to include in context (qwen35vl/qwen3vl)")
    parser.add_argument("--image-max", type=int, default=5,
                        help="Max number of screenshots to keep unfolded in context (qwen35vl)")
    parser.add_argument("--fold-size", type=int, default=2,
                        help="Number of screenshots to fold at a time when exceeding image-max (qwen35vl)")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable thinking mode (Qwen3-VL thinking variant); sets chat_template_kwargs.enable_thinking=true")
    parser.add_argument("--logprobs", action="store_true", default=False,
                        help="Request and record per-token logprobs from the model API. "
                             "Requires an OpenAI-compatible server started with --return-logprobs or equivalent.")

    parser.add_argument("--host-proxy", default=None, help="Host-level proxy")
    parser.add_argument("--vm-proxy", default=None, help="Proxy reachable from VM")
    parser.add_argument("--no-proxy", default="127.0.0.1,localhost,::1,172.17.0.0/16,172.30.0.0/16",
                        help="Hosts to bypass proxy")
    parser.add_argument("--traj-root", default=None,
                        help="Optional root dir for evaldata-style trajectories (default: output_dir/trajectories)")
    parser.add_argument("--num-rollouts", type=int, default=1,
                        help="Run each task N times (rollouts r0..r{N-1}). Resume-aware: skips rollouts whose result.txt exists.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    _configure_logging(debug=args.debug or os.environ.get("RUN_AGENT_RUNTIME_DEBUG") in {"1", "true", "TRUE"})

    if not args.osworld_path:
        raise SystemExit("ERROR: --osworld-path is required (or set OSWORLD_PATH)")
    if not args.vm_path and not args.list_only and not args.dry_run:
        raise SystemExit("ERROR: --vm-path is required (or set OSWORLD_VM_PATH)")

    osworld_path = os.path.abspath(args.osworld_path)
    split_file = Path(osworld_path) / "evaluation_examples" / f"{args.split}.json"
    if not split_file.is_file():
        raise SystemExit(f"ERROR: split file not found: {split_file}")

    desktop_env_py = Path(osworld_path) / "desktop_env" / "desktop_env.py"
    if not args.dry_run and not args.list_only and not desktop_env_py.is_file():
        raise SystemExit(f"ERROR: invalid --osworld-path, missing {desktop_env_py}")

    benchmark = ap.load_benchmark(
        "osworld",
        osworld_path=osworld_path,
        split=args.split,
        action_space=args.action_space,
    )

    filter_kwargs: dict[str, Any] = {}
    if args.domain:
        filter_kwargs["domain"] = args.domain
    all_tasks = benchmark.list_tasks(split=args.split, limit=None, **filter_kwargs)
    if args.offline_only:
        all_tasks = [
            t for t in all_tasks
            if not (t.native_ref or {}).get("task_config", {}).get("proxy", False)
        ]
    tasks = all_tasks[:args.num_tasks] if args.num_tasks else all_tasks

    print(f"\nSelected {len(tasks)} task(s) from {split_file.name}")
    print(f"  domain={args.domain or 'all'}  offline_only={args.offline_only}")
    for task in tasks[:20]:
        print(f"  {task.task_id}  {(task.instruction or '')[:70]}")
    if len(tasks) > 20:
        print(f"  ... and {len(tasks) - 20} more")

    if args.list_only or args.dry_run:
        print("\nValidation complete.")
        return
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        raise SystemExit(1)

    proxy_config = None
    if args.host_proxy or args.vm_proxy:
        proxy_config = ap.ProxyConfig(
            host_proxy=args.host_proxy,
            vm_proxy=args.vm_proxy,
            no_proxy=args.no_proxy,
        )

    backend = ap.backends.osworld(
        osworld_path=osworld_path,
        provider_name=args.provider,
        path_to_vm=args.vm_path,
        action_space=args.action_space,
        headless=args.headless,
        reuse=True,
        num_instances=args.num_vms,
        proxy_config=proxy_config,
        require_a11y_tree=False,
    )

    output_dir = args.output_dir or f"results/osworld_{args.split}_{args.harness}"
    traj_root = Path(args.traj_root) if args.traj_root else Path(output_dir) / "trajectories"

    num_rollouts = max(1, args.num_rollouts)
    run_tasks, rollout_map, skipped = plan_rollouts(
        tasks, traj_root=traj_root, num_rollouts=num_rollouts,
    )
    total_rollouts = len(tasks) * num_rollouts
    print(f"\nRollouts: {num_rollouts} per task  ->  {total_rollouts} total")
    print(f"  Already done (resume): {skipped}")
    print(f"  To run:                {len(run_tasks)}")

    if not run_tasks:
        print("\nAll rollouts already completed. Nothing to run (resume).")
        print(f"  Trajectories: {traj_root.resolve()}")
        return

    harness_args = args
    run_spec = RunSpec(
        name=f"osworld_{args.harness}",
        benchmark=benchmark,
        tasks=run_tasks,
        harness_factory=lambda: create_harness(harness_args),
        backend=backend,
        evaluator_factory=lambda task: benchmark.make_evaluator(task=task),
        concurrency=args.num_vms,
        max_steps=args.max_steps,
        output_dir=None,
        env_options={
            "action_space": args.action_space,
            "settle_time": args.settle_time,
            "sleep_after_execution": args.sleep_after_exec,
        },
        tags={
            "benchmark": "osworld",
            "harness": args.harness,
            "model": args.model,
            "split": args.split,
            "domain": args.domain or "",
        },
    )

    print("\nConfig:")
    print(f"  Split:        {args.split}")
    print(f"  Harness:      {args.harness}")
    print(f"  Model:        {args.model}")
    print(f"  Rollouts:     {num_rollouts}")
    print(f"  VMs:          {args.num_vms}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  API:          {args.api_base or os.environ.get('OPENAI_BASE_URL', '(not set)')}")
    print(f"  Proxy:        host={args.host_proxy}, vm={args.vm_proxy}")
    print(f"  Output:       {output_dir}")
    print(f"  Trajectories: {traj_root.resolve()}")

    writer = RolloutTraceWriter(traj_root=traj_root, rollout_map=rollout_map)

    print(f"\nRunning {len(run_tasks)} rollout(s) with {args.num_vms} concurrent VM(s) ...")
    print("  (real-time saving: traj.jsonl + screenshots written per step)")
    t0 = time.time()

    backend.setup()

    _reset_pool = ThreadPoolExecutor(
        max_workers=args.num_vms,
        thread_name_prefix="vm-reset",
    )
    _original_release = backend._release_env

    def _async_release_env(env: Any) -> None:
        def _reset_then_enqueue() -> None:
            try:
                env.reset(task_config=None)
            except Exception:
                logger.exception("Background VM reset failed; returning env as-is")
            _original_release(env)
        _reset_pool.submit(_reset_then_enqueue)

    backend._release_env = _async_release_env

    results: list[RunTaskResult] = []
    completed = 0
    lock = threading.Lock()

    def _execute_one(task: Any) -> RunTaskResult:
        nonlocal completed
        result = run_task_attempt(run_spec, task, writer=writer)
        with lock:
            completed += 1
            logger.info("Progress: %d/%d completed", completed, len(run_tasks))
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.num_vms) as pool:
            futures = {pool.submit(_execute_one, t): t for t in run_tasks}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    task = futures[fut]
                    logger.error("Task %s raised: %s", task.task_id, exc)
                    results.append(RunTaskResult.failed(task_id=task.task_id, error=exc))
    finally:
        _reset_pool.shutdown(wait=True)
        with contextlib.suppress(Exception):
            backend.teardown()
        writer.close()

    elapsed = time.time() - t0

    successes = sum(1 for r in results if r.evaluation and r.evaluation.success)
    total = len(results)
    scores = [r.evaluation.score for r in results if r.evaluation and r.evaluation.score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'=' * 60}")
    print("Batch Run Complete")
    print(f"{'=' * 60}")
    print(f"  Tasks:        {total}")
    print(f"  Successes:    {successes}/{total}")
    print(f"  Avg score:    {avg_score:.3f}")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Avg per task: {elapsed / max(total, 1):.1f}s")

    print("\n--- Per-task Results ---")
    for result in results:
        task_id = result.trace.task_id if result.trace else "unknown"
        score = result.evaluation.score if result.evaluation else 0.0
        success = result.evaluation.success if result.evaluation else False
        reason = result.evaluation.reason if result.evaluation else "no evaluation"
        steps = len(result.trace.events) if result.trace else 0
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {task_id}  score={score:.2f}  steps={steps}  {reason or ''}")

    report = {
        "summary": {
            "total_tasks": total,
            "succeeded": successes,
            "avg_score": avg_score,
            "duration_seconds": elapsed,
            "concurrency": args.num_vms,
        },
        "config": {
            "split": args.split,
            "harness": args.harness,
            "model": args.model,
            "domain": args.domain,
            "max_steps": args.max_steps,
            "host_proxy": args.host_proxy,
            "vm_proxy": args.vm_proxy,
            "offline_only": args.offline_only,
        },
        "tasks": [
            {
                "task_id": r.trace.task_id if r.trace else "unknown",
                "score": r.evaluation.score if r.evaluation else None,
                "success": r.evaluation.success if r.evaluation else None,
                "reason": r.evaluation.reason if r.evaluation else None,
                "steps": len(r.trace.events) if r.trace else 0,
            }
            for r in results
        ],
        "timestamp": datetime.now().isoformat(),
    }

    report_path = Path(output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report:       {report_path}")
    print(f"  Trajectories: {traj_root.resolve()}  (real-time, layout: domain/task_id/rN/)")

    print("\nDone.")

if __name__ == "__main__":
    main()
