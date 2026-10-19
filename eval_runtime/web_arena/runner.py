from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import signal
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from eval_runtime.guirl.evaluator import build_guirl_config, run_guirl_eval_after_replay
from eval_runtime.passive_baseline import (
    PASSIVE_BASELINE_TYPES,
    build_passive_baseline_config,
    peek_evaluator_type,
    run_passive_baseline_eval_after_replay,
)
from eval_runtime.web_arena.dataset import WebArenaDataset, load_webarena_dataset
from eval_runtime.vagen_active.agent_factory import (
    VAGEN_QWEN_EVALUATOR_TYPE,
    load_vagen_eval_config,
)
from eval_runtime.vagen_active.evaluator import run_vagen_eval_after_replay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WA_ROOT = PROJECT_ROOT / "third_party" / "webarena-verified"
WA_SRC = WA_ROOT / "src"
REPLAY_SCRIPTS = PROJECT_ROOT / "scripts" / "webarena_data_replay"
for path in (str(WA_SRC), str(REPLAY_SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

logger = logging.getLogger("eval_runtime.web_arena.runner")

class RunnerWatchdogTimeout(TimeoutError):

    def __init__(self, operation: str, seconds: int):
        super().__init__(f"{operation} exceeded {seconds}s")
        self.operation = operation
        self.seconds = seconds

@contextlib.contextmanager
def _watchdog(seconds: int, operation: str):
    if seconds <= 0:
        yield
        return

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise RunnerWatchdogTimeout(operation, seconds)

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)

def _task_catalog() -> dict[tuple[int, int], dict[str, Any]]:
    path = WA_SRC / "webarena_verified" / "assets" / "dataset" / "webarena-verified.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(item["task_id"]), int(item.get("revision", 0))): item
        for item in records
    }

def _task_definition(record: dict[str, Any], catalog: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    task = record["task"]
    key = (int(task["task_id"]), int(task.get("revision", 0)))
    definition = catalog.get(key)
    if definition is None:
        raise KeyError(f"WebArena task definition not found: task_id={key[0]} revision={key[1]}")
    return definition

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

def _flush_jsonl(handle: Any) -> None:

    handle.flush()

def _screenshot_bytes(obs: dict[str, Any]) -> bytes:
    screenshot = obs.get("screenshot")
    if isinstance(screenshot, (bytes, bytearray)):
        return bytes(screenshot)
    if screenshot is None:
        raise ValueError("BrowserGym observation has no screenshot")
    image = Image.fromarray(np.asarray(screenshot, dtype=np.uint8))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

def _observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "screenshot": _screenshot_bytes(obs),
        "url": obs.get("url", ""),
        "open_pages_urls": obs.get("open_pages_urls", []),
        "active_page_index": obs.get("active_page_index", 0),
        "last_action_error": obs.get("last_action_error"),
    }

class BrowserGymProbeClient:
    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        policy_result: float | None,
        *,
        action_timeout_seconds: int = 300,
        operation_context: dict[str, Any] | None = None,
    ):
        self.env = env
        self.obs = obs
        self.policy_result = policy_result
        self.action_timeout_seconds = int(action_timeout_seconds)
        self.operation_context = operation_context if operation_context is not None else {}

    def get_obs(self, _lease_id: str = "") -> dict[str, Any]:
        return _observation(self.obs)

    def prepare_for_probe(self, _lease_id: str | None = None) -> None:
        return None

    def step_browser(self, action: str) -> dict[str, Any]:
        self.operation_context.update({
            "phase": "probe",
            "operation": "probe_env.step",
            "action": action,
        })
        with _watchdog(self.action_timeout_seconds, "probe env.step"):
            obs, _reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            raise RuntimeError(f"BrowserGym probe unexpectedly terminated: {info}")
        self.obs = obs
        return self.get_obs()

    def evaluate(self, _lease_id: str = "") -> None:

        return None

def _load_actions(record: dict[str, Any]) -> list[str]:
    payload = json.loads(Path(record["actions_file"]).read_text(encoding="utf-8"))
    from common import rewrite_webarena_action_urls

    return [
        rewrite_webarena_action_urls(str(item["action"]))
        for item in payload["actions"]
    ]

def _link_or_copy(src: Path, dst: Path) -> None:

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def _build_live_action_set(kind: str) -> Any:

    from common import build_action_set

    if kind != "generic_webarena":
        return build_action_set(kind)

    import bgym

    return bgym.HighLevelActionSet(
        subsets=["webarena", "coord"],
        multiaction=True,
        strict=False,
        retry_with_force=True,
        demo_mode="off",
    )

def _materialize_source_episode(
    record: dict[str, Any], definition: dict[str, Any], episode_dir: Path,
    *, dataset_name: str = "webarena",
) -> dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=True)
    actions = _load_actions(record)
    screenshots = [Path(path) for path in record.get("source_screenshots") or []]
    if len(screenshots) < len(actions) + 1:
        raise ValueError(
            f"{record['candidate_id']} has {len(actions)} actions but only "
            f"{len(screenshots)} source screenshots"
        )

    _link_or_copy(screenshots[0], episode_dir / "step_0.png")
    traj_path = episode_dir / "traj.jsonl"
    with traj_path.open("w", encoding="utf-8") as handle:
        for index, action in enumerate(actions, start=1):
            destination = episode_dir / f"step_{index}.png"
            _link_or_copy(screenshots[index], destination)
            handle.write(json.dumps({
                "step_num": index,
                "action_timestamp": f"{index:06d}",
                "action": action,
                "action_tool": "browser_use",
                "response": "",
                "screenshot_file": destination.name,
            }, ensure_ascii=False) + "\n")
    policy_result = float(record.get("original", {}).get("strict_score", 0))
    (episode_dir / "result.txt").write_text(f"{policy_result}\n", encoding="utf-8")
    example = {
        "id": str(record["candidate_id"]),
        "instruction": str(definition["intent"]),
        "sites": definition.get("sites") or [],
    }
    _write_json(episode_dir / "episode.json", {
        "dataset": dataset_name, "candidate_id": record["candidate_id"],
        "source": record["source"], "task": record["task"],
        "instruction": example["instruction"], "policy_result": policy_result,
    })
    return example

def _is_terminal_action(action: str, record: dict[str, Any]) -> bool:
    terminal_action = str((record.get("terminal") or {}).get("action") or "")
    return action == terminal_action or action.lstrip().startswith(("send_msg_to_user(", "report_infeasible("))

def _create_live_episode(
    record: dict[str, Any], definition: dict[str, Any], episode_dir: Path,
    *, instance: Any, skip_full_reset: bool, dataset_name: str = "webarena",
    reset_timeout_seconds: int = 4500,
    env_create_timeout_seconds: int = 600,
    env_reset_timeout_seconds: int = 300,
    action_timeout_seconds: int = 300,
    close_timeout_seconds: int = 120,
    operation_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], BrowserGymProbeClient]:
    from agentlab.experiments.loop import EnvArgs

    episode_dir.mkdir(parents=True, exist_ok=True)
    context = operation_context if operation_context is not None else {}
    if not skip_full_reset:
        context.update({"phase": "environment", "operation": "full_reset"})
        with _watchdog(reset_timeout_seconds, "full_reset"):
            instance.full_reset(skip_if_not_set=False)
    actions = _load_actions(record)
    replay_actions = [action for action in actions if not _is_terminal_action(action, record)]
    action_set = _build_live_action_set(str(record["runtime"]["action_set"]))
    viewport_values = record["runtime"].get("viewport")
    viewport = (
        {"width": int(viewport_values[0]), "height": int(viewport_values[1])}
        if isinstance(viewport_values, list) and len(viewport_values) == 2 else None
    )
    env_args = EnvArgs(
        task_name=str(record["task"]["name"]), task_seed=int(record["task_seed"]),
        max_steps=max(int(record["runtime"]["max_steps"]), len(actions) + 32),
        headless=True, record_video=False, viewport=viewport,
    )
    context.update({"phase": "environment", "operation": "env_create"})
    with _watchdog(env_create_timeout_seconds, "environment creation"):
        env = env_args.make_env(
            action_mapping=action_set.to_python_code, exp_dir=episode_dir,
            use_raw_page_output=False,
        )
    try:
        context.update({"phase": "environment", "operation": "env.reset"})
        with _watchdog(env_reset_timeout_seconds, "env.reset"):
            obs, reset_info = env.reset(seed=int(record["task_seed"]))
        _write_json(episode_dir / "reset_info.json", reset_info)
        traj_path = episode_dir / "traj.jsonl"
        with traj_path.open("w", encoding="utf-8") as handle:
            if not replay_actions:
                (episode_dir / "step_0.png").write_bytes(_screenshot_bytes(obs))
                handle.write(json.dumps({
                    "step_num": 0, "action_timestamp": "000000", "action": "WAIT",
                    "action_tool": "browser_use", "response": "",
                    "screenshot_file": "step_0.png",
                }) + "\n")
                _flush_jsonl(handle)
            for index, action in enumerate(replay_actions, start=1):
                context.update({
                    "phase": "policy_replay",
                    "operation": "env.step",
                    "action_index": index,
                    "action": action,
                })
                with _watchdog(action_timeout_seconds, f"env.step[{index}]"):
                    obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"Saved non-terminal action {index} terminated BrowserGym: {info}"
                    )
                screenshot_name = f"step_{index}.png"
                (episode_dir / screenshot_name).write_bytes(_screenshot_bytes(obs))
                handle.write(json.dumps({
                    "step_num": index, "action_timestamp": f"{index:06d}",
                    "action": action, "action_tool": "browser_use", "response": "",
                    "screenshot_file": screenshot_name,
                }, ensure_ascii=False) + "\n")
                _flush_jsonl(handle)
    except Exception:
        failed_context = dict(context)
        try:
            context.update({"phase": "environment", "operation": "env.close"})
            with _watchdog(close_timeout_seconds, "env.close after episode error"):
                env.close()
        except Exception as close_exc:
            context["close_error"] = repr(close_exc)
            logger.debug("BrowserGym env.close failed while handling an earlier error", exc_info=True)
        context["failed_operation"] = failed_context
        raise
    policy_result = float(record.get("original", {}).get("strict_score", 0))
    (episode_dir / "result.txt").write_text(f"{policy_result}\n", encoding="utf-8")
    example = {
        "id": str(record["candidate_id"]), "instruction": str(definition["intent"]),
        "sites": definition.get("sites") or [],
    }
    _write_json(episode_dir / "episode.json", {
        "dataset": dataset_name, "candidate_id": record["candidate_id"],
        "source": record["source"], "task": record["task"],
        "instruction": example["instruction"], "policy_result": policy_result,
        "terminal_action_skipped_for_probe": True,
        "action_url_routing": {
            "mode": "runtime_webarena_port_rewrite",
            "external_host": os.environ.get("WA_EXTERNAL_HOST"),
            "instance": os.environ.get("WA_INSTANCE", "0"),
            "port_stride": os.environ.get("WA_PORT_STRIDE", "1000"),
        },
    })
    return example, BrowserGymProbeClient(
        env,
        obs,
        policy_result,
        action_timeout_seconds=action_timeout_seconds,
        operation_context=context,
    )

def _error_summary(
    episode_dir: Path,
    record: dict[str, Any],
    evaluator_type: str,
    exc: Exception,
    rollout_id: str = "0",
    dataset_name: str = "webarena",
    error_context: dict[str, Any] | None = None,
) -> None:
    _write_json(episode_dir / "replay_eval" / "eval_summary.json", {
        "domain": "webarena", "example_id": record["candidate_id"], "rollout_id": rollout_id,
        "policy_model": f"{dataset_name}-{record.get('source', 'unknown')}",
        "policy_result": record.get("original", {}).get("strict_score"),
        "evaluator_type": evaluator_type, "final_label": "error", "final_status": "error",
        "error_message": str(exc), "traceback": traceback.format_exc(),
        "error_context": error_context or {},
    })

def _is_resumable_summary(path: Path) -> bool:

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    label = str(summary.get("final_label") or summary.get("final_status") or "").lower()
    return label in {"success", "failure"}

def _select(dataset: WebArenaDataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    records = [record for record in dataset.records if record.get("eligible", True)]
    if args.source:
        records = [record for record in records if record.get("source") == args.source]
    if args.candidate_id:
        wanted = set(args.candidate_id)
        records = [record for record in records if record.get("candidate_id") in wanted]
        missing = wanted - {str(record["candidate_id"]) for record in records}
        if missing:
            raise ValueError(f"Candidate IDs not found: {sorted(missing)}")
    if args.tasks:
        task_ids = {int(value) for value in args.tasks.replace(",", " ").split()}
        records = [record for record in records if int(record["task"]["task_id"]) in task_ids]
    from common import stable_shard
    records = [record for record in records if stable_shard(str(record["candidate_id"]), args.num_shards) == args.shard_index]
    records.sort(key=lambda item: str(item["candidate_id"]))
    return records[:args.limit] if args.limit else records

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="webarena")
    parser.add_argument("--evaluator-config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source", choices=("holo", "guiowl"))
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--rollout-repeats", "--repeats", dest="rollout_repeats", type=int, default=1,
        help="Number of independent judge calls per selected trajectory (passive only).",
    )
    parser.add_argument(
        "--rollout-id", dest="rollout_ids", action="append", type=int, default=[],
        help=(
            "Evaluate only these zero-based rollout IDs. Repeat the option for multiple IDs; "
            "intended for targeted retries with --rollout-repeats set to the full repeat count."
        ),
    )
    parser.add_argument(
        "--summary-path", type=Path,
        help="Write this invocation's summary here instead of the shared shard summary path.",
    )
    parser.add_argument(
        "--skip-manifest", action="store_true",
        help="Do not rewrite suite_manifest.json (useful for concurrent targeted retries).",
    )
    parser.add_argument("--max-eval-rounds", type=int, default=3)
    parser.add_argument("--sleep-after-execution", type=float, default=1.0)
    parser.add_argument("--skip-full-reset", action="store_true")
    parser.add_argument("--reset-timeout-seconds", type=int, default=4500)
    parser.add_argument("--env-create-timeout-seconds", type=int, default=600)
    parser.add_argument("--env-reset-timeout-seconds", type=int, default=300)
    parser.add_argument("--action-timeout-seconds", type=int, default=300)
    parser.add_argument("--close-timeout-seconds", type=int, default=120)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.rollout_repeats < 1:
        raise ValueError("Require --rollout-repeats >= 1")
    if args.rollout_ids:
        if args.rollout_repeats == 1:
            raise ValueError("--rollout-id requires --rollout-repeats greater than 1")
        if len(set(args.rollout_ids)) != len(args.rollout_ids):
            raise ValueError("--rollout-id values must be unique")
        if any(value < 0 or value >= args.rollout_repeats for value in args.rollout_ids):
            raise ValueError(
                f"Require every --rollout-id to be in [0, {args.rollout_repeats})"
            )
        rollout_indices = sorted(args.rollout_ids)
    else:
        rollout_indices = list(range(args.rollout_repeats))
    dataset = load_webarena_dataset(args.dataset, project_root=PROJECT_ROOT)
    evaluator_type = peek_evaluator_type(str(args.evaluator_config))
    if evaluator_type not in {*PASSIVE_BASELINE_TYPES, "guirl", VAGEN_QWEN_EVALUATOR_TYPE}:
        raise ValueError(
            "webarena currently supports zerogui/distrl/digirl/guirl/vagen_qwen, "
            f"got {evaluator_type!r}"
        )
    passive_cfg = (
        build_passive_baseline_config(str(args.evaluator_config))
        if evaluator_type in PASSIVE_BASELINE_TYPES else None
    )
    guirl_cfg = build_guirl_config(str(args.evaluator_config)) if evaluator_type == "guirl" else None
    vagen_cfg = (
        load_vagen_eval_config(str(args.evaluator_config))
        if evaluator_type == VAGEN_QWEN_EVALUATOR_TYPE else None
    )
    if vagen_cfg is not None and str(vagen_cfg.get("platform") or "").strip().lower() not in {
        "webarena", "web_arena"
    }:
        raise ValueError("WebArena vagen_qwen config must set platform: WebArena")
    if guirl_cfg is not None:
        if guirl_cfg["probe_agent_type"] != "WebArenaHolo_Probe":
            raise ValueError("webarena GUIRL requires probe_agent_type: WebArenaHolo_Probe")
        for key, env_name in {
            "judge_base_url": "WA_GUIRL_JUDGE_BASE_URL",
            "judge_model_name": "WA_GUIRL_JUDGE_MODEL_NAME",
            "judge_api_key": "WA_GUIRL_JUDGE_API_KEY",
            "probe_base_url": "WA_GUIRL_PROBE_BASE_URL",
            "probe_model_name": "WA_GUIRL_PROBE_MODEL_NAME",
            "probe_api_key": "WA_GUIRL_PROBE_API_KEY",
        }.items():
            if os.getenv(env_name):
                guirl_cfg[key] = os.environ[env_name]

    instance = None
    if evaluator_type in {"guirl", VAGEN_QWEN_EVALUATOR_TYPE}:
        from common import wa_site_env_defaults
        for variable, default in {
            **wa_site_env_defaults(),
            "PLAYWRIGHT_BROWSERS_PATH": str(WA_ROOT / ".venv" / "ms-playwright"),
        }.items():
            os.environ.setdefault(variable, default)
        from browsergym.webarena.instance import WebArenaInstance
        instance = WebArenaInstance()

    catalog = _task_catalog()
    records = _select(dataset, args)
    args.result_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_manifest:
        _write_json(args.result_root / "suite_manifest.json", {
            "suite": "webarena_eval", "dataset": dataset.name,
            "evaluator_type": evaluator_type, "evaluator_config": str(args.evaluator_config.resolve()),
            "selected": len(records), "num_shards": args.num_shards, "shard_index": args.shard_index,
            "rollout_repeats": args.rollout_repeats,
            "rollout_ids": rollout_indices,
        })
    total_runs = len(records) * len(rollout_indices)
    counts = {
        "selected": len(records), "rollout_repeats": args.rollout_repeats,
        "total_runs": total_runs, "completed": 0, "error": 0, "skipped": 0,
        "watchdog_timeout": 0,
    }
    run_index = 0
    episode_watchdog = False
    for record in records:
        for rollout_idx in rollout_indices:
            run_index += 1
            rollout_id = str(rollout_idx)
            candidate_root = args.result_root / str(record["candidate_id"])
            episode_dir = (
                candidate_root
                if args.rollout_repeats == 1
                else candidate_root / rollout_id
            )
            summary_path = episode_dir / "replay_eval" / "eval_summary.json"
            if args.resume and summary_path.is_file() and _is_resumable_summary(summary_path):
                counts["skipped"] += 1
                continue
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
            env_client = None
            operation_context: dict[str, Any] = {}
            episode_watchdog = False
            started = time.time()
            try:
                definition = _task_definition(record, catalog)
                domain = str((definition.get("sites") or ["webarena"])[0])
                policy_result = float(record.get("original", {}).get("strict_score", 0))
                policy_model = f"{dataset.name}-{record['source']}"
                if evaluator_type in PASSIVE_BASELINE_TYPES:
                    example = _materialize_source_episode(
                        record, definition, episode_dir, dataset_name=dataset.name
                    )
                    eval_args = SimpleNamespace(
                        passive_baseline_config=passive_cfg,
                        replay_eval_config_path=str(args.evaluator_config),
                    )
                    run_passive_baseline_eval_after_replay(
                        example=example, example_result_dir=str(episode_dir), domain=domain,
                        example_id=str(record["candidate_id"]), rollout_id=rollout_id,
                        policy_model=policy_model, policy_result=policy_result, args=eval_args,
                    )
                else:
                    assert instance is not None
                    example, env_client = _create_live_episode(
                        record, definition, episode_dir, instance=instance,
                        skip_full_reset=args.skip_full_reset, dataset_name=dataset.name,
                        reset_timeout_seconds=args.reset_timeout_seconds,
                        env_create_timeout_seconds=args.env_create_timeout_seconds,
                        env_reset_timeout_seconds=args.env_reset_timeout_seconds,
                        action_timeout_seconds=args.action_timeout_seconds,
                        close_timeout_seconds=args.close_timeout_seconds,
                        operation_context=operation_context,
                    )
                    if evaluator_type == "guirl":
                        assert guirl_cfg is not None
                        eval_args = SimpleNamespace(
                            guirl_config=guirl_cfg, max_eval_rounds=args.max_eval_rounds,
                            sleep_after_execution=args.sleep_after_execution,
                            export_guirl_sft_data=False,
                        )
                        run_guirl_eval_after_replay(
                            env_client=env_client, lease_id="", example=example,
                            example_result_dir=str(episode_dir), domain=domain,
                            example_id=str(record["candidate_id"]), rollout_id=rollout_id,
                            policy_model=policy_model, policy_result=policy_result, args=eval_args,
                        )
                    else:
                        assert vagen_cfg is not None
                        eval_args = SimpleNamespace(vagen_eval_config=vagen_cfg)
                        run_vagen_eval_after_replay(
                            env_client=env_client, lease_id="", example=example,
                            example_result_dir=str(episode_dir), domain=domain,
                            example_id=str(record["candidate_id"]), rollout_id=rollout_id,
                            policy_model=policy_model, policy_result=policy_result, args=eval_args,
                        )
                counts["completed"] += 1
                logger.info(
                    "[%d/%d] %s rollout=%s completed in %.1fs",
                    run_index, total_runs, record["candidate_id"], rollout_id,
                    time.time() - started,
                )
            except Exception as exc:
                counts["error"] += 1
                episode_watchdog = isinstance(exc, RunnerWatchdogTimeout)
                if episode_watchdog:
                    counts["watchdog_timeout"] += 1
                logger.exception(
                    "[%d/%d] %s rollout=%s failed",
                    run_index, total_runs, record["candidate_id"], rollout_id,
                )
                episode_dir.mkdir(parents=True, exist_ok=True)
                _error_summary(
                    episode_dir, record, evaluator_type, exc, rollout_id,
                    dataset_name=dataset.name, error_context=operation_context,
                )
            finally:
                if env_client is not None:
                    close_context = dict(operation_context)
                    operation_context.update({
                        "phase": "environment", "operation": "env.close",
                    })
                    try:
                        with _watchdog(args.close_timeout_seconds, "env.close"):
                            env_client.env.close()
                    except RunnerWatchdogTimeout as exc:
                        episode_watchdog = True
                        counts["watchdog_timeout"] += 1
                        operation_context["failed_operation"] = close_context
                        operation_context["close_error"] = repr(exc)
                        _write_json(episode_dir / "watchdog_timeout.json", {
                            "error": str(exc), "error_type": type(exc).__name__,
                            "context": operation_context,
                        })
                        logger.exception("BrowserGym env.close exceeded watchdog")
                    except Exception:
                        logger.debug("BrowserGym env.close failed after episode", exc_info=True)
            if episode_watchdog:
                logger.error(
                    "Shard %d stopping after watchdog timeout; launcher must restart "
                    "a fresh browser process", args.shard_index,
                )
                break
        if episode_watchdog:
            break
    summary_path = args.summary_path or (args.result_root / f"shard_{args.shard_index:02d}_summary.json")
    _write_json(summary_path, counts)
    print(json.dumps(counts, indent=2))
    if counts["watchdog_timeout"]:
        return 75
    return int(counts["error"] > 0)

if __name__ == "__main__":
    raise SystemExit(main())
