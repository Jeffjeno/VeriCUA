from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import time
import traceback as _traceback_mod
import uuid
from collections import Counter, deque
from multiprocessing import Pipe, Process, current_process
from pathlib import Path
from typing import Any

import yaml

import eval_runtime
from eval_runtime.config import load_eval_dataset_config
from eval_runtime.passive_baseline import (
    PASSIVE_BASELINE_TYPES,
    build_passive_baseline_config,
    peek_evaluator_type,
    run_passive_baseline_eval_after_replay,
)
from eval_runtime.env_adapter import (
    AgentRuntimeEnvClient,
    cleanup_docker_containers,
    _ensure_docker_tag,
)
from eval_runtime.guirl.evaluator import (
    build_guirl_config,
    parse_policy_result,
    run_guirl_eval_after_replay,
)
from eval_runtime.task_runner import run_single_example_client

import eval_runtime.agents.replay
from scripts.replay_eval_results import collect_results, gt_label

logger = logging.getLogger("eval_runtime.main")
processes: list[Process] = []
is_terminating = False
_ray_actors: list = []

REPLAY_EVAL_DIR = "replay_eval"
REPLAY_EVAL_SUMMARY_FILE = "eval_summary.json"

def _default_project_dir() -> str:
    env_dir = os.environ.get("PROJECT_DIR", "").strip()
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _resolve_path(root: str, path: str) -> str:

    if not path:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(root, expanded))

_PROXY_CONFIG_REL = os.path.join(
    "evaluation_examples", "settings", "proxy", "dataimpulse.json"
)

def _resolve_proxy_config_file(osworld_path: str, project_dir: str = "") -> str:

    candidates: list[str] = []
    for root in (osworld_path, project_dir):
        if not root:
            continue
        candidate = os.path.join(os.path.abspath(root), _PROXY_CONFIG_REL)
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""

def _apply_osworld_env(osworld_path: str, project_dir: str = "") -> None:

    proxy_cfg = _resolve_proxy_config_file(osworld_path, project_dir)
    if proxy_cfg:
        os.environ["PROXY_CONFIG_FILE"] = proxy_cfg
        return

    import tempfile
    empty_proxy = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="osworld_proxy_empty_", delete=False,
    )
    empty_proxy.write("[]")
    empty_proxy.close()
    os.environ["PROXY_CONFIG_FILE"] = empty_proxy.name

def _resolve_runtime_paths(args: argparse.Namespace) -> None:

    root = args.project_dir

    path_attrs = (
        "result_dir", "test_all_meta_path", "test_config_base_dir",
        "osworld_path", "policy_config_path", "replay_eval_config_path",
        "cache_dir", "path_to_vm", "eval_dataset_config",
    )
    for attr in path_attrs:
        val = getattr(args, attr, "")
        if val:
            setattr(args, attr, _resolve_path(args.project_dir, val))

    inline = getattr(args, "policy_config_inline", None)
    if isinstance(inline, dict):
        for key in ("source_result_root", "config_path"):
            if inline.get(key):
                inline[key] = _resolve_path(args.project_dir, str(inline[key]))

    _apply_osworld_env(args.osworld_path, args.project_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay eval via AgentRuntime")
    parser.add_argument("--osworld_path", type=str, required=True)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--path_to_vm", type=str, default="")
    parser.add_argument("--docker_tag", type=str, default="")
    parser.add_argument("--policy_config_path", type=str, default="",
                        help="Path to policy config YAML. Optional when --eval_dataset_config is provided.")
    parser.add_argument("--replay_eval_config_path", type=str, default="",
                        help="Replay eval config; evaluator_type=guirl enables GUIRL judge/probe; "
                             "zerogui|distrl|digirl enables static passive baseline judge")
    parser.add_argument("--test_all_meta_path", type=str, default="",
                        help="Required unless --eval_dataset_config is provided")
    parser.add_argument("--test_config_base_dir", type=str, default="",
                        help="Required unless --eval_dataset_config is provided")
    parser.add_argument("--examples_subdir", type=str, default="examples")
    parser.add_argument("--result_dir", type=str, default="./results/replay_eval")
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--task_n_combinations", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_eval_rounds", type=int, default=3)
    parser.add_argument("--sampler_max_steps", type=int, default=10)
    parser.add_argument("--sleep_after_execution", type=float, default=2.0)
    parser.add_argument("--sleep_after_reset", type=float, default=30.0,
                        help="Seconds to wait after env reset for VM to stabilize")
    parser.add_argument("--sleep_before_eval", type=float, default=20.0,
                        help="Seconds to wait before evaluation for last action to settle")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", type=str, default="screenshot")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run VM display headless (default: True)")
    parser.add_argument("--eval_dataset_config", type=str, default="",
                        help="Optional dataset config YAML (overrides individual paths)")
    parser.add_argument("--project_dir", type=str, default="",
                        help="Project root for resolving relative paths (default: PROJECT_DIR env "
                             "or repo root). All relative CLI paths are joined against this root.")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Pre-populated OSWorld setup cache (default: <project_dir>/cache). "
                             "Must be absolute on Ray workers; passed via --cache_dir from shell.")

    parser.add_argument("--ray_address", type=str, default="",
                        help="Ray cluster address (e.g. 'auto'). When set, uses Ray actors "
                             "instead of multiprocessing. Leave empty for local multiprocessing.")
    parser.add_argument("--ray_num_actors", type=int, default=0,
                        help="Number of Ray actors (default: same as --num_envs)")
    parser.add_argument("--max-worker-restarts", dest="max_worker_restarts", type=int, default=3,
                        help="Maximum restarts per local worker slot.")
    parser.add_argument("--max-task-attempts", dest="max_task_attempts", type=int, default=2,
                        help="Maximum attempts for a task whose worker dies.")
    parser.add_argument(
        "--export_guirl_sft_data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export successful GUIRL judge sessions to sft_data/guirl_sft_data.jsonl. "
             "Default: off (or GUIRL_EXPORT_SFT_DATA=0/1). Suites disable this; "
             "enable only for dedicated SFT data generation.",
    )
    parser.add_argument(
        "--export_guirl_sft_strip_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When exporting GUIRL SFT data, replace inline base64 images with placeholders "
             "(default: true). Use --no-export_guirl_sft_strip_images to keep raw images.",
    )
    return parser.parse_args()

def setup_logging(log_level: str, project_dir: str = ".") -> None:
    import datetime as dt
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    root.handlers.clear()
    ts = dt.datetime.now().strftime("%Y%m%d@%H%M%S")
    fmt = logging.Formatter("[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"replay_eval-{ts}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, log_level.upper()))
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

    for _ns in ("desktopenv", "desktop_env", "urllib3", "docker", "PIL", "asyncio", "playwright"):
        logging.getLogger(_ns).setLevel(logging.ERROR)

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)

def _normalize_agent_type(agent_type: str, default: str) -> str:
    normalized = str(agent_type or "").strip()
    if not normalized:
        return default
    return normalized[:-5] if normalized.endswith("Agent") else normalized

def _resolve_policy_config(policy_config_path: str) -> dict[str, Any]:
    cfg = _load_yaml(policy_config_path)
    agent_type = _normalize_agent_type(cfg.get("agent_type", "Replay"), "Replay")
    model_name = ""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        model_name = str(model_cfg.get("model_name", "")).strip()
    if not model_name and agent_type == "Replay":
        run_name = str(cfg.get("run_name", "")).strip()
        if run_name:
            model_name = run_name
        else:
            source_root = str(cfg.get("source_result_root", "")).strip()
            model_name = os.path.basename(os.path.normpath(source_root)) if source_root else "Replay"
    return {"agent_type": agent_type, "model_name": model_name or "unknown", "config": cfg}

def _eval_summary_path(example_result_dir: str) -> str:
    return os.path.join(example_result_dir, REPLAY_EVAL_DIR, REPLAY_EVAL_SUMMARY_FILE)


def _task_result_dir(args, task) -> str:
    return os.path.join(args.result_dir, args.action_space, args.observation_type,
                        args.policy_model_name, *task)


def _result_identity(args, task) -> dict[str, Any]:
    policy = getattr(args, "policy_config_inline", None)
    if policy is None:
        policy = _load_yaml(args.policy_config_path) if getattr(args, "policy_config_path", "") else {}
    repeat = max(1, int(policy.get("source_rollout_repeat", 1)))
    source_id = str(int(task[2]) // repeat)
    source_root = str(policy.get("source_result_root", ""))
    return {
        "run_id": getattr(args, "run_id", ""),
        "domain": task[0], "example_id": task[1], "rollout_id": task[2],
        "policy_model": args.policy_model_name,
        "eval_model": getattr(args, "guirl_config", {}).get("judge_model_name")
            or getattr(args, "passive_baseline_config", {}).get("model_name", "environment"),
        "source_trajectory": os.path.join(source_root, task[0], task[1], source_id),
    }


def _persist_task_error(args, task, error, policy_result=None) -> None:
    summary = _result_identity(args, task)
    summary.update(final_label="error", final_status="error", error=str(error),
                   policy_result=policy_result)
    _write_json(_eval_summary_path(_task_result_dir(args, task)), summary)


def _finalize_summary(args, task, summary) -> None:
    summary.update(_result_identity(args, task))
    _write_json(_eval_summary_path(_task_result_dir(args, task)), summary)

def get_unfinished(
    action_space: str, model_name: str, observation_type: str,
    result_dir: str, total_file_json: dict[str, list[str]], rollout_num: int,
    run_id: str = "",
) -> list[tuple[str, str, str]]:
    target_dir = os.path.join(result_dir, action_space, observation_type, model_name)
    unfinished = []
    for domain, example_ids in total_file_json.items():
        for example_id in example_ids:
            if example_id == "onboard":
                continue
            for rollout_idx in range(rollout_num):
                rollout_id = str(rollout_idx)
                rollout_path = os.path.join(target_dir, domain, example_id, rollout_id)
                has_result = os.path.exists(os.path.join(rollout_path, "result.txt"))
                has_eval = os.path.exists(_eval_summary_path(rollout_path))
                if has_result and has_eval:
                    try:
                        with open(_eval_summary_path(rollout_path), "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        if (
                            isinstance(summary, dict)
                            and summary.get("final_label") in {"success", "failure", "uncertain", "yes", "no"}
                            and summary.get("final_status") != "error"
                            and not summary.get("error")
                            and gt_label(summary.get("policy_result")) is not None
                            and (not run_id or summary.get("run_id") == run_id)
                        ):
                            continue
                    except (OSError, ValueError, TypeError):
                        pass
                if os.path.isdir(rollout_path):
                    shutil.rmtree(rollout_path, ignore_errors=True)
                unfinished.append((domain, example_id, rollout_id))
    return unfinished

def _create_policy_agent(args: argparse.Namespace):

    from eval_runtime.agents.base import Agent
    _inline = getattr(args, "policy_config_inline", None)
    if args.policy_config_path:
        return Agent.from_params(
            {"type": args.policy_agent_type, "config_path": args.policy_config_path}
        )
    inline_kwargs = {k: v for k, v in _inline.items() if k != "agent_type"}
    return Agent.from_params({"type": args.policy_agent_type, **inline_kwargs})

def _execute_single_task(
    env_client: AgentRuntimeEnvClient,
    policy_agent,
    domain: str,
    example_id: str,
    rollout_id: str,
    args: argparse.Namespace,
    scores: list,
    worker_name: str = "worker",
) -> dict[str, Any]:

    task_time_list: list[tuple[str, float]] = [("start", time.time())]
    lease = None
    policy_result = None
    task = (domain, example_id, str(rollout_id))
    example_result_dir = os.path.join(
        args.result_dir, args.action_space, args.observation_type,
        args.policy_model_name, domain, example_id, str(rollout_id),
    )
    try:
        config_file = os.path.join(
            args.test_config_base_dir, args.examples_subdir,
            domain, f"{example_id}.json",
        )
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)

        logger.info("[%s] %s/%s/%s: %s",
                    worker_name, domain, example_id, rollout_id,
                    example.get("instruction", "")[:80])

        if os.path.exists(example_result_dir):
            shutil.rmtree(example_result_dir)
        os.makedirs(example_result_dir, exist_ok=True)

        lease = env_client.allocate(
            f"{domain}:{example_id}:{rollout_id}:{uuid.uuid4().hex[:8]}"
        )
        task_time_list.append(("allocate_env", time.time()))

        if hasattr(policy_agent, "set_task_context"):
            policy_agent.set_task_context(domain, example_id, rollout_id)

        policy_result_raw = run_single_example_client(
            policy_agent, env_client, lease, example,
            args.max_steps, example["instruction"], args,
            example_result_dir, scores, task_time_list,
        )
        task_time_list.append(("policy_replay_done", time.time()))
        policy_result = parse_policy_result(policy_result_raw)

        if args.use_guirl_eval:
            summary = run_guirl_eval_after_replay(
                env_client=env_client,
                lease_id=lease["lease_id"],
                example=example,
                example_result_dir=example_result_dir,
                domain=domain,
                example_id=example_id,
                rollout_id=str(rollout_id),
                policy_model=args.policy_model_name,
                policy_result=policy_result,
                args=args,
            )
            task_time_list.append(("guirl_eval_done", time.time()))
            logger.info("[%s] DONE %s/%s/%s: score=%s label=%s",
                        worker_name, domain, example_id, rollout_id,
                        policy_result, summary["final_label"])
            _finalize_summary(args, task, summary)
            return {"score": policy_result, "label": summary["final_label"], "error": None}

        if getattr(args, "use_passive_baseline_eval", False):
            summary = run_passive_baseline_eval_after_replay(
                example=example,
                example_result_dir=example_result_dir,
                domain=domain,
                example_id=example_id,
                rollout_id=str(rollout_id),
                policy_model=args.policy_model_name,
                policy_result=policy_result,
                args=args,
            )
            task_time_list.append(("passive_baseline_eval_done", time.time()))
            logger.info("[%s] DONE %s/%s/%s: score=%s label=%s",
                        worker_name, domain, example_id, rollout_id,
                        policy_result, summary["final_label"])
            _finalize_summary(args, task, summary)
            return {"score": policy_result, "label": summary["final_label"], "error": None}

        label = "yes" if (policy_result is not None and policy_result >= 1.0) else "no"
        summary = {
            "domain": domain,
            "example_id": example_id,
            "rollout_id": rollout_id,
            "policy_model": args.policy_model_name,
            "policy_result": policy_result,
            "final_label": label,
        }
        logger.info("[%s] DONE %s/%s/%s: score=%s label=%s",
                    worker_name, domain, example_id, rollout_id, policy_result, label)
        _finalize_summary(args, task, summary)
        return {"score": policy_result, "label": label, "error": None}

    except Exception as exc:
        tb = _traceback_mod.format_exc()
        logger.error("Exception in %s %s/%s: %s", worker_name, domain, example_id, exc)
        logger.error(tb)
        if example_result_dir:
            try:
                os.makedirs(example_result_dir, exist_ok=True)
                replay_dir = os.path.join(example_result_dir, REPLAY_EVAL_DIR)
                os.makedirs(replay_dir, exist_ok=True)
                with open(os.path.join(replay_dir, "traceback.txt"), "w", encoding="utf-8") as f:
                    f.write(tb)
                _persist_task_error(args, task, exc, policy_result)
            except OSError as write_exc:
                logger.warning("Failed to persist traceback for %s/%s: %s", domain, example_id, write_exc)
        return {"score": None, "label": None, "error": str(exc)}
    finally:
        if lease is not None:
            try:
                env_client.close(lease["lease_id"])
            except Exception as close_exc:
                logger.error("Failed to close lease: %s", close_exc)

def run_env_tasks(connection, args) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    env_client = None
    try:
        _apply_osworld_env(args.osworld_path, args.project_dir)
        env_kwargs = {}
        if getattr(args, "cache_dir", ""):
            env_kwargs["cache_dir"] = args.cache_dir
        env_client = AgentRuntimeEnvClient(
            osworld_path=args.osworld_path,
            provider_name=args.provider_name,
            path_to_vm=args.path_to_vm,
            headless=args.headless,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            sleep_after_execution=args.sleep_after_execution,
            docker_tag=args.docker_tag,
            **env_kwargs,
        )
        policy_agent = _create_policy_agent(args)
        connection.send({"type": "ready"})
        while True:
            task = connection.recv()
            if task is None:
                break
            scores = []
            result = _execute_single_task(
                env_client, policy_agent, *task, args, scores,
                worker_name=current_process().name,
            )
            connection.send({"type": "result", "task": task, "result": result, "scores": scores})
    except EOFError:
        pass
    except Exception:
        logger.exception("Worker %s failed", current_process().name)
        raise
    finally:
        if env_client is not None:
            try:
                env_client.close_all()
            except Exception:
                logger.exception("Worker close_all failed")
        connection.close()


def _run_multiprocessing(tasks: list[tuple[str, str, str]], args: argparse.Namespace) -> tuple[list, list, list]:
    global processes
    if args.num_envs < 1:
        raise ValueError("num_envs must be positive")
    max_restarts = args.max_worker_restarts
    max_attempts = args.max_task_attempts
    if max_restarts < 0 or max_attempts < 1:
        raise ValueError("restart limit must be nonnegative and task attempts must be positive")
    pending = deque(tasks)
    attempts = Counter()
    scores, labels, failures = [], [], []
    slots = []
    processes = []

    def fail(task, error):
        summary_path = _eval_summary_path(_task_result_dir(args, task))
        previous = None
        try:
            with open(summary_path, encoding="utf-8") as f:
                previous = json.load(f).get("policy_result")
        except (OSError, ValueError, AttributeError):
            pass
        _persist_task_error(args, task, error, previous)
        failures.append({"task": list(task), "error": error})

    def start(slot):
        parent, child = Pipe()
        proc = Process(
            target=run_env_tasks, args=(child, args),
            name=f"EvalWorker-{slot['index'] + 1}-{slot['restarts']}",
        )
        proc.daemon = True
        proc.start()
        child.close()
        slot.update(process=proc, connection=parent, ready=False, task=None)
        processes.append(proc)

    try:
        for index in range(min(args.num_envs, len(tasks))):
            slot = {"index": index, "restarts": 0}
            start(slot)
            slots.append(slot)
        completed = 0
        while completed < len(tasks):
            for slot in slots:
                proc = slot.get("process")
                if proc is None:
                    continue
                connection = slot["connection"]
                # Read buffered results before inspecting exit status: the worker
                # may have delivered its result immediately before exiting.
                while connection.poll():
                    try:
                        message = connection.recv()
                    except (EOFError, OSError):
                        break
                    if message["type"] == "ready":
                        slot["ready"] = True
                    elif message["type"] == "result":
                        task = slot["task"]
                        if task is None or tuple(message["task"]) != task:
                            raise RuntimeError("worker returned an unexpected task")
                        result = message["result"]
                        if result.get("error"):
                            fail(task, result["error"])
                        else:
                            scores.extend(message["scores"])
                            if result.get("label") is not None:
                                labels.append(result["label"])
                        slot["task"] = None
                        slot["ready"] = True
                        completed += 1
                if not proc.is_alive():
                    proc.join()
                    connection.close()
                    task = slot["task"]
                    if task is not None:
                        if attempts[task] < max_attempts:
                            pending.appendleft(task)
                            logger.warning("Reclaiming %s after worker exit %s", task, proc.exitcode)
                        else:
                            fail(task, f"worker exited with code {proc.exitcode} after {attempts[task]} attempts")
                            completed += 1
                    slot["task"] = None
                    if slot["restarts"] < max_restarts and completed < len(tasks):
                        slot["restarts"] += 1
                        start(slot)
                    else:
                        slot["process"] = None
                    continue
                if slot["ready"] and slot["task"] is None and pending:
                    task = pending.popleft()
                    # Only the supervisor owns pending/in-flight state. A worker
                    # cannot die between dequeuing a task and recording ownership.
                    slot["task"] = task
                    slot["ready"] = False
                    attempts[task] += 1
                    try:
                        connection.send(task)
                    except (BrokenPipeError, EOFError, OSError):
                        proc.terminate()
                if completed == len(tasks):
                    break
            if not any(slot.get("process") is not None for slot in slots):
                while pending:
                    fail(pending.popleft(), "all worker slots exhausted their restart limit")
                    completed += 1
            if completed < len(tasks):
                time.sleep(0.1)
        return scores, labels, failures
    finally:
        for slot in slots:
            proc = slot.get("process")
            if proc is None:
                continue
            try:
                slot["connection"].send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for slot in slots:
            proc = slot.get("process")
            if proc is None:
                continue
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            slot["connection"].close()
        processes.clear()

def _broadcast_group_cleanup(ray_mod, run_group: str, timeout: int = 60) -> None:

    if not run_group:
        return
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    nodes = [n for n in ray_mod.nodes() if n.get("Alive") and n.get("Resources", {}).get("osworld_vm", 0) > 0]
    if not nodes:
        return

    @ray_mod.remote(num_cpus=0, resources={"osworld_vm": 0.001})
    def _do_cleanup(rg: str) -> int:
        from eval_runtime.env_adapter import cleanup_docker_by_group
        return cleanup_docker_by_group(rg)

    pg = placement_group(
        [{"osworld_vm": 0.001}] * len(nodes),
        strategy="SPREAD",
        name=f"cleanup-{run_group}",
    )
    try:
        ray_mod.get(pg.ready(), timeout=20)
        refs = [
            _do_cleanup.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote(run_group)
            for i in range(len(nodes))
        ]
        results = ray_mod.get(refs, timeout=timeout)
        total = sum(r for r in results if isinstance(r, int))
        if total:
            logger.info("Broadcast cleanup removed %d container(s) across %d node(s)", total, len(nodes))
    except Exception as exc:
        logger.warning("Broadcast group cleanup encountered an error: %s", exc)
    finally:
        try:
            remove_placement_group(pg)
        except Exception:
            pass

def _run_ray_distributed(tasks: list[tuple[str, str, str]], args: argparse.Namespace) -> tuple[list, list, list]:

    import ray

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    num_actors = args.ray_num_actors or args.num_envs
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive")]

    total_osworld_vm = int(sum(
        n.get("Resources", {}).get("osworld_vm", 0) for n in alive_nodes
    ))
    if total_osworld_vm > 0 and num_actors > total_osworld_vm:
        logger.warning(
            "Capping num_actors from %d to %d (total osworld_vm slots in cluster). "
            "To run more actors per node, restart Ray workers with "
            "--resources='{\"osworld_vm\": N}'.",
            num_actors, total_osworld_vm,
        )
        num_actors = total_osworld_vm

    logger.info("Ray cluster: %d alive node(s), creating %d actors (osworld_vm slots: %d)",
                len(alive_nodes), num_actors, total_osworld_vm)

    _HEAD_ENV_VARS = {
        k: os.environ[k]
        for k in ("HF_ENDPOINT", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
                  "https_proxy", "http_proxy", "no_proxy", "OSWORLD_RUN_GROUP",
                  "PROXY_CONFIG_FILE")
        if k in os.environ and os.environ[k]
    }
    logger.info("Forwarding env vars to Ray actors: %s", list(_HEAD_ENV_VARS.keys()))

    run_group = os.environ.get("OSWORLD_RUN_GROUP", "").strip()
    if run_group:
        logger.info("Pre-run: broadcasting container cleanup for run_group=%s across all nodes...", run_group)
        _broadcast_group_cleanup(ray, run_group, timeout=60)

    @ray.remote(num_cpus=1, resources={"osworld_vm": 1})
    class EvalWorkerActor:

        def __init__(self, args_dict: dict[str, Any], actor_idx: int):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import logging as _logging
            for _ns in ("desktopenv", "desktop_env", "urllib3", "docker", "PIL", "asyncio", "playwright"):
                _logging.getLogger(_ns).setLevel(_logging.ERROR)

            for k, v in args_dict.get("_ray_env_vars", {}).items():
                if v:
                    os.environ[k] = v
            self.actor_name = f"RayEvalWorker-{actor_idx}"

            _args = argparse.Namespace(**{k: v for k, v in args_dict.items()
                                          if not k.startswith("_")})
            _apply_osworld_env(
                getattr(_args, "osworld_path", ""),
                getattr(_args, "project_dir", ""),
            )

            import eval_runtime
            import eval_runtime.agents.replay

            self.args = _args
            _env_kwargs: dict[str, Any] = {}
            if getattr(_args, "cache_dir", ""):
                _env_kwargs["cache_dir"] = _args.cache_dir
            self.env_client = AgentRuntimeEnvClient(
                osworld_path=_args.osworld_path,
                provider_name=_args.provider_name,
                path_to_vm=_args.path_to_vm,
                headless=_args.headless,
                screen_width=_args.screen_width,
                screen_height=_args.screen_height,
                sleep_after_execution=_args.sleep_after_execution,
                docker_tag=f"{_args.docker_tag}-ray{actor_idx}",
                **_env_kwargs,
            )
            self.policy_agent = _create_policy_agent(_args)
            self._scores: list = []

        def run_task(self, domain: str, example_id: str, rollout_id: str) -> dict[str, Any]:
            return _execute_single_task(
                self.env_client, self.policy_agent,
                domain, example_id, rollout_id,
                self.args, self._scores,
                worker_name=self.actor_name,
            )

        def get_scores(self) -> list:
            return list(self._scores)

        def shutdown(self) -> None:
            try:
                self.env_client.close_all()
            except Exception:
                pass

    global _ray_actors
    args_dict = dict(vars(args))
    args_dict["_ray_env_vars"] = _HEAD_ENV_VARS
    actors = [EvalWorkerActor.remote(args_dict, i) for i in range(num_actors)]
    _ray_actors = actors

    pending: dict[Any, int] = {}
    all_results: list[dict[str, Any]] = []
    task_idx = 0

    for i, task_tuple in enumerate(tasks):
        actor_i = i % num_actors
        ref = actors[actor_i].run_task.remote(*task_tuple)
        pending[ref] = i
        task_idx += 1

        if len(pending) >= num_actors * 2:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1)
            for r in ready:
                try:
                    all_results.append(ray.get(r))
                except Exception as exc:
                    logger.error("Ray task failed: %s", exc)
                    _persist_task_error(args, tasks[pending[r]], exc)
                    all_results.append({"score": None, "label": None, "error": str(exc)})
                pending.pop(r)
            logger.info("Progress: %d/%d tasks dispatched, %d collected",
                        task_idx, len(tasks), len(all_results))

    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        for r in ready:
            try:
                all_results.append(ray.get(r))
            except Exception as exc:
                logger.error("Ray task failed: %s", exc)
                _persist_task_error(args, tasks[pending[r]], exc)
                all_results.append({"score": None, "label": None, "error": str(exc)})
            pending.pop(r)

    scores: list[float] = []
    for actor in actors:
        try:
            scores.extend(ray.get(actor.get_scores.remote()))
        except Exception:
            pass

    failures = [r for r in all_results if r.get("error")]
    labels = [r["label"] for r in all_results if r.get("label") is not None]

    for actor in actors:
        try:
            ray.get(actor.shutdown.remote())
        except Exception:
            pass

    _ray_actors.clear()

    if run_group:
        logger.info("Post-run: broadcasting container cleanup for run_group=%s...", run_group)
        _broadcast_group_cleanup(ray, run_group, timeout=60)

    ray.shutdown()
    logger.info("Ray run complete: %d tasks, %d scores, %d labels",
                len(all_results), len(scores), len(labels))
    return scores, labels, failures

def signal_handler(signum, frame) -> None:
    global is_terminating, _ray_actors
    if is_terminating:
        return
    is_terminating = True
    logger.info("Received signal %s. Shutting down.", signum)

    for proc in processes:
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass

    if _ray_actors:
        logger.info("Shutting down %d Ray actor(s) and cleaning up Docker containers...", len(_ray_actors))
        try:
            import ray as _ray
            shutdown_refs = [actor.shutdown.remote() for actor in _ray_actors]
            try:
                _ray.get(shutdown_refs, timeout=30)
            except Exception:
                pass

            run_group = os.environ.get("OSWORLD_RUN_GROUP", "").strip()
            if run_group:
                _broadcast_group_cleanup(_ray, run_group, timeout=30)
        except Exception as exc:
            logger.warning("Ray actor shutdown during signal: %s", exc)
        finally:
            _ray_actors.clear()

    time.sleep(1)
    sys.exit(128 + signum)

def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    args.project_dir = _resolve_path(os.getcwd(), args.project_dir or _default_project_dir())
    if not args.cache_dir:
        args.cache_dir = os.path.join(args.project_dir, "cache")
    setup_logging(args.log_level, args.project_dir)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args.docker_tag = _ensure_docker_tag(args.docker_tag)
    logger.info("Docker tag for this run: %s", args.docker_tag)

    cleanup_docker_containers(args.docker_tag)

    if args.eval_dataset_config:
        args.eval_dataset_config = _resolve_path(args.project_dir, args.eval_dataset_config)
        ds_cfg = load_eval_dataset_config(args.eval_dataset_config, args.project_dir)
        if not args.test_all_meta_path:
            args.test_all_meta_path = ds_cfg["test_all_meta_path"]
        if not args.test_config_base_dir:
            args.test_config_base_dir = ds_cfg["test_config_base_dir"]
        args.examples_subdir = ds_cfg.get("examples_subdir", args.examples_subdir)
        args.task_n_combinations = ds_cfg.get("task_n_combinations", args.task_n_combinations)
        args.max_steps = ds_cfg.get("max_steps", args.max_steps)
        if ds_cfg.get("result_dir"):
            args.result_dir = ds_cfg["result_dir"]
        if not args.policy_config_path and ds_cfg.get("policy"):
            args.policy_config_inline = ds_cfg["policy"]

    if not args.test_all_meta_path or not args.test_config_base_dir:
        logger.error(
            "--test_all_meta_path and --test_config_base_dir are required "
            "unless --eval_dataset_config is provided."
        )
        sys.exit(1)

    _resolve_runtime_paths(args)

    inline = getattr(args, "policy_config_inline", None)
    if not args.policy_config_path and not inline:
        logger.error(
            "--policy_config_path is required unless --eval_dataset_config provides a policy section."
        )
        sys.exit(1)

    if inline is not None:
        agent_type = _normalize_agent_type(inline.get("agent_type", "Replay"), "Replay")
        model_name = (
            str(inline.get("run_name") or inline.get("source_model_name") or "").strip()
            or os.path.basename(os.path.normpath(str(inline.get("source_result_root", "Replay"))))
        )
        policy_info = {"agent_type": agent_type, "model_name": model_name or "unknown", "config": inline}
    else:
        policy_info = _resolve_policy_config(args.policy_config_path)
    args.policy_agent_type = policy_info["agent_type"]
    args.policy_model_name = policy_info["model_name"]
    args.guirl_config = {}
    args.use_guirl_eval = False
    args.passive_baseline_config = {}
    args.use_passive_baseline_eval = False
    if args.replay_eval_config_path:
        evaluator_type = peek_evaluator_type(args.replay_eval_config_path)
        if evaluator_type == "guirl":

            args.guirl_config = build_guirl_config(args.replay_eval_config_path)
            args.use_guirl_eval = True
        elif evaluator_type in PASSIVE_BASELINE_TYPES:
            args.passive_baseline_config = build_passive_baseline_config(
                args.replay_eval_config_path
            )
            args.use_passive_baseline_eval = True
        else:
            raise ValueError(f"Unsupported evaluator_type: {evaluator_type!r}")

    if args.export_guirl_sft_data is None:
        env_flag = os.environ.get("GUIRL_EXPORT_SFT_DATA", "").strip().lower()
        args.export_guirl_sft_data = env_flag in {"1", "true", "yes", "on"}

    logger.info("Project dir: %s", args.project_dir)
    logger.info("Policy: type=%s model=%s", args.policy_agent_type, args.policy_model_name)
    logger.info("Eval config: %s", args.replay_eval_config_path)
    logger.info("Use GUIRL eval: %s", args.use_guirl_eval)
    logger.info("Export GUIRL SFT data: %s", args.export_guirl_sft_data)
    logger.info("Use passive baseline eval: %s", args.use_passive_baseline_eval)
    if args.use_passive_baseline_eval:
        logger.info(
            "Passive baseline type: %s",
            args.passive_baseline_config.get("evaluator_type"),
        )
    logger.info("Result dir: %s", args.result_dir)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    if args.task_n_combinations < 1 or args.num_envs < 1:
        raise ValueError("task_n_combinations and num_envs must be positive")
    expected_tasks = [
        (str(domain), str(example), str(i))
        for domain, examples in test_all_meta.items() for example in examples
        if str(example) != "onboard" for i in range(args.task_n_combinations)
    ]
    if len(set(expected_tasks)) != len(expected_tasks) or not expected_tasks:
        raise ValueError("evaluation manifest is empty or contains duplicate tasks")
    identity_config = {
        "policy": policy_info["config"],
        "judge": _load_yaml(args.replay_eval_config_path) if args.replay_eval_config_path else {},
        "max_steps": args.max_steps, "max_eval_rounds": args.max_eval_rounds,
        "sampler_max_steps": args.sampler_max_steps,
    }
    config_hash = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode()).hexdigest()
    inventory_path = os.path.join(args.result_dir, "eval_manifest.json")
    previous_inventory = None
    if os.path.exists(inventory_path):
        with open(inventory_path, encoding="utf-8") as f:
            previous_inventory = json.load(f)
            if previous_inventory.get("config_hash") != config_hash:
                raise ValueError("result directory belongs to a different evaluation; use a fresh directory")
    args.run_id = previous_inventory["run_id"] if previous_inventory else uuid.uuid4().hex
    inventory = {"run_id": args.run_id, "config_hash": config_hash,
                 "tasks": [_result_identity(args, task) for task in expected_tasks]}
    if previous_inventory is not None and previous_inventory != inventory:
        raise ValueError("expected task inventory changed; use a fresh result directory")
    _write_json(inventory_path, inventory)

    tasks = get_unfinished(
        args.action_space, args.policy_model_name, args.observation_type,
        args.result_dir, test_all_meta, args.task_n_combinations,
        args.run_id,
    )
    logger.info("Total unfinished tasks: %d", len(tasks))
    logger.info("By domain: %s", dict(Counter(d for d, _, _ in tasks)))

    scores, labels, failed_tasks = [], [], []
    try:
        if tasks and args.ray_address:
            scores, labels, failed_tasks = _run_ray_distributed(tasks, args)
        elif tasks:
            scores, labels, failed_tasks = _run_multiprocessing(tasks, args)
    finally:
        _, accounting = collect_results(Path(args.result_dir))
        _write_json(os.path.join(args.result_dir, "eval_run_summary.json"), accounting)
        leaked = cleanup_docker_containers(args.docker_tag)
        if leaked:
            logger.warning("Cleaned up %d leaked container(s) after run", leaked)

    if scores:
        logger.info("Average env score: %.4f", sum(scores) / len(scores))
    if labels:
        logger.info("Eval label counts: %s", dict(Counter(labels)))
    logger.info("Evaluation accounting: %s", accounting)
    if failed_tasks or accounting["failed"] or accounting["missing"] or accounting["issues"]:
        logger.error("Evaluation completed with %d failed task(s).", len(failed_tasks))
        for failure in failed_tasks[:20]:
            logger.error("Failed task %s: %s", failure.get("task"), failure.get("error"))
        sys.exit(1)

    logger.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        signal_handler(signal.SIGTERM, None)
