from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
import traceback as _traceback_mod
import uuid
from collections import Counter
from multiprocessing import Manager, Process, current_process
from pathlib import Path
from typing import Any

import eval_runtime.vagen_active
from eval_runtime.config import load_eval_dataset_config
from eval_runtime.env_adapter import (
    AgentRuntimeEnvClient,
    cleanup_docker_containers,
    _ensure_docker_tag,
)
from eval_runtime.guirl.evaluator import parse_policy_result
from eval_runtime.run_replay_eval import (
    _apply_osworld_env,
    _broadcast_group_cleanup,
    _create_policy_agent,
    _default_project_dir,
    _normalize_agent_type,
    _resolve_path,
    _resolve_policy_config,
    _resolve_runtime_paths,
    get_unfinished,
    setup_logging,
)
from eval_runtime.task_runner import run_single_example_client
from eval_runtime.vagen_active.agent_factory import load_vagen_eval_config
from eval_runtime.vagen_active.evaluator import run_vagen_eval_after_replay

import eval_runtime.agents.replay

logger = logging.getLogger("eval_runtime.vagen_active.main")
processes: list[Process] = []
is_terminating = False
_ray_actors: list = []

_EXTRA_HEAD_ENV_VARS = (
    "VAGEN_EVAL_API_URL", "VAGEN_EVAL_API_KEY",
    "VAGEN_QWEN_API_URL", "VAGEN_QWEN_API_KEY", "VAGEN_QWEN_MODEL",
    "VAGEN_QWEN_MAX_ROUND", "VAGEN_QWEN_MAX_TOKENS", "VAGEN_PATH",
    "PRORE_QWEN_API_URL", "PRORE_QWEN_API_KEY", "PRORE_QWEN_MODEL",
    "PRORE_QWEN_MAX_ROUND", "PRORE_QWEN_MAX_TOKENS",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay + VAGEN active judge (Claude or Qwen)")
    parser.add_argument("--osworld_path", type=str, required=True)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--path_to_vm", type=str, default="")
    parser.add_argument("--docker_tag", type=str, default="")
    parser.add_argument("--policy_config_path", type=str, default="",
                        help="Path to policy config YAML. Optional when --eval_dataset_config is provided.")
    parser.add_argument("--vagen_eval_config_path", type=str, required=True,
                        help="VAGEN active judge YAML (evaluator_type: vagen_claude or vagen_qwen)")
    parser.add_argument("--test_all_meta_path", type=str, default="")
    parser.add_argument("--test_config_base_dir", type=str, default="")
    parser.add_argument("--examples_subdir", type=str, default="examples")
    parser.add_argument("--result_dir", type=str, default="./results/vagen_active_eval")
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--task_n_combinations", type=int, default=None,
                        help="Rollouts per task. Overrides the dataset yaml when set "
                             "(unlike run_replay_eval, where the yaml always wins); "
                             "default: dataset yaml value, else 1.")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--sleep_after_execution", type=float, default=2.0)
    parser.add_argument("--sleep_after_reset", type=float, default=30.0)
    parser.add_argument("--sleep_before_eval", type=float, default=20.0)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", type=str, default="screenshot")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--eval_dataset_config", type=str, default="")
    parser.add_argument("--project_dir", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--ray_address", type=str, default="")
    parser.add_argument("--ray_num_actors", type=int, default=0)
    parser.add_argument(
        "--retry_parse_errors_from", type=str, default="",
        help="Run exactly the (domain, example_id, rollout_id) entries whose "
             "source eval_summary has vagen_parse_method=failed.",
    )
    parser.add_argument(
        "--retry_result_errors_from", type=str, default="",
        help="Run exactly the entries whose source eval_summary has final_label=error.",
    )
    return parser.parse_args()

def _execute_single_task_vagen(
    env_client: AgentRuntimeEnvClient,
    policy_agent,
    domain: str,
    example_id: str,
    rollout_id: str,
    args: argparse.Namespace,
    scores: list,
    worker_name: str = "worker",
) -> dict[str, Any]:

    lease = None
    example_result_dir = None
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

        example_result_dir = os.path.join(
            args.result_dir, args.action_space, args.observation_type,
            args.policy_model_name, domain, example_id, str(rollout_id),
        )
        if os.path.exists(example_result_dir):
            shutil.rmtree(example_result_dir)
        os.makedirs(example_result_dir, exist_ok=True)

        lease = env_client.allocate(
            f"{domain}:{example_id}:{rollout_id}:{uuid.uuid4().hex[:8]}"
        )

        if hasattr(policy_agent, "set_task_context"):
            policy_agent.set_task_context(domain, example_id, rollout_id)

        policy_result_raw = run_single_example_client(
            policy_agent, env_client, lease, example,
            args.max_steps, example["instruction"], args,
            example_result_dir, scores, None,
        )
        policy_result = parse_policy_result(policy_result_raw)

        summary = run_vagen_eval_after_replay(
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
        logger.info("[%s] DONE %s/%s/%s: score=%s label=%s",
                    worker_name, domain, example_id, rollout_id,
                    policy_result, summary["final_label"])
        return {"score": policy_result, "label": summary["final_label"], "error": None}

    except Exception as exc:
        tb = _traceback_mod.format_exc()
        logger.error("Exception in %s %s/%s: %s", worker_name, domain, example_id, exc)
        logger.error(tb)
        if example_result_dir:
            os.makedirs(example_result_dir, exist_ok=True)
            with open(os.path.join(example_result_dir, "traceback.txt"), "w") as f:
                f.write(tb)
        return {"score": None, "label": None, "error": str(exc)}
    finally:
        if lease is not None:
            try:
                env_client.close(lease["lease_id"])
            except Exception as close_exc:
                logger.error("Failed to close lease: %s", close_exc)

def run_env_tasks(task_queue, args, shared_scores, eval_labels, remaining_tasks, remaining_lock) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    _apply_osworld_env(
        getattr(args, "osworld_path", ""),
        getattr(args, "project_dir", ""),
    )

    env_client = None
    try:
        env_kwargs: dict[str, Any] = {}
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
        logger.info("Process %s initialized, policy=%s", current_process().name, args.policy_model_name)

        while True:
            try:
                domain, example_id, rollout_id = task_queue.get(timeout=5)
            except Exception:
                break

            result = _execute_single_task_vagen(
                env_client, policy_agent, domain, example_id, rollout_id,
                args, shared_scores, worker_name=current_process().name,
            )
            if result["label"] is not None:
                eval_labels.append(result["label"])

            with remaining_lock:
                remaining_tasks.value -= 1
                logger.info("Remaining tasks: %s", remaining_tasks.value)

    except Exception as exc:
        logger.error("Process-level error in %s: %s", current_process().name, exc)
        logger.error(_traceback_mod.format_exc())
    finally:
        if env_client is not None:
            try:
                env_client.close_all()
            except Exception as exc:
                logger.warning("Worker close_all failed: %s", exc)

def _run_multiprocessing(tasks: list[tuple[str, str, str]], args: argparse.Namespace) -> tuple[list, list]:
    global processes
    with Manager() as manager:
        remaining_tasks = manager.Value("i", len(tasks))
        remaining_lock = manager.Lock()
        shared_scores = manager.list()
        eval_labels = manager.list()
        task_queue = manager.Queue()
        for item in tasks:
            task_queue.put(item)

        processes = []
        for idx in range(args.num_envs):
            proc = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, eval_labels, remaining_tasks, remaining_lock),
                name=f"VagenEvalWorker-{idx + 1}",
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)
            logger.info("Started %s pid=%s", proc.name, proc.pid)

        while True:
            for idx, proc in enumerate(processes):
                if not proc.is_alive() and remaining_tasks.value > 0:
                    logger.warning("Process %s died, restarting.", proc.name)
                    new_proc = Process(
                        target=run_env_tasks,
                        args=(task_queue, args, shared_scores, eval_labels, remaining_tasks, remaining_lock),
                        name=f"VagenEvalWorker-Restart-{idx + 1}",
                    )
                    new_proc.daemon = True
                    new_proc.start()
                    processes[idx] = new_proc
            if remaining_tasks.value == 0:
                break
            time.sleep(5)

        for proc in processes:
            proc.join(timeout=30)

        return list(shared_scores), list(eval_labels)

def _run_ray_distributed(tasks: list[tuple[str, str, str]], args: argparse.Namespace) -> tuple[list, list]:
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
            "Capping num_actors from %d to %d (total osworld_vm slots in cluster).",
            num_actors, total_osworld_vm,
        )
        num_actors = total_osworld_vm

    logger.info("Ray cluster: %d alive node(s), creating %d actors (osworld_vm slots: %d)",
                len(alive_nodes), num_actors, total_osworld_vm)

    _head_env_keys = (
        "HF_ENDPOINT", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
        "https_proxy", "http_proxy", "no_proxy", "OSWORLD_RUN_GROUP",
        "PROXY_CONFIG_FILE",
    ) + _EXTRA_HEAD_ENV_VARS
    _HEAD_ENV_VARS = {
        k: os.environ[k]
        for k in _head_env_keys
        if k in os.environ and os.environ[k]
    }
    logger.info("Forwarding env vars to Ray actors: %s", list(_HEAD_ENV_VARS.keys()))

    run_group = os.environ.get("OSWORLD_RUN_GROUP", "").strip()
    if run_group:
        logger.info("Pre-run: broadcasting container cleanup for run_group=%s...", run_group)
        _broadcast_group_cleanup(ray, run_group, timeout=60)

    @ray.remote(num_cpus=1, resources={"osworld_vm": 1})
    class VagenEvalWorkerActor:
        def __init__(self, args_dict: dict[str, Any], actor_idx: int):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import logging as _logging
            for _ns in ("desktopenv", "desktop_env", "urllib3", "docker", "PIL", "asyncio", "playwright"):
                _logging.getLogger(_ns).setLevel(_logging.ERROR)
            for k, v in args_dict.get("_ray_env_vars", {}).items():
                if v:
                    os.environ[k] = v
            self.actor_name = f"RayVagenEvalWorker-{actor_idx}"

            _args = argparse.Namespace(**{k: v for k, v in args_dict.items()
                                          if not k.startswith("_")})
            _apply_osworld_env(
                getattr(_args, "osworld_path", ""),
                getattr(_args, "project_dir", ""),
            )

            import eval_runtime
            import eval_runtime.vagen_active
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
            return _execute_single_task_vagen(
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
    actors = [VagenEvalWorkerActor.remote(args_dict, i) for i in range(num_actors)]
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
                all_results.append({"score": None, "label": None, "error": str(exc)})
            pending.pop(r)

    scores: list[float] = []
    for actor in actors:
        try:
            scores.extend(ray.get(actor.get_scores.remote()))
        except Exception:
            pass

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
    return scores, labels

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
        logger.info("Shutting down %d Ray actor(s)...", len(_ray_actors))
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

        if args.task_n_combinations is None:
            args.task_n_combinations = ds_cfg.get("task_n_combinations", 1)
        args.max_steps = ds_cfg.get("max_steps", args.max_steps)

        if not args.policy_config_path and ds_cfg.get("policy"):
            args.policy_config_inline = ds_cfg["policy"]

    if not args.test_all_meta_path or not args.test_config_base_dir:
        logger.error(
            "--test_all_meta_path and --test_config_base_dir are required "
            "unless --eval_dataset_config is provided."
        )
        sys.exit(1)
    if args.task_n_combinations is None:
        args.task_n_combinations = 1

    args.vagen_eval_config_path = _resolve_path(args.project_dir, args.vagen_eval_config_path)

    args.vagen_eval_config = load_vagen_eval_config(args.vagen_eval_config_path)

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

    logger.info("Project dir: %s", args.project_dir)
    logger.info("Policy: type=%s model=%s", args.policy_agent_type, args.policy_model_name)
    logger.info("VAGEN eval config: %s (model=%s, max_round=%s)",
                args.vagen_eval_config_path,
                args.vagen_eval_config.get("model"),
                args.vagen_eval_config.get("max_round"))
    logger.info("Result dir: %s", args.result_dir)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    retry_source = args.retry_parse_errors_from or args.retry_result_errors_from
    if retry_source:
        source_root = Path(_resolve_path(args.project_dir, retry_source))
        if not source_root.is_dir():
            logger.error("Parse-error source result dir not found: %s", source_root)
            sys.exit(1)
        tasks = []
        for summary_path in sorted(source_root.rglob("replay_eval/eval_summary.json")):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if args.retry_parse_errors_from:
                selected = str(summary.get("vagen_parse_method") or "").strip().lower() == "failed"
            else:
                selected = str(summary.get("final_label") or "").strip().lower() == "error"
            if not selected:
                continue
            domain = str(summary.get("domain") or "")
            example_id = str(summary.get("example_id") or "")
            rollout_id = str(summary.get("rollout_id") or "")
            if domain and example_id and rollout_id:
                tasks.append((domain, example_id, rollout_id))

        tasks = sorted(set(tasks))
        logger.info("Selected retry source: %s", source_root)
    else:
        tasks = get_unfinished(
            args.action_space, args.policy_model_name, args.observation_type,
            args.result_dir, test_all_meta, args.task_n_combinations,
        )
    logger.info("Total unfinished tasks: %d", len(tasks))
    logger.info("By domain: %s", dict(Counter(d for d, _, _ in tasks)))

    if not tasks:
        logger.info("All tasks complete.")
        return

    if args.ray_address:
        logger.info("Using Ray distributed mode (address=%s, actors=%d)",
                    args.ray_address, args.ray_num_actors or args.num_envs)
        scores, labels = _run_ray_distributed(tasks, args)
    else:
        scores, labels = _run_multiprocessing(tasks, args)

    if scores:
        logger.info("Average env score: %.4f", sum(scores) / len(scores))
    if labels:
        logger.info("Eval label counts: %s", dict(Counter(labels)))

    leaked = cleanup_docker_containers(args.docker_tag)
    if leaked:
        logger.warning("Cleaned up %d leaked container(s) after run", leaked)
    logger.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        signal_handler(signal.SIGTERM, None)
