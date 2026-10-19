from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rollout_runtime

_repo = str(REPO_ROOT)
if _repo in sys.path:
    sys.path.remove(_repo)
sys.path.insert(0, _repo)

import agent_runtime as ap
from agent_runtime.types import Action, ActionType

from rollout_runtime.run_osworld_tasks import (
    create_harness,
    load_benchmark_tasks,
    load_query_tasks,
)
from rollout_runtime.run_osworld_batch import RolloutTraceWriter

from eval_runtime.guirl.evaluator import build_guirl_config
from train.guirl_sft.guirl_runner import run_guirl_for_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train.guirl_sft.collect")

_EXPORT_LOCK = threading.Lock()

GUIRL_DONE_MARKER = "guirl_done.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E GUIRL SFT data collection: actor → judge-probe → export"
    )

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--tasks-dir", default=None)
    task_group.add_argument("--split", default=None)

    parser.add_argument("--osworld-path", default=os.environ.get("OSWORLD_PATH", ""))
    parser.add_argument("--vm-path", default=os.environ.get("OSWORLD_VM_PATH", ""))
    parser.add_argument("--provider", default="docker", choices=["docker", "vmware", "virtualbox"])
    parser.add_argument("--num-vms", type=int, default=2)
    parser.add_argument("--headless", action="store_true", default=True)

    parser.add_argument("--harness", default="holo31")
    parser.add_argument("--model", default="holo3-1-35b-a3b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-trajectory-length", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--client-password", default="password")
    parser.add_argument("--platform", default="ubuntu")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--holo3-max-images", type=int, default=3)
    parser.add_argument("--holo3-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coordinate-type", default="absolute")
    parser.add_argument("--history-n", type=int, default=4)
    parser.add_argument("--image-max", type=int, default=5)
    parser.add_argument("--fold-size", type=int, default=2)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logprobs", action="store_true", default=False)
    parser.add_argument("--action-space", default="pyautogui")
    parser.add_argument("--settle-time", type=float, default=20.0)
    parser.add_argument("--sleep-after-exec", type=float, default=2.0)

    parser.add_argument("--host-proxy", default=None)
    parser.add_argument("--vm-proxy", default=None)
    parser.add_argument("--no-proxy", default=None)

    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--domain", default="all")
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--file-type", default=None)
    parser.add_argument("--vm-upload-dir", default="/home/user/Desktop")

    parser.add_argument("--guirl-config", required=True)
    parser.add_argument("--max-eval-rounds", type=int, default=3)

    parser.add_argument("--output-dir", default="./results/guirl_sft_data")

    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--noop", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

class EnvClientAdapter:

    def __init__(self, env: Any, evaluator: Any = None, sleep_after_execution: float = 2.0):
        self._env = env
        self._evaluator = evaluator
        self._sleep = sleep_after_execution

    def prepare_for_probe(self, lease_id: str | None = None) -> None:

        del lease_id
        if getattr(self._env, "_terminated", False):
            logger.info(
                "Clearing OSWorldEnv._terminated for GUIRL probe (VM state preserved)"
            )
            self._env._terminated = False

    def get_obs(self, lease_id: str) -> dict[str, Any]:
        desktop_env = self._env.desktop_env
        return {"screenshot": desktop_env.controller.get_screenshot()}

    def step(self, lease_id: str, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        from eval_runtime.markers import is_logical_marker
        if is_logical_marker(action):
            obs = self.get_obs(lease_id)
            return obs, 0.0, True, {"marker": action}

        ap_action = Action(type=ActionType.PYTHON, arguments={"code": action})
        _obs, reward, terminated, truncated, info = self._env.step(ap_action)
        time.sleep(self._sleep)
        obs = self.get_obs(lease_id)
        done = bool(terminated or truncated)
        return obs, float(reward or 0.0), done, (info or {})

    def execute(self, lease_id: str, action: str) -> dict[str, Any]:
        obs, _, _, _ = self.step(lease_id, action)
        return obs

    def evaluate(self, lease_id: str) -> float | None:
        if self._evaluator is None:
            return None
        try:
            result = self._evaluator.finalize(env=self._env, trace=self._env.trace)
            if result is None or result.score is None:
                return None
            return float(result.score)
        except Exception as exc:
            logger.warning("evaluate() failed: %s", exc)
            return None

    def heartbeat(self, lease_id: str) -> None:
        pass

def export_sft_sample(
    output_path: str,
    messages: list[dict[str, Any]],
    *,
    task_id: str,
    domain: str,
    reward: float,
    S: float,
    S_prime: float | None,
    judge_final: str,
    num_rounds: int,
    policy_model: str,
    judge_model: str,
) -> None:

    record = {
        "messages": messages,
        "metadata": {
            "task_id": task_id,
            "domain": domain,
            "reward": reward,
            "S": S,
            "S_prime": S_prime,
            "judge_final": judge_final,
            "num_rounds": num_rounds,
            "policy_model": policy_model,
            "judge_model": judge_model,
        },
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with _EXPORT_LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(line)

def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.osworld_path:
        raise SystemExit("ERROR: --osworld-path required (or set OSWORLD_PATH)")
    if not args.vm_path:
        raise SystemExit("ERROR: --vm-path required (or set OSWORLD_VM_PATH)")

    guirl_config = build_guirl_config(args.guirl_config)

    logger.info(
        "GUIRL config: judge=%s probe=%s",
        guirl_config["judge_model_name"],
        guirl_config.get("probe_model_name"),
    )

    if args.split:
        tasks = load_benchmark_tasks(
            args.osworld_path, args.split,
            domain=args.domain if args.domain != "all" else None,
            limit=args.limit,
        )
    else:
        tasks = load_query_tasks(
            args.tasks_dir,
            vm_upload_dir=args.vm_upload_dir,
            limit=args.limit,
            difficulty=args.difficulty,
            file_type=args.file_type,
        )

    print(f"\nLoaded {len(tasks)} task(s)")
    for task in tasks[:10]:
        d = task.metadata.get("domain") or task.suite or "?"
        print(f"  [{d}] {task.task_id}: {str(task.instruction)[:80]}")
    if len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more")

    if args.list_only or args.dry_run:
        print("\n[dry-run] Config validated, tasks listed. Exiting.")
        return
    if not tasks:
        raise SystemExit("No tasks found.")

    output_dir = Path(args.output_dir)
    traj_root = output_dir / "trajectories"
    num_rollouts = max(1, args.num_rollouts)

    run_tasks, rollout_map, skipped = _plan_with_guirl_checkpoint(
        tasks, traj_root=traj_root, num_rollouts=num_rollouts,
    )

    total = len(tasks) * num_rollouts
    print(f"\nResume: {total} total, {skipped} done, {len(run_tasks)} to run")
    if not run_tasks:
        print("All tasks already completed (actor + GUIRL).")
        return

    proxy_config = None
    if args.host_proxy or args.vm_proxy:
        proxy_config = ap.ProxyConfig(
            host_proxy=args.host_proxy,
            vm_proxy=args.vm_proxy,
            no_proxy=args.no_proxy,
        )

    benchmark = ap.load_benchmark(
        "osworld",
        osworld_path=args.osworld_path,
        split=args.split or "osworld_all",
        action_space=args.action_space,
    )
    backend = ap.backends.osworld(
        osworld_path=args.osworld_path,
        provider_name=args.provider,
        path_to_vm=args.vm_path,
        action_space=args.action_space,
        headless=args.headless,
        reuse=True,
        num_instances=args.num_vms,
        proxy_config=proxy_config,
        require_a11y_tree=False,
    )
    backend.setup()

    sft_output_path = str(output_dir / "guirl_sft_data.jsonl")
    writer = RolloutTraceWriter(traj_root=traj_root, rollout_map=rollout_map)

    print(f"\nRunning {len(run_tasks)} task(s) with {args.num_vms} VM(s)")
    print(f"  Actor:  {args.harness} / {args.model}")
    print(f"  Judge:  {guirl_config['judge_model_name']}")
    print(f"  Probe:  {guirl_config.get('probe_model_name', 'N/A')}")
    print(f"  Rounds: max {args.max_eval_rounds}")
    print(f"  Output: {sft_output_path}")

    stats = {"total": 0, "actor_ok": 0, "actor_fail": 0,
             "guirl_run": 0, "reward_1": 0, "exported": 0}
    lock = threading.Lock()

    def _execute_one(task: Any) -> dict[str, Any]:

        task_id = task.task_id
        mapped = rollout_map.get(task_id)
        if mapped:
            orig_task_id, domain, rollout_idx = mapped
        else:
            orig_task_id = task_id
            domain = task.metadata.get("domain") or task.suite or "unknown"
            rollout_idx = 0
        task_traj_dir = str(traj_root / domain / orig_task_id / f"r{rollout_idx}")

        session = None
        env = None
        harness = None
        _actor_stats_counted = False
        try:

            harness = create_harness(args)
            session = backend.create_session(task)
            env = benchmark.make_env(
                task=task, session=session,
                action_space=args.action_space,
                settle_time=args.settle_time,
                sleep_after_execution=args.sleep_after_exec,
            )

            if writer:
                try:
                    writer.begin_task(task_id, task=task)
                    def _on_event(ev, _w=writer, _tid=task_id):
                        _w.record_event(_tid, ev)
                    env.trace.add_event_listener(_on_event)
                except Exception:
                    logger.debug("Writer begin_task failed for %s", task_id)

            env.reset()
            evaluator = benchmark.make_evaluator(task=task)
            result = env.run(
                harness=harness,
                evaluator=evaluator,
                max_steps=args.max_steps,
                timeout=args.request_timeout * args.max_steps,
            )

            if writer:
                try:
                    writer.end_task(
                        task_id,
                        result.trace if result.trace is not None else env.trace,
                        result.evaluation,
                    )
                except Exception:
                    pass

            policy_score = 0.0
            if result.evaluation and result.evaluation.score is not None:
                policy_score = result.evaluation.score

            with lock:
                stats["total"] += 1
                _actor_stats_counted = True
                if result.evaluation and result.evaluation.success:
                    stats["actor_ok"] += 1
                else:
                    stats["actor_fail"] += 1

            env_adapter = EnvClientAdapter(
                env=env,
                evaluator=evaluator,
                sleep_after_execution=args.sleep_after_exec,
            )
            lease_id = f"guirl_{task_id}"

            guirl_result = run_guirl_for_task(
                env_client=env_adapter,
                lease_id=lease_id,
                instruction=str(task.instruction),
                traj_dir=task_traj_dir,
                policy_score=policy_score,
                guirl_config=guirl_config,
                max_eval_rounds=args.max_eval_rounds,
                sleep_after_execution=args.sleep_after_exec,
            )

            with lock:
                stats["guirl_run"] += 1
                reward = guirl_result.get("reward", 0.0)
                if reward >= 1.0:
                    stats["reward_1"] += 1
                    msgs = guirl_result.get("messages", [])
                    if msgs:
                        export_sft_sample(
                            sft_output_path, msgs,
                            task_id=orig_task_id,
                            domain=domain,
                            reward=reward,
                            S=policy_score,
                            S_prime=guirl_result.get("S_prime"),
                            judge_final=guirl_result.get("judge_final", ""),
                            num_rounds=guirl_result.get("num_rounds", 0),
                            policy_model=args.model,
                            judge_model=guirl_config["judge_model_name"],
                        )
                        stats["exported"] += 1
                        logger.info("EXPORTED %s reward=%.1f judge=%s",
                                    orig_task_id, reward, guirl_result.get("judge_final"))

            if guirl_result.get("judge_final") != "error":
                _write_guirl_marker(task_traj_dir, guirl_result, policy_score)

            return {
                "task_id": orig_task_id,
                "policy_score": policy_score,
                "reward": reward,
                "judge_final": guirl_result.get("judge_final"),
                "num_rounds": guirl_result.get("num_rounds", 0),
                "S_prime": guirl_result.get("S_prime"),
            }

        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc, exc_info=args.debug)
            with lock:
                if not _actor_stats_counted:
                    stats["total"] += 1
                stats["actor_fail"] += 1
            return {"task_id": orig_task_id, "policy_score": 0.0, "reward": 0.0,
                    "judge_final": "error", "error": str(exc)}
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
            if session is not None:
                with contextlib.suppress(Exception):
                    session.close()
            if harness is not None and hasattr(harness, "reset") and callable(getattr(harness, "reset", None)):
                with contextlib.suppress(Exception):
                    harness.reset()

    t0 = time.time()
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.num_vms) as pool:
            futures = {pool.submit(_execute_one, t): t for t in run_tasks}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    task = futures[fut]
                    logger.error("Task %s raised: %s", task.task_id, exc)
    finally:
        with contextlib.suppress(Exception):
            backend.teardown()
        if writer:
            writer.close()

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"GUIRL SFT Collection Complete — {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Tasks run:       {stats['total']}")
    print(f"  Actor success:   {stats['actor_ok']}")
    print(f"  Actor fail:      {stats['actor_fail']}")
    print(f"  GUIRL evaluated: {stats['guirl_run']}")
    print(f"  Reward=1.0:      {stats['reward_1']}")
    print(f"  Exported SFT:    {stats['exported']}")
    print(f"  Output:          {sft_output_path}")

    summary_path = output_dir / "collection_summary.json"
    os.makedirs(str(output_dir), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "elapsed_seconds": elapsed,
            "config": {
                "harness": args.harness, "model": args.model,
                "judge_model": guirl_config["judge_model_name"],
                "probe_model": guirl_config.get("probe_model_name"),
                "max_eval_rounds": args.max_eval_rounds,
                "num_vms": args.num_vms, "num_rollouts": num_rollouts,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Summary:         {summary_path}")

def _plan_with_guirl_checkpoint(
    tasks: list[Any],
    *,
    traj_root: Path,
    num_rollouts: int,
) -> tuple[list[Any], dict[str, tuple[str, str, int]], int]:

    import shutil
    traj_root = Path(traj_root)
    run_tasks: list[Any] = []
    rollout_map: dict[str, tuple[str, str, int]] = {}
    skipped = 0

    for task in tasks:
        domain = task.metadata.get("domain") or task.suite or "unknown"
        for r in range(num_rollouts):
            out_dir = traj_root / domain / task.task_id / f"r{r}"
            guirl_marker = out_dir / GUIRL_DONE_MARKER
            if guirl_marker.is_file():
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

def _write_guirl_marker(traj_dir: str, guirl_result: dict[str, Any], policy_score: float) -> None:

    marker_path = os.path.join(traj_dir, GUIRL_DONE_MARKER)
    os.makedirs(traj_dir, exist_ok=True)
    payload = {
        "reward": guirl_result.get("reward", 0.0),
        "judge_final": guirl_result.get("judge_final"),
        "num_rounds": guirl_result.get("num_rounds", 0),
        "S": policy_score,
        "S_prime": guirl_result.get("S_prime"),
    }
    probe_errors = guirl_result.get("probe_errors")
    if probe_errors:
        payload["probe_errors"] = probe_errors
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
