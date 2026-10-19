from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import rollout_runtime

import agent_runtime as ap
from agent_runtime.harnesses.holo3 import Holo3Harness
from agent_runtime.harnesses.holo31 import Holo31Harness
from agent_runtime.runner import RunSpec
from agent_runtime.runtime.local import run_task_attempt
from agent_runtime.types import RunTaskResult

from rollout_runtime.run_osworld_batch import (
    RolloutTraceWriter,
    _rollout_dir,
    _task_domain,
    plan_rollouts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rollout_runtime.osworld")

def _configure_logging(*, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)
    if debug:
        os.environ["RUN_AGENT_RUNTIME_DEBUG"] = "1"
        for name in ("rollout_runtime", "agent_runtime", "agent_runtime.harnesses.holo3", "agent_runtime.harnesses.holo31", "desktopenv"):
            logging.getLogger(name).setLevel(logging.DEBUG)

class NoopHarness(ap.Harness):
    def act(self, observation, context=None):
        return ap.types.Action.terminate(reason="DONE")

def _first_string(obj: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

def _instruction(payload: dict[str, Any]) -> str:
    value = _first_string(
        payload,
        ("instruction", "query", "question", "task", "prompt", "goal", "description"),
    )
    if value:
        return value
    raise ValueError("query.json must contain one of: instruction/query/question/task/prompt/goal")

def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]

def _env_setup(case_dir: Path, payload: dict[str, Any], vm_upload_dir: str) -> dict[str, Any] | None:
    files: list[dict[str, str]] = []
    for entry in payload.get("files") or payload.get("uploads") or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("local_path") or entry.get("path") or entry.get("source")
        if not source:
            continue
        local_path = Path(source)
        if not local_path.is_absolute():
            local_path = case_dir / local_path
        if not local_path.is_file():
            logger.warning("Skipping missing file asset: %s", local_path)
            continue
        target = entry.get("vm_path") or entry.get("target")
        if not target:
            target = f"{vm_upload_dir.rstrip('/')}/{local_path.name}"
        files.append({"local_path": str(local_path), "vm_path": str(target)})

    for entry in payload.get("copied_assets") or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("copied_path")
        if not source:
            continue
        local_path = Path(source)
        if not local_path.is_absolute():
            local_path = case_dir / local_path
        if not local_path.is_file():
            continue
        file_name = str(entry.get("file_name") or "") or local_path.name
        vm_target = f"{vm_upload_dir.rstrip('/')}/{file_name}"
        files.append({"local_path": str(local_path), "vm_path": vm_target})

    setup: dict[str, Any] = {}
    for hook_key in ("pre_task", "post_task"):
        hook = payload.get(hook_key)
        if isinstance(hook, str) and hook.strip():
            hook_path = Path(hook)
            if not hook_path.is_absolute():
                hook_path = case_dir / hook_path
            setup[hook_key] = str(hook_path)
    if files:
        setup["files"] = files
    return setup or None

def load_query_tasks(
    tasks_dir: str,
    *,
    vm_upload_dir: str = "/home/user/Desktop",
    limit: int | None = None,
    difficulty: str | None = None,
    file_type: str | None = None,
) -> list[ap.types.Task]:
    root = Path(tasks_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Tasks directory does not exist: {root}")

    query_files: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        if "query.json" in filenames:
            query_files.append(Path(dirpath) / "query.json")
    query_files.sort(key=lambda p: str(p))

    difficulty_filter = difficulty.lower() if difficulty else None
    file_type_filter = file_type.lower() if file_type else None

    tasks: list[ap.types.Task] = []
    for query_file in query_files:
        payload = json.loads(query_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{query_file} must contain a JSON object")
        case_dir = query_file.parent
        task_id = _first_string(payload, ("id", "task_id", "case_id", "uuid")) or case_dir.name

        diff = str(payload.get("difficulty") or "").strip()
        ftype = str(payload.get("file_type") or "").strip()
        if difficulty_filter and diff.lower() != difficulty_filter:
            continue
        if file_type_filter and ftype.lower() != file_type_filter:
            continue

        task = ap.types.Task(
            task_id=task_id,
            benchmark="osworld",
            suite=str(payload.get("suite") or diff or ftype or "query"),
            instruction=_instruction(payload),
            native_ref={
                "query_json": str(query_file),
                "case_dir": str(case_dir),
                "payload": payload,
            },
            metadata={
                "query_json": str(query_file),
                "case_dir": str(case_dir),
                "domain": payload.get("domain") or "query",
                "difficulty": diff or None,
                "file_type": ftype or None,
                "query": _instruction(payload),
                "deliverables": _string_list(payload.get("deliverables")),
            },
            tags=[item for item in ("query", diff, ftype) if item],
            env_setup=_env_setup(case_dir, payload, vm_upload_dir),
            max_steps=payload.get("max_steps") if isinstance(payload.get("max_steps"), int) else None,
        )
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks

def load_benchmark_tasks(
    osworld_path: str,
    split: str,
    *,
    domain: str | None = None,
    limit: int | None = None,
) -> list[ap.types.Task]:

    benchmark = ap.load_benchmark("osworld", osworld_path=osworld_path, split=split)
    filter_kwargs: dict[str, Any] = {}
    if domain and domain != "all":
        filter_kwargs["domain"] = domain
    return benchmark.list_tasks(split=split, limit=limit, **filter_kwargs)

def create_harness(args: argparse.Namespace) -> ap.Harness:
    if args.noop:
        return NoopHarness()

    if args.harness == "holo3":
        return Holo3Harness(
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "") or "EMPTY",
            base_url=args.api_base or os.environ.get("OPENAI_BASE_URL", ""),
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
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "") or "EMPTY",
            base_url=args.api_base or os.environ.get("OPENAI_BASE_URL", ""),
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

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "") or "EMPTY"
    base_url = args.api_base or os.environ.get("OPENAI_BASE_URL", "")

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
    return harness_cls(
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        base_url=args.api_base or os.environ.get("OPENAI_BASE_URL", ""),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OSWorld tasks via AgentRuntime")

    parser.add_argument("--tasks-dir", default=None,
                        help="Directory containing query task folders (each with query.json)")
    parser.add_argument("--split", default=None,
                        help="OSWorld evaluation split name, e.g. sft_tasks "
                             "(loads evaluation_examples/{split}.json under --osworld-path)")
    parser.add_argument("--domain", default="all",
                        help="Filter tasks by domain when using --split (default: all)")
    parser.add_argument("--osworld-path", default=os.environ.get("OSWORLD_PATH", ""),
                        help="Path to OSWorld root (must contain desktop_env/)")
    parser.add_argument("--vm-path", default=os.environ.get("OSWORLD_VM_PATH", ""),
                        help="Path to Ubuntu.qcow2 VM image")
    parser.add_argument("--provider", default="docker", choices=["docker", "vmware", "virtualbox"])
    parser.add_argument("--num-vms", type=int, default=2,
                        help="Number of parallel VMs")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--vm-upload-dir", default="/home/user/Desktop")
    parser.add_argument("--output-dir", default="./results/osworld_holo3")
    parser.add_argument("--action-space", default="pyautogui")
    parser.add_argument("--settle-time", type=float, default=20.0)
    parser.add_argument("--sleep-after-exec", type=float, default=2.0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of tasks to run")
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--file-type", default=None)
    parser.add_argument("--list-only", action="store_true",
                        help="Only list tasks, don't run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without running")
    parser.add_argument("--noop", action="store_true",
                        help="Use noop harness (test environment only)")

    parser.add_argument("--harness", default="holo3",
                        help="Harness type: holo3, holo31, gpt54-cua, claude-cua, osworld-vlm")
    parser.add_argument("--model", default="holo3-1-35b-a3b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-trajectory-length", type=int, default=12)
    parser.add_argument("--client-password", default="password")
    parser.add_argument("--platform", default="ubuntu")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=600)
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

    parser.add_argument("--host-proxy", default=None)
    parser.add_argument("--vm-proxy", default=None)
    parser.add_argument("--no-proxy", default=None)
    parser.add_argument("--traj-root", default=None,
                        help="Root dir for trajectories (default: output_dir/trajectories). "
                             "Layout: {traj_root}/{domain}/{task_id}/r{N}/")
    parser.add_argument("--num-rollouts", type=int, default=1,
                        help="Run each task N times. Resume-aware: skips rollouts whose "
                             "result.txt already exists; clears partial dirs and re-runs.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose logging and full error tracebacks")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    _configure_logging(debug=args.debug or os.environ.get("RUN_AGENT_RUNTIME_DEBUG") in {"1", "true", "TRUE"})

    if args.debug:
        logger.debug(
            "Debug config: model=%s api_base=%s api_key_set=%s harness=%s",
            args.model,
            args.api_base or os.environ.get("OPENAI_BASE_URL", ""),
            bool(args.api_key or os.environ.get("OPENAI_API_KEY")),
            args.harness,
        )

    if not args.osworld_path:
        raise SystemExit("ERROR: --osworld-path is required (or set OSWORLD_PATH env var)")
    if not args.vm_path:
        raise SystemExit("ERROR: --vm-path is required (or set OSWORLD_VM_PATH env var)")
    if bool(args.tasks_dir) == bool(args.split):
        raise SystemExit("ERROR: specify exactly one of --tasks-dir or --split")

    osworld_desktop_env = os.path.join(args.osworld_path, "desktop_env", "desktop_env.py")
    if not args.dry_run and not os.path.isfile(osworld_desktop_env):
        raise SystemExit(f"ERROR: OSWORLD_PATH invalid, not found: {osworld_desktop_env}")

    task_source = ""
    if args.split:
        split_file = Path(args.osworld_path) / "evaluation_examples" / f"{args.split}.json"
        if not split_file.is_file():
            raise SystemExit(f"ERROR: split file not found: {split_file}")
        tasks = load_benchmark_tasks(
            args.osworld_path,
            args.split,
            domain=args.domain,
            limit=args.limit,
        )
        task_source = f"split={args.split} ({split_file})"
    else:
        tasks = load_query_tasks(
            args.tasks_dir,
            vm_upload_dir=args.vm_upload_dir,
            limit=args.limit,
            difficulty=args.difficulty,
            file_type=args.file_type,
        )
        task_source = str(Path(args.tasks_dir).resolve())

    print(f"\nLoaded {len(tasks)} task(s) from {task_source}")
    for task in tasks[:20]:
        domain = task.metadata.get("domain") or task.suite or "?"
        difficulty = task.metadata.get("difficulty") or ""
        label = difficulty or domain
        query = str(task.metadata.get("query") or task.instruction or "")
        print(f"  [{label}] {task.task_id}: {query[:90]}")
    if len(tasks) > 20:
        print(f"  ... and {len(tasks) - 20} more")

    if args.list_only or args.dry_run:
        print("\nValidation complete.")
        return
    if not tasks:
        raise SystemExit("No tasks found.")

    output_dir = args.output_dir
    traj_root = Path(args.traj_root) if args.traj_root else Path(output_dir) / "trajectories"
    num_rollouts = max(1, args.num_rollouts)

    run_tasks, rollout_map, skipped = plan_rollouts(
        tasks, traj_root=traj_root, num_rollouts=num_rollouts,
    )
    total_rollouts = len(tasks) * num_rollouts
    print(f"\nResume check: {num_rollouts} rollout(s) per task -> {total_rollouts} total")
    print(f"  Already done (skipped): {skipped}")
    print(f"  To run:                 {len(run_tasks)}")

    if not run_tasks:
        print("\nAll rollouts already completed. Nothing to run (resume).")
        print(f"  Trajectories: {traj_root.resolve()}")
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
    )

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
        },
    )

    print(f"\nRunning {len(run_tasks)} rollout(s) with {args.num_vms} concurrent VM(s) ...")
    print(f"  Harness:      {args.harness}")
    print(f"  Model:        {args.model}")
    print(f"  API:          {args.api_base or os.environ.get('OPENAI_BASE_URL', '(not set)')}")
    print(f"  Output:       {output_dir}")
    print(f"  Trajectories: {traj_root.resolve()}")
    print("  (real-time saving: traj.jsonl + screenshots written per step)")

    writer = RolloutTraceWriter(traj_root=traj_root, rollout_map=rollout_map)
    backend.setup()

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

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.num_vms) as pool:
            futures = {pool.submit(_execute_one, t): t for t in run_tasks}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    failed_task = futures[fut]
                    logger.error("Task %s raised: %s", failed_task.task_id, exc)
                    results.append(RunTaskResult.failed(task_id=failed_task.task_id, error=exc))
    finally:
        with contextlib.suppress(Exception):
            backend.teardown()
        writer.close()

    elapsed = time.time() - t0
    succeeded = sum(1 for r in results if r.evaluation and r.evaluation.success)
    scores = [r.evaluation.score for r in results if r.evaluation and r.evaluation.score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\nCompleted {len(results)} task(s) in {elapsed:.1f}s")
    print(f"  Success:   {succeeded}/{len(results)}")
    print(f"  Avg score: {avg_score:.3f}")
    for item in results:
        task_id = item.trace.task_id if item.trace else "unknown"
        success = item.evaluation.success if item.evaluation else None
        score = item.evaluation.score if item.evaluation else None
        reason = item.evaluation.reason if item.evaluation else "no evaluation"
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {task_id}: score={score}  {reason or ''}")
    print(f"\n  Trajectories: {traj_root.resolve()}")
    print("\nDone.")

if __name__ == "__main__":
    main()
