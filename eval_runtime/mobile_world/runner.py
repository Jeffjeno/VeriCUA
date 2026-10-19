from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from eval_runtime.guirl.evaluator import (
    build_guirl_config,
    run_guirl_eval_after_replay,
)
from eval_runtime.passive_baseline.evaluator import (
    PASSIVE_BASELINE_TYPES,
    build_passive_baseline_config,
    run_passive_baseline_eval_after_replay,
)
from eval_runtime.vagen_active.agent_factory import (
    VAGEN_EVALUATOR_TYPE,
    VAGEN_QWEN_EVALUATOR_TYPE,
    load_vagen_eval_config,
)
from eval_runtime.vagen_active.evaluator import run_vagen_eval_after_replay

from .dataset import (
    MobileWorldDataset,
    MobileWorldRecord,
    load_mobile_world_dataset,
    materialize_episode,
    result_episode_dir,
)
from .env_client import MobileWorldEvalEnvClient

logger = logging.getLogger("eval_runtime.mobile_world.runner")

def _read_evaluator_type(config_path: Path) -> str:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return str(
        payload.get("evaluator_type")
        or payload.get("judge_evaluator")
        or payload.get("agent_judge_type")
        or ""
    ).strip().lower()

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def _error_summary(
    *,
    record: MobileWorldRecord,
    evaluator_type: str,
    policy_model: str,
    error: Exception,
    rollout_id: str = "0",
) -> dict[str, Any]:
    return {
        "domain": "mobile_world",
        "example_id": record.task_name,
        "rollout_id": str(rollout_id),
        "policy_model": policy_model,
        "policy_result": None,
        "final_label": "error",
        "final_status": "error",
        "sampler_rounds": 0,
        "rounds": [],
        "created_at": datetime.datetime.now().isoformat(),
        "evaluator_type": evaluator_type,
        "error_message": str(error),
        "candidate_id": record.candidate_id,
    }

def _is_complete(result_dir: Path) -> bool:
    summary_path = result_dir / "replay_eval" / "eval_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary.get("final_label") in {
            "success",
            "failure",
            "uncertain",
            "error",
        }
    except Exception:
        return False

def _has_materialized_screenshots(result_dir: Path) -> bool:
    if not (result_dir / "episode.json").is_file():
        return False
    screenshot_dir = result_dir / "screenshots"
    if not screenshot_dir.is_dir():
        return False
    return any(screenshot_dir.glob("*.png")) or any(screenshot_dir.glob("*.jpg"))

def _copy_materialization(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir,
        dest_dir,
        ignore=shutil.ignore_patterns("replay_eval", "run_error.json"),
    )

def _parse_rollout_ids(raw: str | None) -> list[str]:
    if not raw:
        return ["0"]
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.replace(" ", ",").split(","):
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        values.append(token)
    if not values:
        raise ValueError("--rollout-ids must contain at least one id")
    return values

class MobileWorldEvaluationRunner:
    def __init__(
        self,
        *,
        dataset: MobileWorldDataset,
        evaluator_config_path: Path,
        result_root: Path,
        backends: list[str],
        concurrency: int,
        step_wait_time: float,
        recovery_timeout: float,
        task_retries: int,
        max_eval_rounds: int,
        sleep_after_execution: float,
        resume: bool,
        offline_passive: bool,
        rollout_ids: list[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.evaluator_config_path = evaluator_config_path
        self.result_root = result_root
        self.evaluator_type = _read_evaluator_type(evaluator_config_path)
        self.policy_model = dataset.name
        self.task_retries = max(0, int(task_retries))
        self.max_eval_rounds = int(max_eval_rounds)
        self.sleep_after_execution = float(sleep_after_execution)
        self.resume = resume
        self.offline_passive = bool(offline_passive)
        self.rollout_ids = [str(value) for value in (rollout_ids or ["0"])]
        if not self.rollout_ids:
            raise ValueError("rollout_ids must be non-empty")
        self._task_locks: dict[str, threading.Lock] = {}
        self._task_locks_guard = threading.Lock()

        if self.evaluator_type in PASSIVE_BASELINE_TYPES:
            self.passive_config = build_passive_baseline_config(
                str(evaluator_config_path)
            )
            self.guirl_config = None
            self.vagen_config = None
        elif self.evaluator_type == "guirl":
            self.passive_config = None
            self.guirl_config = build_guirl_config(str(evaluator_config_path))
            env_overrides = {
                "judge_base_url": "MW_GUIRL_JUDGE_BASE_URL",
                "judge_model_name": "MW_GUIRL_JUDGE_MODEL_NAME",
                "probe_base_url": "MW_GUIRL_PROBE_BASE_URL",
                "probe_model_name": "MW_GUIRL_PROBE_MODEL_NAME",
            }
            for config_key, env_key in env_overrides.items():
                if os.getenv(env_key):
                    self.guirl_config[config_key] = os.environ[env_key]
            probe_type = str(self.guirl_config.get("probe_agent_type") or "")
            if not probe_type.startswith("MobileWorld"):
                raise ValueError(
                    "mobileworld GUIRL requires a native MobileWorld probe_agent_type, "
                    f"got {probe_type!r}"
                )
            self.vagen_config = None
        elif self.evaluator_type in (VAGEN_EVALUATOR_TYPE, VAGEN_QWEN_EVALUATOR_TYPE):
            self.passive_config = None
            self.guirl_config = None
            self.vagen_config = load_vagen_eval_config(str(evaluator_config_path))
            if os.getenv("MW_VAGEN_MAX_ROUND"):
                self.vagen_config["max_round"] = int(os.environ["MW_VAGEN_MAX_ROUND"])
                if self.vagen_config["max_round"] <= 0:
                    raise ValueError("MW_VAGEN_MAX_ROUND must be positive")
            if (
                str(self.vagen_config.get("platform") or "").strip().lower()
                != "mobileworld"
            ):
                raise ValueError(
                    "mobileworld VAGEN config must set platform: MobileWorld"
                )
        else:
            raise ValueError(
                f"Unsupported mobileworld evaluator_type: {self.evaluator_type!r}"
            )

        self.client_queue: queue.Queue[MobileWorldEvalEnvClient] = queue.Queue()
        needs_live_replay = not (
            self.evaluator_type in PASSIVE_BASELINE_TYPES
            and self.offline_passive
        )
        if needs_live_replay:
            if not backends:
                raise ValueError(
                    f"{self.evaluator_type} requires at least one --backend "
                    "(or --offline-passive for an explicitly approximate "
                    "passive-only run)"
                )
            for backend in backends:
                self.client_queue.put(
                    MobileWorldEvalEnvClient(
                        backend,
                        step_wait_time=step_wait_time,
                        recovery_timeout=recovery_timeout,
                    )
                )
            self.concurrency = min(max(1, concurrency), len(backends))
        else:
            self.concurrency = max(1, concurrency)

    def _task_lock(self, task_name: str) -> threading.Lock:
        with self._task_locks_guard:
            lock = self._task_locks.get(task_name)
            if lock is None:
                lock = threading.Lock()
                self._task_locks[task_name] = lock
            return lock

    def result_dir(
        self,
        record: MobileWorldRecord,
        rollout_id: str | int = "0",
    ) -> Path:
        return result_episode_dir(
            self.result_root,
            policy_model=self.policy_model,
            task_name=record.task_name,
            rollout_id=str(rollout_id),
        )

    def _find_reusable_materialization(
        self,
        record: MobileWorldRecord,
        rollout_id: str,
    ) -> Path | None:
        task_root = (
            self.result_root
            / "mobile_world_json"
            / "screenshot"
            / self.policy_model
            / "mobile_world"
            / record.task_name
        )
        if not task_root.is_dir():
            return None
        preferred = ["0", *[rid for rid in self.rollout_ids if rid != "0"]]
        seen: set[str] = set()
        candidates: list[str] = []
        for rid in preferred:
            if rid not in seen:
                seen.add(rid)
                candidates.append(rid)
        for path in sorted(task_root.iterdir(), key=lambda item: item.name):
            if path.name not in seen and path.is_dir():
                candidates.append(path.name)
        for rid in candidates:
            if rid == str(rollout_id):
                continue
            candidate = task_root / rid
            if _has_materialized_screenshots(candidate):
                return candidate
        return None

    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            replay_eval_config_path=str(self.evaluator_config_path),
            passive_baseline_config=self.passive_config,
            guirl_config=self.guirl_config,
            vagen_eval_config=self.vagen_config,
            max_eval_rounds=self.max_eval_rounds,
            sleep_after_execution=self.sleep_after_execution,
        )

    def _run_passive_judge_only(
        self,
        record: MobileWorldRecord,
        rollout_id: str,
        output_dir: Path,
        policy_result: float | None,
    ) -> dict[str, Any]:
        summary = run_passive_baseline_eval_after_replay(
            example=self.dataset.load_episode(record).example,
            example_result_dir=str(output_dir),
            domain="mobile_world",
            example_id=record.task_name,
            rollout_id=str(rollout_id),
            policy_model=self.policy_model,
            policy_result=policy_result,
            args=self._args(),
        )
        return {
            "task_name": record.task_name,
            "rollout_id": str(rollout_id),
            "status": "completed",
            "final_label": summary.get("final_label"),
            "policy_result": policy_result,
        }

    def _run_passive(
        self,
        record: MobileWorldRecord,
        rollout_id: str = "0",
    ) -> dict[str, Any]:
        output_dir = self.result_dir(record, rollout_id)
        if self.resume and _is_complete(output_dir):
            return {
                "task_name": record.task_name,
                "rollout_id": str(rollout_id),
                "status": "skipped",
            }
        try:
            episode = self.dataset.load_episode(record)
            reuse_dir = self._find_reusable_materialization(record, str(rollout_id))
            if reuse_dir is not None:
                _copy_materialization(reuse_dir, output_dir)
            else:
                materialize_episode(episode, output_dir, overwrite=True)
            return self._run_passive_judge_only(
                record,
                str(rollout_id),
                output_dir,
                record.original_score,
            )
        except Exception as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = _error_summary(
                record=record,
                evaluator_type=self.evaluator_type,
                policy_model=self.policy_model,
                error=exc,
                rollout_id=str(rollout_id),
            )
            _write_json(
                output_dir / "replay_eval" / "eval_summary.json",
                summary,
            )
            _write_json(
                output_dir / "run_error.json",
                {
                    "task_name": record.task_name,
                    "candidate_id": record.candidate_id,
                    "rollout_id": str(rollout_id),
                    "attempts": 1,
                    "error": repr(exc),
                    "created_at": datetime.datetime.now().isoformat(),
                },
            )
            return {
                "task_name": record.task_name,
                "rollout_id": str(rollout_id),
                "status": "error",
                "error": str(exc),
            }

    def _run_live_attempt(
        self,
        record: MobileWorldRecord,
        client: MobileWorldEvalEnvClient,
        rollout_id: str = "0",
    ) -> dict[str, Any]:
        output_dir = self.result_dir(record, rollout_id)
        episode = self.dataset.load_episode(record)
        replay = client.replay_episode(episode)
        if len(replay.steps) != len(episode.steps):
            raise RuntimeError(
                f"Replay terminated after {len(replay.steps)} of "
                f"{len(episode.steps)} steps"
            )
        materialize_episode(
            episode,
            output_dir,
            initial_screenshot=replay.initial_screenshot,
            post_action_screenshots=[step.screenshot for step in replay.steps],
            native_score=replay.score,
            native_reason=replay.reason,
            overwrite=True,
        )
        args = self._args()
        if self.evaluator_type in PASSIVE_BASELINE_TYPES:
            summary = run_passive_baseline_eval_after_replay(
                example=episode.example,
                example_result_dir=str(output_dir),
                domain="mobile_world",
                example_id=record.task_name,
                rollout_id=str(rollout_id),
                policy_model=self.policy_model,
                policy_result=replay.score,
                args=args,
            )
        elif self.evaluator_type == "guirl":
            summary = run_guirl_eval_after_replay(
                env_client=client,
                lease_id=client.lease_id,
                example=episode.example,
                example_result_dir=str(output_dir),
                domain="mobile_world",
                example_id=record.task_name,
                rollout_id=str(rollout_id),
                policy_model=self.policy_model,
                policy_result=replay.score,
                args=args,
            )
        else:
            summary = run_vagen_eval_after_replay(
                env_client=client,
                lease_id=client.lease_id,
                example=episode.example,
                example_result_dir=str(output_dir),
                domain="mobile_world",
                example_id=record.task_name,
                rollout_id=str(rollout_id),
                policy_model=self.policy_model,
                policy_result=replay.score,
                args=args,
            )
        return {
            "task_name": record.task_name,
            "rollout_id": str(rollout_id),
            "status": "completed",
            "final_label": summary.get("final_label"),
            "policy_result": replay.score,
            "original_result": record.original_score,
            "backend": client.base_url,
        }

    def _run_active(
        self,
        record: MobileWorldRecord,
        rollout_id: str = "0",
    ) -> dict[str, Any]:
        output_dir = self.result_dir(record, rollout_id)
        with self._task_lock(record.task_name):
            if self.resume and _is_complete(output_dir):
                return {
                    "task_name": record.task_name,
                    "rollout_id": str(rollout_id),
                    "status": "skipped",
                }

            if self.evaluator_type in PASSIVE_BASELINE_TYPES:
                reuse_dir = self._find_reusable_materialization(
                    record, str(rollout_id)
                )
                if reuse_dir is not None:
                    try:
                        episode_meta = json.loads(
                            (reuse_dir / "episode.json").read_text(encoding="utf-8")
                        )
                        policy_result = episode_meta.get("native_score")
                        if policy_result is None:
                            policy_result = record.original_score
                        _copy_materialization(reuse_dir, output_dir)
                        return self._run_passive_judge_only(
                            record,
                            str(rollout_id),
                            output_dir,
                            (
                                float(policy_result)
                                if policy_result is not None
                                else None
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "Failed to reuse materialization for %s rollout %s; "
                            "falling back to live replay",
                            record.task_name,
                            rollout_id,
                        )

            last_error: Exception | None = None
            for attempt in range(1, self.task_retries + 2):
                client = self.client_queue.get()
                try:
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    result = self._run_live_attempt(record, client, str(rollout_id))
                    result["attempt"] = attempt
                    return result
                except Exception as exc:
                    last_error = exc
                    logger.exception(
                        "MobileWorld %s failed for %s rollout %s (attempt %d/%d)",
                        self.evaluator_type,
                        record.task_name,
                        rollout_id,
                        attempt,
                        self.task_retries + 1,
                    )
                finally:
                    try:
                        client.tear_down_task(record.task_name)
                    except Exception:
                        logger.warning(
                            "Task teardown failed for %s on %s",
                            record.task_name,
                            client.base_url,
                            exc_info=True,
                        )
                    self.client_queue.put(client)
                if attempt <= self.task_retries:
                    time.sleep(min(10, attempt * 2))

            assert last_error is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = _error_summary(
                record=record,
                evaluator_type=self.evaluator_type,
                policy_model=self.policy_model,
                error=last_error,
                rollout_id=str(rollout_id),
            )
            _write_json(output_dir / "replay_eval" / "eval_summary.json", summary)
            _write_json(
                output_dir / "run_error.json",
                {
                    "task_name": record.task_name,
                    "candidate_id": record.candidate_id,
                    "rollout_id": str(rollout_id),
                    "attempts": self.task_retries + 1,
                    "error": repr(last_error),
                    "created_at": datetime.datetime.now().isoformat(),
                },
            )
            return {
                "task_name": record.task_name,
                "rollout_id": str(rollout_id),
                "status": "error",
                "error": str(last_error),
            }

    def run(self, records: list[MobileWorldRecord]) -> list[dict[str, Any]]:
        worker = (
            self._run_passive
            if self.evaluator_type in PASSIVE_BASELINE_TYPES
            and self.offline_passive
            else self._run_active
        )
        jobs = [
            (record, rollout_id)
            for record in records
            for rollout_id in self.rollout_ids
        ]
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as executor:
            future_to_job = {
                executor.submit(worker, record, rollout_id): (record, rollout_id)
                for record, rollout_id in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                record, rollout_id = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.exception(
                        "Evaluation failed for %s rollout %s",
                        record.task_name,
                        rollout_id,
                    )
                    result = {
                        "task_name": record.task_name,
                        "rollout_id": str(rollout_id),
                        "status": "error",
                        "error": str(exc),
                    }
                results.append(result)
                logger.info(
                    "[%d/%d] %s#%s: %s",
                    len(results),
                    len(jobs),
                    record.task_name,
                    rollout_id,
                    result.get("status"),
                )
        return sorted(
            results,
            key=lambda item: (item["task_name"], str(item.get("rollout_id", "0"))),
        )

def _task_names(args: argparse.Namespace) -> list[str] | None:
    names: list[str] = []
    if args.tasks:
        names.extend(
            value.strip() for value in args.tasks.split(",") if value.strip()
        )
    if args.task_file:
        names.extend(
            line.strip()
            for line in Path(args.task_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return names or None

def _select_records(
    dataset: MobileWorldDataset,
    args: argparse.Namespace,
) -> list[MobileWorldRecord]:
    records = dataset.select(_task_names(args))
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    records = [
        record
        for index, record in enumerate(records)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    return records

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mobileworld")
    parser.add_argument("--evaluator-config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="MobileWorld API URL; repeat for parallel backends",
    )
    parser.add_argument("--tasks", help="Comma-separated task names")
    parser.add_argument("--task-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--step-wait-time", type=float, default=3.0)
    parser.add_argument("--recovery-timeout", type=float, default=180.0)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--max-eval-rounds", type=int, default=5)
    parser.add_argument("--sleep-after-execution", type=float, default=3.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rollout-ids",
        default="0",
        help="Comma-separated rollout ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--offline-passive",
        action="store_true",
        help=(
            "Do not live-replay passive baselines. This uses the source's "
            "fallback final frame and is approximate for max-step trajectories."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    dataset = load_mobile_world_dataset(args.dataset)
    records = _select_records(dataset, args)
    rollout_ids = _parse_rollout_ids(args.rollout_ids)
    evaluator_config = Path(args.evaluator_config).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    evaluator_type = _read_evaluator_type(evaluator_config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "evaluator_type": evaluator_type,
                    "tasks": [record.task_name for record in records],
                    "rollout_ids": rollout_ids,
                    "backends": args.backend,
                    "result_root": str(result_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    runner = MobileWorldEvaluationRunner(
        dataset=dataset,
        evaluator_config_path=evaluator_config,
        result_root=result_root,
        backends=list(args.backend),
        concurrency=args.concurrency,
        step_wait_time=args.step_wait_time,
        recovery_timeout=args.recovery_timeout,
        task_retries=args.task_retries,
        max_eval_rounds=args.max_eval_rounds,
        sleep_after_execution=args.sleep_after_execution,
        resume=args.resume,
        offline_passive=args.offline_passive,
        rollout_ids=rollout_ids,
    )
    started = time.monotonic()
    results = runner.run(records)
    effective_max_eval_rounds = (
        int(runner.vagen_config.get("max_round", args.max_eval_rounds))
        if runner.vagen_config is not None
        else args.max_eval_rounds
    )
    run_summary = {
        "dataset": dataset.name,
        "evaluator_type": runner.evaluator_type,
        "evaluator_config": str(evaluator_config),
        "result_root": str(result_root),
        "backends": args.backend,
        "configuration": {
            "concurrency": runner.concurrency,
            "step_wait_time": args.step_wait_time,
            "recovery_timeout": args.recovery_timeout,
            "task_retries": args.task_retries,
            "max_eval_rounds": args.max_eval_rounds,
            "effective_max_eval_rounds": effective_max_eval_rounds,
            "resume": args.resume,
            "offline_passive": args.offline_passive,
            "rollout_ids": rollout_ids,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
        "counts": {
            "total": len(results),
            "completed": sum(
                result.get("status") == "completed" for result in results
            ),
            "skipped": sum(result.get("status") == "skipped" for result in results),
            "error": sum(result.get("status") == "error" for result in results),
        },
        "duration_seconds": round(time.monotonic() - started, 3),
        "results": results,
        "created_at": datetime.datetime.now().isoformat(),
    }
    if args.shard_count == 1:
        if len(rollout_ids) > 1 or rollout_ids != ["0"]:
            summary_name = f"run_summary_rollouts_{'-'.join(rollout_ids)}.json"
        else:
            summary_name = "run_summary.json"
    else:
        if len(rollout_ids) > 1 or rollout_ids != ["0"]:
            summary_name = (
                f"run_summary_rollouts_{'-'.join(rollout_ids)}"
                f"_shard_{args.shard_index:03d}.json"
            )
        else:
            summary_name = f"run_summary_shard_{args.shard_index:03d}.json"
    _write_json(result_root / summary_name, run_summary)
    print(json.dumps(run_summary["counts"], ensure_ascii=False))
    return 1 if run_summary["counts"]["error"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
