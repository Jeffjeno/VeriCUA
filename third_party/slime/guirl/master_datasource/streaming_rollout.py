from __future__ import annotations

import asyncio
import json
import logging
import time
from argparse import Namespace
from typing import Any

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer
from slime.rollout.sglang_rollout import GenerateState, abort, eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample

from .step_accumulator import BatchInvalidError

logger = logging.getLogger("guirl.streaming_rollout")

def _group_sort_key(group) -> int:
    first = group[0][0] if isinstance(group[0], list) else group[0]
    return first.index

async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:

    assert args.rollout_global_dataset
    assert args.dynamic_sampling_filter_path is None, (
        "guirl streaming rollout does not support dynamic sampling filters"
    )

    state = GenerateState(args)
    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size

    submitted = 0
    data: list = []
    pbar = tqdm(total=target_data_size, desc=f"GUIRL streaming rollout {rollout_id}")

    async def _feed() -> None:

        nonlocal submitted
        while submitted < target_data_size:
            need = target_data_size - submitted
            samples = await asyncio.to_thread(data_source, need)
            if not samples:
                continue
            if len(samples) > need:

                samples = samples[:need]
            state.submit_generate_tasks(samples)
            submitted += len(samples)
            logger.info(
                "GUIRL wave submitted: +%d samples (%d/%d total)",
                len(samples), submitted, target_data_size,
            )

    feeder = asyncio.create_task(_feed())

    try:
        while len(data) < target_data_size:
            if not state.pendings:
                if feeder.done() and feeder.exception() is not None:
                    raise feeder.exception()
                await asyncio.sleep(1.0)
                continue

            snapshot = set(state.pendings)
            done, not_done = await asyncio.wait(
                snapshot, return_when=asyncio.FIRST_COMPLETED
            )
            added_during_wait = state.pendings - snapshot
            state.pendings = not_done | added_during_wait
            for task in done:
                group = task.result()
                assert len(group) == args.n_samples_per_prompt
                data.append(group)
                pbar.update(1)
    except BaseException:
        feeder.cancel()
        raise
    finally:
        pbar.close()

    await feeder

    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, (
        f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    )
    data = sorted(data, key=_group_sort_key)

    state.reset()
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples

def _iter_samples(node):
    if isinstance(node, Sample):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _iter_samples(item)

def _generated_group_failed(group: Any) -> bool:
    samples = list(_iter_samples(group))
    if not samples:
        return True
    return any(
        getattr(getattr(sample, "status", None), "value", getattr(sample, "status", None))
        in {"aborted", "failed"}
        for sample in samples
    )

def _judge_summary(group: Any, entry: dict[str, Any]) -> dict[str, Any]:
    samples = list(_iter_samples(group))
    for sample in samples:
        summary = (getattr(sample, "metadata", None) or {}).get(
            "guirl_judge_artifact"
        )
        if isinstance(summary, dict):
            return summary
    from guirl.slime_custom import build_judge_artifact_summary

    source = (
        ((samples[0].metadata or {}).get("guirl") or {})
        if samples
        else {
            "slot": entry.get("slot"),
            "task_id": entry.get("task_id"),
            "claim_token": entry.get("claim_token"),
            "claim_generation": entry.get("generation"),
        }
    )
    return build_judge_artifact_summary(samples, source_guirl=source)

def _host_batch_state(data_source, rollout_id: int) -> str | None:
    status = data_source.batch_status(rollout_id)
    return None if status is None else str(status.get("state"))

def _raise_terminal_v2_state(data_source, rollout_id: int, state: str | None) -> None:
    if state not in {"FAILED", "INVALID", "COMMITTED", "IN_DOUBT"}:
        return
    accumulator = data_source.accumulator
    run_id = data_source.run_id
    if state in {"FAILED", "INVALID"} and accumulator is not None and run_id is not None:
        accumulator.clear_revision(
            run_id,
            rollout_id,
            terminal_state=state,
        )
    if state == "COMMITTED":
        if accumulator is not None and run_id is not None:
            accumulator.mark_committed(
                run_id,
                rollout_id,
                host_proved_committed=True,
            )
        raise RuntimeError(
            f"Rollout Host v2 step {rollout_id} is already COMMITTED; "
            "refusing to train it again"
        )
    raise RuntimeError(f"Rollout Host v2 step {rollout_id} is {state}")

def _raise_unsafe_local_journal(data_source, rollout_id: int) -> None:
    accumulator = data_source.accumulator
    run_id = data_source.run_id
    if accumulator is None or run_id is None:
        return
    journal = accumulator.journal(run_id, rollout_id)
    if journal is None:
        return
    state = str(journal["state"])
    if state in {
        "OPTIMIZING",
        "OPTIMIZER_SUCCEEDED",
        "CHECKPOINTED",
        "COMMITTING",
        "COMMITTED",
        "IN_DOUBT",
    }:
        raise RuntimeError(
            "GUIRL_V2_OPERATOR_RESOLUTION_REQUIRED: refusing automatic "
            f"optimizer replay for local journal state {state}"
        )

async def generate_rollout_v2_async(
    args: Namespace,
    rollout_id: int,
    data_source,
) -> RolloutFnTrainOutput:

    if not args.rollout_global_dataset:
        raise ValueError("Rollout Host v2 requires rollout_global_dataset")
    if args.dynamic_sampling_filter_path is not None:
        raise ValueError(
            "Rollout Host v2 does not support dynamic sampling filters"
        )
    if int(args.n_samples_per_prompt) != 1:
        raise ValueError("Rollout Host v2 requires n_samples_per_prompt=1")

    status = await asyncio.to_thread(data_source.v2_status)
    host = status.get("host") or {}
    actor_order = list(host.get("actor_order") or [])
    actor_count = int(host.get("actor_count") or len(actor_order))
    if actor_count <= 0:
        raise ValueError("Rollout Host v2 reported no configured actors")
    if int(args.rollout_batch_size) % actor_count:
        raise ValueError(
            "rollout_batch_size must be divisible by the Rollout Host v2 "
            f"actor count: {args.rollout_batch_size} % {actor_count} != 0"
        )
    tasks_per_step = int(args.rollout_batch_size) // actor_count
    if tasks_per_step <= 0:
        raise ValueError("Rollout Host v2 tasks_per_step must be positive")

    await asyncio.to_thread(
        data_source.acquire_batch,
        rollout_id,
        tasks_per_step,
    )
    accumulator = data_source.accumulator
    run_id = data_source.run_id
    if accumulator is None or run_id is None:
        raise RuntimeError(
            "GUIRL_V2_STATE_ERROR: trainer accumulator/run_id is not initialized"
        )
    state_name = await asyncio.to_thread(
        _host_batch_state, data_source, rollout_id
    )
    _raise_terminal_v2_state(data_source, rollout_id, state_name)
    _raise_unsafe_local_journal(data_source, rollout_id)
    if state_name == "SEALED":
        accumulator.assert_replay_safe(run_id, rollout_id)
        if not accumulator.is_complete(run_id, rollout_id):
            raise BatchInvalidError(
                "BATCH_INVALID: Host is SEALED but trainer artifacts are incomplete"
            )
        accumulator.mark_state(run_id, rollout_id, "SEALED")
        return RolloutFnTrainOutput(
            samples=accumulator.ordered_output(run_id, rollout_id),
            metrics=await _collect_v2_metrics(data_source, rollout_id),
        )

    state = GenerateState(args)
    metric_gatherer = MetricGatherer()
    pending: dict[asyncio.Task, dict[str, Any]] = {}
    renewers: dict[asyncio.Task, asyncio.Task] = {}
    renewal_stops: dict[asyncio.Task, asyncio.Event] = {}
    claim_tasks: dict[tuple[int, str], asyncio.Task] = {}
    pbar = tqdm(
        total=len(accumulator.expected_keys(run_id, rollout_id)),
        initial=len(accumulator.accepted_keys(run_id, rollout_id)),
        desc=f"GUIRL v2 rollout {rollout_id}",
    )

    async def renew_until_stopped(
        stop: asyncio.Event,
        entry: dict[str, Any],
        *,
        judge_task: asyncio.Task | None = None,
        immediate: bool = False,
    ) -> None:
        ttl = float(getattr(data_source, "_claim_ttl_seconds", 600.0))
        interval = max(
            0.1,
            min(
                ttl / 3.0,
                float(_getattr(args, "guirl_v2_claim_renew_interval_seconds", 60.0)),
            ),
        )
        safety_margin = max(0.1, min(ttl / 6.0, interval))
        first = True
        while not stop.is_set():
            expires_at = float(entry.get("expires_at") or (time.time() + ttl))
            delay = min(
                interval,
                max(0.0, expires_at - time.time() - safety_margin),
            )
            if immediate and first:
                delay = 0.0
            first = False
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                return
            try:
                result = await asyncio.to_thread(
                    data_source.renew_claim, rollout_id, entry
                )
                if result.get("expires_at") is not None:
                    entry["expires_at"] = float(result["expires_at"])
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if isinstance(exc, asyncio.CancelledError):
                    raise
                entry["_renewal_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                if judge_task is not None and not judge_task.done():
                    judge_task.cancel()
                return

    async def stop_renewer(
        renewer: asyncio.Task | None,
        stop: asyncio.Event | None,
    ) -> None:
        if stop is not None:
            stop.set()
        if renewer is not None:
            await asyncio.gather(renewer, return_exceptions=True)

    async def stop_task_renewer(task: asyncio.Task) -> None:
        await stop_renewer(
            renewers.pop(task, None),
            renewal_stops.pop(task, None),
        )

    def submit_entry(
        entry: dict[str, Any],
        *,
        recovered: bool = False,
    ) -> None:
        identity = accumulator.slot_identity(entry)
        if identity in accumulator.accepted_keys(run_id, rollout_id):
            return
        previous = claim_tasks.get(identity)
        if previous is not None and not previous.done():
            previous_entry = pending.get(previous)
            if previous_entry and (
                previous_entry.get("claim_token") == entry.get("claim_token")
                and int(previous_entry.get("generation", -1))
                == int(entry.get("generation", -2))
            ):
                return
            previous.cancel()
        sample = data_source.entry_to_v2_sample(rollout_id, entry)
        entry["_source_group_id"] = getattr(sample, "group_id", sample.index)
        before = set(state.pendings)
        state.submit_generate_tasks([[sample]])
        created = state.pendings - before
        if len(created) != 1:
            raise RuntimeError(
                f"GenerateState created {len(created)} tasks for one v2 slot"
            )
        task = created.pop()
        pending[task] = entry
        claim_tasks[identity] = task
        stop = asyncio.Event()
        renewal_stops[task] = stop
        renewers[task] = asyncio.create_task(
            renew_until_stopped(
                stop,
                entry,
                judge_task=task,
                immediate=recovered,
            )
        )

    async def best_effort_nack(entry: dict[str, Any], reason: str) -> None:
        try:
            await asyncio.to_thread(data_source.nack, rollout_id, entry, reason)
        except Exception as exc:
            if getattr(exc, "code", None) != "FENCED":
                logger.warning("v2 nack failed for %s: %s", entry.get("slot"), exc)

    async def ack_with_renewal(
        entry: dict[str, Any],
        artifact: dict[str, Any],
        renewer: asyncio.Task,
    ) -> bool:

        final_attempt = False
        total_deadline = asyncio.get_running_loop().time() + float(
            getattr(
                data_source,
                "_ack_total_deadline_seconds",
                max(
                    1.0,
                    float(
                        getattr(
                            data_source,
                            "_claim_ttl_seconds",
                            600.0,
                        )
                    ),
                ),
            )
        )
        while True:
            if (
                not final_attempt
                and asyncio.get_running_loop().time() >= total_deadline
            ):
                logger.warning(
                    "v2 ack total deadline elapsed for %s",
                    entry.get("slot"),
                )
                return False
            try:
                await asyncio.to_thread(
                    data_source.ack,
                    rollout_id,
                    entry,
                    artifact,
                )
                return True
            except Exception as exc:
                if getattr(exc, "code", None) == "FENCED":
                    return False
                renewal_error = entry.get("_renewal_error")
                if renewer.done() and renewal_error is None:
                    renewal_failure = (
                        asyncio.CancelledError("claim renewer was cancelled")
                        if renewer.cancelled()
                        else renewer.exception()
                    )
                    if renewal_failure is not None:
                        renewal_error = (
                            f"{type(renewal_failure).__name__}: "
                            f"{renewal_failure}"
                        )
                        entry["_renewal_error"] = renewal_error
                if renewal_error:
                    if final_attempt:
                        logger.warning(
                            "v2 ack failed after claim renewal stopped for %s: %s",
                            entry.get("slot"),
                            exc,
                        )
                        return False

                    final_attempt = True
                    continue
                if asyncio.get_running_loop().time() >= total_deadline:
                    logger.warning(
                        "v2 ack total deadline elapsed for %s after: %s",
                        entry.get("slot"),
                        exc,
                    )
                    return False
                logger.warning(
                    "bounded v2 ack attempt failed for %s: %s; retrying "
                    "while claim renewal remains healthy",
                    entry.get("slot"),
                    exc,
                )
                await asyncio.sleep(0.1)

    async def recover_pending_acks() -> None:
        for row in accumulator.pending_acks(run_id, rollout_id):
            entry = json.loads(str(row["entry_json"]))
            artifact = {
                "digest": str(row["summary_digest"]),
                "size": int(row["summary_size"]),
                "uri": str(row["summary_uri"]),
                "media_type": "application/json",
            }
            stop = asyncio.Event()
            renewer = asyncio.create_task(
                renew_until_stopped(
                    stop,
                    entry,
                    immediate=True,
                )
            )
            try:
                acked = await ack_with_renewal(entry, artifact, renewer)
                if acked:
                    pbar.update(1)
                else:
                    await best_effort_nack(
                        entry,
                        "recovered-ack-finalization-failed",
                    )
                    accumulator.discard_slot(
                        run_id,
                        rollout_id,
                        int(row["task_index"]),
                        str(row["actor_id"]),
                    )
            finally:
                await stop_renewer(renewer, stop)

    async def process_done(done: set[asyncio.Task]) -> None:
        for task in done:
            state.pendings.discard(task)
            entry = pending.pop(task)
            identity = accumulator.slot_identity(entry)
            is_current_claim = claim_tasks.get(identity) is task
            if is_current_claim:
                claim_tasks.pop(identity, None)
            renewer = renewers.get(task)
            if not is_current_claim:

                try:
                    task.result()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    pass
                await stop_task_renewer(task)
                await best_effort_nack(entry, "superseded-local-claim")
                continue
            try:
                group = task.result()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                renewal_error = entry.get("_renewal_error")
                await best_effort_nack(
                    entry,
                    (
                        f"claim-renewal-failed:{renewal_error}"
                        if renewal_error
                        else f"judge-error:{type(exc).__name__}:{exc}"
                    ),
                )
                await stop_task_renewer(task)
                accumulator.discard_slot(run_id, rollout_id, *identity)
                continue
            if _generated_group_failed(group):
                await stop_task_renewer(task)
                await best_effort_nack(entry, "judge-returned-failed-group")
                accumulator.discard_slot(run_id, rollout_id, *identity)
                continue
            summary = _judge_summary(group, entry)
            persisted = await asyncio.to_thread(
                accumulator.persist_result,
                run_id=run_id,
                train_step_id=rollout_id,
                entry=entry,
                generated_group=group,
                judge_summary=summary,
                source_group_id=entry["_source_group_id"],
            )
            if persisted["status"] == "SUPERSEDED":
                await stop_task_renewer(task)
                await best_effort_nack(entry, "manifest-superseded")
                continue
            if renewer is None:
                raise RuntimeError("v2 claim lost its renewal task before ACK")
            acked = await ack_with_renewal(
                entry,
                persisted["judge_artifact"],
                renewer,
            )
            await stop_task_renewer(task)
            if not acked:
                reason = (
                    "claim-renewal-failed-before-ack:"
                    f"{entry['_renewal_error']}"
                    if entry.get("_renewal_error")
                    else "ack-finalization-failed"
                )
                await best_effort_nack(entry, reason)
                accumulator.discard_slot(run_id, rollout_id, *identity)
                continue
            pbar.update(1)

    try:
        await recover_pending_acks()
        for entry in accumulator.recoverable_claims(run_id, rollout_id):
            submit_entry(entry, recovered=True)

        while True:
            await asyncio.to_thread(
                data_source.acquire_batch,
                rollout_id,
                tasks_per_step,
            )
            current_host_state = await asyncio.to_thread(
                _host_batch_state, data_source, rollout_id
            )
            _raise_terminal_v2_state(
                data_source,
                rollout_id,
                current_host_state,
            )
            _raise_unsafe_local_journal(data_source, rollout_id)

            for task, entry in list(pending.items()):
                if accumulator.entry_is_expected(run_id, rollout_id, entry):
                    continue
                task.cancel()
                await best_effort_nack(entry, "manifest-superseded")
                identity = accumulator.slot_identity(entry)
                accumulator.discard_slot(run_id, rollout_id, *identity)

            await recover_pending_acks()
            if accumulator.is_complete(run_id, rollout_id):
                try:
                    await asyncio.to_thread(data_source.seal, rollout_id)
                except Exception as exc:
                    if getattr(exc, "code", None) != "BATCH_INCOMPLETE":
                        terminal = await asyncio.to_thread(
                            _host_batch_state, data_source, rollout_id
                        )
                        _raise_terminal_v2_state(
                            data_source,
                            rollout_id,
                            terminal,
                        )
                        raise
                    await asyncio.sleep(0.1)
                    continue
                metrics = metric_gatherer.collect()
                metrics.update(
                    await _collect_v2_metrics(data_source, rollout_id)
                )
                return RolloutFnTrainOutput(
                    samples=accumulator.ordered_output(run_id, rollout_id),
                    metrics=metrics,
                )

            accepted = accumulator.accepted_keys(run_id, rollout_id)
            expected = accumulator.expected_keys(run_id, rollout_id)
            capacity = max(1, len(expected - accepted) - len(pending))
            entries = await asyncio.to_thread(
                data_source.pull_ready,
                rollout_id,
                limit=capacity,
                wait_seconds=1.0,
            )
            for entry in entries:
                identity = accumulator.slot_identity(entry)
                if not accumulator.entry_is_expected(run_id, rollout_id, entry):
                    await best_effort_nack(entry, "manifest-superseded")
                    continue
                if identity in accumulator.accepted_keys(run_id, rollout_id):
                    await best_effort_nack(entry, "duplicate-accepted-slot")
                    continue
                submit_entry(entry)

            done = {task for task in pending if task.done()}
            if not done and pending and not entries:
                completed, _ = await asyncio.wait(
                    set(pending),
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                done = set(completed)
            if done:
                await process_done(done)
            elif not pending and not entries:
                await asyncio.sleep(0.1)
    except BatchInvalidError as exc:
        try:
            await asyncio.to_thread(
                data_source.fail,
                rollout_id,
                f"BATCH_INVALID:{exc}",
            )
        except Exception:
            accumulator.clear_revision(
                run_id,
                rollout_id,
                terminal_state="INVALID",
            )
            logger.exception("Failed to mark invalid v2 Host batch FAILED")
        for task, entry in list(pending.items()):
            if not task.done():
                task.cancel()
            await best_effort_nack(entry, "trainer-batch-invalid")
        raise
    except BaseException:
        for task, entry in list(pending.items()):
            if not task.done():
                task.cancel()
            await best_effort_nack(entry, "trainer-rollout-aborted")
        raise
    finally:
        for stop in renewal_stops.values():
            stop.set()
        if renewers:
            await asyncio.gather(*renewers.values(), return_exceptions=True)
        pbar.close()
        state.reset()

def _getattr(args: Namespace, key: str, default: Any) -> Any:
    value = getattr(args, key, default)
    return default if value is None else value

async def _collect_v2_metrics(data_source: Any, rollout_id: int) -> dict[str, float]:

    try:
        response = await asyncio.to_thread(data_source.v2_status)
        host = response.get("host") or {}
    except Exception as exc:
        logger.warning("could not collect guirl_v2 metrics: %s", exc)
        return {}
    batch = None
    run_id = getattr(data_source, "run_id", None)
    for candidate in host.get("batches") or []:
        identity = candidate.get("batch_id") or {}
        if (
            str(identity.get("run_id")) == str(run_id)
            and int(identity.get("train_step_id", -1)) == int(rollout_id)
        ):
            batch = candidate
            break
    metrics: dict[str, float] = {}
    if batch is not None:
        completion = batch.get("step_completion") or {}
        metrics["guirl_v2/step_completion"] = float(
            completion.get("fraction") or 0.0
        )
        counts = batch.get("slot_counts") or {}
        for state in (
            "PENDING",
            "RUNNING",
            "READY",
            "CLAIMED",
            "ACKING",
            "DONE",
            "RETRY_PENDING",
            "FAILED",
        ):
            metrics[f"guirl_v2/slots/{state.lower()}"] = float(
                counts.get(state) or 0
            )
    timings = host.get("timings_seconds") or {}
    for timing_name in ("queue_wait", "actor_execution"):
        distribution = timings.get(timing_name) or {}
        for percentile in ("p50", "p95"):
            metrics[
                f"guirl_v2/timing/{timing_name}_{percentile}_seconds"
            ] = float(distribution.get(percentile) or 0.0)
    utilization = host.get("vm_utilization") or {}
    metrics["guirl_v2/vm_utilization"] = float(
        utilization.get("fraction") or 0.0
    )
    throughput = host.get("throughput") or {}
    metrics["guirl_v2/throughput_slots_per_second"] = float(
        throughput.get("slots_per_second") or 0.0
    )
    claim_age = host.get("claim_age_seconds") or {}
    lease_age = host.get("lease_age_seconds") or {}
    metrics["guirl_v2/claim_age_p95_seconds"] = float(
        claim_age.get("p95") or 0.0
    )
    metrics["guirl_v2/lease_age_p95_seconds"] = float(
        lease_age.get("p95") or 0.0
    )
    return metrics

def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:

    if not args.rollout_global_dataset:
        raise ValueError("GUIRL rollout requires rollout_global_dataset")
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    if bool(getattr(data_source, "v2_enabled", False)):
        return run(generate_rollout_v2_async(args, rollout_id, data_source))

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
