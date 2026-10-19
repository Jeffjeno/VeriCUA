from __future__ import annotations

import asyncio
import sys
import time
import types
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

class _StubModule:
    def __getattr__(self, name):
        return MagicMock()

_saved_modules = dict(sys.modules)

for _mod in [
    "torch", "PIL", "PIL.Image", "pybase64", "numpy", "numpy.typing",
    "sglang", "sglang.srt", "sglang.srt.utils", "sglang_router",
    "aiohttp",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _StubModule()

class _FakeGenerateState:

    instance = None
    judge_delay = 0.01

    def __init__(self, args):
        if _FakeGenerateState.instance is None:
            _FakeGenerateState.instance = self
        self.__dict__ = _FakeGenerateState.instance.__dict__ if _FakeGenerateState.instance is not self else {}
        if not hasattr(self, "pendings"):
            self.pendings = set()
            self.submitted_waves = []

    def submit_generate_tasks(self, samples):
        self.submitted_waves.append(len(samples))
        for group in samples:
            self.pendings.add(asyncio.create_task(self._judge(group)))

    async def _judge(self, group):
        delay = (getattr(group[0], "metadata", None) or {}).get(
            "judge_delay", self.judge_delay
        )
        await asyncio.sleep(delay)
        return group

    def reset(self):
        self.pendings = set()

async def _fake_abort(args, rollout_id):
    return []

async def _fake_eval_rollout(args, rollout_id):
    raise AssertionError("eval path should not run in these tests")

_fake_sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
_fake_sglang_rollout.GenerateState = _FakeGenerateState
_fake_sglang_rollout.abort = _fake_abort
_fake_sglang_rollout.eval_rollout = _fake_eval_rollout

_fake_base_types = types.ModuleType("slime.rollout.base_types")

class _RolloutFnTrainOutput:
    def __init__(self, samples=None, metrics=None):
        self.samples = samples
        self.metrics = metrics

_fake_base_types.RolloutFnTrainOutput = _RolloutFnTrainOutput
_fake_base_types.RolloutFnEvalOutput = object

_fake_filter_base = types.ModuleType("slime.rollout.filter_hub.base_types")

class _MetricGatherer:
    def collect(self):
        return {}

_fake_filter_base.MetricGatherer = _MetricGatherer

_fake_async_utils = types.ModuleType("slime.utils.async_utils")
_fake_async_utils.run = lambda coro: asyncio.new_event_loop().run_until_complete(coro)

_fake_types = types.ModuleType("slime.utils.types")

class _StubSample:
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(
        self,
        index=0,
        group_id=None,
        group_index=None,
        prompt="",
        metadata=None,
        status="completed",
        session_id=None,
        generate_function_path=None,
        **kwargs,
    ):
        self.index = index
        self.group_id = group_id
        self.group_index = group_index
        self.prompt = prompt
        self.metadata = metadata or {}
        self.status = status
        self.session_id = session_id
        self.generate_function_path = generate_function_path
        self.__dict__.update(kwargs)

_fake_types.Sample = _StubSample

sys.modules["slime.rollout.sglang_rollout"] = _fake_sglang_rollout
sys.modules["slime.rollout.base_types"] = _fake_base_types
sys.modules["slime.rollout.filter_hub"] = _StubModule()
sys.modules["slime.rollout.filter_hub.base_types"] = _fake_filter_base
sys.modules["slime.utils.async_utils"] = _fake_async_utils
sys.modules["slime.utils.types"] = _fake_types
sys.modules.setdefault("slime", _StubModule())
sys.modules.setdefault("slime.rollout", _StubModule())
sys.modules.setdefault("slime.utils", _StubModule())

sys.modules.pop("guirl.master_datasource.streaming_rollout", None)
from guirl.master_datasource.step_accumulator import StepAccumulator
from guirl.master_datasource.streaming_rollout import (
    generate_rollout_async,
    generate_rollout_v2_async,
)

for _name in list(sys.modules):
    if _name not in _saved_modules:
        sys.modules.pop(_name, None)
for _name, _module in _saved_modules.items():
    sys.modules[_name] = _module

def _make_args(batch_size):
    return Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=batch_size,
        n_samples_per_prompt=1,
    )

class _WaveDataSource:

    def __init__(self, wave_size):
        self.wave_size = wave_size
        self.calls = []
        self._emitted = 0

    def get_samples(self, num_samples):
        self.calls.append(num_samples)
        n = min(self.wave_size, num_samples)
        wave = [[_StubSample(index=self._emitted + i)] for i in range(n)]
        self._emitted += n
        return wave

@pytest.fixture(autouse=True)
def _fresh_state():
    _FakeGenerateState.instance = None
    yield
    _FakeGenerateState.instance = None

class TestStreamingRollout:
    def test_collects_full_batch_from_waves(self):

        ds = _WaveDataSource(wave_size=16)
        args = _make_args(96)

        async def _run():
            return await generate_rollout_async(args, rollout_id=0, data_source=ds.get_samples)

        output, aborted = asyncio.run(_run())

        assert len(output.samples) == 96
        assert aborted == []

        assert ds.calls == [96, 80, 64, 48, 32, 16]
        indices = [g[0].index for g in output.samples]
        assert indices == sorted(indices)

    def test_single_wave_batch(self):

        ds = _WaveDataSource(wave_size=8)
        args = _make_args(8)

        async def _run():
            return await generate_rollout_async(args, rollout_id=0, data_source=ds.get_samples)

        output, _ = asyncio.run(_run())
        assert len(output.samples) == 8
        assert ds.calls == [8]

    def test_feeder_error_propagates(self):

        args = _make_args(4)

        def broken(_n):
            raise RuntimeError("rollout master is down")

        async def _run():
            return await generate_rollout_async(args, rollout_id=0, data_source=broken)

        with pytest.raises(RuntimeError, match="rollout master is down"):
            asyncio.run(_run())

    def test_v2_exact_slot_coverage_and_manifest_order(self, tmp_path):
        class _V2DataSource:
            v2_enabled = True
            run_id = "run-v2"
            _claim_ttl_seconds = 0.6

            def __init__(self):
                self.accumulator = StepAccumulator(tmp_path)
                self.state = "RUNNING"
                self.pulled = False
                self.pending_ack_installed = False
                self.nacks = []
                self.renewals = 0
                self.entries = [
                    self._entry(1, "b", 0.01),
                    self._entry(0, "b", 0.04),
                    self._entry(1, "a", 0.02),
                    self._entry(0, "a", 0.03),
                ]
                persist = self.accumulator.persist_result

                def slow_persist(**kwargs):
                    if kwargs["entry"]["slot"]["task_index"] == 0 and (
                        kwargs["entry"]["slot"]["actor_id"] == "b"
                    ):
                        time.sleep(0.12)
                    return persist(**kwargs)

                self.accumulator.persist_result = slow_persist

            @staticmethod
            def _entry(task_index, actor_id, delay):
                return {
                    "slot": {
                        "run_id": "run-v2",
                        "train_step_id": 0,
                        "task_index": task_index,
                        "actor_id": actor_id,
                    },
                    "task_id": f"task-{task_index}",
                    "task_group_id": f"group-{task_index}",
                    "task_group_revision": 1,
                    "manifest_revision": 1,
                    "attempt": 1,
                    "claim_token": f"claim-{task_index}-{actor_id}",
                    "generation": 1,
                    "expires_at": time.time() + 0.6,
                    "actor_payload": {"judge_delay": delay},
                }

            @property
            def slots(self):
                return [
                    {
                        "task_index": task_index,
                        "actor_id": actor_id,
                        "task_id": f"task-{task_index}",
                        "task_group_id": f"group-{task_index}",
                    }
                    for task_index in range(2)
                    for actor_id in ("a", "b")
                ]

            def v2_status(self):
                return {
                    "ok": True,
                    "host": {
                        "actor_count": 2,
                        "actor_order": ["a", "b"],
                        "batches": [
                            {
                                "batch_id": {
                                    "run_id": self.run_id,
                                    "train_step_id": 0,
                                },
                                "step_completion": {"fraction": 1.0},
                                "slot_counts": {"DONE": 4},
                            }
                        ],
                        "timings_seconds": {
                            "queue_wait": {"p50": 1.0, "p95": 2.0},
                            "actor_execution": {"p50": 3.0, "p95": 4.0},
                        },
                        "vm_utilization": {"fraction": 0.75},
                        "throughput": {"slots_per_second": 0.5},
                        "claim_age_seconds": {"p95": 5.0},
                        "lease_age_seconds": {"p95": 6.0},
                    },
                }

            def acquire_batch(self, rollout_id, tasks_per_step):
                assert tasks_per_step == 2
                self.accumulator.refresh_manifest(
                    run_id=self.run_id,
                    train_step_id=rollout_id,
                    manifest_revision=1,
                    payload_hash="manifest",
                    slots=self.slots,
                )
                if not self.pending_ack_installed:
                    self.pending_ack_installed = True
                    entry = self.entries.pop(0)
                    self.accumulator.record_claim(
                        self.run_id,
                        rollout_id,
                        entry,
                    )
                    sample = self.entry_to_v2_sample(rollout_id, entry)
                    self.accumulator.persist_result(
                        run_id=self.run_id,
                        train_step_id=rollout_id,
                        entry=entry,
                        generated_group=[sample],
                        judge_summary={"recovered": True},
                        source_group_id=sample.group_id,
                    )
                return {
                    "manifest_revision": 1,
                    "payload_hash": "manifest",
                    "slots": self.slots,
                }

            def batch_status(self, rollout_id):
                return {"state": self.state}

            def pull_ready(self, rollout_id, *, limit, wait_seconds):
                if self.pulled:
                    return []
                self.pulled = True
                for entry in self.entries:
                    self.accumulator.record_claim(self.run_id, rollout_id, entry)
                return list(self.entries)

            def entry_to_v2_sample(self, rollout_id, entry):
                task_index = entry["slot"]["task_index"]
                actor_id = entry["slot"]["actor_id"]
                group_id = task_index * 2 + ["a", "b"].index(actor_id)
                return _StubSample(
                    index=group_id,
                    group_id=group_id,
                    prompt=entry["task_id"],
                    metadata={
                        "judge_delay": entry["actor_payload"]["judge_delay"],
                        "guirl_judge_artifact": {
                            "reward": 1,
                            "source": entry["slot"],
                        },
                    },
                )

            def renew_claim(self, rollout_id, entry):
                self.renewals += 1
                return {"expires_at": time.time() + self._claim_ttl_seconds}

            def ack(self, rollout_id, entry, artifact):
                if (
                    entry["slot"]["actor_id"] == "b"
                ):
                    time.sleep(0.12)
                identity = self.accumulator.slot_identity(entry)
                self.accumulator.mark_acked(
                    self.run_id,
                    rollout_id,
                    *identity,
                    claim_token=entry["claim_token"],
                    generation=entry["generation"],
                )
                return {"ok": True}

            def nack(self, rollout_id, entry, reason):
                self.nacks.append((entry["slot"], reason))
                return {"ok": True}

            def seal(self, rollout_id):
                assert self.accumulator.is_complete(self.run_id, rollout_id)
                self.state = "SEALED"
                self.accumulator.mark_state(self.run_id, rollout_id, "SEALED")
                return {"state": "SEALED"}

            def fail(self, rollout_id, reason):
                raise AssertionError(reason)

        args = _make_args(4)
        args.guirl_v2_claim_renew_interval_seconds = 0.01
        data_source = _V2DataSource()
        output = asyncio.run(generate_rollout_v2_async(args, 0, data_source))
        assert [group[0].group_id for group in output.samples] == [0, 1, 2, 3]
        assert len({group[0].group_id for group in output.samples}) == 4
        assert data_source.nacks == []
        assert data_source.renewals >= 2
        assert data_source.state == "SEALED"
        assert output.metrics["guirl_v2/step_completion"] == 1.0
        assert output.metrics["guirl_v2/slots/done"] == 4.0
        assert output.metrics["guirl_v2/vm_utilization"] == 0.75
        assert output.metrics["guirl_v2/timing/actor_execution_p95_seconds"] == 4.0

    def test_failed_renewal_cancels_and_nacks_stale_judge(self, tmp_path):
        class _RenewalFailureDataSource:
            v2_enabled = True
            run_id = "renew-run"
            _claim_ttl_seconds = 1.0

            def __init__(self):
                self.accumulator = StepAccumulator(tmp_path)
                self.state = "RUNNING"
                self.initial_sent = False
                self.replacement_ready = False
                self.replacement_sent = False
                self.nacks = []

            @property
            def slots(self):
                return [
                    {
                        "task_index": 0,
                        "actor_id": "a",
                        "task_id": "task",
                        "task_group_id": "group",
                    }
                ]

            def entry(self, generation):
                return {
                    "slot": {
                        "run_id": self.run_id,
                        "train_step_id": 0,
                        "task_index": 0,
                        "actor_id": "a",
                    },
                    "task_id": "task",
                    "task_group_id": "group",
                    "task_group_revision": 1,
                    "manifest_revision": 1,
                    "attempt": generation,
                    "claim_token": f"claim-{generation}",
                    "generation": generation,
                }

            def v2_status(self):
                return {
                    "host": {
                        "actor_count": 1,
                        "actor_order": ["a"],
                        "batches": [],
                    }
                }

            def acquire_batch(self, rollout_id, tasks_per_step):
                assert tasks_per_step == 1
                self.accumulator.refresh_manifest(
                    run_id=self.run_id,
                    train_step_id=rollout_id,
                    manifest_revision=1,
                    payload_hash="manifest",
                    slots=self.slots,
                )
                return {"slots": self.slots}

            def batch_status(self, rollout_id):
                return {"state": self.state}

            def pull_ready(self, rollout_id, *, limit, wait_seconds):
                del limit, wait_seconds
                entry = None
                if not self.initial_sent:
                    self.initial_sent = True
                    entry = self.entry(1)
                elif self.replacement_ready and not self.replacement_sent:
                    self.replacement_sent = True
                    entry = self.entry(2)
                if entry is None:
                    return []
                self.accumulator.record_claim(
                    self.run_id, rollout_id, entry
                )
                return [entry]

            def entry_to_v2_sample(self, rollout_id, entry):
                del rollout_id
                generation = entry["generation"]
                return _StubSample(
                    index=0,
                    group_id=0,
                    prompt=f"generation-{generation}",
                    metadata={
                        "judge_delay": 0.4 if generation == 1 else 0.01,
                        "guirl_judge_artifact": {
                            "generation": generation
                        },
                    },
                )

            def renew_claim(self, rollout_id, entry):
                del rollout_id
                if entry["generation"] == 1:
                    raise TimeoutError("renewal timed out before expiry")
                return {"expires_at": 10}

            def ack(self, rollout_id, entry, artifact):
                del artifact
                identity = self.accumulator.slot_identity(entry)
                self.accumulator.mark_acked(
                    self.run_id,
                    rollout_id,
                    *identity,
                    claim_token=entry["claim_token"],
                    generation=entry["generation"],
                )
                return {"ok": True}

            def nack(self, rollout_id, entry, reason):
                del rollout_id
                self.nacks.append((entry["generation"], reason))
                self.replacement_ready = True
                return {"ok": True}

            def seal(self, rollout_id):
                self.state = "SEALED"
                self.accumulator.mark_state(
                    self.run_id, rollout_id, "SEALED"
                )
                return {"state": "SEALED"}

            def fail(self, rollout_id, reason):
                raise AssertionError((rollout_id, reason))

        args = _make_args(1)
        args.guirl_v2_claim_renew_interval_seconds = 0.01
        data_source = _RenewalFailureDataSource()
        output = asyncio.run(
            generate_rollout_v2_async(args, 0, data_source)
        )
        assert output.samples[0][0].prompt == "generation-2"
        assert data_source.nacks
        generation, reason = data_source.nacks[0]
        assert generation == 1
        assert "claim-renewal-failed" in reason
