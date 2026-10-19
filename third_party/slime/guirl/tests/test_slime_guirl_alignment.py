from __future__ import annotations

import asyncio
import re
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SLIME_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SLIME_ROOT))

def _install_slime_stubs() -> None:

    rollout_module = types.ModuleType("slime.rollout")
    sglang_rollout_module = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout_module.GenerateState = lambda args: None
    data_source_module = types.ModuleType("slime.rollout.data_source")

    class _DataSource:
        pass

    data_source_module.DataSource = _DataSource

    utils_module = types.ModuleType("slime.utils")
    http_utils_module = types.ModuleType("slime.utils.http_utils")
    http_utils_module.post = None
    processing_module = types.ModuleType("slime.utils.processing_utils")
    processing_module.encode_image_for_rollout_engine = lambda img: "encoded-image"

    types_module = types.ModuleType("slime.utils.types")

    class _Sample:
        class Status:
            ABORTED = "aborted"
            COMPLETED = "completed"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    types_module.Sample = _Sample

    sys.modules.setdefault("slime.rollout", rollout_module)
    sys.modules.setdefault("slime.rollout.sglang_rollout", sglang_rollout_module)
    sys.modules.setdefault("slime.rollout.data_source", data_source_module)
    sys.modules.setdefault("slime.utils", utils_module)
    sys.modules.setdefault("slime.utils.http_utils", http_utils_module)
    sys.modules.setdefault("slime.utils.processing_utils", processing_module)
    sys.modules.setdefault("slime.utils.types", types_module)

@pytest.fixture()
def slime_custom():
    _install_slime_stubs()
    import guirl.slime_custom as module

    return module

class _FakeTokenizer:
    def __init__(self):
        self.apply_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.apply_calls.append((messages, kwargs))
        if len(messages) == 2 and messages[0]["content"] == "x" and messages[1]["content"] == "y":
            return "dummy"
        return "formatted"

    def encode(self, text, add_special_tokens=False):
        assert text == "dummy"
        assert add_special_tokens is False
        return [10, 11]

    def __call__(self, text, add_special_tokens=False):
        assert text == "formatted"
        assert add_special_tokens is False
        return {"input_ids": [10, 11, 12, 13]}

def test_judge_chat_template_receives_enable_thinking(slime_custom):
    tokenizer = _FakeTokenizer()

    rendered = slime_custom._apply_judge_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        judge_enable_thinking=True,
    )

    assert rendered == "formatted"
    assert tokenizer.apply_calls[-1][1]["enable_thinking"] is True

def test_judge_chat_template_fails_if_thinking_flag_is_ignored(slime_custom):
    class _NoThinkingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return "formatted-without-thinking"

    with pytest.raises(RuntimeError, match="does not accept enable_thinking"):
        slime_custom._apply_judge_chat_template(
            _NoThinkingTokenizer(),
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            judge_enable_thinking=True,
        )

    rendered = slime_custom._apply_judge_chat_template(
        _NoThinkingTokenizer(),
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        judge_enable_thinking=True,
        allow_missing_thinking_template=True,
    )
    assert rendered == "formatted-without-thinking"

def test_followup_encoding_uses_same_thinking_flag_for_trim(slime_custom):
    tokenizer = _FakeTokenizer()
    state = SimpleNamespace(tokenizer=tokenizer, processor=None)

    ids, mm_train, uris = slime_custom._encode_followup_turn(
        state,
        {"role": "user", "content": "next"},
        [],
        judge_enable_thinking=True,
    )

    assert ids == [12, 13]
    assert mm_train is None
    assert uris == []
    assert [call[1]["enable_thinking"] for call in tokenizer.apply_calls] == [True, True]

def test_strip_thinking_for_history_keeps_final_answer(slime_custom):
    response = (
        "<think>\nprivate reasoning\n</think>\n\n"
        "<criteria>\n1. done:success\n</criteria>\nStatus: success\n\n"
        "<think>retry scratchpad</think>\n\n"
    )

    assert slime_custom._strip_thinking_for_history(response) == (
        "<criteria>\n1. done:success\n</criteria>\nStatus: success"
    )

def test_history_response_encoding_removes_all_thinking(slime_custom):
    class _CaptureTokenizer:
        def __init__(self):
            self.text = None

        def __call__(self, text, add_special_tokens=False):
            self.text = text
            assert add_special_tokens is False
            return {"input_ids": [1, 2, 3]}

    tokenizer = _CaptureTokenizer()
    ids = slime_custom._encode_history_response_without_thinking(
        tokenizer,
        "<think>hidden</think>\n\nfinal answer\n\n<think>hidden again</think>",
    )

    assert ids == [1, 2, 3]
    assert tokenizer.text == "final answer"

def test_build_judge_turn_sample_fans_out_current_turn_only(slime_custom):
    from slime.utils.types import Sample

    source = Sample(
        index=7,
        group_index=3,
        group_id=None,
        prompt="task",
        session_id="session-1",
        generate_function_path=None,
    )
    child = slime_custom._build_judge_turn_sample(
        source_sample=source,
        round_idx=1,
        max_rounds=3,
        prompt_ids=[10, 11, 12],
        cur_response="<think>current</think>\nStatus: success",
        cur_token_ids=[20, 21],
        cur_logprobs=[-0.1, -0.2],
        multimodal_train_inputs={"pixel_values": object()},
        metadata={"guirl": {"task_id": "task-1"}},
    )

    assert child.index == 22
    assert child.group_id == 7
    assert child.group_index == 3
    assert child.tokens == [10, 11, 12, 20, 21]
    assert child.response_length == 2
    assert child.loss_mask == [1, 1]
    assert child.rollout_log_probs == [-0.1, -0.2]
    assert child.response == "<think>current</think>\nStatus: success"
    assert child.metadata["guirl"]["task_id"] == "task-1"

def test_eval_probe_config_defaults_match_eval_sft(slime_custom):
    cfg = slime_custom._build_eval_probe_config(Namespace(), history_n=3)

    assert cfg["probe_agent_type"] == "Holo3_Probe"
    assert cfg["probe_enable_thinking"] is True
    assert cfg["probe_reasoning_effort"] == "medium"
    assert cfg["probe_max_trajectory_length"] == 15
    assert cfg["probe_max_images"] == 3
    assert cfg["judge_history_n"] == 3

def test_run_eval_probe_round_sync_uses_rollout_master(monkeypatch, tmp_path, slime_custom):
    calls = {}

    def _fake_post(args, endpoint, payload, *, timeout=700.0):
        calls["request"] = (args, endpoint, payload, timeout)
        return {
            "ok": True,
            "latest_criteria": "1. checked:success",
            "criteria_images": [{"option": 1, "images": "evidence-b64"}],
            "round_record": {"done": False, "steps": 1},
            "probe_agent_type": "Holo3_Probe",
        }

    monkeypatch.setattr(slime_custom, "_post_rollout_master_json", _fake_post)

    result = slime_custom._run_eval_probe_round_sync(
        args=Namespace(
            guirl_rollout_master_url="http://rollout-master",
            guirl_probe_agent_type="Holo3_Probe",
            guirl_probe_model_name="holo3-1-35b-a3b",
            guirl_probe_base_url="http://probe/v1",
            guirl_probe_enable_thinking=True,
            guirl_probe_sleep_after_execution=2.0,
        ),
        env_client=object(),
        lease_id="lease-1",
        criteria="1. checked:uncertain",
        instruction="finish task",
        history_n=3,
        example_result_dir=str(tmp_path),
        sampler_round=1,
    )

    _, endpoint, payload, timeout = calls["request"]
    assert endpoint == "/run_probe_round"
    assert timeout == 700.0
    assert payload["lease_id"] == "lease-1"
    assert payload["criteria"] == "1. checked:uncertain"
    assert payload["instruction"] == "finish task"
    assert payload["probe_enable_thinking"] is True
    assert payload["probe_max_steps"] == 15
    assert payload["probe_sleep_after_execution"] == 2.0
    assert result["latest_criteria"] == "1. checked:success"
    assert result["criteria_images"] == [{"option": 1, "images": "evidence-b64"}]

def test_rollout_datasource_uses_remote_master_without_execution_imports(monkeypatch):
    _install_slime_stubs()
    from guirl.master_datasource.data_source import RolloutDataSource

    forbidden = ("agent_runtime", "mm_agents", "eval_runtime.env_adapter")
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(forbidden):
            raise AssertionError(f"unexpected execution-side import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(
        RolloutDataSource,
        "_get_json",
        lambda self, endpoint, timeout=None: {
            "ok": True,
            "master": {"task_count": 1},
        },
    )
    monkeypatch.setattr(
        RolloutDataSource,
        "_post_json",
        lambda self, endpoint, payload, timeout=None: {
            "ok": True,
            "entries": [
                {
                    "task_id": "task-1",
                    "batch_id": "batch-1",
                    "prompt": {"instruction": "do task"},
                    "lease_id": "lease-1",
                    "trajectory": [{"action": "click"}],
                    "screenshots": ["b64"],
                    "S": 1.0,
                    "task_config": {"instruction": "do task"},
                }
            ],
        },
    )

    data_source = RolloutDataSource(Namespace(guirl_rollout_master_url="http://remote"))
    samples = data_source.get_samples(1)

    sample = samples[0][0]
    assert sample.prompt == "do task"
    assert sample.metadata["guirl"]["lease_id"] == "lease-1"
    assert len(data_source) == 1

def test_sync_env_adapter_exposes_prepare_for_probe(slime_custom):
    calls = []

    class _AsyncClient:
        async def prepare_for_probe(self, lease_id):
            calls.append(lease_id)

    async def _run_check():
        loop = asyncio.get_running_loop()
        adapter = slime_custom._SyncEnvClientAdapter(_AsyncClient(), loop)
        await asyncio.to_thread(adapter.prepare_for_probe, "lease-1")

    asyncio.run(_run_check())
    assert calls == ["lease-1"]

def test_rpc_client_prepare_for_probe_posts_endpoint():
    from guirl.env_service.rpc_env_client import AsyncRpcEnvClient

    calls = []
    client = AsyncRpcEnvClient("http://env-master")

    async def _fake_post(endpoint, payload):
        calls.append((endpoint, payload))
        return {"ok": True}

    client._post = _fake_post
    asyncio.run(client.prepare_for_probe("lease-1"))

    assert calls == [("/prepare_for_probe", {"lease_id": "lease-1"})]

def test_post_judge_evaluate_uses_v2_lease_fence(slime_custom):
    guirl = {
        "v2": True,
        "claim_token": "claim-1",
        "claim_generation": 3,
        "attempt": 2,
    }

    kwargs = slime_custom._post_judge_evaluate_kwargs(guirl, "fence-1")

    assert kwargs == {
        "operation_id": "trainer-post-judge-evaluate:claim-1:3:2",
        "fence_token": "fence-1",
    }
    with pytest.raises(RuntimeError, match="requires lease fence token"):
        slime_custom._post_judge_evaluate_kwargs(guirl, None)
    assert slime_custom._post_judge_evaluate_kwargs({}, None) == {}

def _run_v2_custom_generate(
    monkeypatch,
    slime_custom,
    *,
    judge_status,
    max_rounds,
    probe_error=None,
    evaluate_error=None,
    reward_mode="shaped",
    fanout=False,
):
    from slime.utils.types import Sample

    class _State:
        tokenizer = object()
        processor = None

    class _EnvClient:
        async def evaluate(self, lease_id, **kwargs):
            assert lease_id == "lease-1"
            assert kwargs["fence_token"] == "fence-1"
            if evaluate_error is not None:
                raise evaluate_error
            return 1.0

    async def _encode(*args, **kwargs):
        return [10, 11], None, []

    statuses = iter(judge_status) if isinstance(judge_status, list) else None

    async def _judge(**kwargs):
        status = next(statuses) if statuses is not None else judge_status
        return (
            f"<criteria>\n1. done:{status}\n</criteria>\n"
            f"Status: {status}",
            [20, 21],
            [-0.1, -0.2],
            status,
            f"1. done:{status}",
            1,
        )

    async def _probe(**kwargs):
        if probe_error is not None:
            raise probe_error
        return {
            "latest_criteria": "1. done:success",
            "criteria_images": [],
            "round_record": {"steps": 1},
            "probe_agent_type": "test",
        }

    monkeypatch.setattr(slime_custom, "GenerateState", lambda args: _State())
    monkeypatch.setattr(
        slime_custom.mm_preprocess,
        "init_mm_pool",
        lambda args: None,
    )
    monkeypatch.setattr(slime_custom, "_encode_turn_async", _encode)
    monkeypatch.setattr(slime_custom, "_encode_followup_turn_async", _encode)
    monkeypatch.setattr(
        slime_custom,
        "_generate_parseable_judge_response",
        _judge,
    )
    monkeypatch.setattr(slime_custom, "_run_eval_probe_round", _probe)
    monkeypatch.setattr(slime_custom, "_get_env_client", lambda args: _EnvClient())
    sample = Sample(
        index=0,
        group_id=None,
        group_index=0,
        session_id=None,
        generate_function_path=None,
        prompt="finish task",
        metadata={
            "guirl": {
                "v2": True,
                "run_id": "run",
                "rollout_id": 0,
                "task_id": "task",
                "slot": {"task_index": 0, "actor_id": "actor"},
                "actor_id": "actor",
                "attempt": 1,
                "claim_token": "claim-1",
                "claim_generation": 1,
                "lease_id": "lease-1",
                "lease": {
                    "lease_id": "lease-1",
                    "fence_token": "fence-1",
                },
                "S": 1.0,
                "trajectory": [],
                "screenshots": [],
            }
        },
    )
    args = Namespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        guirl_reward_mode=reward_mode,
        guirl_judge_max_rounds=max_rounds,
        guirl_judge_fanout_turns=fanout,
        guirl_judge_strip_history_thinking=False,
    )
    return asyncio.run(
        slime_custom.custom_generate(args, sample, sampling_params={})
    )

def test_probe_infrastructure_failure_aborts_v2_sample(
    monkeypatch,
    slime_custom,
):
    result = _run_v2_custom_generate(
        monkeypatch,
        slime_custom,
        judge_status="uncertain",
        max_rounds=2,
        probe_error=TimeoutError("lost probe response"),
    )
    assert result.status == result.Status.ABORTED
    assert result.metadata["abort_reason"] == "probe_infrastructure_failure"
    assert result.reward == 0.0

def test_post_judge_evaluate_failure_aborts_v2_sample(
    monkeypatch,
    slime_custom,
):
    result = _run_v2_custom_generate(
        monkeypatch,
        slime_custom,
        judge_status="success",
        max_rounds=1,
        evaluate_error=RuntimeError("fenced"),
    )
    assert result.status == result.Status.ABORTED
    assert result.metadata["abort_reason"] == "post_judge_evaluate_failure"
    assert result.reward == 0.0

def test_legitimate_uncertainty_remains_completed_not_infrastructure_failure(
    monkeypatch,
    slime_custom,
):
    result = _run_v2_custom_generate(
        monkeypatch,
        slime_custom,
        judge_status="uncertain",
        max_rounds=1,
    )
    assert result.status == result.Status.COMPLETED
    assert result.metadata["guirl_result"]["judge_final"] == "uncertain"
    assert result.reward == -1.0

def test_rpc_server_prepare_for_probe_endpoint(monkeypatch):
    pytest.importorskip("flask")
    from guirl.env_service import rpc_env_server

    calls = []

    class _Client:
        def prepare_for_probe(self, lease_id):
            calls.append(lease_id)

    monkeypatch.setattr(rpc_env_server, "_ENV_CLIENT", _Client())
    response = rpc_env_server.app.test_client().post(
        "/prepare_for_probe",
        json={"lease_id": "lease-1"},
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == ["lease-1"]

def test_generate_parseable_judge_response_retries_parse_failure(monkeypatch, slime_custom):
    responses = iter(
        [
            {
                "text": "garbled",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [(-0.3, 101)],
                },
            },
            {
                "text": "<criteria>\n1. done:success\n</criteria>\nStatus: success",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [(-0.1, 201), (-0.2, 202)],
                },
            },
        ]
    )

    async def _fake_post(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(slime_custom, "post", _fake_post)

    result = asyncio.run(
        slime_custom._generate_parseable_judge_response(
            sglang_url="http://sglang/generate",
            payload={"input_ids": [1, 2]},
            round_idx=0,
            max_retries=3,
        )
    )

    assert result == (
        "<criteria>\n1. done:success\n</criteria>\nStatus: success",
        [201, 202],
        [-0.1, -0.2],
        "success",
        "1. done:success",
        2,
    )

def test_default_runtime_config_is_aligned():
    config_path = SLIME_ROOT / "guirl" / "scripts" / "guirl_config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_script = (SLIME_ROOT / "guirl" / "scripts" / "train_judge.sh").read_text(
        encoding="utf-8"
    )

    assert cfg["guirl_judge_max_rounds"] == 3
    assert cfg["guirl_judge_temperature"] == 1.0
    assert cfg["guirl_judge_max_tokens"] == 24576
    assert cfg["guirl_judge_enable_thinking"] is True
    assert cfg["guirl_judge_strip_history_thinking"] is True
    assert cfg["guirl_judge_fanout_turns"] is True
    assert cfg["guirl_allow_missing_thinking_template"] is False
    assert cfg["guirl_judge_parse_max_retries"] == 3
    assert cfg["guirl_probe_agent_type"] == "Holo3_Probe"
    assert cfg["guirl_probe_enable_thinking"] is True
    assert re.search(r"--rollout-max-response-len\s+24576", train_script)
    assert re.search(r"--rollout-temperature\s+1.0", train_script)
    assert "--calculate-per-token-loss" in train_script
    assert "--optimizer-cpu-offload" in train_script

@pytest.mark.parametrize(
    "judge_final,S,used_probe,expected",
    [
        ("parse_failure", 1.0, False, -1.0),
        ("uncertain", 1.0, False, -1.0),
        ("success", None, False, -1.0),
        ("success", 1.0, False, 1.0),
        ("failure", 0.0, False, 1.0),
        ("success", 1.0, True, 0.7),
        ("failure", 0.0, True, 0.7),
        ("success", 0.0, False, -0.7),
        ("failure", 1.0, False, -0.7),
        ("success", 0.0, True, -0.3),
        ("failure", 1.0, True, -0.3),
    ],
)
def test_compute_reward_table(judge_final, S, used_probe, expected):
    from guirl.judge.evaluator import compute_reward

    assert compute_reward(judge_final, S, used_probe=used_probe) == expected

def test_sync_probe_timeout_cancels_background_rpc(slime_custom, monkeypatch):
    import concurrent.futures

    future = concurrent.futures.Future()
    monkeypatch.setattr(
        slime_custom.asyncio, "run_coroutine_threadsafe", lambda coro, loop: future,
    )
    adapter = slime_custom._SyncEnvClientAdapter(object(), object(), timeout=0.001)
    with pytest.raises(TimeoutError):
        adapter._run(object())
    assert future.cancelled()

@pytest.mark.parametrize("status,expected", [("success", 1.0), ("failure", 0.0), ("uncertain", 0.0), ("parse_failure", 0.0)])
def test_binary_mode_reaches_generate_and_fallback_rm(monkeypatch, slime_custom, status, expected):
    result = _run_v2_custom_generate(monkeypatch, slime_custom, judge_status=status, max_rounds=1, reward_mode="binary")
    assert result.reward == expected
    assert result.metadata["guirl_result"]["reward_mode"] == "binary"
    result.reward = None
    assert asyncio.run(slime_custom.custom_rm(Namespace(guirl_reward_mode="binary"), result)) == expected

@pytest.mark.parametrize("mode,expected", [("binary", 1.0), ("shaped", 0.7)])
def test_probe_fanout_retains_all_turns_and_final_reward(monkeypatch, slime_custom, mode, expected):
    results = _run_v2_custom_generate(
        monkeypatch, slime_custom, judge_status=["uncertain", "success"],
        max_rounds=3, reward_mode=mode, fanout=True,
    )
    assert len(results) == 2
    for result in results:
        assert result.reward == expected
        assert result.metadata["guirl_result"]["num_rounds"] == 2
        assert len(result.metadata["guirl_result"]["probe_round_records"]) == 1
        assert all(result.loss_mask)
    assert results[1].response_length < len(results[1].tokens)
