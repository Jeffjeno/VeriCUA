from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eval_runtime import run_replay_eval as replay
from eval_runtime.preflight import check_endpoints, prepare
from scripts.compute_replay_eval_majority_consistency import compute
from scripts.replay_eval_results import collect_results


def worker_recovers(connection, args):
    connection.send({"type": "ready"})
    while (task := connection.recv()) is not None:
        marker = Path(args.result_dir) / "crashed"
        if not marker.exists():
            marker.touch()
            os._exit(137)
        connection.send({"type": "result", "task": task,
                         "result": {"score": 1, "label": "success", "error": None}, "scores": [1]})


def worker_dies(connection, args):
    connection.send({"type": "ready"})
    connection.recv()
    os._exit(137)


def worker_init_fails(connection, args):
    os._exit(1)


def args_for(root):
    return SimpleNamespace(
        result_dir=str(root), action_space="pyautogui", observation_type="screenshot",
        policy_model_name="actor", policy_config_path="", num_envs=1,
        policy_config_inline={"source_result_root": "trajectories", "source_rollout_repeat": 4},
        guirl_config={"judge_model_name": "judge"}, max_worker_restarts=2, max_task_attempts=2,
    )


@pytest.mark.skipif(multiprocessing.get_start_method() != "fork", reason="fault injection requires fork")
@pytest.mark.parametrize("worker,success", [(worker_recovers, True), (worker_dies, False), (worker_init_fails, False)])
def test_supervisor_handles_worker_death(tmp_path, monkeypatch, worker, success):
    monkeypatch.setattr(replay, "run_env_tasks", worker)
    args = args_for(tmp_path)
    scores, labels, failures = replay._run_multiprocessing([("d", "t", "0")], args)
    assert bool(failures) is not success
    assert not replay.processes
    if success:
        assert scores == [1] and labels == ["success"]
    else:
        path = Path(replay._eval_summary_path(replay._task_result_dir(args, ("d", "t", "0"))))
        assert json.loads(path.read_text())["final_label"] == "error"


def test_early_task_error_is_persisted_and_retried_on_resume(tmp_path):
    args = args_for(tmp_path)
    args.test_config_base_dir = str(tmp_path / "absent")
    args.examples_subdir = "examples"
    result = replay._execute_single_task(None, None, "d", "t", "0", args, [])
    assert result["error"]
    path = Path(replay._eval_summary_path(replay._task_result_dir(args, ("d", "t", "0"))))
    assert json.loads(path.read_text())["final_label"] == "error"
    (path.parent.parent / "result.txt").write_text("1")
    assert replay.get_unfinished("pyautogui", "actor", "screenshot", str(tmp_path), {"d": ["t"]}, 1)


def write_results(root, modifications=None, expected_tasks=("t",)):
    tasks = []
    for task in expected_tasks:
        for index in range(4):
            identity = {"domain": "d", "example_id": task, "rollout_id": str(index),
                        "policy_model": "actor", "eval_model": "judge", "run_id": "run",
                        "source_trajectory": f"source/{task}/0"}
            tasks.append(identity)
            if task != "t":
                continue
            summary = dict(identity, policy_result=1, final_label="success")
            if modifications:
                modifications(summary, index)
            path = root / "pyautogui" / "screenshot" / "actor" / "d" / task / str(index) / "replay_eval" / "eval_summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary))
    (root / "eval_manifest.json").write_text(json.dumps({"tasks": tasks}))


def test_valid_vote_and_missing_task_denominator(tmp_path):
    write_results(tmp_path, expected_tasks=("t", "missing"))
    result = compute(tmp_path, 4)
    assert result["strict_rate"] == 0.5
    assert result["task_groups_incomplete"] == 1
    assert result["accounting"]["expected"] == 8
    assert result["accounting"]["missing"] == 4
    assert not result["valid"]


def test_valid_vote_and_raw_score_cli(tmp_path):
    write_results(tmp_path)
    assert compute(tmp_path, 4)["valid"]
    script = Path(__file__).parents[1] / "scripts" / "compute_replay_eval_reward_consistency.py"
    result = subprocess.run([sys.executable, str(script), str(tmp_path), "--json"], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["overall"]["strict_rate"] == 1
    path = next(tmp_path.rglob("eval_summary.json"))
    path.unlink()
    result = subprocess.run([sys.executable, str(script), str(tmp_path), "--json"], capture_output=True, text=True)
    assert result.returncode == 2
    output = json.loads(result.stdout)
    assert output["overall"]["total"] == 4
    assert output["overall"]["strict_rate"] == 0.75


def test_main_exits_nonzero_and_writes_accounting_for_missing_results(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"d":["t"]}')
    policy = tmp_path / "policy.yaml"
    policy.write_text('agent_type: Replay\nsource_result_root: trajectories\nrun_name: actor\n')
    monkeypatch.setattr(sys, "argv", ["replay", "--osworld_path", str(tmp_path),
        "--project_dir", str(tmp_path), "--policy_config_path", str(policy),
        "--test_all_meta_path", str(manifest), "--test_config_base_dir", str(tmp_path),
        "--result_dir", str(tmp_path / "results")])
    monkeypatch.setattr(replay, "cleanup_docker_containers", lambda *_: 0)
    monkeypatch.setattr(replay, "_apply_osworld_env", lambda *_: None)
    monkeypatch.setattr(replay, "setup_logging", lambda *_: None)
    monkeypatch.setattr(replay.signal, "signal", lambda *_: None)
    monkeypatch.setattr(replay, "_run_multiprocessing", lambda *_: ([], [], []))
    with pytest.raises(SystemExit) as exc:
        replay.main()
    assert exc.value.code == 1
    accounting = json.loads((tmp_path / "results" / "eval_run_summary.json").read_text())
    assert accounting["expected"] == accounting["missing"] == 1


@pytest.mark.parametrize("field,values", [
    ("policy_result", [1, 0, 0, 0]),
    ("eval_model", ["a", "a", "b", "b"]),
    ("run_id", ["a", "a", "b", "b"]),
    ("source_trajectory", ["a", "a", "b", "b"]),
    ("rollout_id", ["0", "1", "0", "1"]),
])
def test_invalid_vote_groups_are_rejected(tmp_path, field, values):
    write_results(tmp_path, lambda summary, i: summary.update({field: values[i]}))
    result = compute(tmp_path, 4)
    assert not result["valid"]
    assert result["strict_rate"] != 1


def test_error_with_missing_gt_stays_in_expected_count(tmp_path):
    write_results(tmp_path, lambda summary, i: summary.update(final_label="error", policy_result=None) if i == 0 else None)
    _, accounting = collect_results(tmp_path)
    assert accounting["expected"] == 4
    assert accounting["completed"] == 3 and accounting["failed"] == 1
    assert not compute(tmp_path, 4)["valid"]


def test_preflight_validates_data_and_resolves_nested_model(tmp_path):
    data = tmp_path / "data"
    (data / "examples" / "d").mkdir(parents=True)
    (data / "examples" / "d" / "t.json").write_text('{"instruction":"test"}')
    trajectory = data / "trajectories" / "d" / "t" / "0"
    trajectory.mkdir(parents=True)
    (trajectory / "traj.jsonl").write_text('{}\n')
    (data / "manifest.json").write_text('{"d":["t"]}')
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"paths": {"test_all_meta_path": "data/manifest.json",
        "test_config_base_dir": "data", "source_result_root": "data/trajectories"},
        "policy": {"source_rollout_repeat": 4}, "rollout": {"task_n_combinations": 4}}))
    config = tmp_path / "judge.yaml"
    config.write_text(yaml.safe_dump({"model": {"base_url": "http://old/v1", "model_name": "old"},
                                    "probe_base_url": "http://probe/v1"}))
    vm = tmp_path / "vm.qcow2"
    vm.touch()
    args = SimpleNamespace(project_dir=str(tmp_path), dataset=str(dataset), domain="all", rollouts=4,
        config=str(config), output=str(tmp_path / "effective.yaml"), vm=str(vm),
        judge_url="http://judge:41000/v1", model="new", probe_url="http://probe:42000/v1/models")
    prepare(args)
    effective = yaml.safe_load(Path(args.output).read_text())
    assert effective["model"]["base_url"] == effective["base_url"] == args.judge_url
    assert effective["model"]["model_name"] == "new"
    assert effective["probe_base_url"] == "http://probe:42000/v1"
    (trajectory / "traj.jsonl").unlink()
    with pytest.raises(ValueError, match="missing trajectory"):
        prepare(args)


def test_endpoint_check_uses_effective_urls_and_model_names(tmp_path, monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "probe"}, {"id": "judge"}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    config = tmp_path / "effective.yaml"
    base = f"http://127.0.0.1:{server.server_port}"
    cfg = {"base_url": base + "/judge/v1", "model_name": "judge",
           "probe_base_url": base + "/probe/v1", "probe_model_name": "probe"}
    try:
        config.write_text(yaml.safe_dump(cfg))
        check_endpoints(config)
        assert calls == ["/probe/v1/models", "/judge/v1/models"]
        cfg["model_name"] = "wrong"
        config.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="not serving wrong"):
            check_endpoints(config)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_launcher_rejects_missing_data_before_starting_gpu_services(tmp_path):
    root = Path(__file__).parents[1]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").touch()
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_CALL_LOG"\nexit 0\n')
    docker.chmod(0o755)
    log = tmp_path / "docker_calls"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               DOCKER_CALL_LOG=str(log), PYTHON_BIN=sys.executable,
               DATASET_CONFIG=str(tmp_path / "absent.yaml"), RESULT_ROOT=str(tmp_path / "results"))
    result = subprocess.run(["bash", str(root / "suites/osworld_judge/run.sh"), str(checkpoint)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert log.read_text().splitlines() == ["info"]
