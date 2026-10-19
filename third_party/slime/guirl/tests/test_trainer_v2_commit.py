from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SLIME_ROOT = Path(__file__).resolve().parents[2]

class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)

class _TrainRef:
    def __init__(self, rollout_id: int, fail: bool = False):
        self.rollout_id = rollout_id
        self.fail = fail

class _FakeModel:
    def __init__(
        self,
        events: list[str],
        *,
        fail: bool = False,
        save_fail: bool = False,
        role: str = "actor",
    ):
        self.events = events
        self.fail = fail
        self.save_fail = save_fail
        self.role = role

    def async_train(self, rollout_id, rollout_data, external_data=None):
        return _TrainRef(rollout_id, self.fail)

    def update_weights(self):
        self.events.append("weights")

    def clear_memory(self):
        self.events.append("clear")

    def save_model(self, *args, **kwargs):
        assert kwargs["force_sync"] is True
        if self.save_fail:
            raise RuntimeError("checkpoint failed")
        self.events.append(f"checkpoint:{self.role}:{args[0]}")

class _FakeManager:
    def __init__(self, events: list[str]):
        self.events = events
        self.get_metrics_router_addr = _RemoteMethod(lambda: None)
        self.generate = _RemoteMethod(self._generate)
        self.begin_update = _RemoteMethod(self._begin_update)
        self.optimizer_succeeded = _RemoteMethod(self._optimizer_succeeded)
        self.checkpoint_succeeded = _RemoteMethod(self._checkpoint_succeeded)
        self.commit = _RemoteMethod(self._commit)
        self.mark_in_doubt = _RemoteMethod(self._in_doubt)
        self.dispose = _RemoteMethod(lambda: None)
        self.eval = _RemoteMethod(lambda rollout_id: None)
        self.save = _RemoteMethod(lambda rollout_id: None)
        self.onload_weights = _RemoteMethod(lambda: None)
        self.onload_kv = _RemoteMethod(lambda: None)
        self.offload = _RemoteMethod(lambda: None)
        self.check_weights = _RemoteMethod(lambda action: None)

    def _generate(self, rollout_id):
        self.events.append(f"generate:{rollout_id}")
        return f"data:{rollout_id}"

    def _begin_update(self, rollout_id):
        self.events.append(f"optimizing:{rollout_id}")
        return {"state": "OPTIMIZING"}

    def _optimizer_succeeded(self, rollout_id):
        self.events.append(f"optimizer-succeeded:{rollout_id}")
        return {"state": "OPTIMIZER_SUCCEEDED"}

    def _checkpoint_succeeded(self, rollout_id, marker):
        assert marker["force_sync"] is True
        self.events.append(f"checkpointed:{rollout_id}")
        return {"state": "CHECKPOINTED"}

    def _commit(self, rollout_id):
        self.events.append(f"commit:{rollout_id}")
        return {"state": "COMMITTED"}

    def _in_doubt(self, rollout_id, error):
        self.events.append(f"in-doubt:{rollout_id}")
        return {"state": "IN_DOUBT"}

def _load_train_module(monkeypatch, filename: str):
    ray_module = types.ModuleType("ray")
    ray_module.get = lambda value: value
    placement = types.ModuleType("slime.ray.placement_group")
    placement.create_placement_groups = lambda args: {"rollout": object()}
    placement.create_rollout_manager = lambda args, pg: (None, 1)
    placement.create_training_models = lambda args, pgs, manager: (None, None)
    logging_utils = types.ModuleType("slime.utils.logging_utils")
    for name in (
        "configure_logger",
        "finish_tracking",
        "init_tracking",
        "update_tracking_open_metrics",
    ):
        setattr(logging_utils, name, lambda *args, **kwargs: None)
    arguments = types.ModuleType("slime.utils.arguments")
    arguments.parse_args = lambda: None
    misc = types.ModuleType("slime.utils.misc")
    misc.should_run_periodic_action = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", placement)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", arguments)
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc)

    name = f"_guirl_test_{filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, SLIME_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _args(
    *,
    num_rollout: int = 1,
    use_critic: bool = False,
    num_critic_only_steps: int = 0,
):
    return SimpleNamespace(
        colocate=False,
        offload_rollout=False,
        offload_train=True,
        check_weight_update_equal=False,
        num_rollout=num_rollout,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=True,
        use_critic=use_critic,
        num_critic_only_steps=num_critic_only_steps,
        save_interval=None,
        rollout_global_dataset=False,
        update_weights_interval=100,
        guirl_rollout_host_v2=True,
    )

def _wire(
    module,
    events: list[str],
    *,
    train_fails: bool = False,
    checkpoint_fails: bool = False,
    use_critic: bool = False,
):
    manager = _FakeManager(events)
    actor = _FakeModel(
        events,
        fail=train_fails,
        save_fail=checkpoint_fails,
        role="actor",
    )
    critic = (
        _FakeModel(events, role="critic")
        if use_critic
        else None
    )
    module.create_placement_groups = lambda args: {"rollout": object()}
    module.create_rollout_manager = lambda args, pg: (manager, 1)
    module.create_training_models = lambda args, pgs, rm: (actor, critic)
    module.should_run_periodic_action = lambda *args, **kwargs: False

    def fake_get(value):
        if isinstance(value, _TrainRef):
            events.append(f"optimizer:{value.rollout_id}")
            if value.fail:
                raise RuntimeError("optimizer failed")
        return value

    module.ray.get = fake_get
    return manager

def test_sync_commit_happens_only_after_optimizer_success(monkeypatch) -> None:
    module = _load_train_module(monkeypatch, "train.py")
    events: list[str] = []
    _wire(module, events)
    module.train(_args())
    assert events.index("optimizing:0") < events.index("optimizer:0")
    assert events.index("optimizer:0") < events.index("commit:0")
    assert events.index("checkpoint:actor:0") < events.index("checkpointed:0")
    assert events.index("checkpointed:0") < events.index("commit:0")
    assert not any(event.startswith("in-doubt") for event in events)

def test_sync_optimizer_failure_marks_in_doubt_without_commit(monkeypatch) -> None:
    module = _load_train_module(monkeypatch, "train.py")
    events: list[str] = []
    _wire(module, events, train_fails=True)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        module.train(_args())
    assert "in-doubt:0" in events
    assert "commit:0" not in events

def test_v2_checkpoint_failure_marks_in_doubt_without_commit(
    monkeypatch,
) -> None:
    module = _load_train_module(monkeypatch, "train.py")
    events: list[str] = []
    _wire(module, events, checkpoint_fails=True)
    with pytest.raises(RuntimeError, match="checkpoint failed"):
        module.train(_args())
    assert "optimizer-succeeded:0" in events
    assert "in-doubt:0" in events
    assert "checkpointed:0" not in events
    assert "commit:0" not in events

def test_v2_checkpoints_only_models_updated_in_each_step(monkeypatch) -> None:
    module = _load_train_module(monkeypatch, "train.py")
    events: list[str] = []
    _wire(module, events, use_critic=True)
    module.train(
        _args(
            num_rollout=2,
            use_critic=True,
            num_critic_only_steps=1,
        )
    )
    assert "checkpoint:actor:0" not in events
    assert events.index("checkpoint:critic:0") < events.index(
        "checkpointed:0"
    )
    assert events.index("checkpoint:actor:1") < events.index(
        "checkpointed:1"
    )
    assert events.index("checkpoint:critic:1") < events.index(
        "checkpointed:1"
    )

def test_v2_async_never_prefetches_before_current_commit(monkeypatch) -> None:
    module = _load_train_module(monkeypatch, "train_async.py")
    events: list[str] = []
    _wire(module, events)
    module.train(_args(num_rollout=2))
    assert events.index("commit:0") < events.index("generate:1")
    assert events.index("optimizer:1") < events.index("commit:1")

def test_rollout_manager_v2_has_hard_driver_node_affinity(
    monkeypatch,
) -> None:
    actor_group = types.ModuleType("slime.ray.actor_group")
    actor_group.RayTrainGroup = object
    rollout = types.ModuleType("slime.ray.rollout")

    class _RolloutManager:
        pass

    rollout.RolloutManager = _RolloutManager
    monkeypatch.setitem(sys.modules, "slime.ray.actor_group", actor_group)
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", rollout)
    spec = importlib.util.spec_from_file_location(
        "slime.ray._guirl_affinity_test",
        SLIME_ROOT / "slime/ray/placement_group.py",
    )
    placement_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(placement_module)

    captured: list[dict] = []

    class _Options:
        def __init__(self, options):
            self.options = options

        def remote(self, args, pg):
            del args, pg
            return object()

    monkeypatch.setattr(
        placement_module.RolloutManager,
        "options",
        lambda **options: (
            captured.append(options) or _Options(options)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        placement_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "01" * 28),
    )

    args = SimpleNamespace(
        guirl_rollout_host_v2=True,
        num_rollout=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )
    placement_module.create_rollout_manager(args, object())
    strategy = captured[-1]["scheduling_strategy"]
    assert strategy.node_id == "01" * 28
    assert strategy.soft is False
    assert args.guirl_v2_trainer_node_id == "01" * 28

    legacy = SimpleNamespace(
        guirl_rollout_host_v2=False,
        num_rollout=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )
    placement_module.create_rollout_manager(legacy, object())
    assert "scheduling_strategy" not in captured[-1]
