from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

_logger = logging.getLogger(__name__)

class ModalityKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    JSON = "json"
    HTML = "html"
    A11Y_TREE = "a11y_tree"
    DOM = "dom"
    SCREEN_BBOX = "screen_bbox"
    TOOL_RESULT = "tool_result"
    FILE = "file"
    OTHER = "other"

class Modality(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ModalityKind
    content: Any | None = None
    uri: str | None = None
    mime_type: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):

    model_config = ConfigDict(extra="forbid")

    parts: list[Modality] = Field(default_factory=list)
    instruction: str | None = None
    available_actions: list[str] | None = None
    step_index: int = 0
    terminal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, text: str, *, instruction: str | None = None, **kwargs: Any) -> Observation:
        return cls(
            parts=[Modality(kind=ModalityKind.TEXT, content=text)],
            instruction=instruction,
            **kwargs,
        )

    @classmethod
    def screenshot(
        cls,
        image: bytes | str,
        *,
        a11y_tree: Any | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        parts: list[Modality] = [
            Modality(kind=ModalityKind.IMAGE, content=image, mime_type="image/png"),
        ]
        if a11y_tree is not None:
            parts.append(Modality(kind=ModalityKind.A11Y_TREE, content=a11y_tree))
        return cls(parts=parts, instruction=instruction, **kwargs)

    def first(self, kind: ModalityKind | str) -> Modality | None:
        target = ModalityKind(kind) if isinstance(kind, str) else kind
        for part in self.parts:
            if part.kind is target:
                return part
        return None

    def text_content(self) -> str | None:
        chunks = [
            str(p.content)
            for p in self.parts
            if p.kind is ModalityKind.TEXT and p.content is not None
        ]
        return "\n".join(chunks) if chunks else None

    def to_model_input(self) -> dict[str, Any]:

        return {
            "parts": [p.model_dump() for p in self.parts],
            "instruction": self.instruction,
        }

class ActionType(str, Enum):
    TOOL_CALL = "tool_call"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY_PRESS = "key_press"
    HOTKEY = "hotkey"
    HOVER = "hover"
    SCREENSHOT = "screenshot"
    SWIPE = "swipe"
    TAP = "tap"
    LONG_PRESS = "long_press"
    BASH = "bash"
    PYTHON = "python"
    PATCH = "patch"
    MESSAGE = "message"
    ASK_USER = "ask_user"
    NOOP = "noop"
    TERMINATE = "terminate"
    SUBMIT = "submit"
    CUSTOM = "custom"

class Action(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = Field(
        default=None,
        description="Original model output for replay.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Action:
        if self.type is ActionType.TOOL_CALL and not self.name:
            raise ValueError("TOOL_CALL action requires a non-empty `name`.")
        if self.type is ActionType.CUSTOM and not self.name:
            raise ValueError("CUSTOM action requires a `name` discriminator.")
        return self

    @classmethod
    def tool_call(cls, name: str, arguments: dict[str, Any] | None = None, **kw: Any) -> Action:
        return cls(type=ActionType.TOOL_CALL, name=name, arguments=arguments or {}, **kw)

    @classmethod
    def message(cls, text: str, **kw: Any) -> Action:
        return cls(type=ActionType.MESSAGE, arguments={"text": text}, **kw)

    @classmethod
    def submit(cls, answer: Any | None = None, **kw: Any) -> Action:
        args: dict[str, Any] = {}
        if answer is not None:
            args["answer"] = answer
        return cls(type=ActionType.SUBMIT, arguments=args, **kw)

    @classmethod
    def terminate(cls, reason: str | None = None, **kw: Any) -> Action:
        args: dict[str, Any] = {}
        if reason is not None:
            args["reason"] = reason
        return cls(type=ActionType.TERMINATE, arguments=args, **kw)

    @classmethod
    def noop(cls, **kw: Any) -> Action:
        return cls(type=ActionType.NOOP, **kw)

    @classmethod
    def patch(cls, diff: str, **kw: Any) -> Action:
        return cls(type=ActionType.PATCH, arguments={"diff": diff}, **kw)

    @classmethod
    def bash(cls, command: str, **kw: Any) -> Action:
        return cls(type=ActionType.BASH, arguments={"command": command}, **kw)

    @classmethod
    def click(cls, x: int, y: int, **kw: Any) -> Action:
        return cls(type=ActionType.CLICK, arguments={"x": x, "y": y}, **kw)

    @classmethod
    def type_text(cls, text: str, **kw: Any) -> Action:
        return cls(type=ActionType.TYPE, arguments={"text": text}, **kw)

    @classmethod
    def custom(cls, name: str, arguments: dict[str, Any] | None = None, **kw: Any) -> Action:
        return cls(type=ActionType.CUSTOM, name=name, arguments=arguments or {}, **kw)

    @classmethod
    def from_model_response(cls, response: Any) -> Action:

        if isinstance(response, Action):
            return response
        if isinstance(response, dict):
            return cls.model_validate(response)
        if isinstance(response, str):
            return cls.message(response)
        raise TypeError(f"Cannot convert {type(response)} to Action")

class Task(BaseModel):

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark: str
    suite: str | None = None
    instruction: str | None = None
    native_ref: dict[str, Any] | None = Field(
        default=None,
        description="Opaque reference back to the native benchmark task.",
    )
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    answer: Any | None = None
    max_steps: int | None = None
    tags: list[str] = Field(default_factory=list)
    env_setup: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Hooks for environment setup scripts and file provisioning. "
            "Keys: 'pre_task' (str, script path run after reset, before agent), "
            "'post_task' (str, script path run after agent, before evaluation), "
            "'files' (list[dict], files to upload from host to VM before hooks. "
            "Each entry: {'local_path': str, 'vm_path': str})."
        ),
    )

class HarnessContext(BaseModel):

    model_config = ConfigDict(extra="forbid")

    history: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    task: Task | None = None
    step_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

class EvaluationEvent(BaseModel):

    model_config = ConfigDict(extra="forbid")

    phase: str = "step"
    kind: str = "progress"
    score: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class EvaluationResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    success: bool | None = None
    score: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    error_type: str | None = None
    native_result: dict[str, Any] | None = None
    events: list[EvaluationEvent] = Field(default_factory=list)

class TraceEvent(BaseModel):

    model_config = ConfigDict(extra="forbid")

    step_index: int = 0
    kind: str = "step"
    observation: Observation | None = None
    action: Action | None = None
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class Trace(BaseModel):

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    benchmark: str = ""
    events: list[TraceEvent] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _event_listeners: list[Callable[[TraceEvent], None]] = PrivateAttr(
        default_factory=list
    )

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def evaluation_events(self) -> list[EvaluationEvent]:
        return self.evaluation.events if self.evaluation else []

    def add_event_listener(self, fn: Callable[[TraceEvent], None]) -> None:

        self._event_listeners.append(fn)

    def remove_event_listener(self, fn: Callable[[TraceEvent], None]) -> None:

        import contextlib as _ctx
        with _ctx.suppress(ValueError):
            self._event_listeners.remove(fn)

    def _fire_listeners(self, event: TraceEvent) -> None:
        for fn in list(self._event_listeners):
            try:
                fn(event)
            except Exception:
                _logger.exception(
                    "Trace event listener raised; swallowing to keep episode alive",
                )

    def record_event(self, event: TraceEvent) -> None:
        self.events.append(event)
        self._fire_listeners(event)

    def record_step(
        self,
        observation: Observation,
        action: Action | None = None,
        reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            step_index=len(self.events),
            kind="step",
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )
        self.events.append(event)
        self._fire_listeners(event)
        return event

    def record_evaluation(self, evaluation: EvaluationResult) -> None:
        self.evaluation = evaluation

    def finalize(self) -> None:
        self.ended_at = time.time()

    def save(self, path: str | Path) -> None:

        from agent_runtime.storage.writer import TraceWriter

        p = Path(path)
        run_id = p.name
        with TraceWriter(p, run_id=run_id) as w:
            w.write(self)

class RunTaskResult(BaseModel):

    model_config = ConfigDict(extra="forbid")

    trace: Trace
    evaluation: EvaluationResult | None = None
    native_result: Any | None = None

    @classmethod
    def failed(cls, task_id: str, error: Exception) -> RunTaskResult:
        trace = Trace(task_id=task_id)
        trace.finalize()
        evaluation = EvaluationResult(
            success=False,
            score=0.0,
            error_type=f"{type(error).__module__}.{type(error).__name__}",
            reason=str(error),
        )
        return cls(trace=trace, evaluation=evaluation)

class RunSummary(BaseModel):

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    total_tasks: int = 0
    succeeded: int = 0
    failed: int = 0
    error_count: int = 0
    avg_score: float | None = None
    success_rate: float | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update(self, result: RunTaskResult) -> None:
        self.total_tasks += 1
        if result.evaluation and result.evaluation.success:
            self.succeeded += 1
        elif result.evaluation and result.evaluation.error_type:
            self.error_count += 1
            self.failed += 1
        else:
            self.failed += 1
        if self.total_tasks > 0:
            self.success_rate = self.succeeded / self.total_tasks

class EnvCapabilities(BaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: bool = False
    native_runner: bool = False
    checkpoint: bool = False
    screenshot: bool = False
    a11y_tree: bool = False

@dataclass(frozen=True)
class JudgeConfig:

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512

    def resolve(
        self,
        default_model: str = "",
        default_base_url: str = "",
        default_api_key: str = "",
    ) -> tuple[str, str, str]:

        model = self.model or os.environ.get("AP_JUDGE_MODEL") or default_model
        base_url = self.base_url or os.environ.get("AP_JUDGE_BASE_URL") or default_base_url
        api_key = self.api_key or os.environ.get("AP_JUDGE_API_KEY") or default_api_key
        return model, base_url, api_key

__all__ = [
    "Action",
    "ActionType",
    "EnvCapabilities",
    "EvaluationEvent",
    "EvaluationResult",
    "HarnessContext",
    "JudgeConfig",
    "Modality",
    "ModalityKind",
    "Observation",
    "RunSummary",
    "RunTaskResult",
    "Task",
    "Trace",
    "TraceEvent",
]
