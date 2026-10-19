from __future__ import annotations

import base64
import logging
import os
import re
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable

from agent_runtime.harness import Harness
from agent_runtime.types import Action, HarnessContext, ModalityKind, Observation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MOBILE_WORLD_SRC = os.environ.get("MOBILE_WORLD_SRC", "")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def setup_mobile_world_imports(src_path: str | None = None) -> None:

    import sys

    target = src_path or _MOBILE_WORLD_SRC
    try:
        import mobile_world
    except ImportError:
        if target and target not in sys.path:
            sys.path.insert(0, target)

def strip_thinking(text: str) -> str:

    return _THINK_RE.sub("", text).strip()

def _b64_to_pil(b64_str: str) -> Any:

    from PIL import Image

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[-1]
    raw = base64.b64decode(b64_str)
    return Image.open(BytesIO(raw)).convert("RGB")

def _observation_to_mw(obs: Observation) -> dict[str, Any]:

    screenshot = None
    ask_user_response = None
    tool_call = None

    for part in obs.parts:
        if part.kind is ModalityKind.IMAGE:
            if part.content and isinstance(part.content, str):
                screenshot = _b64_to_pil(part.content)
            elif part.uri:
                from PIL import Image
                screenshot = Image.open(part.uri).convert("RGB")
        elif part.kind is ModalityKind.TEXT and part.name == "ask_user_response" and part.content:
            ask_user_response = str(part.content)
        elif part.kind is ModalityKind.TOOL_RESULT and part.name == "tool_call" and part.content:
            tool_call = part.content

    return {
        "screenshot": screenshot,
        "tool_call": tool_call,
        "ask_user_response": ask_user_response,
    }

def _mw_native_to_action(json_action: Any, raw: str | None = None) -> Action:

    from agent_runtime.benchmarks.mobile_world import native_to_action
    native_dict = json_action.model_dump(exclude_none=True)
    return native_to_action(native_dict, raw=raw)

def _aw_native_to_action(json_action: Any, raw: str | None = None) -> Action:

    from agent_runtime.benchmarks.android_world import native_to_action
    native_dict = json_action.model_dump(exclude_none=True)
    return native_to_action(native_dict, raw=raw)

def _patch_strip_thinking(agent: Any) -> None:

    original = agent.openai_chat_completions_create

    def patched(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if isinstance(result, str):
            return strip_thinking(result)
        return result

    agent.openai_chat_completions_create = patched

class UpstreamAgentHarness(Harness):

    def __init__(
        self,
        agent: Any,
        *,
        action_converter: Callable[[Any, str | None], Action] = _mw_native_to_action,
        strip_think_tags: bool = True,
    ) -> None:
        self._agent = agent
        self._initialized = False
        self._action_converter = action_converter
        if strip_think_tags and hasattr(agent, "openai_chat_completions_create"):
            _patch_strip_thinking(agent)

    @property
    def agent(self) -> Any:
        return self._agent

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        if not self._initialized:
            instruction = observation.instruction or ""
            if context and context.task and context.task.instruction:
                instruction = context.task.instruction
            self._agent.initialize(instruction)
            self._initialized = True

        mw_obs = _observation_to_mw(observation)

        if mw_obs["screenshot"] is None:
            logger.warning("No screenshot found in observation; sending NOOP")
            return Action.noop()

        raw_response, json_action = self._agent.predict(mw_obs)

        if raw_response is None:
            logger.warning("Agent returned None response; sending TERMINATE")
            return Action.terminate(reason="agent prediction failed")

        action_type = getattr(json_action, "action_type", None)
        if action_type == "unknown" or action_type is None:
            logger.warning("Agent returned unknown/unparseable action; sending TERMINATE")
            return Action.terminate(reason="agent returned unparseable action", raw=raw_response)

        return self._action_converter(json_action, raw_response)

    def reset(self) -> None:

        if self._initialized:
            try:
                self._agent.done()
            except Exception:
                logger.debug("agent.done() raised; ignoring", exc_info=True)
            self._initialized = False

    def get_token_usage(self) -> dict[str, int]:

        if hasattr(self._agent, "get_total_token_usage"):
            return self._agent.get_total_token_usage()
        return {}

class MobileWorldAgentHarness(UpstreamAgentHarness):

    def __init__(self, agent: Any, *, strip_think_tags: bool = True) -> None:
        super().__init__(
            agent,
            action_converter=_mw_native_to_action,
            strip_think_tags=strip_think_tags,
        )

class AndroidWorldAgentHarness(UpstreamAgentHarness):

    def __init__(self, agent: Any, *, strip_think_tags: bool = True) -> None:
        super().__init__(
            agent,
            action_converter=_aw_native_to_action,
            strip_think_tags=strip_think_tags,
        )

__all__ = [
    "AndroidWorldAgentHarness",
    "MobileWorldAgentHarness",
    "UpstreamAgentHarness",
    "setup_mobile_world_imports",
    "strip_thinking",
]
