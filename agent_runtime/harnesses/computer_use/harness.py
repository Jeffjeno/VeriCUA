from __future__ import annotations

import logging
from typing import Any, Literal

from agent_runtime.harness import Harness
from agent_runtime.types import Action, ActionType, HarnessContext, Observation

logger = logging.getLogger(__name__)

ProviderName = Literal["openai"]
ChainMode = Literal["stateless", "stateful"]

class ComputerUseHarness(Harness):

    def __init__(
        self,
        *,
        model: Any,
        provider: ProviderName = "openai",
        chain_mode: ChainMode = "stateless",
        auto_ack_safety: bool = True,
        instructions: str | None = None,
        truncation: str | None = "auto",
    ) -> None:
        if provider != "openai":
            raise NotImplementedError(
                f"ComputerUseHarness provider={provider!r} not implemented yet; "
                "only 'openai' is available."
            )
        self.model = model
        self.provider = provider
        self.chain_mode = chain_mode
        self.auto_ack_safety = auto_ack_safety
        self.instructions = instructions
        self.truncation = truncation

        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None
        self._pending_safety_checks: list[dict[str, Any]] = []

        self._stateless_input: list[dict[str, Any]] = []

    def act(self, observation: Observation, context: HarnessContext | None = None) -> Action:
        if self.provider == "openai":
            return self._act_openai(observation, context)
        raise NotImplementedError(self.provider)

    def _act_openai(self, observation: Observation, context: HarnessContext | None) -> Action:
        from agent_runtime.harnesses.computer_use import (
            action_translator,
            observation_builder,
        )
        from agent_runtime.harnesses.computer_use.providers.openai import (
            build_openai_tool,
        )

        if observation.step_index == 0:
            self._reset_chain()

        instruction = (
            (context.task.instruction if context and context.task else None)
            or observation.instruction
            or self.instructions
        )

        if self.chain_mode == "stateful":
            input_items, previous_id = self._stateful_input(
                observation, instruction, observation_builder
            )
        else:
            input_items, previous_id = self._stateless_input_for_turn(
                observation, instruction, observation_builder
            )

        result = self.model.responses_create(
            input=input_items,
            tools=[build_openai_tool()],
            previous_response_id=previous_id,
            truncation=self.truncation,
        )
        self._previous_response_id = result.response_id

        calls = result.computer_calls()
        if not calls:
            messages = result.messages()
            self._pending_call_id = None
            self._pending_safety_checks = []
            if messages:
                text = _text_from_message_blocks(messages)
                return Action.message(text or "(model returned no action)")
            return Action.terminate(reason="model_returned_no_call")

        call = calls[0]
        action = action_translator.openai_to_agent_runtime(call)
        self._pending_call_id = action_translator.call_id_of(action)
        self._pending_safety_checks = action_translator.acknowledged_safety_checks(action)

        if self.chain_mode == "stateless":
            self._stateless_input.append(_strip_computer_call(call))

        return action

    def _stateful_input(
        self,
        observation: Observation,
        instruction: str | None,
        observation_builder,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._previous_response_id is None or self._pending_call_id is None:
            return observation_builder.initial_input(observation, instruction), None
        acked = self._pending_safety_checks if self.auto_ack_safety else []
        items = observation_builder.computer_call_output(
            call_id=self._pending_call_id,
            observation=observation,
            acknowledged_safety_checks=acked,
        )
        return items, self._previous_response_id

    def _stateless_input_for_turn(
        self,
        observation: Observation,
        instruction: str | None,
        observation_builder,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not self._stateless_input:
            self._stateless_input = list(
                observation_builder.initial_input(observation, instruction)
            )
            return list(self._stateless_input), None

        acked = self._pending_safety_checks if self.auto_ack_safety else []
        if self._pending_call_id is not None:
            self._stateless_input.extend(
                observation_builder.computer_call_output(
                    call_id=self._pending_call_id,
                    observation=observation,
                    acknowledged_safety_checks=acked,
                )
            )
        return list(self._stateless_input), None

    def _reset_chain(self) -> None:
        self._previous_response_id = None
        self._pending_call_id = None
        self._pending_safety_checks = []
        self._stateless_input = []

    def reset(self) -> None:

        self._reset_chain()

def _text_from_message_blocks(messages: list[dict[str, Any]]) -> str | None:
    chunks: list[str] = []
    for block in messages:
        for part in block.get("content", []) or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks) if chunks else None

def _strip_computer_call(call: dict[str, Any]) -> dict[str, Any]:

    keep = {"type", "call_id", "actions", "action", "pending_safety_checks"}
    return {k: v for k, v in call.items() if k in keep}

__all__ = ["ComputerUseHarness"]
