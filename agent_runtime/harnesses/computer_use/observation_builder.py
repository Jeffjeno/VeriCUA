from __future__ import annotations

import base64
from typing import Any

from agent_runtime.types import ModalityKind, Observation

def _image_data_url(image: bytes | str) -> str:
    if isinstance(image, str):

        if image.startswith("data:"):
            return image
        return f"data:image/png;base64,{image}"
    payload = base64.b64encode(image).decode("ascii")
    return f"data:image/png;base64,{payload}"

def initial_input(observation: Observation, instruction: str | None) -> list[dict[str, Any]]:

    parts: list[dict[str, Any]] = []
    text = instruction or observation.instruction
    if text:
        parts.append({"type": "input_text", "text": text})
    image_part = observation.first(ModalityKind.IMAGE)
    if image_part is not None and image_part.content is not None:
        parts.append({"type": "input_image", "image_url": _image_data_url(image_part.content)})
    return [{"role": "user", "content": parts}]

def computer_call_output(
    *,
    call_id: str,
    observation: Observation,
    acknowledged_safety_checks: list[dict[str, Any]] | None = None,
    current_url: str | None = None,
) -> list[dict[str, Any]]:

    image_part = observation.first(ModalityKind.IMAGE)
    if image_part is None or image_part.content is None:

        output: dict[str, Any] = {
            "type": "computer_screenshot",
            "image_url": "",
        }
    else:
        output = {
            "type": "computer_screenshot",
            "image_url": _image_data_url(image_part.content),
        }
    if current_url is not None:
        output["current_url"] = current_url

    item: dict[str, Any] = {
        "type": "computer_call_output",
        "call_id": call_id,
        "output": output,
    }
    if acknowledged_safety_checks:
        item["acknowledged_safety_checks"] = acknowledged_safety_checks
    return [item]

__all__ = ["initial_input", "computer_call_output"]
