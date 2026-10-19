from __future__ import annotations

import copy
from typing import Any

def extract_logprobs_data(response: dict[str, Any]) -> dict[str, Any] | None:

    choices = response.get("choices")
    if not choices:
        return None

    logprobs_obj = choices[0].get("logprobs")
    if not logprobs_obj:
        return None

    content_list = logprobs_obj.get("content")
    if not content_list:
        return None

    tokens: list[dict[str, Any]] = []
    total = 0.0
    minimum = float("inf")

    for token_info in content_list:
        lp = token_info.get("logprob")
        if lp is None:
            continue
        tokens.append({"token": token_info.get("token", ""), "logprob": lp})
        total += lp
        minimum = min(minimum, lp)

    n = len(tokens)
    if n == 0:
        return None

    return {
        "summary": {
            "sequence_logprob": total,
            "mean_token_logprob": total / n,
            "min_token_logprob": minimum,
            "num_generated_tokens": n,
        },
        "tokens": tokens,
    }

def sanitize_messages_for_save(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    sanitized = copy.deepcopy(messages)
    image_index = 0

    for msg in sanitized:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            if isinstance(url, str) and url.startswith("data:"):
                content[i] = {"type": "image_ref", "image_index": image_index}
                image_index += 1

    return sanitized
