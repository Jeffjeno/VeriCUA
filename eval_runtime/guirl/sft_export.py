from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("eval_runtime.guirl.sft_export")

def _strip_base64_images(content: Any) -> Any:

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                if url.startswith("data:image/"):
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": "<base64_image_placeholder>"},
                    })
                else:
                    result.append(item)
            else:
                result.append(item)
        return result
    return content

def export_sft_sample(
    messages: list[dict[str, Any]],
    *,
    output_path: str,
    example_id: str = "",
    domain: str = "",
    reward: float = 1.0,
    extra_metadata: dict[str, Any] | None = None,
    strip_images: bool = False,
) -> None:

    if strip_images:
        exported_messages = []
        for msg in messages:
            exported_messages.append({
                "role": msg["role"],
                "content": _strip_base64_images(msg.get("content", "")),
            })
    else:
        exported_messages = messages

    record = {
        "messages": exported_messages,
        "metadata": {
            "example_id": example_id,
            "domain": domain,
            "reward": reward,
            **(extra_metadata or {}),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("SFT sample exported: example_id=%s, reward=%.2f", example_id, reward)

def maybe_export_sft_from_eval_summary(
    summary: dict[str, Any],
    *,
    output_dir: str,
    reward_threshold: float = 1.0,
    strip_images: bool = False,
) -> bool:

    messages = summary.get("messages")
    if not messages:
        return False

    final_status = summary.get("final_judge_status", "failure")
    policy_env_success = summary.get("policy_env_success", False)
    post_probe_env_success = summary.get("post_probe_env_success")

    if final_status != "success":
        return False
    if not policy_env_success:
        return False
    if post_probe_env_success is not None and not post_probe_env_success:
        return False

    output_path = os.path.join(output_dir, "guirl_sft_data.jsonl")
    export_sft_sample(
        messages,
        output_path=output_path,
        example_id=summary.get("example_id", ""),
        domain=summary.get("domain", ""),
        reward=1.0,
        extra_metadata={
            "rollout_id": summary.get("rollout_id", ""),
            "policy_model": summary.get("policy_model", ""),
            "eval_model": summary.get("eval_model", ""),
            "judge_rounds": len(summary.get("judge_records", [])),
        },
        strip_images=strip_images,
    )
    return True
