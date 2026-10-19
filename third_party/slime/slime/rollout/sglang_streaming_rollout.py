import json
import logging
from argparse import Namespace
from typing import Any

import numpy as np
import pybase64

from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids
from slime.utils import http_utils
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span
from slime.utils.types import Sample

__all__ = ["generate_streaming"]

logger = logging.getLogger(__name__)

async def generate_streaming(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:

    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in (
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    payload: dict[str, Any] = {
        "sampling_params": sampling_params,
        "return_logprob": True,
        "stream": True,
    }
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}

    base_tokens = list(sample.tokens)
    base_response = sample.response or ""
    base_response_length = sample.response_length
    base_log_probs = list(sample.rollout_log_probs or [])
    base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

    last_meta_info: dict[str, Any] = {}
    call_tokens: list[int] = []
    call_log_probs: list[float] = []
    call_text: str = ""

    client = http_utils._http_client
    assert client is not None, "http client not initialized; call init_http_client first"

    with trace_span(
        sample, "sglang_generate_stream", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}
    ) as span:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:") :].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("sglang_streaming: skipping non-JSON chunk: %r", data_str[:120])
                    continue

                meta = chunk.get("meta_info") or {}
                last_meta_info = meta

                call_text = chunk.get("text", call_text)
                if "output_token_logprobs" in meta:
                    call_tokens = [item[1] for item in meta["output_token_logprobs"]]
                    call_log_probs = [item[0] for item in meta["output_token_logprobs"]]

                sample.tokens = base_tokens + call_tokens
                sample.response = base_response + call_text
                sample.response_length = base_response_length + len(call_tokens)
                sample.rollout_log_probs = base_log_probs + call_log_probs
                if base_loss_mask is not None:
                    assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
                    sample.loss_mask = base_loss_mask + [1] * len(call_tokens)

                if state.aborted:
                    break

        if last_meta_info.get("finish_reason"):
            span.update(build_sglang_meta_trace_attrs(last_meta_info))

    if "routed_experts" in last_meta_info:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(last_meta_info["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    if last_meta_info.get("finish_reason"):
        sample.update_from_meta_info(args, last_meta_info)
    elif state.aborted:
        sample.status = Sample.Status.ABORTED

    return sample
