from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from argparse import Namespace
from typing import Any

import torch
from PIL import Image

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

from .judge.evaluator import compute_reward, validate_reward_mode
from .judge.judge_prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_content,
    parse_judge_output,
    JudgeStatus,
)
from .env_service.rpc_env_client import AsyncRpcEnvClient
from . import mm_preprocess
from . import minicpmv_support
from . import minicpmv46_support

logger = logging.getLogger("guirl.slime_custom")

_ENV_CLIENT_CACHE: dict[str, AsyncRpcEnvClient] = {}

_DUMMY_MESSAGES = [
    {"role": "system", "content": "x"},
    {"role": "user", "content": "y"},
]
_THINKING_BLOCK_RE = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

def _get_arg(args: Namespace, key: str, default: Any = None) -> Any:
    if hasattr(args, key):
        val = getattr(args, key)
        if val is not None:
            return val
    return default

def _get_env_client(args: Namespace) -> AsyncRpcEnvClient:
    url = str(_get_arg(args, "guirl_env_master_url", "http://127.0.0.1:9900"))
    if url not in _ENV_CLIENT_CACHE:
        _ENV_CLIENT_CACHE[url] = AsyncRpcEnvClient(base_url=url)
    return _ENV_CLIENT_CACHE[url]

def _strip_thinking_for_history(content: str) -> str:

    stripped = _THINKING_BLOCK_RE.sub("", content or "")
    return stripped.lstrip("\n")

def _encode_history_response_without_thinking(tokenizer, response_text: str) -> list[int]:
    return tokenizer(
        _strip_thinking_for_history(response_text),
        add_special_tokens=False,
    )["input_ids"]

class _SyncEnvClientAdapter:

    def __init__(self, async_client: AsyncRpcEnvClient, loop: asyncio.AbstractEventLoop, timeout: float = 600.0):
        self._async_client = async_client
        self._loop = loop
        self._timeout = timeout

    def _run(self, coro):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            raise RuntimeError("_SyncEnvClientAdapter must be called from a worker thread")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self._timeout)
        except TimeoutError:
            future.cancel()
            raise

    def get_obs(self, lease_id: str) -> dict[str, Any]:
        return self._run(self._async_client.get_obs(lease_id))

    def step(self, lease_id: str, action: str):
        return self._run(self._async_client.step(lease_id, action))

    def prepare_for_probe(self, lease_id: str | None = None) -> None:
        if lease_id is None:
            raise ValueError("lease_id is required for prepare_for_probe()")
        return self._run(self._async_client.prepare_for_probe(lease_id))

    def evaluate(self, lease_id: str) -> float:
        return self._run(self._async_client.evaluate(lease_id))

    def close(self, lease_id: str) -> None:
        return self._run(self._async_client.close(lease_id))

def _b64_to_pil(b64_str: str) -> Image.Image:

    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _extract_pil_images_from_message(message: dict[str, Any]) -> list[Image.Image]:

    pics: list[Image.Image] = []
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image/") and "," in url:
                    try:
                        pics.append(_b64_to_pil(url.split(",", 1)[1]))
                    except Exception:
                        pass
    return pics

def _abort_sample(sample: Sample, reason: str) -> Sample:

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = None
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("GUIRL abort: %s", reason)
    return sample

def build_judge_artifact_summary(
    samples: Sample | list[Sample],
    *,
    source_guirl: dict[str, Any] | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:

    group = samples if isinstance(samples, list) else [samples]
    source = dict(source_guirl or {})
    result = dict(result_metadata or {})
    sample_descriptors = []
    for item in group:
        status = getattr(item, "status", None)
        status_value = getattr(status, "value", status)
        sample_descriptors.append(
            {
                "index": getattr(item, "index", None),
                "group_id": getattr(item, "group_id", None),
                "reward": getattr(item, "reward", None),
                "status": status_value,
                "response_length": getattr(item, "response_length", 0),
            }
        )
    return _json_artifact_value(
        {
            "schema": "guirl.judge-artifact.v2",
            "reward": [getattr(item, "reward", None) for item in group],
            "probe": {
                "judge_final": result.get("judge_final"),
                "num_rounds": result.get("num_rounds"),
                "S": result.get("S"),
                "S_prime": result.get("S_prime"),
                "probe_round_records": result.get("probe_round_records", []),
                "probe_artifact_dir": result.get("probe_artifact_dir"),
                "judge_parse_attempts": result.get("judge_parse_attempts", []),
            },
            "source": {
                "run_id": source.get("run_id"),
                "rollout_id": source.get("rollout_id"),
                "slot": source.get("slot"),
                "task_id": source.get("task_id"),
                "task_group_id": source.get("task_group_id"),
                "task_group_revision": source.get("task_group_revision"),
                "manifest_revision": source.get("manifest_revision"),
                "actor_id": source.get("actor_id") or source.get("actor"),
                "attempt": source.get("attempt"),
                "claim_token": source.get("claim_token"),
                "claim_generation": source.get("claim_generation"),
                "source_group_id": source.get("source_group_id"),
            },
            "samples": sample_descriptors,
            "sample_serialization": {
                "kind": "trainer-step-accumulator",
                "artifact": None,
            },
        }
    )

def _json_artifact_value(value: Any) -> Any:

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_artifact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_artifact_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return _json_artifact_value(enum_value)
    return str(value)

def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:

    if not chunks:
        return None
    values_by_key: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is not None:
                values_by_key.setdefault(key, []).append(val)
    merged = {}
    for key, values in values_by_key.items():
        if any(isinstance(value, list) for value in values):
            merged[key] = [
                part
                for value in values
                for part in (value if isinstance(value, list) else [value])
            ]
        elif all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
        elif len(values) == 1:
            merged[key] = values[0]
        else:

            merged[key] = values
    return merged or None

def _build_judge_turn_sample(
    *,
    source_sample: Sample,
    round_idx: int,
    max_rounds: int,
    prompt_ids: list[int],
    cur_response: str,
    cur_token_ids: list[int],
    cur_logprobs: list[float],
    multimodal_train_inputs: dict | None,
    metadata: dict[str, Any],
) -> Sample:

    if source_sample.index is None:
        index = round_idx
        group_id = round_idx if source_sample.group_id is None else source_sample.group_id
    else:
        index = source_sample.index * max_rounds + round_idx
        group_id = source_sample.index if source_sample.group_id is None else source_sample.group_id

    return Sample(
        group_index=source_sample.group_index,
        index=index,
        group_id=group_id,
        prompt=source_sample.prompt,
        tokens=list(prompt_ids) + list(cur_token_ids),
        response=cur_response,
        response_length=len(cur_token_ids),
        loss_mask=[1] * len(cur_token_ids),
        rollout_log_probs=list(cur_logprobs),
        status=Sample.Status.COMPLETED,
        metadata=metadata,
        multimodal_train_inputs=multimodal_train_inputs,
        session_id=source_sample.session_id,
        generate_function_path=source_sample.generate_function_path,
    )

def _apply_judge_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    judge_enable_thinking: bool,
    allow_missing_thinking_template: bool = False,
) -> str:

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=judge_enable_thinking,
        )
    except TypeError as exc:
        if judge_enable_thinking and not allow_missing_thinking_template:
            raise RuntimeError(
                "Judge thinking is enabled, but tokenizer.apply_chat_template() "
                "does not accept enable_thinking. Use a Qwen3-VL thinking template "
                "or set guirl_allow_missing_thinking_template=true explicitly."
            ) from exc
        logger.warning(
            "Tokenizer chat template does not accept enable_thinking; "
            "falling back without the flag."
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

def _encode_turn(
    state,
    messages: list[dict[str, Any]],
    pil_images: list[Image.Image],
    *,
    judge_enable_thinking: bool,
    allow_missing_thinking_template: bool = False,
):

    is_minicpm = minicpmv_support.is_minicpmv_processor(state.processor)
    is_minicpm46 = minicpmv46_support.is_minicpmv46_processor(state.processor)
    template_messages = (
        minicpmv_support.flatten_messages_for_minicpmv(messages) if is_minicpm else messages
    )
    prompt_text = _apply_judge_chat_template(
        state.tokenizer,
        template_messages,
        add_generation_prompt=True,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    if state.processor and pil_images:
        if is_minicpm:
            ids, mm_train = minicpmv_support.encode_prompt(
                state.processor, prompt_text, pil_images
            )
            uris = [encode_image_for_rollout_engine(img) for img in pil_images]
            return ids, mm_train, uris
        if is_minicpm46:
            ids, mm_train = minicpmv46_support.encode_prompt(
                state.processor, prompt_text, pil_images
            )
            uris = [encode_image_for_rollout_engine(img) for img in pil_images]
            return ids, mm_train, uris
        out = state.processor(text=prompt_text, images=pil_images)
        ids = out["input_ids"][0]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        mm_train = {
            k: v
            for k, v in out.items()
            if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
        } or None
        uris = [encode_image_for_rollout_engine(img) for img in pil_images]
    else:
        ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        mm_train = None
        uris = _extract_base64_data_uris(messages)
    return ids, mm_train, uris

def _encode_followup_turn(
    state,
    user_message: dict[str, Any],
    pil_images: list[Image.Image],
    *,
    judge_enable_thinking: bool,
    allow_missing_thinking_template: bool = False,
    image_index_offset: int = 0,
):

    tokenizer = state.tokenizer
    is_minicpm = minicpmv_support.is_minicpmv_processor(state.processor)
    is_minicpm46 = minicpmv46_support.is_minicpmv46_processor(state.processor)
    followup_messages = (
        minicpmv_support.flatten_messages_for_minicpmv([user_message])
        if is_minicpm
        else [user_message]
    )
    dummy = _apply_judge_chat_template(
        tokenizer,
        _DUMMY_MESSAGES,
        add_generation_prompt=False,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    formatted = _apply_judge_chat_template(
        tokenizer,
        _DUMMY_MESSAGES + followup_messages,
        add_generation_prompt=True,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    if is_minicpm:

        trim = minicpmv_support.encode_text_len(tokenizer, dummy)
    else:
        trim = len(tokenizer.encode(dummy, add_special_tokens=False))

    if state.processor and pil_images:
        if is_minicpm:
            ids, mm_train = minicpmv_support.encode_prompt(
                state.processor,
                formatted,
                pil_images,
                image_index_offset=image_index_offset,
            )
            uris = [encode_image_for_rollout_engine(img) for img in pil_images]
            return ids[trim:], mm_train, uris
        if is_minicpm46:
            ids, mm_train = minicpmv46_support.encode_prompt(
                state.processor,
                formatted,
                pil_images,
                image_index_offset=image_index_offset,
            )
            dummy_ids = state.processor(text=dummy)["input_ids"][0]
            if isinstance(dummy_ids, torch.Tensor):
                dummy_ids = dummy_ids.tolist()
            assert len(dummy_ids) == trim, (
                f"tokenizer/processor trim mismatch: tokenizer={trim}, "
                f"processor={len(dummy_ids)}"
            )
            uris = [encode_image_for_rollout_engine(img) for img in pil_images]
            return ids[trim:], mm_train, uris
        out = state.processor(text=formatted, images=pil_images)
        ids = out["input_ids"][0]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        dummy_ids = state.processor(text=dummy)["input_ids"][0]
        if isinstance(dummy_ids, torch.Tensor):
            dummy_ids = dummy_ids.tolist()
        assert len(dummy_ids) == trim, (
            f"tokenizer/processor trim mismatch: tokenizer={trim}, "
            f"processor={len(dummy_ids)}. The model's processor encodes the "
            f"chat-template preamble differently from its tokenizer."
        )
        mm_train = {
            k: v
            for k, v in out.items()
            if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
        } or None
        uris = [encode_image_for_rollout_engine(img) for img in pil_images]
    else:
        ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        mm_train = None
        uris = _extract_base64_data_uris([user_message])
    return ids[trim:], mm_train, uris

async def _encode_prompt_with_mm_pool(
    args: Namespace,
    state,
    prompt_text: str,
    image_data: list[str],
    image_cache: dict[str, dict[str, Any]],
    image_index_offset: int = 0,
) -> tuple[list[int], dict | None, list[str]]:

    is_minicpm = minicpmv_support.is_minicpmv_processor(state.processor)
    is_minicpm46 = minicpmv46_support.is_minicpmv46_processor(state.processor)
    if (
        is_minicpm
        or is_minicpm46
        or not bool(_get_arg(args, "guirl_mm_per_image_cache", True))
    ):

        result = await mm_preprocess.process_full(
            prompt_text, image_data, image_index_offset
        )
        return result["prompt_ids"], result["mm_train"], result["image_data"]

    missing = list(dict.fromkeys(data for data in image_data if data not in image_cache))
    if missing:
        processed = await asyncio.gather(
            *(mm_preprocess.process_single_image(data) for data in missing)
        )
        image_cache.update(zip(missing, processed, strict=True))

    entries = [image_cache[data] for data in image_data]
    processor = state.processor
    expanded_text = mm_preprocess.expand_image_pad_tokens(
        prompt_text,
        [entry["image_grid_thw"] for entry in entries],
        image_token=processor.image_token,
        merge_size=processor.image_processor.merge_size,
    )
    prompt_ids = state.tokenizer(expanded_text, return_tensors=None)["input_ids"]
    mm_train = {
        key: [entry[key] for entry in entries]
        for key in entries[0]
        if key != "image_data"
    }
    image_uris = [entry["image_data"] for entry in entries]
    return prompt_ids, mm_train, image_uris

async def _encode_turn_async(
    args: Namespace,
    state,
    messages: list[dict[str, Any]],
    *,
    image_cache: dict[str, dict[str, Any]],
    judge_enable_thinking: bool,
    allow_missing_thinking_template: bool = False,
):

    is_minicpm = minicpmv_support.is_minicpmv_processor(state.processor)
    template_messages = (
        minicpmv_support.flatten_messages_for_minicpmv(messages) if is_minicpm else messages
    )
    prompt_text = _apply_judge_chat_template(
        state.tokenizer,
        template_messages,
        add_generation_prompt=True,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    image_data = _extract_base64_data_uris(messages)
    if state.processor and image_data and mm_preprocess.mm_pool_enabled():
        return await _encode_prompt_with_mm_pool(
            args, state, prompt_text, image_data, image_cache
        )

    pil_images = [
        image
        for message in messages
        for image in _extract_pil_images_from_message(message)
    ]
    return _encode_turn(
        state,
        messages,
        pil_images,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )

async def _encode_followup_turn_async(
    args: Namespace,
    state,
    user_message: dict[str, Any],
    *,
    image_cache: dict[str, dict[str, Any]],
    judge_enable_thinking: bool,
    allow_missing_thinking_template: bool = False,
    image_index_offset: int = 0,
):

    tokenizer = state.tokenizer
    is_minicpm = minicpmv_support.is_minicpmv_processor(state.processor)
    followup_messages = (
        minicpmv_support.flatten_messages_for_minicpmv([user_message])
        if is_minicpm
        else [user_message]
    )
    dummy = _apply_judge_chat_template(
        tokenizer,
        _DUMMY_MESSAGES,
        add_generation_prompt=False,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    formatted = _apply_judge_chat_template(
        tokenizer,
        _DUMMY_MESSAGES + followup_messages,
        add_generation_prompt=True,
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
    )
    if is_minicpm:

        trim = minicpmv_support.encode_text_len(tokenizer, dummy)
    else:
        trim = len(tokenizer.encode(dummy, add_special_tokens=False))
    image_data = _extract_base64_data_uris([user_message])

    if state.processor and image_data and mm_preprocess.mm_pool_enabled():
        encoded, dummy_ids = await asyncio.gather(
            _encode_prompt_with_mm_pool(
                args, state, formatted, image_data, image_cache, image_index_offset
            ),
            mm_preprocess.process_text(dummy),
        )
        assert len(dummy_ids) == trim, (
            f"tokenizer/processor trim mismatch: tokenizer={trim}, "
            f"processor={len(dummy_ids)}"
        )
        ids, mm_train, uris = encoded
        return ids[trim:], mm_train, uris

    return _encode_followup_turn(
        state,
        user_message,
        _extract_pil_images_from_message(user_message),
        judge_enable_thinking=judge_enable_thinking,
        allow_missing_thinking_template=allow_missing_thinking_template,
        image_index_offset=image_index_offset,
    )

def _build_judge_sampling_params(
    sampling_params: dict[str, Any], judge_max_tokens: int, judge_temperature: float
) -> dict[str, Any]:

    merged = dict(sampling_params or {})
    merged["max_new_tokens"] = judge_max_tokens
    merged["temperature"] = judge_temperature
    return merged

async def _generate_parseable_judge_response(
    *,
    sglang_url: str,
    payload: dict[str, Any],
    round_idx: int,
    max_retries: int,
) -> tuple[str, list[int], list[float], JudgeStatus, str, int] | None:

    last_error: Exception | None = None
    last_response: str = ""
    last_token_ids: list[int] = []
    last_logprobs: list[float] = []

    for attempt in range(1, max_retries + 1):
        output = await post(sglang_url, payload, max_retries=3)

        try:
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                return None

            response_text = output["text"]
            token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            logprobs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        except (KeyError, TypeError, IndexError) as exc:
            last_error = exc
            logger.warning(
                "SGLang output malformed (round %d, attempt %d/%d): %s",
                round_idx,
                attempt,
                max_retries,
                exc,
            )
            continue

        last_response = response_text
        last_token_ids = token_ids
        last_logprobs = logprobs

        try:
            parsed = parse_judge_output(response_text)
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "Judge parse failed (round %d, attempt %d/%d): %s",
                round_idx,
                attempt,
                max_retries,
                exc,
            )
            continue

        return response_text, token_ids, logprobs, parsed.status, parsed.criteria_text, attempt

    logger.warning(
        "Judge parse exhausted retries (round %d): %s",
        round_idx,
        last_error,
    )
    return last_response, last_token_ids, last_logprobs, "parse_failure", "", max_retries

def _build_eval_probe_config(args: Namespace, *, history_n: int) -> dict[str, Any]:

    return {
        "probe_agent_type": str(_get_arg(args, "guirl_probe_agent_type", "Holo3_Probe")),
        "probe_model_name": str(_get_arg(args, "guirl_probe_model_name", "holo3-1-35b-a3b")),
        "probe_api_key": str(_get_arg(args, "guirl_probe_api_key", "EMPTY")),
        "probe_base_url": str(_get_arg(args, "guirl_probe_base_url", "http://127.0.0.1:40001/v1")),
        "probe_max_tokens": int(_get_arg(args, "guirl_probe_max_tokens", 8192)),
        "probe_temperature": float(_get_arg(args, "guirl_probe_temperature", 0.0)),
        "probe_history_n": int(_get_arg(args, "guirl_probe_history_n", 4)),
        "probe_max_steps": int(_get_arg(args, "guirl_probe_max_steps", 15)),
        "probe_enable_thinking": bool(_get_arg(args, "guirl_probe_enable_thinking", True)),
        "probe_reasoning_effort": str(_get_arg(args, "guirl_probe_reasoning_effort", "medium")),
        "probe_max_trajectory_length": int(_get_arg(args, "guirl_probe_max_trajectory_length", 15)),
        "probe_max_images": int(_get_arg(args, "guirl_probe_max_images", 3)),
        "probe_client_password": str(_get_arg(args, "guirl_probe_client_password", "password")),
        "judge_history_n": history_n,
    }

def _probe_artifact_dir(args: Namespace, guirl: dict[str, Any], sample: Sample) -> str:
    base_dir = str(_get_arg(args, "guirl_probe_artifact_dir", "results/guirl_rl_probe"))
    task_id = str(guirl.get("task_id") or f"sample_{sample.index}")
    batch_id = str(guirl.get("batch_id") or "batch_unknown")
    safe = lambda text: "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return os.path.join(base_dir, safe(batch_id), safe(task_id), str(sample.index))

def _rollout_master_url(args: Namespace) -> str:
    return str(_get_arg(args, "guirl_rollout_master_url", "http://127.0.0.1:9910")).rstrip("/")

def _post_rollout_master_json(
    args: Namespace,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float = 700.0,
) -> dict[str, Any]:
    url = f"{_rollout_master_url(args)}{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-object JSON: {data!r}")
    if not data.get("ok", False):
        raise RuntimeError(f"POST {url} returned not-ok: {data}")
    return data

def _run_eval_probe_round_sync(
    *,
    args: Namespace,
    env_client: _SyncEnvClientAdapter,
    lease_id: str,
    fence_token: str | None = None,
    criteria: str,
    instruction: str,
    history_n: int,
    example_result_dir: str,
    sampler_round: int,
) -> dict[str, Any]:

    guirl_config = _build_eval_probe_config(args, history_n=history_n)
    payload = {
        **guirl_config,
        "lease_id": lease_id,
        "fence_token": fence_token,
        "criteria": criteria,
        "instruction": instruction,
        "history_n": history_n,
        "example_result_dir": example_result_dir,
        "sampler_round": sampler_round,
        "probe_sleep_after_execution": float(_get_arg(args, "guirl_probe_sleep_after_execution", 2.0)),
    }
    result = _post_rollout_master_json(args, "/run_probe_round", payload)
    return {
        "latest_criteria": result.get("latest_criteria", criteria),
        "criteria_images": result.get("criteria_images", []),
        "round_record": result.get("round_record", {}),
        "probe_agent_type": result.get("probe_agent_type", guirl_config["probe_agent_type"]),
    }

async def _run_eval_probe_round(
    *,
    args: Namespace,
    async_env_client: AsyncRpcEnvClient,
    lease_id: str,
    fence_token: str | None = None,
    criteria: str,
    instruction: str,
    history_n: int,
    example_result_dir: str,
    sampler_round: int,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _run_eval_probe_round_sync,
        args=args,
        env_client=None,
        lease_id=lease_id,
        fence_token=fence_token,
        criteria=criteria,
        instruction=instruction,
        history_n=history_n,
        example_result_dir=example_result_dir,
        sampler_round=sampler_round,
    )

def _post_judge_evaluate_kwargs(
    guirl: dict[str, Any],
    lease_fence_token: str | None,
) -> dict[str, str]:

    if not guirl.get("v2"):
        return {}
    if not lease_fence_token:
        raise RuntimeError("v2 post-judge evaluate requires lease fence token")
    return {
        "operation_id": (
            "trainer-post-judge-evaluate:"
            f"{guirl.get('claim_token')}:"
            f"{guirl.get('claim_generation')}:"
            f"{guirl.get('attempt')}"
        ),
        "fence_token": lease_fence_token,
    }

async def custom_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:

    reward_mode = validate_reward_mode(_get_arg(args, "guirl_reward_mode", "shaped"))
    state = GenerateState(args)

    mm_preprocess.init_mm_pool(args)
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    if sample.metadata is None:
        sample.metadata = {}

    guirl = sample.metadata.get("guirl", {})
    if not guirl:
        return _abort_sample(sample, "missing_guirl_metadata")

    lease_id = guirl.get("lease_id")
    lease_fence_token = (guirl.get("lease") or {}).get(
        "fence_token"
    ) or guirl.get("fence_token")
    close_env = bool(_get_arg(args, "guirl_close_env_after_generate", True))
    if guirl.get("v2") and not lease_id:
        return _abort_sample(sample, "missing_v2_lease")
    if guirl.get("v2") and not lease_fence_token:
        return _abort_sample(sample, "missing_v2_lease_fence")

    try:
        trajectory = guirl.get("trajectory", [])
        screenshots = guirl.get("screenshots", [])
        S = guirl.get("S")
        task_config = guirl.get("task_config", {})

        instruction = ""
        if isinstance(sample.prompt, str):
            instruction = sample.prompt
        elif isinstance(sample.prompt, dict):
            instruction = sample.prompt.get("instruction", "")
        elif isinstance(task_config, dict):
            instruction = task_config.get("instruction", "")

        if not instruction:
            return _abort_sample(sample, "missing_instruction")

        action_history = [
            entry.get("action", "") for entry in trajectory if entry.get("action")
        ]

        max_rounds = int(_get_arg(args, "guirl_judge_max_rounds", 3))
        history_n = int(_get_arg(args, "guirl_judge_history_n", 3))
        judge_max_tokens = int(_get_arg(args, "guirl_judge_max_tokens", 24576))
        judge_temperature = float(_get_arg(args, "guirl_judge_temperature", 1.0))
        judge_enable_thinking = bool(_get_arg(args, "guirl_judge_enable_thinking", True))
        judge_strip_history_thinking = bool(
            _get_arg(args, "guirl_judge_strip_history_thinking", True)
        )
        judge_fanout_turns = bool(_get_arg(args, "guirl_judge_fanout_turns", True))
        allow_missing_thinking_template = bool(
            _get_arg(args, "guirl_allow_missing_thinking_template", False)
        )
        judge_parse_max_retries = int(_get_arg(args, "guirl_judge_parse_max_retries", 3))
        judge_sampling_params = _build_judge_sampling_params(
            sampling_params, judge_max_tokens, judge_temperature
        )
        env_client = _get_env_client(args)
        probe_artifact_dir = _probe_artifact_dir(args, guirl, sample)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        ]
        user_content_r0 = build_judge_user_content(
            task=instruction,
            task_traj_imgs=screenshots,
            prev_criteria=None,
            action_history=action_history,
            history_n=history_n,
            criteria_option_screenshot=None,
        )
        messages.append({"role": "user", "content": user_content_r0})

        messages_r1_for_encode = [messages[0], messages[1]]
        mm_image_cache: dict[str, dict[str, Any]] = {}
        prompt_ids, mm_train_0, image_uris = await _encode_turn_async(
            args,
            state,
            messages_r1_for_encode,
            image_cache=mm_image_cache,
            judge_enable_thinking=judge_enable_thinking,
            allow_missing_thinking_template=allow_missing_thinking_template,
        )

        multimodal_train_inputs_buffer: list[dict | None] = []
        if mm_train_0:
            multimodal_train_inputs_buffer.append(mm_train_0)
        image_data_for_sglang: list[str] = list(image_uris)
        is_minicpmv46 = minicpmv46_support.is_minicpmv46_processor(
            state.processor
        )
        sglang_prompt_ids = (
            minicpmv46_support.collapse_image_spans_for_sglang(
                prompt_ids, state.tokenizer
            )
            if is_minicpmv46
            else list(prompt_ids)
        )

        response_token_ids: list[int] = []
        sglang_response_token_ids: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] = []
        response_text = ""

        current_criteria: str | None = None
        current_status: JudgeStatus = "uncertain"
        judge_responses: list[str] = []
        probe_round_records: list[dict[str, Any]] = []
        judge_parse_attempts: list[int] = []
        turn_samples: list[Sample] = []

        for round_idx in range(max_rounds):
            all_input_ids = prompt_ids + response_token_ids
            sglang_input_ids = sglang_prompt_ids + sglang_response_token_ids

            payload: dict[str, Any] = {
                "input_ids": sglang_input_ids,
                "sampling_params": judge_sampling_params,
                "return_logprob": True,
            }
            if image_data_for_sglang:
                payload["image_data"] = image_data_for_sglang

            generated = await _generate_parseable_judge_response(
                sglang_url=sglang_url,
                payload=payload,
                round_idx=round_idx,
                max_retries=judge_parse_max_retries,
            )
            if generated is None:
                return _abort_sample(sample, "sglang_abort")

            (
                cur_response,
                cur_token_ids,
                cur_logprobs,
                current_status,
                current_criteria,
                parse_attempts,
            ) = generated

            fanout_prompt_ids = list(all_input_ids)

            response_text += cur_response
            response_token_ids += cur_token_ids
            sglang_response_token_ids += cur_token_ids
            loss_mask += [1] * len(cur_token_ids)
            rollout_log_probs += cur_logprobs
            judge_responses.append(cur_response)
            judge_parse_attempts.append(parse_attempts)

            messages.append({"role": "assistant", "content": cur_response})

            is_last_round = (
                current_status == "parse_failure"
                or current_status != "uncertain"
                or round_idx >= max_rounds - 1
                or lease_id is None
            )

            if not is_last_round and judge_strip_history_thinking and cur_token_ids:
                history_response_ids = _encode_history_response_without_thinking(
                    state.tokenizer,
                    cur_response,
                )
                history_response_text = _strip_thinking_for_history(cur_response)
                response_token_ids = response_token_ids[: -len(cur_token_ids)] + history_response_ids
                sglang_response_token_ids = (
                    sglang_response_token_ids[: -len(cur_token_ids)]
                    + history_response_ids
                )
                loss_mask = loss_mask[: -len(cur_token_ids)] + [0] * len(history_response_ids)
                rollout_log_probs = rollout_log_probs[: -len(cur_token_ids)] + [0.0] * len(history_response_ids)
                if response_text.endswith(cur_response):
                    response_text = response_text[: -len(cur_response)] + history_response_text
                messages[-1] = {"role": "assistant", "content": history_response_text}

            if judge_fanout_turns:
                turn_metadata = {
                    **(sample.metadata or {}),
                    "guirl_turn": {
                        "round_idx": round_idx,
                        "judge_status": current_status,
                        "criteria": current_criteria,
                        "parse_attempts": parse_attempts,
                    },
                }
                turn_samples.append(
                    _build_judge_turn_sample(
                        source_sample=sample,
                        round_idx=round_idx,
                        max_rounds=max_rounds,
                        prompt_ids=fanout_prompt_ids,
                        cur_response=cur_response,
                        cur_token_ids=cur_token_ids,
                        cur_logprobs=cur_logprobs,
                        multimodal_train_inputs=_merge_multimodal_train_inputs(
                            multimodal_train_inputs_buffer
                        ),
                        metadata=turn_metadata,
                    )
                )

            if is_last_round:
                break

            try:
                probe_result = await _run_eval_probe_round(
                    args=args,
                    async_env_client=env_client,
                    lease_id=lease_id,
                    fence_token=lease_fence_token,
                    criteria=current_criteria or "",
                    instruction=instruction,
                    history_n=history_n,
                    example_result_dir=probe_artifact_dir,
                    sampler_round=round_idx + 1,
                )
            except Exception as exc:
                logger.error("Probe round failed: %s", exc)
                return _abort_sample(sample, "probe_infrastructure_failure")

            current_criteria = probe_result["latest_criteria"] or current_criteria
            probe_round_records.append(probe_result["round_record"])

            user_content_rN = build_judge_user_content(
                task=instruction,
                task_traj_imgs=screenshots,
                prev_criteria=current_criteria,
                action_history=None,
                history_n=history_n,
                criteria_option_screenshot=probe_result["criteria_images"] or None,
            )
            evidence_msg = {"role": "user", "content": user_content_rN}
            messages.append(evidence_msg)

            evidence_ids, evidence_mm, evidence_uris = await _encode_followup_turn_async(
                args,
                state,
                evidence_msg,
                image_cache=mm_image_cache,
                judge_enable_thinking=judge_enable_thinking,
                allow_missing_thinking_template=allow_missing_thinking_template,

                image_index_offset=len(image_data_for_sglang),
            )

            response_text += "\n[probe evidence]\n"
            response_token_ids += evidence_ids
            sglang_response_token_ids += (
                minicpmv46_support.collapse_image_spans_for_sglang(
                    evidence_ids, state.tokenizer
                )
                if is_minicpmv46
                else evidence_ids
            )
            loss_mask += [0] * len(evidence_ids)
            rollout_log_probs += [0.0] * len(evidence_ids)
            if evidence_uris:
                image_data_for_sglang += evidence_uris
            if evidence_mm:
                multimodal_train_inputs_buffer.append(evidence_mm)

        S_prime = None
        if lease_id:
            try:
                S_prime = await env_client.evaluate(
                    lease_id,
                    **_post_judge_evaluate_kwargs(
                        guirl,
                        lease_fence_token,
                    ),
                )
            except Exception as exc:
                logger.error("Post-judge evaluate failed: %s", exc)
                return _abort_sample(
                    sample,
                    "post_judge_evaluate_failure",
                )

        used_probe = len(probe_round_records) > 0
        reward = compute_reward(current_status, S, used_probe=used_probe, reward_mode=reward_mode)

        if not response_token_ids:
            return _abort_sample(sample, "no_response_generated")

        result_metadata = {
            "reward_mode": reward_mode,
            "judge_final": current_status,
            "num_rounds": len(judge_responses),
            "S": S,
            "S_prime": S_prime,
            "reward": reward,
            "messages": messages,
            "probe_round_records": probe_round_records,
            "probe_artifact_dir": probe_artifact_dir,
            "judge_enable_thinking": judge_enable_thinking,
            "judge_strip_history_thinking": judge_strip_history_thinking,
            "judge_fanout_turns": judge_fanout_turns,
            "judge_parse_attempts": judge_parse_attempts,
        }

        if judge_fanout_turns and turn_samples:
            for turn_sample in turn_samples:
                turn_sample.reward = reward
            artifact_summary = build_judge_artifact_summary(
                turn_samples,
                source_guirl=guirl,
                result_metadata=result_metadata,
            )
            for turn_sample in turn_samples:
                turn_sample.metadata = {
                    **(turn_sample.metadata or {}),
                    "guirl_result": result_metadata,
                    "guirl_judge_artifact": artifact_summary,
                }
            return turn_samples

        sample.tokens = prompt_ids + response_token_ids
        sample.response = response_text
        sample.response_length = len(response_token_ids)
        sample.loss_mask = loss_mask
        sample.rollout_log_probs = rollout_log_probs
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(
            multimodal_train_inputs_buffer
        )

        sample.metadata["guirl_result"] = result_metadata
        sample.metadata["guirl_judge_artifact"] = build_judge_artifact_summary(
            sample,
            source_guirl=guirl,
            result_metadata=result_metadata,
        )
        return sample

    finally:

        if close_env and lease_id and not bool(guirl.get("v2")):
            try:
                await _get_env_client(args).close(lease_id)
            except Exception:
                pass

async def custom_rm(args: Namespace, sample: Sample | list[Sample]) -> float | list[float]:

    if isinstance(sample, list):
        return [float(await custom_rm(args, item)) for item in sample]

    if sample.reward is not None:
        return float(sample.reward)
    guirl_result = (sample.metadata or {}).get("guirl_result", {})
    judge_final = guirl_result.get("judge_final", "failure")
    S = guirl_result.get("S")
    used_probe = bool(guirl_result.get("probe_round_records"))
    return compute_reward(
        judge_final, S, used_probe=used_probe,
        reward_mode=_get_arg(args, "guirl_reward_mode", "shaped"),
    )

def _extract_base64_data_uris(messages: list[dict[str, Any]]) -> list[str]:

    images: list[str] = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/"):
                        images.append(url)
    return images
