from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any

import ray

logger = logging.getLogger(__name__)

_POOL: list = []
_POOL_SIZE = 0
_RR = 0

def _strip_data_uri(image_data: str) -> str:
    return image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data

def _decode_to_rgb(image_data: str):
    from PIL import Image

    image = Image.open(BytesIO(base64.b64decode(_strip_data_uri(image_data))))
    return image.convert("RGB")

def _input_ids(processor_output) -> list[int]:
    ids = processor_output["input_ids"][0]
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)

@ray.remote(num_cpus=1)
class GUIRLMMProcessorActor:

    def __init__(self, hf_checkpoint: str):
        os.environ["OMP_NUM_THREADS"] = "1"

        import torch
        from slime.utils.processing_utils import load_processor

        from . import minicpmv_support, minicpmv46_support

        torch.set_num_threads(1)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        if self.processor is None:
            raise RuntimeError(f"load_processor returned None for {hf_checkpoint}")
        self.is_minicpmv = minicpmv_support.is_minicpmv_processor(self.processor)
        self.is_minicpmv46 = minicpmv46_support.is_minicpmv46_processor(
            self.processor
        )
        patch_size = getattr(self.processor.image_processor, "patch_size", None)
        if self.is_minicpmv or self.is_minicpmv46:
            if patch_size != 14:
                raise RuntimeError(
                    f"MiniCPM-V processors use patch_size=14, got {patch_size}"
                )
        elif patch_size != 16:
            raise RuntimeError(
                f"GUIRL MM pool requires Qwen3-VL patch_size=16 (or a MiniCPM-V "
                f"processor), got {patch_size}"
            )

    def healthy(self) -> bool:
        return True

    def process_full(
        self, prompt_text: str, images_data: list[str], image_index_offset: int = 0
    ) -> dict[str, Any]:

        import torch
        from slime.utils.processing_utils import encode_image_for_rollout_engine

        from . import minicpmv_support, minicpmv46_support

        images = [_decode_to_rgb(image_data) for image_data in images_data]
        if self.is_minicpmv:
            prompt_ids, mm_train = minicpmv_support.encode_prompt(
                self.processor,
                prompt_text,
                images,
                image_index_offset=image_index_offset,
            )
            return {
                "prompt_ids": prompt_ids,
                "mm_train": mm_train,
                "image_data": [encode_image_for_rollout_engine(image) for image in images],
            }
        if self.is_minicpmv46:
            prompt_ids, mm_train = minicpmv46_support.encode_prompt(
                self.processor,
                prompt_text,
                images,
                image_index_offset=image_index_offset,
            )
            return {
                "prompt_ids": prompt_ids,
                "mm_train": mm_train,
                "image_data": [encode_image_for_rollout_engine(image) for image in images],
            }
        output = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
        )
        mm_train = {
            key: (value.numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in output.items()
            if key not in ("input_ids", "attention_mask", "mm_token_type_ids")
        } or None
        return {
            "prompt_ids": _input_ids(output),
            "mm_train": mm_train,
            "image_data": [encode_image_for_rollout_engine(image) for image in images],
        }

    def process_single_image(self, image_data: str) -> dict[str, Any]:

        import torch
        from slime.utils.processing_utils import encode_image_for_rollout_engine

        if self.is_minicpmv or self.is_minicpmv46:

            raise RuntimeError("per-image cache is not supported for MiniCPM-V")
        image = _decode_to_rgb(image_data)
        output = self.processor.image_processor(images=[image], return_tensors="pt")
        result = {
            key: (value.numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in output.items()
        }
        result["image_data"] = encode_image_for_rollout_engine(image)
        return result

    def process_text(self, text: str) -> list[int]:

        if self.is_minicpmv:

            return list(self.processor.tokenizer.encode(text))
        if self.is_minicpmv46:
            return list(
                self.processor.tokenizer.encode(text, add_special_tokens=False)
            )
        return _input_ids(self.processor(text=text))

def init_mm_pool(args: Any) -> bool:

    global _POOL, _POOL_SIZE
    if _POOL:
        return True

    pool_size = int(getattr(args, "guirl_mm_pool_size", 0) or 0)
    if pool_size <= 0:
        return False

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=True)

    available_cpus = int(ray.available_resources().get("CPU", 0))
    if available_cpus <= 0:
        logger.warning("guirl mm_preprocess: guirl_mm_pool_size=%d but no Ray CPU is free; preprocessing inline", pool_size)
        return False
    if pool_size > available_cpus:
        logger.warning(
            "guirl mm_preprocess: guirl_mm_pool_size=%d but only %d Ray CPU(s) are free; using %d worker(s)",
            pool_size,
            available_cpus,
            available_cpus,
        )
        pool_size = available_cpus

    _POOL = [
        GUIRLMMProcessorActor.options(scheduling_strategy=strategy).remote(args.hf_checkpoint)
        for _ in range(pool_size)
    ]
    _POOL_SIZE = pool_size
    ray.get([actor.healthy.remote() for actor in _POOL])
    logger.info("Started %d GUIRL MM processor workers on node %s", pool_size, node_id)
    return True

def mm_pool_enabled() -> bool:
    return bool(_POOL)

def _next_actor():
    global _RR
    actor = _POOL[_RR % _POOL_SIZE]
    _RR += 1
    return actor

async def process_full(
    prompt_text: str, images_data: list[str], image_index_offset: int = 0
) -> dict[str, Any]:
    return await _next_actor().process_full.remote(
        prompt_text, images_data, image_index_offset
    )

async def process_single_image(image_data: str) -> dict[str, Any]:
    return await _next_actor().process_single_image.remote(image_data)

async def process_text(text: str) -> list[int]:
    return await _next_actor().process_text.remote(text)

def expand_image_pad_tokens(
    prompt_text: str,
    grids,
    *,
    image_token: str,
    merge_size: int,
) -> str:

    import numpy as np

    parts = prompt_text.split(image_token)
    if len(parts) - 1 != len(grids):
        raise ValueError(f"prompt has {len(parts) - 1} image tokens but {len(grids)} grids")

    merge_length = merge_size**2
    output = [parts[0]]
    for grid, tail in zip(grids, parts[1:]):
        t, h, w = (int(value) for value in np.asarray(grid).reshape(-1)[:3])
        output.append(image_token * (t * h * w // merge_length))
        output.append(tail)
    return "".join(output)
