from __future__ import annotations

from typing import Any

import torch

from slime_plugins.minicpmv46_common import (
    flatten_pixel_values,
    validate_image_alignment,
)

IMAGE_PATTERN = "<|image_pad|>"

def is_minicpmv46_processor(processor: Any) -> bool:
    return (
        processor is not None
        and type(processor).__name__ == "MiniCPMV4_6Processor"
    )

def encode_text_len(tokenizer, text: str) -> int:

    return len(tokenizer.encode(text, add_special_tokens=False))

def _offset_image_ids(
    ids: list[int],
    tokenizer,
    image_count: int,
    image_index_offset: int,
) -> list[int]:

    if not image_index_offset:
        return ids
    start_id = tokenizer.convert_tokens_to_ids(tokenizer.image_id_start_token)
    end_id = tokenizer.convert_tokens_to_ids(tokenizer.image_id_end_token)
    starts = [i for i, token in enumerate(ids) if token == start_id]
    ends = [i for i, token in enumerate(ids) if token == end_id]
    if len(starts) != image_count or len(ends) != image_count:
        raise ValueError(
            f"expected {image_count} image-id spans, found "
            f"{len(starts)} starts and {len(ends)} ends"
        )

    result: list[int] = []
    cursor = 0
    for local_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end <= start:
            raise ValueError(f"malformed image-id span: start={start}, end={end}")
        result.extend(ids[cursor : start + 1])
        result.extend(
            tokenizer.encode(
                str(image_index_offset + local_idx),
                add_special_tokens=False,
            )
        )
        result.append(end_id)
        cursor = end + 1
    result.extend(ids[cursor:])
    return result

def collapse_image_spans_for_sglang(
    ids: list[int],
    tokenizer,
) -> list[int]:

    image_id_start = tokenizer.convert_tokens_to_ids(
        tokenizer.image_id_start_token
    )
    image_id_end = tokenizer.convert_tokens_to_ids(tokenizer.image_id_end_token)
    image_start = tokenizer.convert_tokens_to_ids("<image>")
    image_end = tokenizer.convert_tokens_to_ids("</image>")
    slice_start = tokenizer.convert_tokens_to_ids("<slice>")
    slice_end = tokenizer.convert_tokens_to_ids("</slice>")
    image_pad = tokenizer.convert_tokens_to_ids(IMAGE_PATTERN)

    def consume_span(index: int, start: int, end: int) -> int:
        if index >= len(ids) or ids[index] != start:
            raise ValueError(f"expected span start {start} at token {index}")
        try:
            return ids.index(end, index + 1) + 1
        except ValueError as exc:
            raise ValueError(f"unterminated image span starting at {index}") from exc

    result: list[int] = []
    index = 0
    collapsed = 0
    while index < len(ids):
        if ids[index] != image_id_start:
            result.append(ids[index])
            index += 1
            continue

        index = consume_span(index, image_id_start, image_id_end)
        index = consume_span(index, image_start, image_end)

        while index < len(ids):
            if ids[index] == slice_start:
                index = consume_span(index, slice_start, slice_end)
                continue
            probe = index
            while probe < len(ids) and not tokenizer.decode(
                ids[index : probe + 1],
                skip_special_tokens=False,
            ).strip():
                probe += 1
            if probe < len(ids) and ids[probe] == slice_start:
                index = consume_span(probe, slice_start, slice_end)
                continue
            break

        result.append(image_pad)
        collapsed += 1

    expected = sum(token == image_id_start for token in ids)
    if collapsed != expected:
        raise ValueError(
            f"collapsed {collapsed} MiniCPM image spans, expected {expected}"
        )
    return result

def encode_prompt(
    processor,
    prompt_text: str,
    pil_images: list,
    *,
    image_index_offset: int = 0,
) -> tuple[list[int], dict | None]:

    if not is_minicpmv46_processor(processor):
        raise TypeError(f"expected MiniCPMV4_6Processor, got {type(processor)!r}")
    if not pil_images:
        output = processor(text=prompt_text, return_tensors=None)
        return list(output["input_ids"][0]), None

    marker_count = prompt_text.count(IMAGE_PATTERN)
    if marker_count != len(pil_images):
        raise ValueError(
            f"prompt has {marker_count} image markers but {len(pil_images)} images"
        )
    output = processor(
        text=prompt_text,
        images=pil_images,
        return_tensors="pt",
    )
    ids = output["input_ids"][0]
    ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    validate_image_alignment(ids, output["target_sizes"])
    ids = _offset_image_ids(
        ids,
        processor.tokenizer,
        len(pil_images),
        image_index_offset,
    )
    mm_train = flatten_pixel_values(
        output["pixel_values"],
        output["target_sizes"],
    )
    return ids, mm_train
