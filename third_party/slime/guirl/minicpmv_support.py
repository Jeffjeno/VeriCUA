from __future__ import annotations

from typing import Any

from slime_plugins.minicpmv_common import flatten_image_inputs

IMAGE_PATTERN = "(<image>./</image>)"

_VISION_TAGS = (
    "<image>", "</image>", "<slice>", "</slice>", "<image_id>", "</image_id>",
    "<ref>", "</ref>", "<box>", "</box>", "<quad>", "</quad>",
    "<point>", "</point>", "<unit>", "</unit>",
)

def ensure_image_special_tokens(tokenizer) -> None:

    if tokenizer is None or getattr(tokenizer, "_guirl_vision_tags_ok", False):
        return
    probe = tokenizer.encode("<image>", add_special_tokens=False)
    if probe != [101]:
        from tokenizers import AddedToken

        backend = getattr(tokenizer, "backend_tokenizer", None) or getattr(
            tokenizer, "_tokenizer", None
        )
        if backend is None:
            raise RuntimeError(
                "MiniCPM tokenizer lost vision special tokens and has no fast "
                f"backend to repair (encode('<image>') -> {probe})"
            )
        backend.add_special_tokens(
            [AddedToken(tag, special=True, normalized=False) for tag in _VISION_TAGS]
        )
        assert tokenizer.encode("<image>", add_special_tokens=False) == [101]
    tokenizer._guirl_vision_tags_ok = True

def is_minicpmv_processor(processor: Any) -> bool:
    return processor is not None and type(processor).__name__ == "MiniCPMVProcessor"

def flatten_messages_for_minicpmv(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    flat: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            flat.append(message)
            continue
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
            elif item.get("type") == "image_url":
                parts.append(IMAGE_PATTERN)
            elif item.get("type") == "text":
                parts.append(item.get("text", ""))
        flat.append({**message, "content": "\n".join(parts)})
    return flat

def encode_text_len(tokenizer, text: str) -> int:

    ensure_image_special_tokens(tokenizer)
    return len(tokenizer.encode(text))

def encode_prompt(
    processor,
    prompt_text: str,
    pil_images: list,
    *,
    image_index_offset: int = 0,
) -> tuple[list[int], dict | None]:

    ensure_image_special_tokens(processor.tokenizer)
    if not pil_images:
        ids = processor.tokenizer.encode(prompt_text)
        return list(ids), None

    n_markers = prompt_text.count(IMAGE_PATTERN)
    assert n_markers == len(pil_images), (
        f"prompt has {n_markers} image markers but {len(pil_images)} images"
    )

    image_processor = processor.image_processor
    original = image_processor.get_slice_image_placeholder

    def _offset_placeholder(image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        return original(image_size, image_idx + image_index_offset, max_slice_nums, use_image_id)

    if image_index_offset:
        image_processor.get_slice_image_placeholder = _offset_placeholder
    try:
        output = processor(text=prompt_text, images=pil_images, return_tensors="pt")
    finally:
        if image_index_offset:
            image_processor.get_slice_image_placeholder = original

    ids = output["input_ids"][0]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    mm_train = flatten_image_inputs(output["pixel_values"], output["tgt_sizes"])
    return ids, mm_train
