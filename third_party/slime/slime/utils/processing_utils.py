import base64
import io
import json
import logging
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14

def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)

def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result

def _try_load_glm4v_processor(name_or_path: str, **kwargs):

    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc

def _load_minicpmv46_processor(name_or_path: str, **kwargs):

    config_path = Path(name_or_path) / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as config_file:
        if json.load(config_file).get("model_type") != "minicpmv4_6":
            return None
    try:
        from transformers.models.minicpmv4_6 import MiniCPMV4_6Processor
    except ImportError as exc:
        raise RuntimeError(
            "MiniCPM-V 4.6 requires the pinned Transformers 5.12.1 upstream "
            "MiniCPMV4_6Processor; activate the GUIRL Transformers overlay"
        ) from exc
    proc = MiniCPMV4_6Processor.from_pretrained(name_or_path, **kwargs)
    max_slice_nums = os.environ.get("MINICPMV46_MAX_SLICE_NUMS")
    if max_slice_nums is not None:
        try:
            max_slice_nums_int = int(max_slice_nums)
        except ValueError as exc:
            raise ValueError(
                "MINICPMV46_MAX_SLICE_NUMS must be a positive integer, "
                f"got {max_slice_nums!r}"
            ) from exc
        if max_slice_nums_int <= 0:
            raise ValueError(
                "MINICPMV46_MAX_SLICE_NUMS must be positive, "
                f"got {max_slice_nums_int}"
            )
        for component_name in ("image_processor", "video_processor"):
            component = getattr(proc, component_name, None)
            if component is not None and hasattr(component, "max_slice_nums"):
                component.max_slice_nums = max_slice_nums_int
        logger.info(
            "Configured MiniCPMV4_6Processor max_slice_nums=%d from runtime",
            max_slice_nums_int,
        )
    logger.info("Loaded upstream MiniCPMV4_6Processor for %s", name_or_path)
    return proc

def load_processor(name_or_path: str, **kwargs):
    if proc := _load_minicpmv46_processor(name_or_path, **kwargs):
        return proc
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):

        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc

def _extract_images_from_messages(messages):

    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:

                        images.append(Image.open(image_data))
    return images

def process_vision_info(prompt, processor):

    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    except Exception:

        images = _extract_images_from_messages(prompt) or None
        videos = None

    return {"images": images, "videos": videos}

def encode_image_for_rollout_engine(image) -> str:

    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
