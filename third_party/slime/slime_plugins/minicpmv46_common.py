from __future__ import annotations

import numpy as np
import torch

IMAGE_TOKEN_ID = 248056
DEFAULT_PATCH_SIZE = 14
DEFAULT_DOWNSAMPLE_DIVISOR = 16
MINICPMV46_MM_KEYS = ("pixel_values_flat", "target_sizes")

def flatten_pixel_values(pixel_values, target_sizes) -> dict | None:

    if pixel_values is None:
        return None
    if isinstance(pixel_values, torch.Tensor):
        pixels = pixel_values.detach().cpu().numpy()
    else:
        pixels = np.asarray(pixel_values)
    if isinstance(target_sizes, torch.Tensor):
        sizes = target_sizes.detach().cpu().numpy()
    else:
        sizes = np.asarray(target_sizes)

    if pixels.ndim == 3:
        pixels = pixels[None, ...]
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError(
            "MiniCPM-V 4.6 pixels must be [1, 3, patch_size, sum_L], "
            f"got {pixels.shape}"
        )
    sizes = sizes.reshape(-1, 2).astype(np.int32, copy=False)
    patch_size = int(pixels.shape[2])
    expected_length = int(np.prod(sizes, axis=1).sum()) * patch_size
    if pixels.shape[3] != expected_length:
        raise ValueError(
            f"packed pixel length {pixels.shape[3]} != "
            f"sum(h*w)*patch_size {expected_length}"
        )

    flat = pixels[0].reshape(3 * patch_size, -1).transpose(1, 0)
    return {
        "pixel_values_flat": np.ascontiguousarray(flat, dtype=np.float32),
        "target_sizes": np.ascontiguousarray(sizes),
    }

def unflatten_pixel_values(
    pixel_values_flat: torch.Tensor,
    target_sizes: torch.Tensor,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:

    if pixel_values_flat.ndim != 2:
        raise ValueError(
            f"pixel_values_flat must be rank 2, got {pixel_values_flat.shape}"
        )
    if pixel_values_flat.shape[1] != 3 * patch_size:
        raise ValueError(
            f"pixel feature width {pixel_values_flat.shape[1]} != {3 * patch_size}"
        )
    sizes = target_sizes.reshape(-1, 2)
    expected_length = int(
        (sizes[:, 0].long() * sizes[:, 1].long()).sum().item()
    ) * patch_size
    if pixel_values_flat.shape[0] != expected_length:
        raise ValueError(
            f"pixel rows {pixel_values_flat.shape[0]} != "
            f"sum(h*w)*patch_size {expected_length}"
        )
    return (
        pixel_values_flat.transpose(0, 1)
        .contiguous()
        .reshape(1, 3, patch_size, expected_length)
    )

def expected_image_features(
    target_sizes: torch.Tensor | np.ndarray,
    divisor: int = DEFAULT_DOWNSAMPLE_DIVISOR,
) -> int:

    sizes = torch.as_tensor(target_sizes).reshape(-1, 2).long()
    merge_side = int(divisor**0.5)
    if merge_side * merge_side != divisor:
        raise ValueError(f"downsample divisor must be a square, got {divisor}")
    if bool(torch.any(sizes % merge_side)):
        raise ValueError(
            f"target height and width must each be divisible by {merge_side}: "
            f"{sizes.tolist()}"
        )
    areas = sizes[:, 0] * sizes[:, 1]
    return int((areas // divisor).sum().item())

def count_image_tokens(token_ids: torch.Tensor | list[int]) -> int:
    return int((torch.as_tensor(token_ids).reshape(-1) == IMAGE_TOKEN_ID).sum())

def validate_image_alignment(
    token_ids: torch.Tensor | list[int],
    target_sizes: torch.Tensor | np.ndarray,
    divisor: int = DEFAULT_DOWNSAMPLE_DIVISOR,
) -> int:

    tokens = count_image_tokens(token_ids)
    features = expected_image_features(target_sizes, divisor)
    if tokens != features:
        raise ValueError(
            f"MiniCPM-V 4.6 image tokens/features mismatch: {tokens} != {features}"
        )
    return features
