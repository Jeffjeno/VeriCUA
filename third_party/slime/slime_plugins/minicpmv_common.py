from __future__ import annotations

import numpy as np
import torch

IM_START_ID = 101
IM_END_ID = 102
SLICE_START_ID = 111
SLICE_END_ID = 112
UNK_ID = 0
DEFAULT_QUERY_NUM = 64
DEFAULT_PATCH_SIZE = 14

MINICPMV_MM_KEYS = ("pixel_values_flat", "tgt_sizes")

def flatten_slice(slice_arr) -> np.ndarray:

    if isinstance(slice_arr, torch.Tensor):
        slice_arr = slice_arr.numpy()
    arr = np.asarray(slice_arr, dtype=np.float32)
    assert arr.ndim == 3, f"expected [3, patch, L], got {arr.shape}"
    c, p, length = arr.shape
    return np.ascontiguousarray(arr.reshape(c * p, length).transpose(1, 0))

def flatten_image_inputs(pixel_values_nested, tgt_sizes_nested) -> dict | None:

    flats: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    for batch_slices, batch_tgt in zip(pixel_values_nested, tgt_sizes_nested, strict=True):
        if batch_slices is None or len(batch_slices) == 0:
            continue
        for s in batch_slices:
            flats.append(flatten_slice(s))
        tgt = batch_tgt.numpy() if isinstance(batch_tgt, torch.Tensor) else np.asarray(batch_tgt)
        tgts.append(tgt.reshape(-1, 2).astype(np.int32))
    if not flats:
        return None
    tgt_sizes = np.concatenate(tgts, axis=0)
    n_from_tgt = int(tgt_sizes.shape[0])
    assert n_from_tgt == len(flats), f"tgt_sizes rows {n_from_tgt} != num slices {len(flats)}"
    return {
        "pixel_values_flat": np.concatenate(flats, axis=0),
        "tgt_sizes": tgt_sizes,
    }

def unflatten_pixel_values(
    pixel_values_flat: torch.Tensor,
    tgt_sizes: torch.Tensor,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> list[torch.Tensor]:

    lengths = (tgt_sizes[:, 0].long() * tgt_sizes[:, 1].long() * patch_size).tolist()
    total = int(sum(lengths))
    assert pixel_values_flat.shape[0] == total, (
        f"pixel_values_flat rows {pixel_values_flat.shape[0]} != sum(h*w*patch)={total}; "
        "tgt_sizes and pixel_values_flat are out of sync"
    )
    return list(torch.split(pixel_values_flat, lengths, dim=0))

def scan_image_bounds(
    token_ids: torch.Tensor,
    query_num: int = DEFAULT_QUERY_NUM,
) -> torch.Tensor:

    flat = token_ids.reshape(-1)
    open_mask = (flat == IM_START_ID) | (flat == SLICE_START_ID)
    close_mask = (flat == IM_END_ID) | (flat == SLICE_END_ID)
    close_positions = torch.where(close_mask)[0].tolist()

    pairs: list[tuple[int, int]] = []
    for e in close_positions:
        s = e - query_num
        if s < 1:
            continue
        if not bool(open_mask[s - 1]):
            continue
        if not bool(torch.all(flat[s:e] == UNK_ID)):
            continue
        pairs.append((s, e))

    n_open = int(open_mask.sum())
    n_close = len(close_positions)
    if n_open != len(pairs) or n_close != len(pairs):
        import logging

        logging.getLogger(__name__).warning(
            "scan_image_bounds: %d open / %d close tags but %d validated spans "
            "(stray model-sampled tags ignored)",
            n_open,
            n_close,
            len(pairs),
        )
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=token_ids.device)
    return torch.tensor(pairs, dtype=torch.long, device=token_ids.device)
