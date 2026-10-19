from __future__ import annotations

import logging
from typing import Any

from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)

def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:

    if use_dynamic_batch_size:
        assert max_per_bin is not None
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]

def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    group_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:

    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    group_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, group_id in enumerate(group_indices):
        group_id_to_samples.setdefault(group_id, []).append(sample_pos)
    group_ids = list(group_id_to_samples.keys())

    num_steps = len(group_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_groups ({len(group_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one group per step."
    )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_groups = group_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for group_id in step_groups for pos in group_id_to_samples[group_id]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
        )

        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        if args.balance_data:
            mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_token_sums, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
