import logging
import os
from pathlib import Path

import torch

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

def _resolve_path(args, rollout_id: int, evaluation: bool) -> str | None:
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError(
            "--load-forge-rollout-data not set. Pass the dump path, "
            "e.g. /path/to/rollout_data/0.pt (literal) or "
            "/path/to/rollout_data/{rollout_id}.pt (template)."
        )

    if evaluation and "{rollout_id}" not in tpl:
        return None
    rid_str = ("eval_" if evaluation else "") + str(rollout_id)
    path = tpl.format(rollout_id=rid_str)
    if os.path.exists(path):
        return path

    if not evaluation:
        fallback = tpl.format(rollout_id="0")
        if os.path.exists(fallback):
            logger.info("forge_load: %s missing, falling back to %s", path, fallback)
            return fallback
    return None

def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:

        if path is None:
            logger.info("forge_load: no eval dump found; returning empty eval result")
            return RolloutFnEvalOutput(data={})
        logger.info("forge_load: loading eval samples from %s", path)
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]

        reward_key = args.eval_reward_key or args.reward_key
        rewards = [s.reward if (not reward_key or s.reward is None) else s.reward[reward_key] for s in samples]
        return RolloutFnEvalOutput(
            data={
                "forge_eval": {
                    "rewards": [r if r is not None else 0.0 for r in rewards],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
                    "samples": samples,
                }
            }
        )

    if path is None:
        raise RuntimeError(
            f"forge_load: no dump found for rollout_id={rollout_id} "
            f"(--load-forge-rollout-data={args.load_forge_rollout_data!r})"
        )

    logger.info("forge_load: loading samples from %s", path)
    blob = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(s) for s in blob["samples"]]

    logger.info(
        "forge_load: loaded %d samples for rollout_id=%d from %s",
        len(samples),
        rollout_id,
        Path(path).name,
    )
    return RolloutFnTrainOutput(samples=samples)
