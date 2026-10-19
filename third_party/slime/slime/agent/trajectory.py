from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class TurnRecord:

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(frozen=True)
class TokenSegment:

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(frozen=True)
class TurnSegment:

    turns: list[TurnRecord]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnSegment:

    frozen_turns = list(turns)
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault("finish_reason", frozen_turns[-1].finish_reason if frozen_turns else "")
    return TurnSegment(turns=frozen_turns, metadata=segment_metadata)

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _output_log_probs(turn: TurnRecord) -> list[float]:
    if len(turn.output_log_probs) == len(turn.output_ids):
        return list(turn.output_log_probs)
    logger.warning(
        "[trajectory] turn logprob length mismatch; zeroing output logprobs (%d ids, %d logprobs)",
        len(turn.output_ids),
        len(turn.output_log_probs),
    )
    return [0.0] * len(turn.output_ids)

def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:

    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_spans: list[tuple[int, int]] = []

    for i, turn in enumerate(turns):
        if i > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning("[trajectory] merge prompt base changed; starting segment from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
                rollout_log_probs = []
                output_spans = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    logger.warning(
                        "[trajectory] merge prefix drift; truncating %d unstitched response tokens",
                        len(response_ids) - matched_len,
                    )
                    for start, end in output_spans:
                        if start < matched_len < end:
                            loss_mask[start:matched_len] = [0] * (matched_len - start)
                            rollout_log_probs[start:matched_len] = [0.0] * (matched_len - start)
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]
                    rollout_log_probs = rollout_log_probs[:matched_len]
                    output_spans = [
                        (start, min(end, matched_len)) for start, end in output_spans if start < matched_len
                    ]

                context_tail = prompt_suffix[matched_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
                rollout_log_probs.extend([0.0] * len(context_tail))

        output_start = len(response_ids)
        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        rollout_log_probs.extend(_output_log_probs(turn))
        output_spans.append((output_start, len(response_ids)))

    rollout_log_probs = [logprob if mask else 0.0 for logprob, mask in zip(rollout_log_probs, loss_mask, strict=True)]

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        metadata=dict(metadata or {}),
    )

def merge_turn_segments(segments: list[TurnSegment]) -> list[TokenSegment]:

    out: list[TokenSegment] = []
    for turn_segment in segments:
        token_segment = merge_turns(turn_segment.turns, metadata=turn_segment.metadata)
        if token_segment is None:
            continue
        if token_segment.response_ids:
            out.append(token_segment)
    return out

def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:

    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    sample.response = tokenizer.decode(segment.response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED

def fan_out_sample_segments(
    sample: Sample,
    segments: list[TokenSegment],
    reward: float,
    tokenizer,
    *,
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:

    k = len(segments)
    per_segment_reward = float(reward) / max(1, k)
    shared_group_id = sample.group_id if sample.group_id is not None else sample.index
    base_metadata = {**(sample.metadata or {}), **(metadata or {})}

    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, per_segment_reward, tokenizer)
        sub.group_id = shared_group_id
        sub.metadata = {
            **base_metadata,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
