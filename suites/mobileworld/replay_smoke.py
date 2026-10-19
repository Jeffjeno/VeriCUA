#!/usr/bin/env python3
"""Replay one mobileworld trajectory on MobileWorld without calling an LLM."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_runtime.mobile_world.dataset import (
    load_mobile_world_dataset,
    materialize_episode,
)
from eval_runtime.mobile_world.env_client import MobileWorldEvalEnvClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AcceptMeetingTask")
    parser.add_argument("--backend", default="http://127.0.0.1:6800")
    parser.add_argument("--step-wait-time", type=float, default=3.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = load_mobile_world_dataset("mobileworld", project_root=PROJECT_ROOT)
    episode = dataset.load_episode(args.task)
    client = MobileWorldEvalEnvClient(
        args.backend,
        step_wait_time=args.step_wait_time,
    )
    replay = None
    try:
        replay = client.replay_episode(episode)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            materialize_episode(
                episode,
                output,
                initial_screenshot=replay.initial_screenshot,
                post_action_screenshots=[
                    step.screenshot for step in replay.steps
                ],
                native_score=replay.score,
                native_reason=replay.reason,
                overwrite=True,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="mobileworld_replay_smoke_") as tmp:
                materialize_episode(
                    episode,
                    Path(tmp) / args.task,
                    initial_screenshot=replay.initial_screenshot,
                    post_action_screenshots=[
                        step.screenshot for step in replay.steps
                    ],
                    native_score=replay.score,
                    native_reason=replay.reason,
                )
    finally:
        try:
            client.tear_down_task(args.task)
        finally:
            client.close()

    assert replay is not None
    payload = {
        "task": args.task,
        "backend": args.backend,
        "steps": len(replay.steps),
        "expected_steps": len(episode.steps),
        "original_score": episode.record.original_score,
        "replay_score": replay.score,
        "reason": replay.reason,
        "score_matches": (
            (replay.score >= 1.0)
            == (episode.record.original_score >= 1.0)
        ),
        "output": str(Path(args.output).expanduser().resolve()) if args.output else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["score_matches"] and payload["steps"] == payload["expected_steps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
