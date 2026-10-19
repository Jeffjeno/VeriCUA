from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

LOGICAL_MARKERS: frozenset[str] = frozenset({"DONE", "FAIL", "WAIT"})

def is_logical_marker(action: object) -> bool:

    return isinstance(action, str) and action.strip().upper() in LOGICAL_MARKERS

def is_terminal_marker(action: object) -> bool:

    return isinstance(action, str) and action.strip().upper() in {"DONE", "FAIL"}

def step_or_handle_marker(
    env_client: Any,
    lease_id: str,
    action: str,
    sleep_after_execution: float,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:

    if is_logical_marker(action):
        time.sleep(sleep_after_execution)
        obs = env_client.get_obs(lease_id)
        done = is_terminal_marker(action)
        return obs, 0.0, done, {}

    return env_client.step(lease_id, action)
