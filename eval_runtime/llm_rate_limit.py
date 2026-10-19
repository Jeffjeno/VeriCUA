from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

def throttle_shared_llm_request() -> None:

    lock_path = os.environ.get("AGENTLAB_LLM_RATE_LIMIT_FILE", "").strip()
    if not lock_path:
        return
    interval = float(os.environ.get("AGENTLAB_LLM_MIN_INTERVAL", "0"))
    if interval <= 0:
        return

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        lock_file.seek(0)
        raw_last = lock_file.read().strip()
        last = float(raw_last) if raw_last else 0.0
        now = time.monotonic()
        delay = last + interval - now
        if delay > 0:
            time.sleep(delay)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(time.monotonic()))
        lock_file.flush()
        os.fsync(lock_file.fileno())
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
