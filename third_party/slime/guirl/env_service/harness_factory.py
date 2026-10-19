from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("guirl.harness_factory")

def _ensure_agent_runtime_importable() -> None:

    agent_runtime_src = os.environ.get(
        "AGENT_RUNTIME_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."),
    )
    agent_runtime_src = os.path.abspath(agent_runtime_src)
    if agent_runtime_src not in sys.path:
        sys.path.insert(0, agent_runtime_src)

def create_guirl_harness(harness_type: str, kwargs: dict):

    _ensure_agent_runtime_importable()

    if harness_type == "holo3":
        from agent_runtime.harnesses.holo3 import Holo3Harness
        return Holo3Harness(**kwargs)

    if harness_type == "holo31":
        from agent_runtime.harnesses.holo31 import Holo31Harness
        return Holo31Harness(**kwargs)

    if harness_type == "qwen35vl":
        from agent_runtime.harnesses.qwen35vl import Qwen35VLHarness
        return Qwen35VLHarness(**kwargs)

    if harness_type == "qwen3vl":
        from agent_runtime.harnesses.qwen3vl import Qwen3VLHarness
        return Qwen3VLHarness(**kwargs)

    if harness_type == "qwen3vl-probe":
        from agent_runtime.harnesses.qwen3vl_probe import Qwen3VLProbeHarness
        return Qwen3VLProbeHarness(**kwargs)

    if harness_type == "holo31-probe":
        from agent_runtime.harnesses.holo31_probe import Holo31ProbeHarness
        return Holo31ProbeHarness(**kwargs)

    if harness_type == "holo3-probe":
        from agent_runtime.harnesses.holo3_probe import Holo3ProbeHarness
        return Holo3ProbeHarness(**kwargs)

    if harness_type == "evocua":
        from agent_runtime.harnesses.evocua import EvoCUAHarness
        return EvoCUAHarness(**kwargs)

    if harness_type == "owl15":
        from agent_runtime.harnesses.owl15 import Owl15Harness
        return Owl15Harness(**kwargs)

    import agent_runtime as ap
    harness_cls = ap.resolve_harness(harness_type)
    logger.info("Using generic fallback for harness %r -> %s", harness_type, harness_cls.__name__)
    return harness_cls(**kwargs)
