from __future__ import annotations

import os
import sys

_AGENT_RUNTIME_PATH = os.environ.get(
    "AGENT_RUNTIME_PATH",
    os.path.join(os.path.dirname(__file__), ".."),
)
_AGENT_RUNTIME_PATH = os.path.abspath(_AGENT_RUNTIME_PATH)
if _AGENT_RUNTIME_PATH not in sys.path:
    sys.path.insert(0, _AGENT_RUNTIME_PATH)
