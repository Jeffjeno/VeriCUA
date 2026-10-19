from __future__ import annotations

import os
import sys

_AGENT_RUNTIME_SRC = os.environ.get(
    "AGENT_RUNTIME_PATH",
    os.path.join(os.path.dirname(__file__), ".."),
)
_AGENT_RUNTIME_SRC = os.path.abspath(_AGENT_RUNTIME_SRC)
if _AGENT_RUNTIME_SRC not in sys.path:
    sys.path.insert(0, _AGENT_RUNTIME_SRC)

_OSWORLD_PATH = os.environ.get(
    "OSWORLD_PATH",
    os.path.join(os.path.dirname(__file__), ".."),
)
_OSWORLD_PATH = os.path.abspath(_OSWORLD_PATH)
if _OSWORLD_PATH not in sys.path:
    sys.path.insert(0, _OSWORLD_PATH)
