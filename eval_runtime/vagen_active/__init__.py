from __future__ import annotations

import os
import sys

import eval_runtime

_VAGEN_PATH = os.environ.get(
    "VAGEN_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "VAGEN"),
)
_VAGEN_PATH = os.path.abspath(_VAGEN_PATH)

if _VAGEN_PATH not in sys.path:
    sys.path.append(_VAGEN_PATH)
