from __future__ import annotations

import logging
import os
import sys

_DEFAULT_LEVEL = os.environ.get("AGENT_RUNTIME_LOG_LEVEL", "INFO").upper()
_INITIALISED = False

def _init_root_logger() -> None:
    global _INITIALISED
    if _INITIALISED:
        return

    root = logging.getLogger("agent_runtime")
    root.setLevel(_DEFAULT_LEVEL)
    root.propagate = False

    if not root.handlers:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
        root.addHandler(handler)

    _INITIALISED = True

def get_logger(name: str | None = None) -> logging.Logger:

    _init_root_logger()
    if name is None or name == "agent_runtime":
        return logging.getLogger("agent_runtime")
    if name.startswith("agent_runtime."):
        return logging.getLogger(name)
    return logging.getLogger(f"agent_runtime.{name}")

def set_level(level: str | int) -> None:
    _init_root_logger()
    logging.getLogger("agent_runtime").setLevel(level)

__all__ = ["get_logger", "set_level"]
