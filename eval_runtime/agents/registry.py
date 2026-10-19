from __future__ import annotations

from typing import Any

_AGENT_REGISTRY: dict[str, type] = {}

def register(name: str):

    def decorator(cls):
        _AGENT_REGISTRY[name] = cls
        return cls
    return decorator

def create_agent(agent_type: str, **kwargs: Any):

    if agent_type not in _AGENT_REGISTRY:
        raise KeyError(
            f"Unknown agent type '{agent_type}'. "
            f"Available: {sorted(_AGENT_REGISTRY.keys())}"
        )
    cls = _AGENT_REGISTRY[agent_type]
    return cls(**kwargs)

def list_agents() -> list[str]:
    return sorted(_AGENT_REGISTRY.keys())
