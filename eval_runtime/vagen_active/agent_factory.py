from __future__ import annotations

import logging
import os
import re
from typing import Any

import yaml

import eval_runtime.vagen_active

logger = logging.getLogger("eval_runtime.vagen_active.agent_factory")

VAGEN_EVALUATOR_TYPE = "vagen_claude"
VAGEN_QWEN_EVALUATOR_TYPE = "vagen_qwen"
VAGEN_EVALUATOR_TYPES = (VAGEN_EVALUATOR_TYPE, VAGEN_QWEN_EVALUATOR_TYPE)


_DEFAULTS: dict[str, Any] = {
    "platform": "Ubuntu",
    "max_tokens": 4096,

    "max_round": 35,
    "only_n_most_recent_images_eval": 10,

    "temperature": 0.0,
    "top_p": 1.0,
    "screen_width": 1920,
    "screen_height": 1080,
    "system_prompt_suffix": "",
    "shell_timeout": 120,
    "low_confidence_as_uncertain": False,

    "api_format": "anthropic",
}

def _expand(value: Any) -> Any:

    if isinstance(value, str):

        value = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]+([^}]*)\}",
            lambda m: os.environ.get(m.group(1), m.group(2)),
            value,
        )
        return os.path.expandvars(value)
    return value

def load_vagen_eval_config(config_path: str) -> dict[str, Any]:

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid vagen eval config: {config_path}")

    evaluator_type = str(raw.get("evaluator_type") or "").strip().lower()
    if evaluator_type not in (VAGEN_EVALUATOR_TYPE, VAGEN_QWEN_EVALUATOR_TYPE):
        raise ValueError(
            f"vagen active baseline expects evaluator_type in "
            f"({VAGEN_EVALUATOR_TYPE!r}, {VAGEN_QWEN_EVALUATOR_TYPE!r}), "
            f"got {evaluator_type!r} in {config_path}"
        )

    cfg = dict(_DEFAULTS)

    _NULLABLE = {"temperature", "top_p"}
    cfg.update({
        k: v for k, v in raw.items()
        if v is not None or k in _NULLABLE
    })

    for key, value in list(cfg.items()):
        cfg[key] = _expand(value)
    cfg["evaluator_type"] = evaluator_type
    cfg["config_path"] = os.path.abspath(config_path)

    cfg["model"] = _expand(cfg.get("model") or "")
    cfg["api_url"] = _expand(cfg.get("api_url") or "")
    cfg["api_key"] = _expand(cfg.get("api_key") or "")
    if not cfg["model"]:
        raise ValueError(f"vagen eval config missing 'model': {config_path}")
    if not cfg["api_url"]:
        raise ValueError(
            f"vagen eval config missing 'api_url' (set it in the YAML or via "
            f"VAGEN_EVAL_API_URL env expansion): {config_path}"
        )
    return cfg

def peek_is_vagen_config(config_path: str) -> bool:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return str(raw.get("evaluator_type") or "").strip().lower() in (
            VAGEN_EVALUATOR_TYPE, VAGEN_QWEN_EVALUATOR_TYPE
        )
    except Exception:
        return False

def build_vagen_eval_agent(cfg: dict[str, Any]):

    if cfg.get("evaluator_type") == VAGEN_QWEN_EVALUATOR_TYPE:
        platform = str(cfg.get("platform") or "Ubuntu").strip().lower()
        if platform == "mobileworld":
            from eval_runtime.mobile_world.vagen_agent import (
                MobileWorldQwenVagenEvalAgent,
            )
            agent_cls = MobileWorldQwenVagenEvalAgent
        elif platform in {"webarena", "web_arena"}:
            from eval_runtime.web_arena.vagen_agent import WebArenaQwenVagenEvalAgent
            agent_cls = WebArenaQwenVagenEvalAgent
        elif platform in {"ubuntu", "osworld"}:
            from eval_runtime.vagen_active.qwen_eval import QwenVagenEvalAgent
            agent_cls = QwenVagenEvalAgent
        else:
            raise ValueError(
                "vagen_qwen does not support platform "
                f"{cfg.get('platform')!r}; use a platform-specific adapter"
            )
        if platform in {"mobileworld", "webarena", "web_arena"}:
            return agent_cls(
                platform=cfg["platform"],
                model=cfg["model"],
                api_url=cfg["api_url"],
                api_key=cfg.get("api_key") or "EMPTY",
                max_tokens=int(cfg["max_tokens"]),
                temperature=cfg.get("temperature"),
                top_p=cfg.get("top_p"),
                screen_size=(int(cfg["screen_width"]), int(cfg["screen_height"])),
                only_n_most_recent_images_eval=int(
                    cfg["only_n_most_recent_images_eval"]
                ),
                request_timeout=float(cfg.get("request_timeout", 120.0)),
            )
        return agent_cls(
            model=cfg["model"],
            api_url=cfg["api_url"],
            api_key=cfg.get("api_key") or "EMPTY",
            max_tokens=int(cfg["max_tokens"]),
            temperature=cfg.get("temperature"),
            top_p=cfg.get("top_p"),
            screen_size=(int(cfg["screen_width"]), int(cfg["screen_height"])),
            only_n_most_recent_images_eval=int(cfg["only_n_most_recent_images_eval"]),
            enable_thinking=bool(cfg.get("enable_thinking", False)),
                max_retries=int(cfg.get("max_retries", 3)),
                format_retry_rounds=int(cfg.get("format_retry_rounds", 2)),
                request_timeout=float(cfg.get("request_timeout", 120.0)),
        )

    if str(cfg.get("platform") or "").strip().lower() == "mobileworld":
        from eval_runtime.mobile_world.vagen_agent import (
            MobileWorldClaudeEvalAgent as AnthropicAgent45Eval,
        )
    else:
        from src.agent.anthropic.main_45_eval import AnthropicAgent45Eval

    api_url = cfg["api_url"]
    api_format = str(cfg.get("api_format") or "anthropic").strip().lower()
    if api_format == "anthropic":
        from eval_runtime.vagen_active.anthropic_transport import get_shim_url
        api_url = get_shim_url(cfg["api_url"], fallback_api_key=cfg.get("api_key") or "")
    elif api_format != "openai":
        raise ValueError(f"api_format must be 'anthropic' or 'openai', got {api_format!r}")

    agent = AnthropicAgent45Eval(
        platform=cfg["platform"],
        model=cfg["model"],
        max_tokens=int(cfg["max_tokens"]),
        api_url=api_url,
        api_key=cfg["api_key"],
        system_prompt_suffix=cfg.get("system_prompt_suffix") or "",
        only_n_most_recent_images_eval=int(cfg["only_n_most_recent_images_eval"]),
        screen_size=(int(cfg["screen_width"]), int(cfg["screen_height"])),
        temperature=cfg.get("temperature"),
        top_p=cfg.get("top_p"),
        customer_tools=(
            []
            if str(cfg.get("platform") or "").strip().lower() == "mobileworld"
            else AnthropicAgent45Eval.DEFAULT_TOOLS
        ),
    )
    logger.info(
        "Built VAGEN eval agent: model=%s api_format=%s upstream=%s max_round=%s",
        cfg["model"], api_format, cfg["api_url"], cfg["max_round"],
    )
    return agent
