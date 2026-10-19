from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from desktop_env.evaluators import getters, metrics

logger = logging.getLogger("desktopenv.inline_eval")


def last_action_is_fail(env: Any) -> bool:
    action_history = getattr(env, "action_history", [])
    if not action_history:
        return False
    last_action = action_history[-1]
    return last_action == "FAIL" or (
        isinstance(last_action, dict) and last_action.get("action_type") == "FAIL"
    )


def call_inline_verifier(verifier: Callable[..., Any], env: Any) -> Any:
    signature = inspect.signature(verifier)
    required_params = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if not required_params:
        return verifier()
    return verifier(env)


def inline_eval_result_to_score(result: Any) -> float:
    if result is None:
        return 1.0
    if isinstance(result, dict):
        for key in ("score", "success", "passed", "pass"):
            if key in result:
                return inline_eval_result_to_score(result[key])
        return 1.0 if result else 0.0
    if isinstance(result, str):
        normalized = result.strip().lower()
        if normalized in {"1", "true", "pass", "passed", "success", "yes"}:
            return 1.0
        if normalized in {"0", "false", "fail", "failed", "failure", "no", ""}:
            return 0.0
    return 1.0 if bool(result) else 0.0


def evaluate_inline_code_on_env(
    env: Any,
    code: str,
    *,
    respect_fail_action: bool = True,
    source_name: str = "<inline_eval>",
) -> float:
    if respect_fail_action and last_action_is_fail(env):
        return 0.0
    if not isinstance(code, str):
        raise TypeError("Inline evaluator code must be a string.")
    if not code.strip():
        raise ValueError("Inline evaluator code cannot be empty.")

    namespace = {
        "__name__": "__inline_eval__",
        "__file__": source_name,
        "env": env,
        "getters": getters,
        "metrics": metrics,
        "logger": logger,
    }

    try:
        exec(compile(code, source_name, "exec"), namespace, namespace)
        verifier = namespace.get("verifier")
        if callable(verifier):
            result = call_inline_verifier(verifier, env)
        elif "score" in namespace:
            result = namespace["score"]
        else:
            raise ValueError("Inline eval code must define verifier or score.")
        return inline_eval_result_to_score(result)
    except AssertionError as e:
        logger.info("Inline verifier assertion failed: %s", e)
        return 0.0
    except Exception:
        logger.exception("Inline verifier failed.")
        raise
