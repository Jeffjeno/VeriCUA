from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("eval_runtime.vagen_active.env_adapter")

class VagenEnvAdapter:

    def __init__(self, env_client: Any, lease_id: str, shell_timeout: int = 120):
        self._client = env_client
        self._lease_id = lease_id
        self._shell_timeout = shell_timeout

    def _resolve_controller(self):

        sessions = getattr(self._client, "_sessions", None)
        entry = (sessions or {}).get(self._lease_id)
        env = (entry or {}).get("env")
        desktop_env = getattr(env, "desktop_env", None)
        controller = getattr(desktop_env, "controller", None)
        if controller is None:
            raise RuntimeError(
                "VagenEnvAdapter could not resolve the DesktopEnv controller "
                f"for lease_id={self._lease_id!r}. The eval_runtime internals "
                "(_sessions[lease]['env'].desktop_env.controller) may have changed."
            )
        return controller

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, Any]:

        action_type = str(action.get("action_type") or "")
        command = action.get("command") or ""

        if action_type == "EXECUTE_PYTHON":
            controller = self._resolve_controller()
            result = controller.run_python_script(command)
            logger.info("EXECUTE_PYTHON via controller: %s", str(result)[:200])

            return {}, 0.0, False, result

        if action_type == "EXECUTE_SHELL":
            controller = self._resolve_controller()
            result = controller.run_bash_script(command, timeout=self._shell_timeout)
            logger.info("EXECUTE_SHELL via controller: %s", str(result)[:200])
            return {}, 0.0, False, result

        obs, reward, done, info = self._client.step(self._lease_id, command)

        return obs, float(reward or 0.0), False, info
