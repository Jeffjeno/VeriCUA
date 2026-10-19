from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any

import aiohttp

logger = logging.getLogger("guirl.env_rpc_client")

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=30)
HEALTH_TIMEOUT = aiohttp.ClientTimeout(total=10, connect=5)
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 2.0

class RpcEnvError(RuntimeError):
    def __init__(
        self,
        endpoint: str,
        *,
        status: int,
        code: str | None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{endpoint} failed: {message}")
        self.endpoint = endpoint
        self.status = int(status)
        self.code = str(code) if code is not None else None
        self.details = dict(details or {})

def _decode_obs(obs: dict[str, Any]) -> dict[str, Any]:

    out = dict(obs or {})
    screenshot_b64 = out.pop("screenshot_b64", None)
    if screenshot_b64:
        out["screenshot"] = base64.b64decode(screenshot_b64)
    else:
        out["screenshot"] = b""
    return out

class AsyncRpcEnvClient:

    def __init__(
        self,
        base_url: str,
        max_retries: int | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries or int(os.getenv("GUI_ENV_HTTP_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        self.timeout = timeout or DEFAULT_TIMEOUT
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:

        if endpoint in {
            "/reset", "/step", "/get_obs", "/prepare_for_probe", "/evaluate", "/close",
        } and payload.get("operation_id") is None:
            payload = {**payload, "operation_id": uuid.uuid4().hex}
        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if 200 <= resp.status < 300:
                        return data
                    error_msg = data.get("error", f"HTTP {resp.status}")
                    if isinstance(error_msg, dict):
                        error_code = error_msg.get("code") or data.get("code")
                        error_details = error_msg.get("details") or data.get("details")
                        error_message = error_msg.get("message") or str(error_msg)
                    else:
                        error_code = data.get("code")
                        error_details = data.get("details")
                        error_message = str(error_msg)
                    raise RpcEnvError(
                        endpoint,
                        status=resp.status,
                        code=error_code,
                        message=error_message,
                        details=error_details,
                    )
            except RpcEnvError as exc:
                last_exc = exc
                if exc.code in {
                    "fenced",
                    "fence_required",
                    "idempotency_conflict",
                    "lease_lost",
                    "lease_not_active",
                    "allocation_cancelled",
                }:
                    break
                if attempt < self.max_retries:
                    delay = DEFAULT_RETRY_DELAY * attempt
                    logger.warning(
                        "%s attempt %d/%d failed: %s, retrying in %.1fs",
                        endpoint, attempt, self.max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = DEFAULT_RETRY_DELAY * attempt
                    logger.warning(
                        "%s attempt %d/%d failed: %s, retrying in %.1fs",
                        endpoint, attempt, self.max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)

        if isinstance(last_exc, RpcEnvError):
            raise last_exc
        raise RuntimeError(f"{endpoint} failed after {self.max_retries} attempts: {last_exc}")

    async def _get(self, endpoint: str) -> dict[str, Any]:
        session = await self._get_session()
        async with session.get(f"{self.base_url}{endpoint}") as resp:
            data = await resp.json()
            if not 200 <= resp.status < 300:
                raise RuntimeError(
                    f"{endpoint} failed: {data.get('error', f'HTTP {resp.status}')}"
                )
            return data

    @staticmethod
    def _operation_payload(
        payload: dict[str, Any],
        *,
        operation_id: str | None,
        fence_token: str | None,
    ) -> dict[str, Any]:
        if operation_id is not None:
            payload["operation_id"] = operation_id
        if fence_token is not None:
            payload["fence_token"] = fence_token
        return payload

    async def allocate(
        self,
        episode_id: str,
        *,
        allocation_key: str | None = None,
        idempotency_key: str | None = None,
        owner: dict[str, Any] | None = None,
        fence_token: str | None = None,
        generation: int | None = None,
    ) -> dict[str, Any]:

        payload: dict[str, Any] = {"episode_id": episode_id}
        key = allocation_key if allocation_key is not None else idempotency_key
        if key is not None:
            payload["allocation_key"] = key
        if owner is not None:
            payload["owner"] = owner
        if fence_token is not None:
            payload["fence_token"] = fence_token
        if generation is not None:
            payload["generation"] = generation
        result = await self._post("/allocate", payload)
        if not result.get("ok", False):
            raise RuntimeError(f"allocate failed: {result}")
        return result

    async def cancel_allocation(
        self,
        allocation_key: str,
        *,
        operation_id: str,
        fence_token: str,
        generation: int,
    ) -> dict[str, Any]:
        result = await self._post(
            "/cancel_allocation",
            {
                "allocation_key": allocation_key,
                "operation_id": operation_id,
                "fence_token": fence_token,
                "generation": int(generation),
            },
        )
        if not result.get("ok", False):
            raise RuntimeError(f"cancel allocation failed: {result}")
        return result

    async def reset(
        self,
        lease_id: str,
        task_config: dict[str, Any] | None,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> dict[str, Any]:

        result = await self._post(
            "/reset",
            self._operation_payload(
                {"lease_id": lease_id, "task_config": task_config},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"reset failed: {result}")
        return _decode_obs(result["observation"])

    async def step(
        self,
        lease_id: str,
        action: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, dict]:

        result = await self._post(
            "/step",
            self._operation_payload(
                {"lease_id": lease_id, "action": action},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"step failed: {result}")
        obs = _decode_obs(result["observation"])
        return obs, float(result.get("reward", 0.0)), bool(result.get("done", False)), result.get("info", {})

    async def get_obs(
        self,
        lease_id: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> dict[str, Any]:

        result = await self._post(
            "/get_obs",
            self._operation_payload(
                {"lease_id": lease_id},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"get_obs failed: {result}")
        return _decode_obs(result["observation"])

    async def prepare_for_probe(
        self,
        lease_id: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> None:

        result = await self._post(
            "/prepare_for_probe",
            self._operation_payload(
                {"lease_id": lease_id},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"prepare_for_probe failed: {result}")

    async def evaluate(
        self,
        lease_id: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> float:

        result = await self._post(
            "/evaluate",
            self._operation_payload(
                {"lease_id": lease_id},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"evaluate failed: {result}")
        return float(result["score"])

    async def close_strict(
        self,
        lease_id: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> dict[str, Any]:

        result = await self._post(
            "/close",
            self._operation_payload(
                {"lease_id": lease_id},
                operation_id=operation_id,
                fence_token=fence_token,
            ),
        )
        if not result.get("ok", False):
            raise RuntimeError(f"close failed: {result}")
        return result

    async def close(
        self,
        lease_id: str,
        *,
        operation_id: str | None = None,
        fence_token: str | None = None,
    ) -> None:

        try:
            await self.close_strict(
                lease_id,
                operation_id=operation_id,
                fence_token=fence_token,
            )
        except Exception as exc:
            logger.warning("close failed for lease %s: %s", lease_id, exc)

    async def list_leases(self) -> list[dict[str, Any]]:
        result = await self._get("/leases")
        if not result.get("ok", False):
            raise RuntimeError(f"list leases failed: {result}")
        return list(result.get("leases") or [])

    async def get_lease(self, lease_id: str) -> dict[str, Any]:
        result = await self._get(f"/leases/{lease_id}")
        if not result.get("ok", False):
            raise RuntimeError(f"get lease failed: {result}")
        return dict(result["lease"])

    async def heartbeat(
        self,
        lease_id: str,
        *,
        fence_token: str | None = None,
    ) -> None:

        try:
            await self._post(
                "/heartbeat",
                self._operation_payload(
                    {"lease_id": lease_id},
                    operation_id=None,
                    fence_token=fence_token,
                ),
            )
        except Exception:
            pass

    async def healthcheck(self) -> bool:

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/healthz", timeout=HEALTH_TIMEOUT) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def status(self) -> dict[str, Any]:

        session = await self._get_session()
        async with session.get(f"{self.base_url}/status", timeout=HEALTH_TIMEOUT) as resp:
            if resp.status != 200:
                raise RuntimeError(f"/status returned HTTP {resp.status}")
            return await resp.json()

    async def close_session(self) -> None:

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
