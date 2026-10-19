from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None

def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    return _http_client

async def _post(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any] | None,
    max_retries: int = 60,
    headers: dict[str, str] | None = None,
    raise_for_status: bool = True,
) -> Any:
    retry_count = 0
    while retry_count < max_retries:
        response: httpx.Response | None = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            if raise_for_status:
                response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
            if (
                not raise_for_status
                and isinstance(output, dict)
                and response.status_code >= 400
            ):
                output.setdefault("ok", False)
                output.setdefault("_http_status", response.status_code)
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
                status_code = e.response.status_code
            else:
                response_text = None
                status_code = None

            logger.info(
                "Error: %s, retrying... (attempt %s/%s, url=%s, response=%s)",
                e,
                retry_count,
                max_retries,
                url,
                response_text,
            )
            if retry_count >= max_retries:
                logger.info("Max retries (%s) reached, failing... (url=%s)", max_retries, url)
                if isinstance(e, httpx.HTTPStatusError):
                    detail = response_text.strip() if response_text else "<empty response>"
                    raise RuntimeError(
                        f"HTTP {status_code} from {url}: {detail}"
                    ) from e
                raise
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output

async def post(
    url: str,
    payload: dict[str, Any] | None,
    max_retries: int = 60,
    headers: dict[str, str] | None = None,
    raise_for_status: bool = True,
) -> Any:
    return await _post(
        _get_http_client(),
        url,
        payload,
        max_retries=max_retries,
        headers=headers,
        raise_for_status=raise_for_status,
    )

async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
