from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests

logger = logging.getLogger("eval_runtime.vagen_active.anthropic_transport")

ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT = 110

def _openai_part_to_anthropic_block(part: dict[str, Any]) -> dict[str, Any] | None:
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "text", "text": part.get("text", "")}
    if ptype == "image_url":
        url = (part.get("image_url") or {}).get("url", "")

        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = header.removeprefix("data:").split(";")[0] or "image/png"
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        return None
    return None

def _convert_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:

    tools: list[dict[str, Any]] = []
    for tool in openai_tools or []:
        if tool.get("type") == "function":
            fn = tool.get("function") or {}
            tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object"},
            })
        else:

            tools.append(tool)
    return tools

def openai_payload_to_anthropic(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:

    system_text = ""
    anthropic_messages: list[dict[str, Any]] = []

    def _append_user_blocks(blocks: list[dict[str, Any]]) -> None:

        if anthropic_messages and anthropic_messages[-1]["role"] == "user":
            anthropic_messages[-1]["content"].extend(blocks)
        else:
            anthropic_messages.append({"role": "user", "content": list(blocks)})

    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                system_text = content
            continue

        if role == "tool":
            _append_user_blocks([{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": str(content or ""),
            }])
            continue

        if role == "user":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    block = _openai_part_to_anthropic_block(part)
                    if block is not None:
                        blocks.append(block)
            if blocks:
                _append_user_blocks(blocks)
            continue

        if role == "assistant":
            blocks = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if blocks:
                anthropic_messages.append({"role": "assistant", "content": blocks})
            continue

    anthropic_payload: dict[str, Any] = {
        "model": payload.get("model", ""),
        "max_tokens": payload.get("max_tokens", 4096),
        "messages": _patch_orphan_tool_uses(anthropic_messages),
    }
    if system_text:
        anthropic_payload["system"] = system_text

    tools = _convert_tools(payload.get("tools") or [])
    if tools:
        anthropic_payload["tools"] = tools

    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    if temperature is not None:
        anthropic_payload["temperature"] = temperature

        if top_p is not None:
            logger.debug("Dropping top_p=%s (temperature=%s already set)", top_p, temperature)
    elif top_p is not None:
        anthropic_payload["top_p"] = top_p

    betas = list(payload.get("anthropic_beta") or [])
    return anthropic_payload, betas

def _patch_orphan_tool_uses(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    patched: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        patched.append(msg)
        if msg["role"] != "assistant":
            continue
        tool_ids = [b["id"] for b in msg["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not tool_ids:
            continue
        next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
        answered: set[str] = set()
        if next_msg is not None and next_msg["role"] == "user":
            answered = {b.get("tool_use_id") for b in next_msg["content"]
                        if isinstance(b, dict) and b.get("type") == "tool_result"}
        missing = [tid for tid in tool_ids if tid not in answered]
        if not missing:
            continue
        synthetic = [{
            "type": "tool_result",
            "tool_use_id": tid,
            "content": "[Tool execution skipped: the evaluation was asked to "
                       "conclude before this tool ran.]",
        } for tid in missing]
        if next_msg is not None and next_msg["role"] == "user":

            next_msg = {"role": "user", "content": synthetic + list(next_msg["content"])}
            messages[idx + 1] = next_msg
        else:
            patched.append({"role": "user", "content": synthetic})
        logger.debug("Patched %d orphan tool_use id(s) at message %d", len(missing), idx)
    return patched

def anthropic_response_to_openai(resp_json: dict[str, Any]) -> dict[str, Any]:

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in resp_json.get("content") or []:
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp_json.get("usage") or {}
    return {
        "id": resp_json.get("id", "unknown-id"),
        "object": "chat.completion",
        "model": resp_json.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    }

class _ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            auth = self.headers.get("Authorization") or ""
            api_key = auth.removeprefix("Bearer ").strip()
            status, body = self.server.shim.forward(payload, api_key)
        except Exception as exc:
            logger.error("Shim request failed: %s", exc, exc_info=True)
            status, body = 500, {"error": {"type": "shim_error", "message": str(exc)}}

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

class LocalAnthropicShim:

    def __init__(self, upstream_url: str, fallback_api_key: str = ""):
        self.upstream_url = upstream_url
        self.fallback_api_key = fallback_api_key
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.local_url = ""

    def start(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ShimHandler)
        server.daemon_threads = True
        server.shim = self
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="vagen-anthropic-shim", daemon=True
        )
        self._thread.start()
        port = server.server_address[1]
        self.local_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        logger.info("Anthropic transport shim started: %s -> %s",
                    self.local_url, self.upstream_url)

    def forward(self, payload: dict[str, Any], api_key: str) -> tuple[int, dict[str, Any]]:
        anthropic_payload, betas = openai_payload_to_anthropic(payload)
        tools = anthropic_payload.get("tools") or []
        logger.info(
            "Anthropic upstream request model=%s messages=%d tools=%s "
            "temperature=%s top_p=%s betas=%s",
            anthropic_payload.get("model"),
            len(anthropic_payload.get("messages") or []),
            [str(tool.get("name") or tool.get("type") or "") for tool in tools],
            anthropic_payload.get("temperature"),
            anthropic_payload.get("top_p"),
            betas,
        )
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key or self.fallback_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if betas:
            headers["anthropic-beta"] = ",".join(betas)

        try:
            resp = requests.post(
                self.upstream_url, headers=headers, json=anthropic_payload,
                timeout=UPSTREAM_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return 504, {"error": {"type": "timeout", "message": "upstream gateway timeout"}}
        except requests.exceptions.RequestException as exc:
            return 502, {"error": {"type": "connection_error", "message": str(exc)}}

        if resp.status_code != 200:

            try:
                body = resp.json()
            except ValueError:
                body = {"error": {"type": "upstream_error", "message": resp.text[:2000]}}
            logger.warning("Upstream gateway returned %s: %s",
                           resp.status_code, str(body)[:300])
            return resp.status_code, body

        response_json = resp.json()
        content = response_json.get("content") or []
        logger.info(
            "Anthropic upstream response model=%s stop_reason=%s "
            "block_types=%s tool_calls=%s usage=%s",
            response_json.get("model"),
            response_json.get("stop_reason"),
            [block.get("type") for block in content],
            [
                block.get("name")
                for block in content
                if block.get("type") == "tool_use"
            ],
            response_json.get("usage") or {},
        )
        return 200, anthropic_response_to_openai(response_json)

_shim_lock = threading.Lock()
_shim: LocalAnthropicShim | None = None

def get_shim_url(upstream_url: str, fallback_api_key: str = "") -> str:

    global _shim
    with _shim_lock:
        if _shim is None or _shim.upstream_url != upstream_url:
            _shim = LocalAnthropicShim(upstream_url, fallback_api_key)
            _shim.start()
        return _shim.local_url
