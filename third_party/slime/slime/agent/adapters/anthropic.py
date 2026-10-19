from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import ADAPTER_KEY, REASONING_PARSER_KEY, TOKENIZER_KEY, TOOL_PARSER_KEY
from slime.agent.adapters.common import AdapterChain as Chain
from slime.agent.adapters.common import (
    BaseAdapter,
    call_sglang_generate,
    ok_response,
    render_token_ids,
    request_session_id,
)
from slime.agent.adapters.common import stable_hash as _hash
from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments

logger = logging.getLogger(__name__)

_SUBAGENT_TOOLS = {"Task", "Agent"}

@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None
    pending_dispatch_id: str = ""
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)

class AnthropicAdapter(BaseAdapter):

    session_cls = Session

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        super().__init__(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        self.app.router.add_post("/v1/messages", _handle_request)
        self.app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        s = self.store.pop(sid, None)
        if s is None:
            return []
        if s.active_sub is not None and s.active_sub.turns:
            s.segments.append(make_turn_segment(s.active_sub.turns, kind="subagent"))
        if s.main.turns:
            s.segments.append(make_turn_segment(s.main.turns, kind="final"))

        return merge_turn_segments(s.segments)

def _select_chain(s: Session, body: dict) -> tuple[Chain, bool, str]:

    all_msgs = body.get("messages") or []
    msg_hashes = [_hash(m) for m in all_msgs]
    req_system_hash = _hash(body.get("system")) if "system" in body else s.main.system_hash

    if s.pending_dispatch_id and s.active_sub is not None:
        tu_id = s.pending_dispatch_id
        for m in all_msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            done = any(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tu_id
                for b in content
            )
            if done:
                if s.active_sub.turns:
                    s.segments.append(make_turn_segment(s.active_sub.turns, kind="subagent"))
                s.active_sub = None
                s.pending_dispatch_id = ""
                break

    if s.active_sub is None:
        target, is_sub = s.main, False
    else:
        main_continues = (
            req_system_hash == s.main.system_hash
            and len(msg_hashes) >= s.main.seen_msgs
            and msg_hashes[: s.main.seen_msgs] == s.main.msg_hashes[: s.main.seen_msgs]
        )
        target, is_sub = (s.main, False) if main_continues else (s.active_sub, True)

    if target.seen_msgs == 0:
        kind = "new"
    else:
        is_append = (
            req_system_hash == target.system_hash
            and len(msg_hashes) >= target.seen_msgs
            and msg_hashes[: target.seen_msgs] == target.msg_hashes[: target.seen_msgs]
        )
        if is_append:
            kind = "append"
        else:
            if target.turns:
                s.segments.append(make_turn_segment(target.turns, kind="wipe"))
            kind = "wipe"

    return target, is_sub, kind

def _flatten(c: Any) -> str:

    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append(_flatten(b.get("content")))
            elif t == "image":
                parts.append("[image omitted]")
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)

def _translate_anthropic(msgs: list[dict], system: Any) -> list[dict]:

    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": _flatten(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    translated.append({"role": "tool", "content": _flatten(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": _flatten(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    tcs.append({"function": {"name": b.get("name", "tool"), "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": _flatten(content)})
    return translated

def _anthropic_tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:

    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None

def _replace_chat_messages(target: Chain, body: dict) -> None:

    all_msgs = body.get("messages") or []
    target.chat_messages = _translate_anthropic(all_msgs, body.get("system"))
    if "system" in body:
        target.system_hash = _hash(body.get("system"))
    target.turns.clear()
    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))

def _extend_chat_messages(target: Chain, body: dict) -> None:

    all_msgs = body.get("messages") or []
    translated = _translate_anthropic(all_msgs[target.seen_msgs :], None)
    target.chat_messages.extend(translated)

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))

def _build_prompt(target: Chain, body: dict, kind: str, tok) -> list[int]:

    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, body)
    return render_token_ids(target, tok)

async def _generate(
    prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None
) -> TurnRecord:

    return await call_sglang_generate(
        prompt_ids,
        s,
        body,
        app,
        max_token_keys=("max_tokens",),
        stop_keys=("stop_sequences",),
        log_prefix="anthropic_adapter",
        logger=logger,
        session_id=session_id,
    )

def _build_reply(target: Chain, output_ids: list[int], finish: str, app) -> tuple[list[dict], str, str]:

    tok = app[TOKENIZER_KEY]

    raw_output = tok.decode(output_ids, skip_special_tokens=False) if output_ids else ""
    parsed = parse_model_output(
        raw_output,
        tools_schema=target.tools_schema,
        tool_parser_name=app[TOOL_PARSER_KEY],
        reasoning_parser_name=app[REASONING_PARSER_KEY],
    )
    blocks, dispatch_id = _anthropic_blocks(parsed.reasoning, parsed.text, parsed.tool_uses)
    return blocks, _stop_reason(parsed.tool_uses, finish), dispatch_id

def _anthropic_blocks(thinking: str, visible: str, tool_uses: list[dict]) -> tuple[list[dict], str]:

    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    dispatch_id = ""
    for tu in tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        if tu["name"] in _SUBAGENT_TOOLS:
            dispatch_id = tu_id
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks, dispatch_id

def _stop_reason(tool_uses: list[dict], finish: str) -> str:
    if tool_uses:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"

def _start_sub_chain(s: Session, dispatch_id: str) -> None:

    s.pending_dispatch_id = dispatch_id
    if s.active_sub is None:
        s.active_sub = Chain()

def _request_session_id(request: web.Request) -> str:
    return request_session_id(request, include_x_api_key=True)

async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = _request_session_id(request)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:
        return web.Response(status=503, text="session closed")
    app = request.app
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:
            target, is_sub, kind = _select_chain(s, body)
            ideal_ids = _build_prompt(target, body, kind, app[TOKENIZER_KEY])
            turn = await _generate(ideal_ids, s, body, app, session_id=sid)
            blocks, stop, did = _build_reply(target, turn.output_ids, turn.finish_reason, app)
            target.turns.append(turn)
            if did and not is_sub:
                _start_sub_chain(s, did)
            in_tok, out_tok = len(ideal_ids), len(turn.output_ids)
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await _stream_response(request, blocks, stop, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop, in_tok, out_tok))
    finally:
        adapter.inflight.get(sid, set()).discard(task)

def _message_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "slime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }

async def _stream_response(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:

    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "slime-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out

async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})

async def _ok(request: web.Request) -> web.Response:
    return await ok_response(request)
