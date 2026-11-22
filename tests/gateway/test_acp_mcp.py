"""The loopback MCP server and the per-session tool bridge.

These tests speak HTTP to the server directly, so they cover the framing a
real MCP client (the agent) uses: one JSON-RPC message per POST, a batch, an
unknown path token, and a ``tools/call`` that stays open until the API client
returns its result.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

from llm_rosetta_gateway.transport.acp_mcp import AcpMcpServer, AcpToolBridge

ECHO = {
    "name": "echo",
    "description": "echo the text back",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


async def _request(
    url: str,
    payload: Any,
    *,
    method: str = "POST",
    path: str | None = None,
) -> tuple[int, Any]:
    """Send one HTTP request to *url* and return ``(status, JSON body)``."""
    parts = urlsplit(url)
    assert parts.hostname is not None and parts.port is not None
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    target = path or parts.path
    head = (
        f"{method} {target} HTTP/1.1\r\n"
        f"Host: {parts.netloc}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
    writer.write(head + body)
    await writer.drain()
    try:
        status_line = await reader.readline()
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length") or 0)
        data = await reader.readexactly(length) if length else b""
    finally:
        writer.close()
        await writer.wait_closed()
    return int(status_line.split()[1]), json.loads(data) if data else None


def _rpc(rpc_id: Any, method: str, params: dict[str, Any] | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}


async def _wait_pending(bridge: AcpToolBridge) -> None:
    """Wait until the server has recorded a call."""
    for _ in range(200):
        if bridge.has_pending():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the bridge recorded no pending call")


class TestTheBridge:
    """The tool list and the pending-call bookkeeping."""

    def test_tools_are_exposed_in_mcp_shape(self):
        bridge = AcpToolBridge([ECHO])
        assert bridge.tools() == [
            {
                "name": "echo",
                "description": "echo the text back",
                "inputSchema": ECHO["parameters"],
            }
        ]
        assert bridge.names() == {"echo"}

    def test_a_tool_without_parameters_gets_an_empty_schema(self):
        bridge = AcpToolBridge([{"name": "ping"}])
        assert bridge.tools()[0]["inputSchema"] == {
            "type": "object",
            "properties": {},
        }

    def test_a_permission_request_matches_a_prefixed_mcp_name(self):
        bridge = AcpToolBridge([ECHO])
        assert bridge.may_allow({"name": "mcp_llm_rosetta_tools_echo"})
        assert bridge.may_allow({"title": "echo"})
        assert not bridge.may_allow({"name": "search_web"})

    async def test_close_fails_a_pending_call(self):
        bridge = AcpToolBridge([ECHO])
        task = asyncio.create_task(bridge.begin_call("echo", {"text": "hi"}))
        await _wait_pending(bridge)
        bridge.close()
        result = await asyncio.wait_for(task, 1)
        assert result["isError"] is True
        assert "closed" in result["content"][0]["text"]


class TestTheServer:
    """The HTTP endpoint the agent's MCP client talks to."""

    async def test_tools_list_returns_the_declared_tools(self):
        server = AcpMcpServer()
        bridge = AcpToolBridge([ECHO])
        url = await server.register(bridge)
        try:
            status, reply = await _request(url, _rpc(1, "tools/list"))
        finally:
            await server.close()
        assert status == 200
        assert reply["result"]["tools"][0]["name"] == "echo"
        assert reply["result"]["tools"][0]["inputSchema"] == ECHO["parameters"]

    async def test_initialize_echoes_the_requested_version(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        try:
            status, reply = await _request(
                url,
                _rpc(
                    1,
                    "initialize",
                    {"protocolVersion": "2025-11-25", "capabilities": {}},
                ),
            )
        finally:
            await server.close()
        assert status == 200
        assert reply["result"]["protocolVersion"] == "2025-11-25"
        assert reply["result"]["capabilities"]["tools"]["listChanged"] is False

    async def test_a_batch_is_answered_as_a_list(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        batch = [_rpc(1, "initialize"), _rpc(2, "tools/list")]
        try:
            status, reply = await _request(url, batch)
        finally:
            await server.close()
        assert status == 200
        assert [item["id"] for item in reply] == [1, 2]

    async def test_a_tools_call_waits_for_the_client_result(self):
        server = AcpMcpServer()
        bridge = AcpToolBridge([ECHO])
        url = await server.register(bridge)
        pending = asyncio.create_task(
            _request(
                url,
                _rpc(
                    7,
                    "tools/call",
                    {"name": "echo", "arguments": {"text": "hello"}},
                ),
            )
        )
        try:
            await _wait_pending(bridge)
            call_id = bridge.pending_ids()[0]
            assert bridge.pending()[0].arguments == {"text": "hello"}
            assert bridge.resolve(call_id, "hello back") is True
            status, reply = await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await server.close()
        assert status == 200
        assert reply["result"] == {
            "content": [{"type": "text", "text": "hello back"}],
            "isError": False,
        }

    async def test_a_client_error_is_reported_as_an_mcp_error(self):
        server = AcpMcpServer()
        bridge = AcpToolBridge([ECHO])
        url = await server.register(bridge)
        pending = asyncio.create_task(
            _request(url, _rpc(8, "tools/call", {"name": "echo", "arguments": {}}))
        )
        try:
            await _wait_pending(bridge)
            assert bridge.resolve(bridge.pending_ids()[0], "boom", is_error=True)
            status, reply = await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await server.close()
        assert status == 200
        assert reply["result"]["isError"] is True

    async def test_an_unknown_tool_is_an_error_result(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        try:
            status, reply = await _request(
                url, _rpc(9, "tools/call", {"name": "nope", "arguments": {}})
            )
        finally:
            await server.close()
        assert status == 200
        assert reply["result"]["isError"] is True
        assert "unknown tool" in reply["result"]["content"][0]["text"]

    async def test_an_unknown_path_token_is_404(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        bogus = url.rsplit("/", 1)[0] + "/not-a-token"
        try:
            status, reply = await _request(bogus, _rpc(1, "tools/list"))
        finally:
            await server.close()
        assert status == 404
        assert reply["error"] == "unknown session"

    async def test_invalid_json_is_400(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        try:
            status, reply = await _request(url, b"{not json")
        finally:
            await server.close()
        assert status == 400
        assert reply["error"] == "invalid JSON"

    async def test_a_cancelled_rpc_id_fails_its_call(self):
        server = AcpMcpServer()
        bridge = AcpToolBridge([ECHO])
        url = await server.register(bridge)
        pending = asyncio.create_task(
            _request(url, _rpc(11, "tools/call", {"name": "echo", "arguments": {}}))
        )
        try:
            await _wait_pending(bridge)
            assert bridge.cancel_rpc(11, "the agent withdrew the call")
            status, reply = await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await server.close()
        assert status == 200
        assert reply["result"]["isError"] is True
        assert "withdrew" in reply["result"]["content"][0]["text"]

    async def test_closing_the_server_fails_a_pending_call(self):
        server = AcpMcpServer()
        bridge = AcpToolBridge([ECHO])
        url = await server.register(bridge)
        pending = asyncio.create_task(
            _request(url, _rpc(12, "tools/call", {"name": "echo", "arguments": {}}))
        )
        await _wait_pending(bridge)
        await server.close()
        try:
            status, reply = await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
        assert status == 200
        assert reply["result"]["isError"] is True
        assert "closed" in reply["result"]["content"][0]["text"]

    async def test_each_registration_gets_its_own_token(self):
        server = AcpMcpServer()
        first = await server.register(AcpToolBridge([ECHO]))
        second = await server.register(AcpToolBridge([ECHO]))
        try:
            assert first != second
            assert urlsplit(first).port == urlsplit(second).port
        finally:
            await server.close()

    async def test_the_server_restarts_after_its_last_registration_goes(self):
        server = AcpMcpServer()
        first = await server.register(AcpToolBridge([ECHO]))
        server.unregister(first)
        second = await server.register(AcpToolBridge([ECHO]))
        try:
            status, reply = await _request(second, _rpc(1, "tools/list"))
        finally:
            await server.close()
        assert status == 200
        assert reply["result"]["tools"][0]["name"] == "echo"

    async def test_a_get_probe_is_405_so_the_agent_stays_on_post(self):
        server = AcpMcpServer()
        url = await server.register(AcpToolBridge([ECHO]))
        try:
            status, _ = await _request(url, b"", method="GET")
        finally:
            await server.close()
        assert status == 405

    def test_a_bridge_without_tools_is_empty(self):
        assert AcpToolBridge([]).empty is True
        assert AcpToolBridge([ECHO]).empty is False
