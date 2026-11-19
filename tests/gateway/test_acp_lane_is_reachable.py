"""The lane a request is served by is the one its provider names.

``routes_llm`` resolves ``provider_info.transport`` against the app's transport
registry on every chat request.  That single lookup is what makes an ACP lane
reachable at all: without it every request would be sent to the HTTP transport,
which for an agent-served provider means dialing a placeholder host.

This is a route-level test on purpose.  The transport's own tests hand the
transport a body directly, so they stay green whether or not the route ever
picks it — which is exactly the failure this pins.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.transport._base import UpstreamResponse

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class _Recorder:
    """A transport that answers the way its lane would, and records the call."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        self.calls.append(body)
        raw = json.dumps(self.payload).encode()
        return UpstreamResponse(status_code=200, body=self.payload, raw_content=raw)

    async def send_streaming(self, *args: Any, **kw: Any) -> Any:
        raise AssertionError("this test never streams")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""


def _config(tmp_path: Any, transport: str) -> GatewayConfig:
    provider: dict[str, Any] = {
        "type": "acp",
        "api_key": "k",
        "base_url": "http://agy.local",
    }
    if transport:
        provider["transport"] = transport
    return GatewayConfig(
        {
            "providers": {"g": provider},
            "models": {"m": {"providers": [{"name": "g"}]}},
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )


async def _post(port: int, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    raw = json.dumps(body).encode()
    request = (
        f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
    ).encode() + raw
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    status = int(head.decode().split("\r\n")[0].split()[1])
    return status, json.loads(payload or b"{}")


@pytest.fixture()
def serve(tmp_path):
    """Serve the app with one stub per lane; yields (port, acp, http)."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _serve(transport: str):
        app = await create_app(_config(tmp_path, transport))
        acp = _Recorder(
            {
                "updates": [
                    {"type": "session_start", "session_id": "s1", "model": "m"},
                    {"type": "text", "text": "hi"},
                    {"type": "stop", "stop_reason": "end_turn"},
                ],
                "stop_reason": "end_turn",
                "session_id": "s1",
                "model": "m",
            }
        )
        http = _Recorder(
            {
                "id": "resp-1",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}}
                ],
            }
        )
        wired: Any = app
        wired.transport = http
        wired.transports = {"acp": acp}
        listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
        try:
            yield listener.sockets[0].getsockname()[1], acp, http
        finally:
            listener.close()
            await listener.wait_closed()

    return _serve


async def test_a_provider_that_names_the_acp_lane_is_served_by_it(serve):
    async with serve("acp") as (port, acp, http):
        status, payload = await _post(port, "/v1/chat/completions", _CHAT)

    assert status == 200
    assert len(acp.calls) == 1
    assert http.calls == []
    # The body reaching the lane is that lane's own shape, not a chat body.
    assert acp.calls[0]["prompt"]
    assert payload["choices"][0]["message"]["content"]


async def test_a_provider_with_no_transport_uses_the_http_lane(serve):
    async with serve("") as (port, acp, http):
        status, _payload = await _post(port, "/v1/chat/completions", _CHAT)

    assert status == 200
    assert http.calls and acp.calls == []
