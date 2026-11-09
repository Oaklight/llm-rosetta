"""The request log names the lane that served (#1430).

One model name may route to two upstream models of one provider.  Without the
lane in the row, every request of that model reads as the same provider, and
the operator cannot tell the cheap lane from the fast one.

This drives the whole stack — the app, the proxy, the ops record — because the
lane has to survive three hand-offs: the route handler records it, the failover
moves it, and the ops layer writes it into the row.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.observability import PersistenceManager, RequestLog
from llm_rosetta_gateway.ops import OpsContext
from llm_rosetta_gateway.transport._base import UpstreamResponse

_OK_BODY: dict[str, Any] = {
    "id": "resp-1",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0}],
}

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _ok() -> UpstreamResponse:
    raw = json.dumps(_OK_BODY).encode()
    return UpstreamResponse(status_code=200, body=_OK_BODY, raw_content=raw)


def _server_error() -> UpstreamResponse:
    raw = b'{"error": {"message": "internal error", "type": "server_error"}}'
    return UpstreamResponse(status_code=500, body=None, raw_content=raw)


class _StubTransport:
    """Answers from a script, recording the model each attempt asked for."""

    def __init__(self, *responses: UpstreamResponse) -> None:
        self._responses = list(responses)
        self.models: list[Any] = []

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        self.models.append(body.get("model"))
        return self._responses.pop(0)

    async def send_streaming(self, *args: Any, **kw: Any) -> Any:
        raise AssertionError("this test never streams")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""


class Served:
    def __init__(self, port: int, pm: PersistenceManager) -> None:
        self.port = port
        self.pm = pm

    async def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        raw = json.dumps(body).encode()
        request = (
            f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
        ).encode() + raw
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(request)
        await writer.drain()
        writer.write_eof()
        data = await reader.read()
        writer.close()
        head, _, payload = data.partition(b"\r\n\r\n")
        status_line = head.decode().split("\r\n")[0]
        return int(status_line.split()[1]), json.loads(payload or b"{}")


def _config(tmp_path: Any) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                "a": {
                    "type": "chat_completions",
                    "api_key": "sk-a",
                    "base_url": "https://a.test",
                },
            },
            "models": {
                "m": {
                    "providers": [
                        {"name": "a", "upstream_model": "fast", "weight": 100},
                        {"name": "a", "upstream_model": "eco"},
                    ]
                }
            },
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )


@pytest.fixture()
def serve(tmp_path):
    """Serve the app on a socket with a scripted upstream; yields a starter."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _serve(*responses: UpstreamResponse):
        pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
        app = await create_app(_config(tmp_path))
        request_log = RequestLog(persistence=pm)
        wired: Any = app
        wired.ops_ctx = OpsContext(request_log=request_log, persistence=pm)
        wired.request_log = request_log
        wired.persistence = pm
        transport = _StubTransport(*responses)
        wired.transport = transport
        listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
        try:
            yield Served(listener.sockets[0].getsockname()[1], pm), transport
        finally:
            listener.close()
            await listener.wait_closed()
            await pm.close()

    return _serve


async def test_the_row_names_the_lane_that_succeeded_the_first_time(serve):
    """The route handler records the lane before the request is sent."""
    async with serve(_ok()) as (server, transport):
        status, _payload = await server.post("/v1/chat/completions", _CHAT)

        assert status == 200
        assert transport.models == ["fast"]
        entries, _total = await server.pm.query_log_entries(limit=10)
        (row,) = entries
        assert (row["target_provider_name"], row["upstream_model"]) == ("a", "fast")
        # The client's model is unchanged: the lane is an extra fact, not a
        # rename, and a row without a failover carries no failover entry.
        assert row["model"] == "m"
        assert "failover" not in (row.get("profile") or {})


async def test_the_row_follows_the_lane_a_failover_moved_to(serve):
    async with serve(_server_error(), _ok()) as (server, transport):
        status, _payload = await server.post("/v1/chat/completions", _CHAT)

        assert status == 200
        assert transport.models == ["fast", "eco"]
        entries, _total = await server.pm.query_log_entries(limit=10)
        (row,) = entries
        assert (row["target_provider_name"], row["upstream_model"]) == ("a", "eco")
        assert row["model"] == "m"
        assert row["profile"]["failover"]["scope"] == "lane"
