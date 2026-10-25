"""A model the gateway does not serve is a 404, on every chat route (#1220).

The chat handler built its selection context outside the guard that turns a
missing model into a 404, so the ``KeyError`` reached the lifecycle and the
client got a 500 with ``Internal server error: 'nope'``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.observability import PersistenceManager, RequestLog
from llm_rosetta_gateway.ops import OpsContext


class Served:
    def __init__(self, port: int, pm: PersistenceManager) -> None:
        self.port = port
        self.pm = pm

    async def post(
        self, path: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
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
        status_line, *header_lines = head.decode().split("\r\n")
        headers = {
            key.strip().lower(): value.strip()
            for key, _, value in (line.partition(":") for line in header_lines)
        }
        return int(status_line.split()[1]), headers, json.loads(payload or b"{}")


@pytest.fixture()
async def served(tmp_path) -> AsyncIterator[Served]:
    pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    config = GatewayConfig(
        {
            "providers": {
                "Mock": {
                    "type": "chat_completions",
                    "base_url": "http://127.0.0.1:1/v1",
                    "api_key": "x",
                }
            },
            "models": {"real-model": {"provider": "Mock"}},
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )
    app = await create_app(config)
    request_log = RequestLog(persistence=pm)
    wired: Any = app
    wired.ops_ctx = OpsContext(request_log=request_log, persistence=pm)
    wired.request_log = request_log
    wired.persistence = pm
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    try:
        yield Served(listener.sockets[0].getsockname()[1], pm)
    finally:
        listener.close()
        await listener.wait_closed()
        await pm.close()


CHAT = {"model": "nope", "messages": [{"role": "user", "content": "hi"}]}


def _openai_shape(body: dict[str, Any]) -> bool:
    return (
        body["error"]["code"] == "model_not_found"
        and body["error"]["type"] == "invalid_request_error"
    )


def _anthropic_shape(body: dict[str, Any]) -> bool:
    return body["type"] == "error" and body["error"]["type"] == "not_found_error"


def _google_shape(body: dict[str, Any]) -> bool:
    return body["error"]["code"] == 404 and body["error"]["status"] == "NOT_FOUND"


@pytest.mark.parametrize(
    ("path", "body", "has_shape"),
    [
        pytest.param("/v1/chat/completions", CHAT, _openai_shape, id="chat"),
        pytest.param(
            "/v1/responses",
            {"model": "nope", "input": "hi"},
            _openai_shape,
            id="responses",
        ),
        pytest.param(
            "/v1/messages", {**CHAT, "max_tokens": 8}, _anthropic_shape, id="messages"
        ),
        pytest.param(
            "/v1beta/models/nope:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            _google_shape,
            id="google",
        ),
    ],
)
async def test_an_unknown_model_is_a_404_in_the_clients_format(
    served, path, body, has_shape
):
    status, headers, payload = await served.post(path, body)

    assert status == 404
    assert has_shape(payload), payload
    message = json.dumps(payload)
    assert "Unknown model: 'nope'" in message
    assert "real-model" in message, "the message lists what is configured"
    assert headers["x-request-id"]


async def test_the_request_log_records_the_404_not_a_traceback(served):
    status, _, _ = await served.post("/v1/chat/completions", CHAT)

    assert status == 404
    rows, total = await served.pm.query_log_entries()
    assert total == 1
    assert rows[0]["status_code"] == 404
    assert "Unknown model: 'nope'" in rows[0]["error_detail"]
    assert "Internal server error" not in rows[0]["error_detail"]


async def test_a_configured_model_gets_past_the_guard(served):
    # The upstream is unreachable, so this fails later and differently: the
    # point is only that the model lookup does not refuse it.
    status, _, payload = await served.post(
        "/v1/chat/completions", {**CHAT, "model": "real-model"}
    )

    assert status != 404, payload
