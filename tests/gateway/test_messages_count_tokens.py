"""``POST /v1/messages/count_tokens`` — delegated to the upstream (#1404).

The route forwards the client's body unchanged (bar the model name) to the
target's own ``/count_tokens`` endpoint and relays the answer verbatim.  A
target that does not speak Messages answers 501: there is no counting endpoint
to call and no IR form to convert a count into.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig


class _Recorder:
    """A stand-in Anthropic upstream: records the request, answers a count."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response_body: dict[str, Any] = {"input_tokens": 7}
        self.status = 200

    def handler(self) -> type[BaseHTTPRequestHandler]:
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers.items()),
                        "body": json.loads(raw or b"{}"),
                    }
                )
                payload = json.dumps(recorder.response_body).encode()
                self.send_response(recorder.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler


@pytest.fixture()
def upstream() -> Iterator[tuple[_Recorder, str]]:
    recorder = _Recorder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), recorder.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield recorder, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class Served:
    def __init__(self, port: int) -> None:
        self.port = port

    async def post(
        self, path: str, body: dict[str, Any] | str
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
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


@asynccontextmanager
async def _serve(raw_config: dict[str, Any]) -> AsyncIterator[Served]:
    app = await create_app(GatewayConfig(raw_config))
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    try:
        yield Served(listener.sockets[0].getsockname()[1])
    finally:
        listener.close()
        await listener.wait_closed()


def _config(provider_url: str, provider_type: str = "messages") -> dict[str, Any]:
    return {
        "providers": {
            "Up": {
                "type": provider_type,
                "base_url": provider_url,
                "api_key": "k",
            }
        },
        "models": {"claude-alias": {"provider": "Up", "upstream_model": "claude-real"}},
        "server": {"open_on_no_keys": True},
    }


@pytest.fixture()
async def served(upstream: tuple[_Recorder, str]) -> AsyncIterator[Served]:
    _, url = upstream
    async with _serve(_config(url)) as app:
        yield app


_COUNT_BODY: dict[str, Any] = {
    "model": "claude-alias",
    "messages": [{"role": "user", "content": "Hello"}],
    "system": "You are helpful.",
    "tools": [{"name": "t", "input_schema": {"type": "object"}}],
}


@pytest.mark.asyncio
async def test_body_is_forwarded_unchanged_and_answer_relayed(
    served: Served, upstream: tuple[_Recorder, str]
) -> None:
    recorder, _ = upstream

    status, headers, body = await served.post("/v1/messages/count_tokens", _COUNT_BODY)

    assert status == 200
    assert body == {"input_tokens": 7}
    assert headers["x-rosetta-conversion"] == "passthrough"

    assert len(recorder.requests) == 1
    forwarded = recorder.requests[0]
    assert forwarded["path"] == "/v1/messages/count_tokens"
    # Only the model name changes: the client alias becomes the upstream id,
    # and every other field — messages, system, tools — crosses untouched.
    assert forwarded["body"] == dict(_COUNT_BODY, model="claude-real")


@pytest.mark.asyncio
async def test_upstream_error_is_relayed_verbatim(
    served: Served, upstream: tuple[_Recorder, str]
) -> None:
    recorder, _ = upstream
    recorder.status = 400
    recorder.response_body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "max_tokens: nope"},
    }

    status, _, body = await served.post(
        "/v1/messages/count_tokens", dict(_COUNT_BODY, max_tokens=100)
    )

    assert status == 400
    assert body["error"]["message"] == "max_tokens: nope"


@pytest.mark.asyncio
async def test_unparseable_body_is_a_400_in_the_messages_envelope(
    served: Served,
) -> None:
    status, _, body = await served.post("/v1/messages/count_tokens", "{not json")

    assert status == 400
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_missing_model_is_a_400(served: Served) -> None:
    status, _, body = await served.post(
        "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}]}
    )

    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_non_messages_target_is_a_501(upstream: tuple[_Recorder, str]) -> None:
    recorder, url = upstream
    raw = _config(url, provider_type="open_responses")

    async with _serve(raw) as app:
        status, _, body = await app.post("/v1/messages/count_tokens", _COUNT_BODY)

    assert status == 501
    assert body["type"] == "error"
    assert "messages upstream" in body["error"]["message"]
    # The refusal happens before any upstream call is made.
    assert recorder.requests == []


@pytest.mark.asyncio
async def test_route_is_registered() -> None:
    app = await create_app(
        GatewayConfig({"providers": {}, "server": {"open_on_no_keys": True}})
    )

    patterns = {route.pattern.pattern for route in app._routes}
    assert "^/v1/messages/count_tokens$" in patterns

    count_route = next(
        route
        for route in app._routes
        if route.pattern.fullmatch("/v1/messages/count_tokens")
    )
    assert "POST" in count_route.methods
