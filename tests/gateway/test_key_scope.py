"""A key's model allowlist, driven through the real app (#1502 §13).

The allowlist is an authorization control on three routes that take a model —
chat, compaction and token counting — plus what the listings answer.  A test
of the predicate alone cannot see a route that forgot to call it, so these
build the app, open a socket and send the request a client would send; the
source-level tests stay next door in ``test_key_limits`` for the store and
the limiter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.keystore import KeyStore

PROVIDERS = {
    "up": {"type": "openai", "base_url": "https://api.example.com", "api_key": "k"}
}
MODELS = {
    "m-a": {"providers": [{"name": "up", "model": "m-a"}]},
    "m-b": {"providers": [{"name": "up", "model": "m-b"}]},
}


class Served:
    """A running gateway on a real socket, plus the keys it holds."""

    def __init__(self, port: int, key: str) -> None:
        self.port = port
        self.key = key

    async def send(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, bytes]:
        raw = b"" if body is None else __import__("json").dumps(body).encode()
        headers = {
            "Host": "test",
            "Connection": "close",
            "Content-Length": str(len(raw)),
            "Content-Type": "application/json",
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        head = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in headers.items()]
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write("\r\n".join(head).encode() + b"\r\n\r\n" + raw)
        await writer.drain()
        data = await reader.read()
        writer.close()
        status_line, _, payload = data.partition(b"\r\n\r\n")
        return int(status_line.split()[1]), payload


async def _serve(tmp_path: Any, allowed: list[str] | None) -> AsyncIterator[Served]:
    keys_db = tmp_path / "keys.db"
    store = await KeyStore.create(keys_db)
    _, raw_key = await store.create_key("scoped", allowed_models=allowed)
    await store.close()
    config = GatewayConfig(
        {
            "providers": PROVIDERS,
            "models": MODELS,
            "server": {"data_dir": str(tmp_path / "data"), "api_keys_db": str(keys_db)},
        }
    )
    app = await create_app(config)
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        yield Served(port, raw_key)
    finally:
        listener.close()
        await listener.wait_closed()


@pytest.fixture()
async def scoped(tmp_path) -> AsyncIterator[Served]:
    async for gateway in _serve(tmp_path, ["m-a"]):
        yield gateway


# "m-z" is not configured: a 403 for it proves the allowlist runs before the
# model lookup, where a 404 would prove the opposite.
CHAT = {"model": "m-z", "messages": [{"role": "user", "content": "hi"}]}


class TestTheAllowlistRefusesBeforeTheLookup:
    async def test_a_model_outside_the_allowlist_is_403_on_chat(self, scoped):
        status, payload = await scoped.send("POST", "/v1/chat/completions", CHAT)

        # 403, not 404: the model is not configured either, so the refusal
        # proves the allowlist runs before the lookup.
        assert status == 403
        assert b"This API key is not allowed to use model 'm-z'" in payload

    async def test_the_count_tokens_route_answers_the_same_refusal(self, scoped):
        status, payload = await scoped.send("POST", "/v1/messages/count_tokens", CHAT)

        assert status == 403
        assert b"This API key is not allowed to use model 'm-z'" in payload

    async def test_the_compact_route_answers_the_same_refusal(self, scoped):
        status, payload = await scoped.send(
            "POST", "/v1/responses/compact", {"model": "m-z", "input": []}
        )

        assert status == 403
        assert b"This API key is not allowed to use model 'm-z'" in payload

    async def test_a_listing_names_only_what_the_key_may_call(self, scoped):
        status, payload = await scoped.send("GET", "/v1/models")

        assert status == 200
        body = payload.decode()
        assert '"m-a"' in body
        assert '"m-b"' not in body, "a scoped key reads only what it may call"

    async def test_the_google_listing_answers_the_same_set(self, scoped):
        status, payload = await scoped.send("GET", "/v1beta/models")

        assert status == 200
        body = payload.decode()
        assert "models/m-a" in body
        assert "models/m-b" not in body

    async def test_the_404_note_does_not_name_the_catalogue_to_a_scoped_key(
        self, tmp_path
    ):
        # The key may call "m-z", which no provider serves: the answer is 404,
        # and the configured set is not spelled out to a key that may not see
        # all of it.
        async for gateway in _serve(tmp_path, ["m-a", "m-z"]):
            status, payload = await gateway.send(
                "POST", "/v1/chat/completions", {"model": "m-z", "messages": []}
            )

        assert status == 404
        assert b"Configured models" not in payload


@pytest.fixture()
async def unscoped(tmp_path) -> AsyncIterator[Served]:
    async for gateway in _serve(tmp_path, None):
        yield gateway


async def test_an_unscoped_key_still_reads_the_whole_listing(unscoped):
    status, payload = await unscoped.send("GET", "/v1/models")

    assert status == 200
    assert b'"m-a"' in payload and b'"m-b"' in payload
