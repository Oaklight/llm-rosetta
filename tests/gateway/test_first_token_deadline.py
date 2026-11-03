"""The model's first-token deadline: a quiet lane is not a serving lane.

``first_token_ms`` is the one routing-block key an operator can set that says
"a provider that answers headers and then stalls is not usable".  The deadline
is enforced before the ``StreamingResponse`` exists, which is the only point
where the request can still be answered by another provider or with an honest
``504``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llm_rosetta_gateway._vendor.httpserver import StreamingResponse
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import ProviderMetadataStore, handle_streaming

_REQUEST: dict[str, Any] = {
    "model": "m",
    "messages": [{"role": "user", "content": "hello"}],
}

_CHUNK: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
}


class _StalledStream:
    """An upstream that accepted the request and then sent nothing."""

    status_code = 200
    is_error = False

    def __init__(self) -> None:
        self.closed = False

    async def read_error(self) -> str:  # pragma: no cover - never an error
        return ""

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(5)
        raise StopAsyncIteration  # pragma: no cover - the wait never ends


class _StalledStreamThatFailsToClose(_StalledStream):
    """A quiet stream whose socket also refuses a clean close."""

    async def close(self) -> None:
        raise ConnectionResetError("peer reset during close")


class _ChunkStream:
    """An upstream that sends the scripted chunks, then ends."""

    status_code = 200
    is_error = False

    def __init__(self, *chunks: Any) -> None:
        self._chunks = list(chunks)
        self.closed = False

    async def read_error(self) -> str:  # pragma: no cover - never an error
        return ""

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _Transport:
    """An upstream whose streaming calls answer from the scripted streams."""

    def __init__(self, *streams: Any) -> None:
        self._streams = list(streams)
        self.calls: list[str] = []

    async def send_streaming(
        self, provider_info: Any, url: str, body: Any, **kw: Any
    ) -> Any:
        provider_info.auth_headers()
        self.calls.append(provider_info.base_url)
        return self._streams.pop(0)

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        raise AssertionError("this test never takes the non-streaming path")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""


def _config(
    *,
    providers: tuple[str, ...] = ("a", "b"),
    budget: int | None = 50,
    opt_in: bool = True,
) -> GatewayConfig:
    entry: dict[str, Any] = {"providers": list(providers)}
    if budget is not None:
        entry["routing"] = {"first_token_ms": budget}
    if not opt_in:
        # The switch belongs to the model entry, and master turns it on by
        # default, so only the opt-out case says anything.
        entry["failover_on_error"] = False
    config = GatewayConfig(
        {
            "providers": {
                name: {
                    "type": "chat_completions",
                    "api_key": f"sk-{name}",
                    "base_url": f"https://{name}.test",
                }
                for name in providers
            },
            "models": {"m": entry},
        }
    )
    return config


def _run_stream(config: GatewayConfig, transport: _Transport):
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = {**_REQUEST, "stream": True}

    async def _go():
        return await handle_streaming(
            route,
            pinfo,
            body,
            reported_model="m",
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    return asyncio.run(_go())


class TestTheDeadline:
    def test_a_stalled_lane_answers_504(self):
        """No other lane: the client gets the deadline, not a dead stream."""
        config = _config(providers=("a",))
        transport = _Transport(_StalledStream())

        response, profile = _run_stream(config, transport)

        assert response.status_code == 504
        error = json.loads(response.body)["error"]
        assert "no first token within 50 ms" in error["message"]
        assert "'a'" in error["message"]
        assert profile["first_token_timeout_ms"] == 50

    def test_the_stalled_stream_is_closed(self):
        config = _config(providers=("a",))
        stalled = _StalledStream()

        _run_stream(config, _Transport(stalled))

        assert stalled.closed is True

    def test_a_failed_close_does_not_replace_the_504(self):
        """The socket that went quiet is the one likely to refuse a close."""
        config = _config(providers=("a",))

        response, profile = _run_stream(
            config, _Transport(_StalledStreamThatFailsToClose())
        )

        assert response.status_code == 504
        assert profile["first_token_timeout_ms"] == 50

    def test_the_second_lane_serves_when_the_first_stalls(self):
        config = _config()
        transport = _Transport(_StalledStream(), _ChunkStream(_CHUNK))

        response, profile = _run_stream(config, transport)

        assert isinstance(response, StreamingResponse)
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["from"] == "a"
        assert profile["failover"]["to"] == "b"
        assert profile["failover"]["reason"] == "no first token within 50 ms"

    def test_a_model_with_failover_off_keeps_the_504(self):
        config = _config(opt_in=False)
        transport = _Transport(_StalledStream(), _ChunkStream(_CHUNK))

        response, profile = _run_stream(config, transport)

        assert response.status_code == 504
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile

    def test_without_a_budget_nothing_waits(self):
        """A model that sets no deadline keeps the old behaviour."""
        config = _config(budget=None)
        transport = _Transport(_StalledStream())

        response, profile = _run_stream(config, transport)

        assert isinstance(response, StreamingResponse)
        assert "first_token_timeout_ms" not in profile


class TestThePrefetchedChunk:
    def test_the_first_chunk_reaches_the_client(self):
        """The deadline consumes the first chunk; the stream must replay it."""
        config = _config(providers=("a",))
        transport = _Transport(_ChunkStream(_CHUNK))

        response, _profile = _run_stream(config, transport)

        async def _drain() -> str:
            parts: list[str] = []
            async for chunk in response._generator:  # noqa: SLF001 - the only reader
                parts.append(chunk if isinstance(chunk, str) else chunk.decode())
            return "".join(parts)

        body = asyncio.run(_drain())

        assert "hi" in body

    def test_an_empty_stream_is_not_a_missed_deadline(self):
        """The upstream closed without an event: an empty stream, not a 504."""
        config = _config(providers=("a",))

        response, profile = _run_stream(config, _Transport(_ChunkStream()))

        assert isinstance(response, StreamingResponse)
        assert "first_token_timeout_ms" not in profile


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The quota state is process-wide; a failure here must not leak."""
    from llm_rosetta_gateway.quota.state import QUOTA_STATE

    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()
