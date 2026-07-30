"""Tests for SSE [DONE] terminator emission in gateway streaming."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.gateway.proxy import _stream_event_generator
from llm_rosetta.gateway.transport.sse_format import format_sse_done


class _FakeStream:
    """Async iterator that yields chunks, usable as an async context manager."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = list(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeProcessor:
    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        return [chunk]


def _identity_format(chunk: dict[str, Any]) -> str:
    return f"data: {chunk}\n\n"


async def _collect(source_provider: ProviderType) -> list[str]:
    chunks: list[dict[str, Any]] = [{"type": "response.completed"}]
    events: list[str] = []
    gen = _stream_event_generator(
        source_provider=source_provider,
        stream=_FakeStream(chunks),
        processor=_FakeProcessor(),
        model="test-model",
        format_sse=_identity_format,
    )
    async for event in gen:
        events.append(event)
    return events


DONE_MARKER = format_sse_done()


@pytest.mark.parametrize(
    "provider",
    ["openai_chat", "openai_responses", "open_responses"],
)
def test_done_emitted(provider: str) -> None:
    events = asyncio.run(_collect(cast(ProviderType, provider)))
    assert events[-1] == DONE_MARKER


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "google"],
)
def test_done_not_emitted(provider: str) -> None:
    events = asyncio.run(_collect(cast(ProviderType, provider)))
    assert DONE_MARKER not in events
