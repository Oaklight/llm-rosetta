"""Tests for terminal event emission when an upstream stream dies mid-flight.

Regression coverage for #481: the generator used to re-raise without telling
the downstream client anything, so clients waiting on a terminal event saw
only an abrupt socket close.

Refactored in #485: replaced test doubles (_FakeContext, _FakeProcessor)
with real StreamContext and ConversionPipeline objects to eliminate
interface drift between fakes and production code.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.gateway.proxy import _stream_event_generator
from llm_rosetta.gateway.transport.sse_format import (
    SSE_FORMATTERS,
    build_stream_error_events,
    format_sse_done,
)
from llm_rosetta.pipeline import ConversionPipeline

UPSTREAM_MESSAGE = "connection reset by peer"


class _AbortingStream:
    """Yields some chunks, then fails the way a dropped upstream would."""

    def __init__(self, chunks: list[dict[str, Any]], exc: BaseException):
        self._chunks = list(chunks)
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._chunks:
            return self._chunks.pop(0)
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(source: str, target: str | None = None) -> Any:
    """Build a real stream processor from a ConversionPipeline."""
    target = target or source
    pipeline = ConversionPipeline(source, target)
    pipeline.convert_request(
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        if source in ("openai_responses", "open_responses")
        else {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    )
    return pipeline.create_stream_processor()


async def _drain(
    provider: ProviderType,
    *,
    exc: BaseException,
    processor: Any = None,
) -> tuple[list[str], BaseException | None]:
    """Run the generator to failure, returning emitted SSE and the exception."""
    if processor is None:
        processor = _make_processor(provider)

    gen = _stream_event_generator(
        source_provider=provider,
        stream=_AbortingStream([{"type": "response.output_text.delta"}], exc),
        processor=processor,
        model="test-model",
        format_sse=SSE_FORMATTERS[provider],
    )
    events: list[str] = []
    raised: BaseException | None = None
    try:
        async for event in gen:
            events.append(event)
    except BaseException as e:  # noqa: BLE001 - we assert on what propagated
        raised = e
    return events, raised


# ---------------------------------------------------------------------------
# Core abort-path tests (real contexts only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    ["openai_chat", "openai_responses", "open_responses", "anthropic", "google"],
)
def test_terminal_event_emitted_on_abort(provider: str) -> None:
    """Every format gets a terminal notice carrying the upstream reason."""
    events, raised = asyncio.run(
        _drain(cast(ProviderType, provider), exc=ConnectionError(UPSTREAM_MESSAGE))
    )

    assert isinstance(raised, ConnectionError)
    joined = "".join(events)
    assert UPSTREAM_MESSAGE in joined
    assert "Upstream stream ended before completion" in joined


def test_responses_emits_response_failed() -> None:
    """Responses clients get `response.failed`, the event Codex waits on."""
    processor = _make_processor("openai_responses")
    ctx = processor.source_context
    ctx.response_id = "abc"
    ctx.options["response_id_prefix"] = "resp_"

    events, _ = asyncio.run(
        _drain(
            cast(ProviderType, "openai_responses"),
            exc=ConnectionError(UPSTREAM_MESSAGE),
            processor=processor,
        )
    )

    failed = [e for e in events if e.startswith("event: response.failed")]
    assert len(failed) == 1

    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["id"] == "resp_abc"
    assert UPSTREAM_MESSAGE in payload["response"]["error"]["message"]
    assert isinstance(payload["sequence_number"], int)

    assert events[-1] == format_sse_done()


def test_no_terminal_event_when_stream_already_ended() -> None:
    """A failure during teardown must not append a second terminal event."""
    processor = _make_processor("openai_responses")
    processor.source_context.mark_ended()

    events, raised = asyncio.run(
        _drain(
            cast(ProviderType, "openai_responses"),
            exc=ConnectionError("late failure"),
            processor=processor,
        )
    )

    assert isinstance(raised, ConnectionError)
    assert not any("response.failed" in e for e in events)


def test_cancellation_emits_nothing() -> None:
    """Client disconnects leave nobody to notify, and must stay cancellations."""
    events, raised = asyncio.run(
        _drain(
            cast(ProviderType, "openai_responses"),
            exc=asyncio.CancelledError(),
        )
    )

    assert isinstance(raised, asyncio.CancelledError)
    assert not any("response.failed" in e for e in events)


def test_original_exception_survives_builder_failure() -> None:
    """A broken context must not replace the real upstream error.

    This uses an intentionally broken object — not a fake that mirrors the
    real interface, but a sabotaged one that proves the guard catches
    context failures without masking the original exception.
    """

    class _ExplodingContext:
        @property
        def is_ended(self) -> bool:
            raise RuntimeError("context is broken")

    class _ExplodingProcessor:
        @property
        def source_context(self) -> Any:
            return _ExplodingContext()

        def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
            return [chunk]

    events, raised = asyncio.run(
        _drain(
            cast(ProviderType, "openai_responses"),
            exc=ConnectionError(UPSTREAM_MESSAGE),
            processor=_ExplodingProcessor(),
        )
    )

    assert isinstance(raised, ConnectionError)
    assert str(raised) == UPSTREAM_MESSAGE
    assert not any("response.failed" in e for e in events)


def test_builder_returns_empty_for_unknown_format() -> None:
    """Unknown source formats have no terminal convention to honour."""
    assert build_stream_error_events("mystery", "boom") == []


# ---------------------------------------------------------------------------
# Context contract verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    ["openai_chat", "openai_responses", "anthropic", "google"],
)
def test_context_exposes_abort_path_attributes(provider: str) -> None:
    """Every converter's stream context satisfies the abort-path contract.

    `_terminal_error_sse` reads `is_ended`, `outbound_response_id`, and
    `next_sequence_number` through a try guard.  A missing or mistyped
    attribute degrades to "emit nothing" rather than failing loudly.
    Assert the shape here so drift is caught at test time.
    """
    processor = _make_processor(provider)
    ctx = processor.source_context

    assert isinstance(ctx.is_ended, bool)
    seq = ctx.next_sequence_number
    assert seq is None or isinstance(seq, int)
    assert isinstance(ctx.outbound_response_id, str)

    ctx.mark_ended()
    assert ctx.is_ended is True

    ctx.response_id = "xyz"
    ctx.options["response_id_prefix"] = "pfx_"
    assert ctx.outbound_response_id == "pfx_xyz"


def test_responses_context_supplies_sequence_number() -> None:
    """`next_sequence_number` resolves on the real Responses context."""
    processor = _make_processor("openai_responses")
    ctx = processor.source_context

    assert isinstance(ctx.next_sequence_number, int)


def test_terminal_event_id_matches_the_stream_it_ends() -> None:
    """The synthesized event must carry the same ID as the events before it.

    ``response_id`` on the context is the bare stem — the source prefix is
    stripped on ingest and re-added by the converter on output. Reading it
    directly emitted ``real1`` while every earlier event in the same stream
    said ``resp_real1``, i.e. two IDs for one response.
    """
    upstream_created = {
        "type": "response.created",
        "response": {
            "id": "resp_real1",
            "object": "response",
            "status": "in_progress",
            "model": "m",
            "output": [],
        },
    }

    processor = _make_processor("openai_responses")
    gen = _stream_event_generator(
        source_provider=cast(ProviderType, "openai_responses"),
        stream=_AbortingStream([upstream_created], ConnectionError(UPSTREAM_MESSAGE)),
        processor=processor,
        model="test-model",
        format_sse=SSE_FORMATTERS["openai_responses"],
    )

    events: list[str] = []

    async def run() -> None:
        try:
            async for event in gen:
                events.append(event)
        except ConnectionError:
            pass

    asyncio.run(run())

    ids = {
        json.loads(e.split("data: ", 1)[1].strip())["response"]["id"]
        for e in events
        if "data: " in e and '"response"' in e
    }
    assert ids == {"resp_real1"}, f"inconsistent response ids across stream: {ids}"
