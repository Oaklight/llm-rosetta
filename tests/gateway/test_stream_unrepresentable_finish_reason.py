"""A stream that ends in a terminal state the client's dialect cannot say.

The JSON leg answers 502 (#1234); a stream has already committed its status by
the time the converter reaches the finish, so the same classification has to
ride the in-band terminal error instead: the dialect's own envelope, the
``502`` / ``UNAVAILABLE`` pair the JSON leg would have used, and the reason the
converter refused (#1237).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta_gateway.proxy import _stream_event_generator
from llm_rosetta_gateway.proxy.streaming import _terminal_error_sse
from llm_rosetta_gateway.transport.sse_format import (
    SSE_FORMATTERS,
    build_stream_error_events,
)

_REQUESTS: dict[str, dict[str, Any]] = {
    "chat_completions": {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    },
    "messages": {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    },
    "google_generate": {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    },
}

#: A Google candidate whose call the model itself malformed: the reader files it
#: under the IR's ``error``, and Chat and Messages have no word for it.
_MALFORMED_CHUNK = {
    "candidates": [
        {
            "index": 0,
            "content": {"role": "model", "parts": [{"text": "par"}]},
            "finishReason": "MALFORMED_FUNCTION_CALL",
        }
    ]
}


class _ScriptedStream:
    """An upstream that yields the scripted chunks and then ends."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        return False


def _processor(source: str) -> Any:
    pipeline = ConversionPipeline(source, "google_generate")
    pipeline.convert_request(_REQUESTS[source])
    return pipeline.create_stream_processor()


async def _drain(source: str) -> tuple[list[str], BaseException | None]:
    gen = _stream_event_generator(
        source_provider=cast(Any, source),
        stream=_ScriptedStream([dict(_MALFORMED_CHUNK)]),
        processor=_processor(source),
        model="m",
        format_sse=SSE_FORMATTERS[source],
    )
    out: list[str] = []
    raised: BaseException | None = None
    try:
        async for event in gen:
            out.append(event)
    except BaseException as exc:  # noqa: BLE001 - the raise is part of the test
        raised = exc
    return out, raised


def _json_bodies(events: list[str]) -> list[Any]:
    """Every SSE data payload that is JSON, in order."""
    bodies: list[Any] = []
    for event in events:
        if "data: " not in event:
            continue
        raw = event.split("data: ", 1)[1].strip()
        try:
            bodies.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return bodies


@pytest.mark.parametrize("source", ["chat_completions", "messages"])
def test_a_terminal_failure_reaches_a_streaming_client(source: str) -> None:
    """The stream says the turn failed rather than ending normally."""
    events, raised = asyncio.run(_drain(source))

    assert isinstance(raised, UnrepresentableFinishReason)
    assert any("cannot represent" in event for event in events)

    error = next(body for body in _json_bodies(events) if "error" in body)
    if source == "messages":
        # Anthropic's in-band event carries the JSON leg's type.
        assert error["type"] == "error"
        assert error["error"]["type"] == "api_error"
    else:
        # Chat's in-band ``type`` carries the JSON leg's classification; a
        # plan-limit code still replaces it (see test_stream_upstream_error).
        assert error["error"]["type"] == "api_error"


def test_a_google_word_survives_a_google_stream() -> None:
    """Same dialect, so the word is expressible and nothing is refused."""
    events, raised = asyncio.run(_drain("google_generate"))

    assert raised is None
    assert not any("error" in body for body in _json_bodies(events))
    finish = _json_bodies(events)[-1]["candidates"][0]
    assert finish["finishReason"] == "MALFORMED_FUNCTION_CALL"


class _Context:
    """The slice of a stream context the terminal-error path reads."""

    is_ended = False
    outbound_response_id = ""
    next_sequence_number = 3


class _FakeProcessor:
    source_context = _Context()


def test_a_refusal_is_a_502_in_band_and_a_gateway_fault_stays_500():
    """The in-band pair matches the JSON leg's envelope (#1237)."""
    refusal = UnrepresentableFinishReason(
        "Google generateContent cannot represent a response ending with 'error' "
        "(response_id='g_x')",
        reason="error",
        response_id="g_x",
    )

    refusal_sse = "".join(
        _terminal_error_sse(
            cast(Any, "google_generate"),
            _FakeProcessor(),
            SSE_FORMATTERS["google_generate"],
            refusal,
        )
    )
    body = json.loads(refusal_sse.split("data: ", 1)[1].strip())

    assert body["error"]["code"] == 502
    assert body["error"]["status"] == "UNAVAILABLE"
    assert "cannot represent" in body["error"]["message"]
    assert "g_x" in body["error"]["message"]

    # A broken upstream connection is a gateway-side fault, not a refusal.
    fault = "".join(
        _terminal_error_sse(
            cast(Any, "google_generate"),
            _FakeProcessor(),
            SSE_FORMATTERS["google_generate"],
            ConnectionError("socket closed"),
        )
    )
    fault_body = json.loads(fault.split("data: ", 1)[1].strip())
    assert fault_body["error"]["code"] == 500
    assert fault_body["error"]["status"] == "INTERNAL"


def test_the_builder_takes_the_status_from_its_caller():
    """The default keeps every existing caller's envelope unchanged."""
    event = build_stream_error_events("google_generate", "boom")[0]

    assert event["error"]["code"] == 500
    assert event["error"]["status"] == "INTERNAL"

    # Chat's generic fault keeps OpenAI's own stream default; the 502 that a
    # proxy answers with is `api_error` on both legs.
    assert build_stream_error_events("chat_completions", "boom")[0]["error"][
        "type"
    ] == ("server_error")
    assert (
        build_stream_error_events("chat_completions", "boom", status=502)[0]["error"][
            "type"
        ]
        == "api_error"
    )

    mapped = build_stream_error_events(
        "google_generate", "boom", status=502, google_status="UNAVAILABLE"
    )[0]
    assert mapped["error"]["code"] == 502
    assert mapped["error"]["status"] == "UNAVAILABLE"
