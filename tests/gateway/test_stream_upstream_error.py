"""Tests for in-band upstream errors on the streaming path.

Some upstreams report a request error inside a 200 SSE stream rather than by
HTTP status.  ARGO answers an over-limit `tools` array with `event: error` on
a 200.  Such a chunk converts to zero source events, so without explicit
handling the client receives a successful but completely empty stream.
"""

import json

import asyncio

import pytest

from llm_rosetta_gateway.proxy import _stream_event_generator
from llm_rosetta_gateway.transport.sse_format import (
    build_stream_error_events,
    classify_upstream_error,
    is_upstream_error_chunk,
)

ERROR_CHUNK = {
    "error": {
        "message": "Invalid 'tools': array too long. Expected an array with "
        "maximum length 128, but got an array with length 129 instead.",
        "type": "BadRequestError",
        "code": "internal_error",
    }
}


class _FakeStream:
    """Minimal stand-in for an upstream SSE stream."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.status_code = 200
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _FakeProcessor:
    """Converts content chunks; an error chunk yields nothing, as in real life."""

    def process_chunk(self, chunk):
        if chunk is None:
            return []
        if "error" in chunk:
            return []
        return [{"type": "response.output_text.delta", "delta": chunk.get("text", "")}]


def _format_sse(event):
    return f"data: {json.dumps(event)}\n\n"


async def _collect(chunks, source_provider="open_responses"):
    out = []
    async for piece in _stream_event_generator(
        source_provider=source_provider,
        stream=_FakeStream(chunks),
        processor=_FakeProcessor(),
        model="gpt-5.6-sol",
        format_sse=_format_sse,
    ):
        out.append(piece)
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_bare_error_envelope():
    assert is_upstream_error_chunk(ERROR_CHUNK)


@pytest.mark.parametrize(
    "chunk",
    [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "error": None},
        # A provider shipping content alongside an error field is not an error.
        {"error": {"message": "x"}, "choices": [{"delta": {}}]},
        {"type": "response.output_text.delta", "error": {"message": "x"}},
        "not a dict",
    ],
)
def test_content_chunks_are_not_errors(chunk):
    assert not is_upstream_error_chunk(chunk)


def test_error_event_uses_source_envelope():
    assert (
        build_stream_error_events("messages", "boom")[0]["error"]["type"] == "api_error"
    )
    assert build_stream_error_events("google", "boom")[0]["error"]["code"] == 500
    assert (
        build_stream_error_events("openai_responses", "boom")[0]["type"]
        == "response.failed"
    )


# ---------------------------------------------------------------------------
# response.failed carries the code the client branches on (#1060)
# ---------------------------------------------------------------------------

FAILED_EVENT = {
    "type": "response.failed",
    "sequence_number": 12,
    "response": {
        "id": "resp_chatgpt_plan_1",
        "object": "response",
        "status": "failed",
        "error": {
            "code": "subscription_sharing_usage_limit_exceeded",
            "message": "You have reached your plan's usage limit.",
        },
    },
}


def test_a_response_failed_event_is_an_error():
    assert is_upstream_error_chunk(FAILED_EVENT)
    classified = classify_upstream_error(FAILED_EVENT)
    assert classified is not None
    assert classified.code == "subscription_sharing_usage_limit_exceeded"
    assert "usage limit" in classified.message
    assert classified.request_id == "resp_chatgpt_plan_1"


def test_a_responses_error_event_is_an_error():
    classified = classify_upstream_error(
        {"type": "error", "code": "server_error", "message": "boom"}
    )
    assert classified is not None
    assert classified.code == "server_error"
    assert classified.message == "boom"


def test_the_terminal_event_keeps_the_upstream_code_and_request_id():
    events = build_stream_error_events(
        "openai_responses",
        "plan limit",
        code="subscription_sharing_usage_limit_exceeded",
        request_id="resp_chatgpt_plan_1",
    )
    error = events[0]["response"]["error"]
    assert error["code"] == "subscription_sharing_usage_limit_exceeded"
    assert events[0]["response"]["id"] == "resp_chatgpt_plan_1"

    # Without a code, the previous generic shape is unchanged.
    generic = build_stream_error_events("openai_responses", "boom")
    assert generic[0]["response"]["error"]["code"] == "server_error"


def test_the_code_reaches_a_chat_client_as_the_error_type():
    event = build_stream_error_events(
        "chat_completions", "boom", code="subscription_sharing_user_not_eligible"
    )[0]
    assert event["error"]["type"] == "subscription_sharing_user_not_eligible"
    assert event["error"]["code"] == "subscription_sharing_user_not_eligible"


def test_a_failed_event_terminates_the_downstream_stream():
    out = asyncio.run(_collect([FAILED_EVENT]))
    payload = json.loads(out[0].removeprefix("data: ").strip())
    assert payload["type"] == "response.failed"
    assert (
        payload["response"]["error"]["code"]
        == "subscription_sharing_usage_limit_exceeded"
    )
    assert payload["response"]["id"] == "resp_chatgpt_plan_1"


# ---------------------------------------------------------------------------
# Generator behaviour
# ---------------------------------------------------------------------------


def test_in_band_error_reaches_the_client():
    out = asyncio.run(_collect([ERROR_CHUNK]))
    assert out, "an upstream error must not produce an empty stream"
    payload = json.loads(out[0].removeprefix("data: ").strip())
    assert payload["type"] == "response.failed"
    assert "array too long" in payload["response"]["error"]["message"]


def test_stream_stops_after_an_error_but_stays_well_formed():
    out = asyncio.run(_collect([ERROR_CHUNK, {"text": "should not appear"}]))
    # The error, then the terminator. Content after the error is dropped.
    assert len(out) == 2
    assert "should not appear" not in "".join(out)
    assert out[-1] == "data: [DONE]\n\n"


def test_healthy_stream_is_unaffected():
    out = asyncio.run(_collect([{"text": "a"}, {"text": "b"}]))
    assert len(out) == 3  # two deltas plus the [DONE] terminator
    assert "error" not in out[0]


def test_stream_chunk_error_persists_tool_collisions(tmp_path):
    """The mid-stream dump path carries the collision map from DumpContext.

    A dump insert needs its ``request_log`` parent row to exist; the request-log
    entry is written when the request finishes, so this test provides it up
    front, the way a corrected lifecycle would.
    """
    from llm_rosetta_gateway.proxy import DumpContext
    from llm_rosetta_gateway.observability.persistence import PersistenceManager

    async def run():
        pm = await PersistenceManager.create(str(tmp_path))
        entry_id = "stream-907"
        await pm._conn.execute(
            "INSERT INTO request_log "
            "(id, timestamp, model, source_provider, target_provider, "
            "is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, 1, 200, 1.0)",
            (
                entry_id,
                "2026-01-01T00:00:00+00:00",
                "test-model",
                "chat_completions",
                "chat_completions",
            ),
        )
        await pm._conn.commit()

        dump_ctx = DumpContext(
            persistence=pm,
            request_body={"model": "test-model", "messages": []},
            converted_body={"model": "test-model", "messages": []},
            source_provider="chat_completions",
            target_provider="chat_completions",
            provider_name="test-provider",
            upstream_url="https://example.invalid/v1/chat/completions",
            tool_name_collisions={"exec": 2},
        )
        async for _ in _stream_event_generator(
            source_provider="chat_completions",
            stream=_FakeStream([ERROR_CHUNK]),
            processor=_FakeProcessor(),
            model="test-model",
            format_sse=_format_sse,
            entry_id=entry_id,
            dump_ctx=dump_ctx,
        ):
            pass

        entries, _ = await pm.query_error_dumps()
        await pm.close()
        return entries

    entries = asyncio.run(run())
    assert len(entries) == 1
    assert entries[0]["error_phase"] == "stream_chunk"
    assert entries[0]["tool_name_collisions"] == '{"exec": 2}'
