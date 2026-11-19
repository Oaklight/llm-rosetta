"""The stream leg: one transport chunk becomes one group of IR events.

The chunk vocabulary is the ACP transport's, not a wire format's.  What these
tests pin is the contract between the two halves: which chunk produces which
event, that the stream ends exactly once, and that an agent error is reported
as a loss rather than as an empty successful turn.
"""

from __future__ import annotations

from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.llm.acp import AcpConverter


def _converter() -> AcpConverter:
    return AcpConverter()


def _context() -> StreamContext:
    return StreamContext()


def _types(events) -> list[str]:
    return [dict(event)["type"] for event in events]


def _only(events) -> dict[str, Any]:
    """The one event a chunk produced, as a plain mapping ty can index."""
    assert len(events) == 1
    return dict(events[0])


class TestTheChunks:
    def test_session_start_opens_the_stream_once(self):
        ctx = _context()
        first = _converter().stream_response_from_provider(
            {"type": "session_start", "session_id": "s1", "model": "gemini-x"},
            ctx,
        )
        second = _converter().stream_response_from_provider(
            {"type": "session_start", "session_id": "s1", "model": "gemini-x"},
            ctx,
        )

        assert _types(first) == ["stream_start"]
        assert _only(first)["response_id"] == "s1"
        assert second == []

    def test_text_and_thought_become_deltas(self):
        converter = _converter()
        ctx = _context()

        text = converter.stream_response_from_provider(
            {"type": "text", "text": "Hello"}, ctx
        )
        thought = converter.stream_response_from_provider(
            {"type": "thought", "text": "hm"}, ctx
        )

        assert text == [{"type": "text_delta", "text": "Hello"}]
        assert thought == [{"type": "reasoning_delta", "reasoning": "hm"}]

    def test_an_empty_delta_emits_nothing(self):
        assert (
            _converter().stream_response_from_provider({"type": "text", "text": ""})
            == []
        )

    def test_usage_is_an_input_side_figure(self):
        """ACP's ``used`` is context occupancy, not one turn's bill."""
        events = _converter().stream_response_from_provider(
            {"type": "usage", "used": 42, "size": 1000}
        )

        usage = cast(dict[str, Any], _only(events)["usage"])
        assert usage["prompt_tokens"] == 42
        assert usage["completion_tokens"] == 0

    def test_stop_maps_the_reason(self):
        events = _converter().stream_response_from_provider(
            {"type": "stop", "stop_reason": "max_tokens"}
        )

        assert events == [{"type": "finish", "finish_reason": {"reason": "length"}}]

    def test_end_of_stream_is_emitted_exactly_once(self):
        converter = _converter()
        ctx = _context()

        first = converter.stream_response_from_provider(None, ctx)
        second = converter.stream_response_from_provider(None, ctx)

        assert _types(first) == ["stream_end"]
        assert second == []


class TestTheErrorChunk:
    def test_an_agent_error_is_a_loss_and_an_error_finish(self):
        ctx = _context()
        events = _converter().stream_response_from_provider(
            {"type": "error", "message": "model call failed"}, ctx
        )

        assert events == [{"type": "finish", "finish_reason": {"reason": "error"}}]
        assert any("model call failed" in str(w) for w in ctx.warnings)
