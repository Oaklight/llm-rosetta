"""The EOF sentinel every stream reader must accept (#1038).

Callers signal end-of-stream by passing ``None`` where a chunk would go:
``BaseConverter.iter_stream_from_provider`` does it once its iterable is
exhausted, and the gateway's ``StreamProcessor`` does it in
``_flush_deferred()`` when the upstream connection closes.  A reader that
normalizes ``None`` like a chunk raises instead, which past the end of a
response is unrecoverable — the gateway turns it into a terminal error event
where the client was waiting for ``[DONE]``.
"""

from typing import Any, cast

import pytest

from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.pipeline import ConversionPipeline

#: Every chat format, as the *upstream* the gateway proxies to.
FORMATS = [
    "chat_completions",
    "open_responses",
    "messages",
    "google_generate",
    "google_interactions",
]

#: A first chunk of each dialect: enough to start the stream and produce
#: something, so the sentinel is exercised after real state was built up.
FIRST_CHUNK: dict[str, dict[str, Any]] = {
    "chat_completions": {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "m",
        "created": 1,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
    },
    "open_responses": {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hi",
    },
    "messages": {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    },
    "google_generate": {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "hi"}]}, "index": 0}
        ]
    },
    "google_interactions": {"event_type": "step.delta", "delta": {"text": "hi"}},
}


class TestReadersAcceptTheSentinel:
    """Each reader ignores the sentinel instead of normalizing it."""

    @pytest.mark.parametrize("name", FORMATS)
    def test_none_returns_no_events(self, name: str):
        converter = get_converter_for_provider(name)
        assert converter.stream_response_from_provider(None) == []

    @pytest.mark.parametrize("name", FORMATS)
    def test_none_after_a_chunk_returns_no_events(self, name: str):
        converter = get_converter_for_provider(name)
        ctx = converter.create_stream_context()
        converter.stream_response_from_provider(FIRST_CHUNK[name], context=ctx)
        events = converter.stream_response_from_provider(None, context=ctx)
        assert events == []

    def test_the_reader_that_defers_still_flushes(self):
        """The sentinel flushes deferred state; it is not just ignored.

        Chat Completions holds its ``StreamEndEvent`` back while it waits for
        a usage chunk, so the sentinel is how that stream ends.  A guard that
        returned ``[]`` unconditionally would quietly cut it off.
        """
        converter = ChatCompletionsConverter()
        ctx = converter.create_stream_context()
        converter.stream_response_from_provider(
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]}, context=ctx
        )
        # A finish chunk with no usage defers the stream end.
        converter.stream_response_from_provider(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            context=ctx,
        )
        events = cast(
            list[Any],
            converter.stream_response_from_provider(None, context=ctx),
        )
        assert [event["type"] for event in events] == ["stream_end"]


class TestGatewayEofFlush:
    """``StreamProcessor.process_chunk(None)`` survives every upstream."""

    @pytest.mark.parametrize("upstream", FORMATS)
    def test_flush_after_a_partial_stream_adds_nothing(self, upstream: str):
        # The client speaks Responses; `upstream` is what the gateway proxies
        # to, so the flush runs that dialect's reader.  Nothing is pending in
        # this turn — the finishes here are not deferred ones — so the tail
        # has to be empty rather than merely well-formed.
        pipeline = ConversionPipeline("open_responses", upstream, baseline=False)
        pipeline.convert_request({"model": "m", "input": "hi", "stream": True})
        processor = pipeline.create_stream_processor()
        processor.process_chunk(FIRST_CHUNK[upstream])
        assert processor.process_chunk(None) == []

    @pytest.mark.parametrize("upstream", FORMATS)
    def test_flush_before_any_chunk_adds_nothing(self, upstream: str):
        """A stream that ends before its first chunk still ends cleanly."""
        pipeline = ConversionPipeline("open_responses", upstream, baseline=False)
        pipeline.convert_request({"model": "m", "input": "hi", "stream": True})
        processor = pipeline.create_stream_processor()
        assert processor.process_chunk(None) == []

    def test_flush_delivers_a_deferred_terminal_event(self):
        """The flush consults the reader, it does not just discard the call.

        A Chat upstream that reports its finish without usage leaves the
        stream end held back, and the EOF flush is what hands the client its
        terminal event.  Anything that returned early — a processor that
        skipped the reader, a reader that answered ``[]`` for everything —
        would leave the client without a ``response.completed`` at all.
        """
        pipeline = ConversionPipeline(
            "open_responses", "chat_completions", baseline=False
        )
        pipeline.convert_request({"model": "m", "input": "hi", "stream": True})
        processor = pipeline.create_stream_processor()
        processor.process_chunk(FIRST_CHUNK["chat_completions"])
        processor.process_chunk(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
        tail = processor.process_chunk(None)
        assert "response.completed" in [event.get("type") for event in tail]
