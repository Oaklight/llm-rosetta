"""Several ``UsageEvent`` of one stream are increments (#1749).

The contract is on :class:`llm_rosetta.ir.model.stream.UsageEvent`: a consumer
adds them, and a reader of a wire that repeats the running total on every chunk
emits the difference against the previous snapshot.  The live stream behind
#1749 (Google ``streamGenerateContent``, six chunks) ended at prompt 15,
completion 110, total 228 and repeated its last chunk.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.ir.context import StreamContext, add_usage
from llm_rosetta.pipeline.chat.streaming import StreamProcessor


def _google_chunk(
    completion: int,
    *,
    thoughts: int | None = None,
    cached: int | None = None,
    text: str = "x",
    finish: str | None = None,
) -> dict[str, Any]:
    """One streamGenerateContent chunk with the running usage."""
    thoughts_n = thoughts or 0
    usage: dict[str, Any] = {
        "promptTokenCount": 15,
        "candidatesTokenCount": completion,
        "totalTokenCount": 15 + completion + thoughts_n,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 15}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": completion}],
    }
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    if cached is not None:
        usage["cachedContentTokenCount"] = cached
    candidate: dict[str, Any] = {
        "index": 0,
        "content": {"role": "model", "parts": [{"text": text}]},
    }
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate], "usageMetadata": usage}


#: Completion and thoughts grow; the last chunk is sent twice, as live.
_LIVE = [
    _google_chunk(3, thoughts=20, cached=4),
    _google_chunk(10, thoughts=45, cached=4),
    _google_chunk(40, thoughts=70, cached=4),
    _google_chunk(80, thoughts=90, cached=4),
    _google_chunk(110, thoughts=103, cached=4),
    _google_chunk(110, thoughts=103, cached=4, finish="STOP"),
]


def _usage_events(converter: Any, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context = converter.create_stream_context()
    events = list(converter.iter_stream_from_provider(chunks, context=context))
    return [e["usage"] for e in events if e["type"] == "usage"]


def _sum(events: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] | None = None
    for event in events:
        total = add_usage(total, event)
    return total or {}


class TestUsageIncrement:
    def test_the_first_snapshot_is_all_new(self):
        ctx = StreamContext()
        assert ctx.usage_increment(
            {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
        ) == {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}

    def test_a_growing_snapshot_gives_the_difference(self):
        ctx = StreamContext()
        ctx.usage_increment(
            {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
        )
        assert ctx.usage_increment(
            {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        ) == {"prompt_tokens": 0, "completion_tokens": 7, "total_tokens": 7}

    def test_an_identical_snapshot_adds_nothing(self):
        ctx = StreamContext()
        snap = {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
        ctx.usage_increment(snap)
        assert ctx.usage_increment(dict(snap)) is None

    def test_nested_details_are_diffed_field_by_field(self):
        ctx = StreamContext()
        base = {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
        ctx.usage_increment({**base, "prompt_tokens_details": {"text_tokens": 1}})
        inc = ctx.usage_increment(
            {**base, "prompt_tokens_details": {"text_tokens": 1, "image_tokens": 4}}
        )
        assert inc is not None
        assert inc["prompt_tokens_details"] == {"image_tokens": 4}

    def test_the_first_snapshot_passes_whole_with_its_zeros(self):
        ctx = StreamContext()
        first = {
            "prompt_tokens": 20,
            "completion_tokens": 0,
            "total_tokens": 20,
            "cache_read_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        }
        assert ctx.usage_increment(first) == first

    def test_a_decrease_resets_nothing_a_high_water_mark_holds(self, caplog):
        ctx = StreamContext()
        snaps = [(10, 5), (0, 0), (10, 5)]
        incs = []
        with caplog.at_level(logging.WARNING, logger="llm_rosetta.ir.context"):
            for prompt, completion in snaps:
                incs.append(
                    ctx.usage_increment(
                        {
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "total_tokens": prompt + completion,
                        }
                    )
                )
        assert _sum([i for i in incs if i is not None]) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert incs[1:] == [None, None]

    def test_a_snapshot_that_dips_counts_only_the_rise_over_the_maximum(self, caplog):
        ctx = StreamContext()
        total = 0
        with caplog.at_level(logging.WARNING, logger="llm_rosetta.ir.context"):
            for value in (100, 90, 120):
                inc = ctx.usage_increment(
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": value,
                        "total_tokens": value + 1,
                    }
                )
                total += inc["completion_tokens"] if inc else 0
        assert total == 120
        assert caplog.text.count("went down") == 1  # logged once per stream

    def test_a_field_newly_reported_as_zero_is_not_an_event(self):
        ctx = StreamContext()
        base = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        ctx.usage_increment(base)
        assert ctx.usage_increment({**base, "cache_read_tokens": 0}) is None

    def test_add_usage_sums_numbers_and_details_and_skips_unstated(self):
        total = add_usage(
            {"prompt_tokens": 5, "total_tokens": 5, "d": {"a": 1}},
            {"completion_tokens": 2, "total_tokens": 2, "d": {"b": 3}, "x": None},
        )
        assert total == {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "d": {"a": 1, "b": 3},
            "x": None,
        }
        assert add_usage(total, {"prompt_tokens": 0, "x": None})["x"] is None

    def test_add_usage_does_not_modify_its_arguments(self):
        first = {"d": {"a": 1}}
        add_usage(first, {"d": {"a": 2}})
        assert first == {"d": {"a": 1}}


class TestGoogleReader:
    def test_the_events_of_a_live_shaped_stream_sum_to_the_last_snapshot(self):
        reader = GoogleGenerateConverter()
        events = _usage_events(reader, _LIVE)
        body = reader._build_p_usage_to_ir(_LIVE[-1]["usageMetadata"])
        assert _sum(events) == dict(body)
        assert (
            _sum(events)["prompt_tokens"],
            _sum(events)["completion_tokens"],
            _sum(events)["total_tokens"],
        ) == (15, 110, 228)

    def test_the_repeated_last_chunk_emits_no_event(self):
        events = _usage_events(GoogleGenerateConverter(), _LIVE)
        assert len(events) == 5

    def test_thoughts_cache_and_modality_details_are_covered(self):
        total = _sum(_usage_events(GoogleGenerateConverter(), _LIVE))
        assert total["reasoning_tokens"] == 103
        assert total["cache_read_tokens"] == 4
        assert total["prompt_tokens_details"] == {"text_tokens": 15}
        assert total["completion_tokens_details"] == {"text_tokens": 110}

    def test_without_a_context_the_snapshot_is_passed_as_is(self):
        reader = GoogleGenerateConverter()
        events = list(reader.iter_stream_from_provider(_LIVE[:2], context=None))
        assert len([e for e in events if e["type"] == "usage"]) == 2


class TestChatReader:
    def test_a_repeating_endpoint_sums_to_the_last_usage(self):
        def chunk(delta: dict, usage: dict | None, finish: str | None = None) -> dict:
            body: dict[str, Any] = {
                "id": "x",
                "object": "chat.completion.chunk",
                "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                body["usage"] = usage
            return body

        u1 = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        u2 = {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}
        events = _usage_events(
            ChatCompletionsConverter(),
            [
                chunk({"role": "assistant", "content": "a"}, u1),
                chunk({"content": "b"}, u2),
                chunk({}, u2, "stop"),
            ],
        )
        assert _sum(events) == u2

    def test_a_single_final_usage_is_unchanged(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}
        events = _usage_events(
            ChatCompletionsConverter(),
            [
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [
                        {"index": 0, "delta": {"content": "a"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            ],
        )
        assert _sum(events) == usage


def _gateway_record(target: Any, source: Any, chunks: list[dict[str, Any]]) -> Any:
    """The usage the gateway records: ``StreamProcessor.get_accumulated_usage``."""
    processor = StreamProcessor(
        target_converter=target,
        source_converter=source,
        from_ctx=target.create_stream_context(),
        to_ctx=source.create_stream_context(),
    )
    for chunk in chunks:
        processor.process_chunk(chunk)
    processor.process_chunk(None)
    return processor.get_accumulated_usage()


class TestGatewayRecord:
    def test_a_cumulative_google_stream_is_recorded_once(self):
        usage = _gateway_record(
            GoogleGenerateConverter(), ChatCompletionsConverter(), _LIVE
        )
        assert usage is not None
        assert (
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
        ) == (
            15,
            110,
            228,
        )

    def test_split_messages_usage_is_recorded_as_the_sum(self):
        """``message_start`` carries the input and ``message_delta`` the output;
        the total is theirs added, where the old rule kept only the last one."""
        chunks = [
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [],
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 30},
            },
            {"type": "message_stop"},
        ]
        usage = _gateway_record(MessagesConverter(), ChatCompletionsConverter(), chunks)
        assert usage is not None
        assert (
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
        ) == (
            12,
            30,
            42,
        )


def test_the_chat_writer_reports_the_total_of_a_cumulative_google_stream():
    """Google cumulative chunks written as Chat: the final usage is the total."""
    reader = GoogleGenerateConverter()
    writer = ChatCompletionsConverter()
    r_ctx = reader.create_stream_context()
    w_ctx = writer.create_stream_context()
    usages: list[dict[str, Any]] = []
    for event in reader.iter_stream_from_provider(_LIVE, context=r_ctx):
        out = writer.stream_response_to_provider(event, context=w_ctx)
        for chunk in out if isinstance(out, list) else [out]:
            if chunk and chunk.get("usage"):
                usages.append(chunk["usage"])
    assert usages, "the Chat writer emitted no usage chunk"
    assert usages[-1]["total_tokens"] == 228
    assert usages[-1]["completion_tokens"] == 110
    assert usages[-1]["prompt_tokens"] == 15


def _messages_stream(
    start: dict[str, int], delta: dict[str, int]
) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [],
                "usage": start,
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": delta},
        {"type": "message_stop"},
    ]


class TestMessagesIsACumulativeWire:
    """Anthropic's ``message_delta`` repeats the input beside the output so far."""

    def test_the_live_shape_counts_the_input_once(self):
        # Measured live: start input 24 / output 1; delta input 24 / output 83.
        events = _usage_events(
            MessagesConverter(),
            _messages_stream(
                {"input_tokens": 24, "output_tokens": 1},
                {"input_tokens": 24, "cache_read_input_tokens": 0, "output_tokens": 83},
            ),
        )
        total = _sum(events)
        assert (
            total["prompt_tokens"],
            total["completion_tokens"],
            total["total_tokens"],
        ) == (24, 83, 107)

    def test_a_delta_that_carries_only_the_output_still_adds_to_the_start(self):
        events = _usage_events(
            MessagesConverter(),
            _messages_stream(
                {"input_tokens": 12, "output_tokens": 1}, {"output_tokens": 30}
            ),
        )
        total = _sum(events)
        assert (
            total["prompt_tokens"],
            total["completion_tokens"],
            total["total_tokens"],
        ) == (12, 30, 42)

    def test_cache_read_in_both_events_is_counted_once(self):
        start = {"input_tokens": 100, "cache_read_input_tokens": 40, "output_tokens": 1}
        delta = {"input_tokens": 100, "cache_read_input_tokens": 40, "output_tokens": 2}
        total = _sum(_usage_events(MessagesConverter(), _messages_stream(start, delta)))
        assert total["total_tokens"] == 142  # 100 + 40 + 2
        assert total["cache_read_tokens"] == 40
        record = _gateway_record(
            MessagesConverter(),
            ChatCompletionsConverter(),
            _messages_stream(start, delta),
        )
        assert record is not None
        assert record["total_tokens"] == 142


def test_the_interactions_writer_adds_usage_events():
    """A client of Google Interactions gets the sum, not the last event (#1750)."""
    from llm_rosetta.converters.llm.google_interactions import (
        GoogleInteractionsConverter,
    )

    writer = GoogleInteractionsConverter()
    ctx = writer.create_stream_context()
    for usage in (
        {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
        {"prompt_tokens": 0, "completion_tokens": 107, "total_tokens": 107},
    ):
        event: Any = {"type": "usage", "usage": usage}
        writer.stream_response_to_provider(event, context=ctx)
    assert ctx.metadata["_interactions_usage"] == {
        "total_input_tokens": 15,
        "total_output_tokens": 110,
        "total_tokens": 125,
    }
