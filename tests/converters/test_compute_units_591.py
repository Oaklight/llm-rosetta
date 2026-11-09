"""`compute_units` survives the IR in both OpenAI dialects (#591).

OpenAI added `compute_units` (`integer | null`, minimum 0, currently always
``null``) to the usage object of Chat Completions and Responses on 2025-08-27.
The converters read only the fields they knew, so the value was dropped in both
directions.  The field is nullable on the wire, so "absent" and "present but
null" are different facts and each is checked here.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.converters.llm.open_responses.stream_context import (
    OpenResponsesStreamContext,
)
from llm_rosetta.types.ir import (
    FinishEvent,
    IRResponse,
    StreamEndEvent,
    UsageEvent,
    UsageInfo,
)

# `None` here means "the key is present and null"; the absent case is separate.
CASES = [pytest.param(7, id="value"), pytest.param(None, id="null")]


def _chat_body(**usage_extra: object) -> dict:
    return {
        "id": "chatcmpl-591",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            **usage_extra,
        },
    }


def _responses_body(**usage_extra: object) -> dict:
    return {
        "id": "resp_591",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_591",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            **usage_extra,
        },
    }


def _ir_usage(**extra: object) -> UsageInfo:
    return cast(
        UsageInfo,
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            **extra,
        },
    )


class TestChatCompletions:
    def setup_method(self):
        self.converter = ChatCompletionsConverter()

    @pytest.mark.parametrize("value", CASES)
    def test_reads_the_field(self, value):
        ir = self.converter.response_from_provider(_chat_body(compute_units=value))
        assert "compute_units" in ir["usage"]
        assert ir["usage"]["compute_units"] == value

    def test_absent_stays_absent(self):
        ir = self.converter.response_from_provider(_chat_body())
        assert "compute_units" not in ir["usage"]

    @pytest.mark.parametrize("value", CASES)
    def test_writes_the_field(self, value):
        ir = cast(
            IRResponse,
            {
                "id": "chatcmpl-591",
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": {"reason": "stop"},
                    }
                ],
                "usage": _ir_usage(compute_units=value),
            },
        )
        body = self.converter.response_to_provider(ir)
        assert "compute_units" in body["usage"]
        assert body["usage"]["compute_units"] == value

    def test_absent_is_not_written(self):
        ir = cast(
            IRResponse,
            {
                "id": "chatcmpl-591",
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": {"reason": "stop"},
                    }
                ],
                "usage": _ir_usage(),
            },
        )
        body = self.converter.response_to_provider(ir)
        assert "compute_units" not in body["usage"]


class TestOpenResponses:
    def setup_method(self):
        self.converter = OpenResponsesConverter()

    @pytest.mark.parametrize("value", CASES)
    def test_reads_the_field(self, value):
        ir = self.converter.response_from_provider(_responses_body(compute_units=value))
        assert "compute_units" in ir["usage"]
        assert ir["usage"]["compute_units"] == value

    def test_absent_stays_absent(self):
        ir = self.converter.response_from_provider(_responses_body())
        assert "compute_units" not in ir["usage"]

    @pytest.mark.parametrize("value", CASES)
    def test_writes_the_field(self, value):
        ir = self.converter.response_from_provider(_responses_body(compute_units=value))
        body = self.converter.response_to_provider(ir)
        assert "compute_units" in body["usage"]
        assert body["usage"]["compute_units"] == value

    def test_absent_is_not_written(self):
        ir = self.converter.response_from_provider(_responses_body())
        body = self.converter.response_to_provider(ir)
        assert "compute_units" not in body["usage"]

    @pytest.mark.parametrize("value", CASES)
    def test_the_context_free_stream_terminal_keeps_it(self, value):
        event = UsageEvent(type="usage", usage=_ir_usage(compute_units=value))
        chunk = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=None),
        )
        assert "compute_units" in chunk["response"]["usage"]
        assert chunk["response"]["usage"]["compute_units"] == value

    def test_the_context_free_stream_terminal_omits_it_when_absent(self):
        event = UsageEvent(type="usage", usage=_ir_usage())
        chunk = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=None),
        )
        assert "compute_units" not in chunk["response"]["usage"]


class TestCrossFormat:
    """Chat body → IR → Responses body → IR → Chat body."""

    def setup_method(self):
        self.chat = ChatCompletionsConverter()
        self.responses = OpenResponsesConverter()

    @pytest.mark.parametrize("value", CASES)
    def test_chat_to_responses_and_back(self, value):
        ir1 = self.chat.response_from_provider(_chat_body(compute_units=value))
        assert ir1["usage"]["compute_units"] == value

        responses_body = self.responses.response_to_provider(ir1)
        assert responses_body["usage"]["compute_units"] == value

        ir2 = self.responses.response_from_provider(responses_body)
        assert ir2["usage"]["compute_units"] == value

        chat_body = self.chat.response_to_provider(ir2)
        assert chat_body["usage"]["compute_units"] == value

    def test_absent_survives_the_cross_format_trip(self):
        ir1 = self.chat.response_from_provider(_chat_body())
        assert "compute_units" not in ir1["usage"]

        responses_body = self.responses.response_to_provider(ir1)
        assert "compute_units" not in responses_body["usage"]

        ir2 = self.responses.response_from_provider(responses_body)
        assert "compute_units" not in ir2["usage"]

        chat_body = self.chat.response_to_provider(ir2)
        assert "compute_units" not in chat_body["usage"]


def _stream_end() -> StreamEndEvent:
    return StreamEndEvent(type="stream_end")


def _finish() -> FinishEvent:
    return FinishEvent(type="finish", finish_reason={"reason": "stop"})


class TestChatStreamTerminal:
    """The chat usage chunk is written from the buffered usage at stream end."""

    def setup_method(self):
        self.converter = ChatCompletionsConverter()

    @pytest.mark.parametrize("value", CASES)
    def test_the_usage_chunk_keeps_it(self, value):
        context = StreamContext()
        self.converter.stream_response_to_provider(
            UsageEvent(type="usage", usage=_ir_usage(compute_units=value)), context
        )
        chunk = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(_stream_end(), context),
        )
        assert chunk["usage"]["compute_units"] == value

    def test_the_usage_chunk_omits_it_when_absent(self):
        context = StreamContext()
        self.converter.stream_response_to_provider(
            UsageEvent(type="usage", usage=_ir_usage()), context
        )
        chunk = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(_stream_end(), context),
        )
        assert "compute_units" not in chunk["usage"]


class TestOpenResponsesStreamTerminal:
    """With a context the body is emitted at stream end, from the finish payload."""

    def setup_method(self):
        self.converter = OpenResponsesConverter()

    def _completed(self, events):
        """Drive one context through the events and return the completed payload."""
        context = OpenResponsesStreamContext()
        completed: dict[str, Any] = {}
        for event in events:
            result = self.converter.stream_response_to_provider(event, context)
            for chunk in result if isinstance(result, list) else [result]:
                if (
                    isinstance(chunk, dict)
                    and chunk.get("type") == "response.completed"
                ):
                    completed = cast(dict[str, Any], chunk["response"])
        assert completed, "the stream never emitted response.completed"
        return completed

    @pytest.mark.parametrize("value", CASES)
    def test_usage_before_finish_keeps_it(self, value):
        completed = self._completed(
            [
                UsageEvent(type="usage", usage=_ir_usage(compute_units=value)),
                _finish(),
                _stream_end(),
            ]
        )
        assert completed["usage"]["compute_units"] == value

    def test_usage_before_finish_omits_it_when_absent(self):
        completed = self._completed(
            [UsageEvent(type="usage", usage=_ir_usage()), _finish(), _stream_end()]
        )
        assert "compute_units" not in completed["usage"]

    @pytest.mark.parametrize("value", CASES)
    def test_usage_after_finish_keeps_it(self, value):
        """Usage arriving after FinishEvent is merged in at stream end."""
        completed = self._completed(
            [
                _finish(),
                UsageEvent(type="usage", usage=_ir_usage(compute_units=value)),
                _stream_end(),
            ]
        )
        assert completed["usage"]["compute_units"] == value

    def test_usage_after_finish_omits_it_when_absent(self):
        """The merge must not invent a null when the usage never had the key."""
        completed = self._completed(
            [_finish(), UsageEvent(type="usage", usage=_ir_usage()), _stream_end()]
        )
        assert "compute_units" not in completed["usage"]


class TestDialectsWithoutTheField:
    """A dialect with no place for the field drops it, rather than writing null."""

    @pytest.fixture(scope="class")
    def ir(self) -> IRResponse:
        return ChatCompletionsConverter().response_from_provider(
            _chat_body(compute_units=7)
        )

    @pytest.mark.parametrize(
        ("converter", "usage_key"),
        [
            pytest.param(MessagesConverter(), "usage", id="messages"),
            pytest.param(
                GoogleGenerateConverter(), "usageMetadata", id="google_generate"
            ),
            pytest.param(
                GoogleInteractionsConverter(), "usage", id="google_interactions"
            ),
        ],
    )
    def test_the_key_is_absent_not_null(self, ir, converter, usage_key):
        body = converter.response_to_provider(ir)
        assert "compute_units" not in body[usage_key]


class TestTheBaseUsageMapping:
    """The required base-contract mapping carries the field like the live ones.

    ``_build_ir_usage_to_p`` writes the same usage shape ``_build_finish_usage``
    does but has no call site in this dialect; the docstring says a caller may
    reach it directly, so two builders of one shape must not disagree about a
    field (#1446 review).
    """

    @pytest.mark.parametrize("value", CASES)
    def test_keeps_the_field(self, value):
        usage = OpenResponsesConverter._build_ir_usage_to_p(
            _ir_usage(compute_units=value)
        )
        assert "compute_units" in usage
        assert usage["compute_units"] == value

    def test_omits_it_when_absent(self):
        usage = OpenResponsesConverter._build_ir_usage_to_p(_ir_usage())
        assert "compute_units" not in usage


class TestCrossFormatReverse:
    """Responses body → IR → Chat body → IR → Responses body."""

    def setup_method(self):
        self.chat = ChatCompletionsConverter()
        self.responses = OpenResponsesConverter()

    @pytest.mark.parametrize("value", CASES)
    def test_responses_to_chat_and_back(self, value):
        ir1 = self.responses.response_from_provider(
            _responses_body(compute_units=value)
        )
        assert ir1["usage"]["compute_units"] == value

        chat_body = self.chat.response_to_provider(ir1)
        assert chat_body["usage"]["compute_units"] == value

        ir2 = self.chat.response_from_provider(chat_body)
        assert ir2["usage"]["compute_units"] == value

        responses_body = self.responses.response_to_provider(ir2)
        assert responses_body["usage"]["compute_units"] == value

    def test_absent_survives_the_reverse_trip(self):
        ir1 = self.responses.response_from_provider(_responses_body())
        assert "compute_units" not in ir1["usage"]

        chat_body = self.chat.response_to_provider(ir1)
        assert "compute_units" not in chat_body["usage"]

        ir2 = self.chat.response_from_provider(chat_body)
        assert "compute_units" not in ir2["usage"]

        responses_body = self.responses.response_to_provider(ir2)
        assert "compute_units" not in responses_body["usage"]


def _chat_stream_chunk(**usage_extra: object) -> dict:
    """A final Chat SSE chunk that carries usage."""
    return {
        "id": "chatcmpl-591-stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            **usage_extra,
        },
    }


def _responses_completed(**usage_extra: object) -> dict:
    """The Responses terminal event that carries usage."""
    body = _responses_body(**usage_extra)
    return {"type": "response.completed", "response": body}


def _usage_events(events: list[Any]) -> list[Any]:
    return [event for event in events if event.get("type") == "usage"]


class TestStreamReadsTheField:
    """The read half: an upstream chunk carrying the field reaches IR usage."""

    def setup_method(self):
        self.chat = ChatCompletionsConverter()
        self.responses = OpenResponsesConverter()

    def test_the_chat_stream_reads_the_value(self):
        events = self.chat.stream_response_from_provider(
            _chat_stream_chunk(compute_units=7)
        )
        usages = _usage_events(events)
        assert usages, "the chunk produced no UsageEvent"
        assert usages[0]["usage"]["compute_units"] == 7

    def test_the_responses_stream_reads_the_null(self):
        events = self.responses.stream_response_from_provider(
            _responses_completed(compute_units=None)
        )
        usages = _usage_events(events)
        assert usages, "the event produced no UsageEvent"
        assert "compute_units" in usages[0]["usage"]
        assert usages[0]["usage"]["compute_units"] is None

    def test_the_chat_stream_omits_it_when_absent(self):
        events = self.chat.stream_response_from_provider(_chat_stream_chunk())
        usages = _usage_events(events)
        assert usages
        assert "compute_units" not in usages[0]["usage"]
