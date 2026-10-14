"""End-to-end google_generate intrinsic (server-tool) path (#986).

Google runs server tools itself: ``code_execution`` shows up as an
``executableCode`` part plus a ``codeExecutionResult`` part inside the model
turn, and ``google_search`` shows up only as candidate ``groundingMetadata``
(out of scope, #984).  Before this, the converter read/wrote the whole path as
if the call were a client ``functionCall``, raised on a code-execution turn,
and dropped every server part on the streaming side.

These tests cover the four cells the issue measured:

- non-streaming read: a code-execution turn reads into a validated IR response
  without raising, keeping the parts in order;
- non-streaming write: an intrinsic call writes back as ``executableCode`` /
  ``codeExecutionResult``, never a client ``functionCall``;
- streaming read: ``executableCode`` becomes the intrinsic call event and
  ``codeExecutionResult`` the one-shot result event;
- streaming write: Google's own part types are re-emitted, and a server call
  never comes out as a client call.

The shape follows ``tests/converters/messages/test_stream_server_tool_result.py``.
"""

import json
from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenResponsesConverter
from llm_rosetta.types.ir.response import IRResponse
from llm_rosetta.types.ir.stream import (
    FinishEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from llm_rosetta.types.ir.validation import validate_ir_response

_CODE_CHUNK: dict[str, Any] = {
    "candidates": [
        {
            "index": 0,
            "content": {
                "role": "model",
                "parts": [
                    {
                        "executableCode": {
                            "language": "PYTHON",
                            "code": "print(17 * 23 + 5)",
                        }
                    }
                ],
            },
        }
    ]
}
_RESULT_CHUNK: dict[str, Any] = {
    "candidates": [
        {
            "index": 0,
            "content": {
                "role": "model",
                "parts": [
                    {
                        "codeExecutionResult": {
                            "outcome": "OUTCOME_OK",
                            "output": "396\n",
                        }
                    }
                ],
            },
        }
    ]
}
_TEXT_CHUNK: dict[str, Any] = {
    "candidates": [
        {"index": 0, "content": {"role": "model", "parts": [{"text": "The computed"}]}}
    ]
}
_FINISH_CHUNK: dict[str, Any] = {
    "candidates": [
        {
            "index": 0,
            "content": {
                "role": "model",
                "parts": [{"text": " number is 396."}],
            },
            "finishReason": "STOP",
        }
    ]
}

# A captured non-streaming code-execution turn.
_RAW_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "index": 0,
            "content": {
                "role": "model",
                "parts": [
                    {
                        "executableCode": {
                            "code": "print(17 * 23 + 5)",
                            "language": "PYTHON",
                        }
                    },
                    {
                        "codeExecutionResult": {
                            "output": "396\n",
                            "outcome": "OUTCOME_OK",
                        }
                    },
                    {"text": "396"},
                ],
            },
            "finishReason": "STOP",
        }
    ]
}


def _ir_message(
    parts: list[dict[str, Any]],
) -> IRResponse:
    """Wrap content parts in a minimal response-shaped IR."""
    return cast(
        IRResponse,
        {
            "id": "x",
            "object": "response",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": parts},
                    "finish_reason": {"reason": "stop"},
                }
            ],
        },
    )


def _code_execution_parts() -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "code_execution",
            "tool_input": {"code": "print(1)", "language": "PYTHON"},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        },
        {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "1\n",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        },
        {"type": "text", "text": "done"},
    ]


def _chunks(result: Any) -> list[dict[str, Any]]:
    """Normalize a writer result (dict or list) to a list of chunks."""
    return cast(list[dict[str, Any]], result if isinstance(result, list) else [result])


def _emitted_parts(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for chunk in chunks:
        for candidate in chunk.get("candidates", []):
            parts.extend(candidate.get("content", {}).get("parts", []))
    return parts


class TestNonStreamingRead:
    """A model turn keeps the parts it owns instead of splitting into a list."""

    def test_code_execution_turn_reads_without_raising(self):
        ir = GoogleGenerateConverter().response_from_provider(dict(_RAW_RESPONSE))

        validate_ir_response(dict(ir))

    def test_parts_stay_inline_in_order(self):
        ir = GoogleGenerateConverter().response_from_provider(dict(_RAW_RESPONSE))
        message = cast(dict[str, Any], ir["choices"][0]["message"])
        content = cast(list[dict[str, Any]], message["content"])

        assert message["role"] == "assistant"
        assert [p["type"] for p in content] == ["tool_call", "tool_result", "text"]
        call, result, _text = content
        assert call["tool_type"] == "intrinsic"
        assert result["tool_type"] == "intrinsic"
        assert result["result"] == "396\n"
        # The server result answers the call that produced it.
        assert result["tool_call_id"] == call["tool_call_id"]


class TestNonStreamingWrite:
    """The response writer routes through the intrinsic-aware part writer."""

    def test_intrinsic_call_and_result_write_google_parts(self):
        out = GoogleGenerateConverter().response_to_provider(
            _ir_message(_code_execution_parts())
        )

        parts = out["candidates"][0]["content"]["parts"]
        assert [next(iter(p)) for p in parts] == [
            "executableCode",
            "codeExecutionResult",
            "text",
        ]
        assert parts[0]["executableCode"]["code"] == "print(1)"
        assert parts[1]["codeExecutionResult"]["outcome"] == "OUTCOME_OK"

    def test_server_call_is_never_a_client_function_call(self):
        out = GoogleGenerateConverter().response_to_provider(
            _ir_message(_code_execution_parts())
        )

        assert "functionCall" not in json.dumps(out)

    def test_failed_result_round_trips_as_failed(self):
        parts = _code_execution_parts()
        parts[1]["is_error"] = True
        out = GoogleGenerateConverter().response_to_provider(_ir_message(parts))

        result = out["candidates"][0]["content"]["parts"][1]
        assert result["codeExecutionResult"]["outcome"] == "OUTCOME_FAILED"

    def test_unrenderable_intrinsic_kind_is_dropped_not_client(self):
        """``google_search`` has no inline part; it must not become a call."""
        remote_parts: list[dict[str, Any]] = [
            {
                "type": "tool_call",
                "tool_call_id": "s1",
                "tool_name": "google_search",
                "tool_input": {"query": "q"},
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "google_search"},
            },
            {
                "type": "tool_result",
                "tool_call_id": "s1",
                "result": "Paris",
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "google_search"},
            },
            {"type": "text", "text": "Paris"},
        ]

        out = GoogleGenerateConverter().response_to_provider(_ir_message(remote_parts))

        parts = out["candidates"][0]["content"]["parts"]
        assert parts == [{"text": "Paris"}]

    def test_client_result_is_not_written_into_a_model_turn(self):
        """A client result belongs to a tool turn, not the model turn (#986)."""
        parts: list[dict[str, Any]] = [
            {
                "type": "tool_call",
                "tool_call_id": "fc_1",
                "tool_name": "get_weather",
                "tool_input": {"city": "Paris"},
            },
            {
                "type": "tool_result",
                "tool_call_id": "fc_1",
                "result": "sunny",
            },
            {"type": "text", "text": "It is sunny."},
        ]

        out = GoogleGenerateConverter().response_to_provider(_ir_message(parts))

        written = out["candidates"][0]["content"]["parts"]
        assert [next(iter(p)) for p in written] == ["functionCall", "text"]
        assert "functionResponse" not in json.dumps(out)


class TestStreamingRead:
    """``executableCode`` / ``codeExecutionResult`` reach the IR."""

    def setup_method(self):
        self.converter = GoogleGenerateConverter()
        self.context = StreamContext()

    def test_executable_code_becomes_intrinsic_call_event(self):
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(
                dict(_CODE_CHUNK), self.context
            ),
        )

        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["tool_type"] == "intrinsic"
        assert starts[0]["tool_name"] == "code_execution"
        assert get_intrinsic_kind(starts[0]) == "code_execution"
        assert "functionCall" not in json.dumps(events)

    def test_input_rides_a_delta_for_other_formats(self):
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(
                dict(_CODE_CHUNK), self.context
            ),
        )

        deltas = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(deltas) == 1
        assert json.loads(deltas[0]["arguments_delta"])["code"] == "print(17 * 23 + 5)"

    def test_code_execution_result_becomes_one_shot_event(self):
        self.converter.stream_response_from_provider(dict(_CODE_CHUNK), self.context)
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(
                dict(_RESULT_CHUNK), self.context
            ),
        )

        (event,) = [e for e in events if e["type"] == "tool_result"]
        event = cast(ToolResultEvent, event)
        assert event["result"] == "396\n"
        assert event["tool_type"] == "intrinsic"
        assert event["is_error"] is False
        assert get_intrinsic_kind(event) == "code_execution"
        # Paired with the call it answers, by kind (Google sends no id).
        assert event["tool_call_id"] == self.context.tool_call_ids[0]


class TestStreamingWrite:
    """Google's own part types are re-emitted; never a client ``functionCall``."""

    def _convert_all(self) -> list[dict[str, Any]]:
        converter = GoogleGenerateConverter()
        context = StreamContext()
        chunks: list[dict[str, Any]] = []
        for source in (_CODE_CHUNK, _RESULT_CHUNK, _TEXT_CHUNK, _FINISH_CHUNK):
            for event in cast(
                list[Any],
                converter.stream_response_from_provider(dict(source), context),
            ):
                chunks.extend(
                    _chunks(converter.stream_response_to_provider(event, context))
                )
        return chunks

    def test_writer_reemits_google_part_types(self):
        parts = _emitted_parts(self._convert_all())

        assert [next(iter(p)) for p in parts] == [
            "executableCode",
            "codeExecutionResult",
            "text",
            "text",
        ]
        assert "functionCall" not in json.dumps(parts)

    def test_same_dialect_round_trip_preserves_parts(self):
        """Google -> IR -> Google reproduces the raw part sequence."""
        expected = _emitted_parts([_CODE_CHUNK, _RESULT_CHUNK])

        assert _emitted_parts(self._convert_all())[:2] == expected

    def test_unrenderable_result_event_is_dropped(self):
        """A server result Google cannot render yields no chunk."""
        converter = GoogleGenerateConverter()
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="s1",
            result="Paris",
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "google_search"},
        )

        out = converter.stream_response_to_provider(event, StreamContext())

        assert out == {}

    def test_flush_drops_a_server_call_that_never_received_input(self):
        """No input arrived, so the finish flush writes no empty part (#986)."""
        converter = GoogleGenerateConverter()
        context = StreamContext()
        events: list[Any] = [
            ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id="c1",
                tool_name="code_execution",
                tool_call_index=0,
                tool_type="intrinsic",
                provider_metadata={"intrinsic_kind": "code_execution"},
            ),
            FinishEvent(type="finish", finish_reason={"reason": "stop"}),
        ]
        chunks: list[dict[str, Any]] = []
        for event in events:
            chunks.extend(
                _chunks(converter.stream_response_to_provider(event, context))
            )

        assert _emitted_parts(chunks) == []


class TestCrossFormat:
    """A server call never becomes a client call on a cross-format hop."""

    def test_into_google_streaming_uses_the_server_part(self):
        """An intrinsic call from another format is not a client call.

        It arrives without a Google-native part, so the writer rebuilds
        ``executableCode`` from the accumulated input at finish.
        """
        converter = GoogleGenerateConverter()
        context = StreamContext()
        chunks: list[dict[str, Any]] = []
        start = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="c1",
            tool_name="code_execution",
            tool_type="intrinsic",
        )
        start["provider_metadata"] = {"intrinsic_kind": "code_execution"}
        chunks.extend(_chunks(converter.stream_response_to_provider(start, context)))
        delta = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id="c1",
            arguments_delta=json.dumps({"code": "print(1)", "language": "PYTHON"}),
        )
        chunks.extend(_chunks(converter.stream_response_to_provider(delta, context)))
        finish = FinishEvent(type="finish", finish_reason={"reason": "stop"})
        chunks.extend(_chunks(converter.stream_response_to_provider(finish, context)))

        parts = _emitted_parts(chunks)
        assert parts == [{"executableCode": {"code": "print(1)", "language": "PYTHON"}}]
        assert "functionCall" not in json.dumps(parts)

    def test_to_messages_keeps_a_server_call(self):
        """Google -> IR -> Messages renders a server call, not a client one."""
        google = GoogleGenerateConverter()
        read_ctx = StreamContext()
        events: list[Any] = []
        for source in (_CODE_CHUNK, _RESULT_CHUNK):
            events.extend(
                cast(
                    list[Any],
                    google.stream_response_from_provider(dict(source), read_ctx),
                )
            )

        messages = MessagesConverter()
        write_ctx = StreamContext()
        blocks: list[dict[str, Any]] = []
        for event in events:
            result = messages.stream_response_to_provider(event, write_ctx)
            blocks.extend(result if isinstance(result, list) else [result])

        block_types = [
            cast(dict[str, Any], b["content_block"])["type"]
            for b in blocks
            if "content_block" in b
        ]
        assert "server_tool_use" in block_types
        assert "tool_use" not in block_types

    def test_to_open_responses_stays_server_typed(self):
        """The reader hands open_responses an intrinsic-typed call.

        Its output shape is that format's own path (#981, #985); what this
        pins is that google_generate never downgrades the call before it gets
        there.
        """
        google = GoogleGenerateConverter()
        context = StreamContext()
        events = cast(
            list[Any],
            google.stream_response_from_provider(dict(_CODE_CHUNK), context),
        )
        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert all(e["tool_type"] == "intrinsic" for e in starts)

        # open_responses consumes the events without raising.
        open_responses = OpenResponsesConverter()
        write_ctx = StreamContext()
        for event in events:
            open_responses.stream_response_to_provider(event, write_ctx)
