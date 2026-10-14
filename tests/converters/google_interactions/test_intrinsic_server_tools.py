"""Google Interactions intrinsic (server-tool) path through the IR (#992).

The Interactions API runs server tools (``google_search``,
``code_execution``) itself and returns their calls and results as native
steps in the turn.  Every direction of the conversion was broken: the
non-streaming reader dropped the ``role: "tool"`` message that holds the
result and lost the call's ``signature``; the non-streaming writer put a
``name`` on the server call (the API answers ``400 Unknown parameter
'name'``); the streaming reader emitted no intrinsic events; and the
streaming writer downgraded the call to a client ``function_call`` and
dropped the result.

Everything here mirrors the messages converter, which is the reference for
what intrinsic looks like on every path.  The kind is always folded from
the step type, never inferred from a step or tool name.
"""

import json
from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.types.ir.stream import ToolCallStartEvent, ToolResultEvent
from llm_rosetta.types.ir.validation import validate_ir_response

# ---------------------------------------------------------------------------
# Captured shapes (measured live against /v1beta/interactions, 2026-10-13)
# ---------------------------------------------------------------------------

_SEARCH_CALL: dict[str, Any] = {
    "id": "c1",
    "signature": "SIG-CALL",
    "type": "google_search_call",
    "arguments": {"queries": ["mars rover"]},
    "search_type": "web_search",
}

_SEARCH_RESULT: dict[str, Any] = {
    "call_id": "c1",
    "signature": "SIG-RESULT",
    "type": "google_search_result",
    "result": [{"rendered_content": "<div>x</div>"}],
}

_CODE_CALL: dict[str, Any] = {
    "id": "c2",
    "signature": "SIG-CODE",
    "type": "code_execution_call",
    "arguments": {"language": "python", "code": "print(42)"},
}

_CODE_RESULT: dict[str, Any] = {
    "call_id": "c2",
    "type": "code_execution_result",
    "result": "42\n",
    "is_error": False,
}

_SEARCH_STREAM: list[dict[str, Any]] = [
    {
        "event_type": "step.start",
        "index": 0,
        "step": {
            "id": "c1",
            "signature": "",
            "type": "google_search_call",
            "search_type": "web_search",
        },
    },
    {
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "signature": "SIG-CALL",
            "type": "google_search_call",
            "arguments": {"queries": ["mars rover"]},
        },
    },
    {"event_type": "step.stop", "index": 0},
    {
        "event_type": "step.start",
        "index": 1,
        "step": {"call_id": "c1", "signature": "", "type": "google_search_result"},
    },
    {
        "event_type": "step.delta",
        "index": 1,
        "delta": {
            "signature": "SIG-RESULT",
            "type": "google_search_result",
            "result": [{"rendered_content": "<div>x</div>"}],
        },
    },
    {"event_type": "step.stop", "index": 1},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _converter() -> GoogleInteractionsConverter:
    return GoogleInteractionsConverter()


def _read_response(steps: list[dict[str, Any]]) -> Any:
    return _converter().response_from_provider(
        cast(Any, {"id": "i", "status": "completed", "steps": steps})
    )


def _content(ir: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], ir["choices"][0]["message"]["content"])


def _read_events(
    chunks: list[dict[str, Any]], context: StreamContext | None = None
) -> list[dict[str, Any]]:
    converter = _converter()
    ctx = context if context is not None else StreamContext()
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        events.extend(
            cast(
                list[dict[str, Any]],
                converter.stream_response_from_provider(cast(Any, chunk), ctx),
            )
        )
    return events


def _write(event: Any, context: StreamContext | None = None) -> dict[str, Any]:
    out = _converter().stream_response_to_provider(
        event, context if context is not None else StreamContext()
    )
    chunks = out if isinstance(out, list) else [out]
    # Tool block openers return ``{}`` (the paired event emits the step).
    non_empty = [c for c in chunks if c]
    assert len(non_empty) == 1, non_empty
    return non_empty[0]


def _write_response(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ir = {
        "id": "i",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": {"reason": "tool_calls"},
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    return _converter().response_to_provider(cast(Any, ir))["steps"]


# ---------------------------------------------------------------------------
# Cell 1 — non-streaming read
# ---------------------------------------------------------------------------


class TestNonStreamingRead:
    """The server result is kept in the turn, and the signature comes along."""

    def test_server_result_is_kept_in_the_turn(self):
        ir = _read_response([_SEARCH_CALL, _SEARCH_RESULT, _text_step("hi")])

        assert [p["type"] for p in _content(ir)] == ["tool_call", "tool_result", "text"]

    def test_call_keeps_the_thought_signature(self):
        (call, _) = _content(_read_response([_SEARCH_CALL, _SEARCH_RESULT]))

        assert call["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(call) == "google_search"
        assert call["provider_metadata"]["google"]["thought_signature"] == "SIG-CALL"

    def test_call_identity_is_the_canonical_kind(self):
        (call, _) = _content(_read_response([_SEARCH_CALL, _SEARCH_RESULT]))

        assert call["tool_name"] == "google_search"

    def test_result_keeps_its_payload_shape(self):
        """A search result is an array on the wire and stays one in the IR."""
        (_, result) = _content(_read_response([_SEARCH_CALL, _SEARCH_RESULT]))

        assert result["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(result) == "google_search"
        assert result["result"] == [{"rendered_content": "<div>x</div>"}]

    def test_code_execution_result_keeps_its_string(self):
        (_, result) = _content(_read_response([_CODE_CALL, _CODE_RESULT]))

        assert result["result"] == "42\n"
        assert get_intrinsic_kind(result) == "code_execution"

    def test_kind_comes_from_the_type_not_a_name(self):
        step = dict(_SEARCH_CALL, name="totally_not_search")

        (call, _) = _content(_read_response([step, _SEARCH_RESULT]))

        assert get_intrinsic_kind(call) == "google_search"

    def test_the_read_validates(self):
        validate_ir_response(
            _read_response([_SEARCH_CALL, _SEARCH_RESULT, _text_step("hi")])
        )


def _text_step(text: str) -> dict[str, Any]:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Cell 2 — non-streaming write
# ---------------------------------------------------------------------------


class TestNonStreamingWrite:
    """A server call carries no ``name`` and gets its signature back."""

    def _call_part(self, **overrides: Any) -> dict[str, Any]:
        part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "google_search",
            "tool_input": {"queries": ["mars rover"]},
            "tool_type": "intrinsic",
            "provider_metadata": {
                "intrinsic_kind": "google_search",
                "google": {"thought_signature": "SIG-CALL"},
            },
        }
        part.update(overrides)
        return part

    def test_server_call_omits_name_and_restores_the_signature(self):
        (call,) = _write_response([self._call_part()])

        assert call["type"] == "google_search_call"
        assert "name" not in call
        assert call["signature"] == "SIG-CALL"
        assert call["arguments"] == {"queries": ["mars rover"]}

    def test_server_call_without_a_signature_omits_it(self):
        part = self._call_part(provider_metadata={"intrinsic_kind": "google_search"})

        (call,) = _write_response([part])

        assert call["type"] == "google_search_call"
        assert "signature" not in call
        assert "name" not in call

    def test_server_result_emits_the_native_result_step(self):
        result = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": [{"rendered_content": "<div>x</div>"}],
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "google_search"},
        }

        (step,) = _write_response([result])

        assert step["type"] == "google_search_result"
        assert step["call_id"] == "c1"
        assert step["result"] == [{"rendered_content": "<div>x</div>"}]
        assert "name" not in step

    def test_unknown_intrinsic_kind_is_dropped(self):
        """Never downgrade a server call to a client ``function_call``.

        A kind this dialect cannot render has no native step type
        (``special_search_call`` is not one the API defines), so the call is
        dropped rather than fabricated or downgraded — matching the streaming
        writer and the sibling converters.
        """
        part = self._call_part(
            tool_name="special_search",
            provider_metadata={"intrinsic_kind": "special_search"},
        )

        assert _write_response([part]) == []

    def test_client_call_still_gets_its_name(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "fc1",
            "tool_name": "get_weather",
            "tool_input": {"city": "SF"},
        }

        (call,) = _write_response([part])

        assert call["type"] == "function_call"
        assert call["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Cell 3 — streaming read
# ---------------------------------------------------------------------------


class TestStreamingRead:
    """The reader turns server steps into intrinsic IR events."""

    def test_call_step_emits_the_intrinsic_call_event(self):
        events = _read_events(_SEARCH_STREAM)
        call = next(e for e in events if e["type"] == "tool_call_start")

        assert call["tool_call_id"] == "c1"
        assert call["tool_name"] == "google_search"
        assert call["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(call) == "google_search"

    def test_call_step_opens_a_tool_block(self):
        events = _read_events(_SEARCH_STREAM)
        start = next(e for e in events if e["type"] == "content_block_start")

        assert start["block_type"] == "server_tool_use"

    def test_call_delta_serializes_the_arguments(self):
        events = _read_events(_SEARCH_STREAM)
        delta = next(e for e in events if e["type"] == "tool_call_delta")

        assert delta["tool_call_id"] == "c1"
        assert json.loads(delta["arguments_delta"]) == {"queries": ["mars rover"]}

    def test_result_delta_emits_the_result_event(self):
        events = _read_events(_SEARCH_STREAM)
        result = next(e for e in events if e["type"] == "tool_result")

        assert result["tool_call_id"] == "c1"
        assert result["result"] == [{"rendered_content": "<div>x</div>"}]
        assert result["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(result) == "google_search"
        assert result["block_index"] == 1

    def test_result_event_mirrors_the_non_streaming_part(self):
        part = _content(_read_response([_SEARCH_CALL, _SEARCH_RESULT]))[1]
        events = _read_events(_SEARCH_STREAM)
        result = next(e for e in events if e["type"] == "tool_result")

        assert result["tool_call_id"] == part["tool_call_id"]
        assert result["result"] == part["result"]
        assert get_intrinsic_kind(result) == get_intrinsic_kind(part)

    def test_no_events_without_a_context_needed(self):
        """The reader still emits both events when driven without a context."""
        events = _read_events(_SEARCH_STREAM, StreamContext())
        types = [e["type"] for e in events]

        assert "tool_call_start" in types
        assert "tool_result" in types

    def test_other_step_types_do_not_emit_intrinsic_events(self):
        events = _read_events(
            [
                {
                    "event_type": "step.start",
                    "index": 0,
                    "step": {"type": "model_output"},
                }
            ]
        )

        assert all(e["type"] not in ("tool_call_start", "tool_result") for e in events)


# ---------------------------------------------------------------------------
# Cell 4 — streaming write
# ---------------------------------------------------------------------------


class TestStreamingWrite:
    """The writer re-emits the provider's own server step types."""

    def _call_event(self, **overrides: Any) -> ToolCallStartEvent:
        event: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": "c1",
            "tool_name": "google_search",
            "tool_call_index": 0,
            "tool_type": "intrinsic",
            "provider_metadata": {
                "intrinsic_kind": "google_search",
                "google": {"thought_signature": "SIG-CALL"},
            },
        }
        event.update(overrides)
        return cast(ToolCallStartEvent, event)

    def test_intrinsic_call_start_emits_the_native_step(self):
        out = _write(self._call_event())

        assert out["event_type"] == "step.start"
        assert out["step"]["type"] == "google_search_call"
        assert out["step"]["id"] == "c1"
        assert "name" not in out["step"]

    def test_intrinsic_call_start_restores_the_signature(self):
        out = _write(self._call_event())

        assert out["step"]["signature"] == "SIG-CALL"

    def test_client_call_start_still_emits_function_call(self):
        out = _write(
            cast(
                ToolCallStartEvent,
                {
                    "type": "tool_call_start",
                    "tool_call_id": "fc1",
                    "tool_name": "get_weather",
                    "tool_call_index": 0,
                },
            )
        )

        assert out["step"]["type"] == "function_call"
        assert out["step"]["name"] == "get_weather"

    def test_unknown_kind_is_dropped(self):
        out = _converter().stream_response_to_provider(
            self._call_event(
                tool_name="special_search",
                provider_metadata={"intrinsic_kind": "special_search"},
            ),
            StreamContext(),
        )
        chunks = out if isinstance(out, list) else [out]

        assert not [chunk for chunk in chunks if chunk]

    def test_intrinsic_delta_reemits_arguments_as_a_dict(self):
        context = StreamContext()
        _write(self._call_event(), context)
        out = _write(
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"queries": ["mars rover"]}),
                "block_index": 0,
            },
            context,
        )

        assert out["delta"]["type"] == "google_search_call"
        assert out["delta"]["arguments"] == {"queries": ["mars rover"]}

    def test_tool_result_emits_the_native_result_step(self):
        out = _write(
            cast(
                ToolResultEvent,
                {
                    "type": "tool_result",
                    "tool_call_id": "c1",
                    "result": [{"rendered_content": "<div>x</div>"}],
                    "tool_type": "intrinsic",
                    "block_index": 1,
                    "provider_metadata": {"intrinsic_kind": "google_search"},
                },
            )
        )

        assert out["event_type"] == "step.start"
        assert out["step"]["type"] == "google_search_result"
        assert out["step"]["call_id"] == "c1"
        assert out["step"]["result"] == [{"rendered_content": "<div>x</div>"}]

    def test_unknown_result_kind_is_dropped(self):
        out = _converter().stream_response_to_provider(
            cast(
                ToolResultEvent,
                {
                    "type": "tool_result",
                    "tool_call_id": "c1",
                    "result": "x",
                    "tool_type": "intrinsic",
                    "block_index": 1,
                    "provider_metadata": {"intrinsic_kind": "special_search"},
                },
            ),
            StreamContext(),
        )
        chunks = out if isinstance(out, list) else [out]

        assert not [chunk for chunk in chunks if chunk]

    def test_tool_result_handler_is_registered(self):
        assert GoogleInteractionsConverter._IR_TO_P_DISPATCH["tool_result"] == (
            "_handle_ir_tool_result_to_p"
        )

    def test_base_dispatch_does_not_register_tool_result(self):
        """Only this converter's own dispatch adds ``tool_result`` (#992)."""
        from llm_rosetta.converters.base import BaseConverter

        assert "tool_result" not in BaseConverter._IR_TO_P_DISPATCH


# ---------------------------------------------------------------------------
# Same-dialect round trip
# ---------------------------------------------------------------------------


class TestSameFormatRoundTrip:
    """Google Interactions -> IR -> Google Interactions keeps every step."""

    def test_non_streaming_round_trip(self):
        written = _write_response(
            _content(_read_response([_SEARCH_CALL, _SEARCH_RESULT, _text_step("hi")]))
        )

        assert [s["type"] for s in written] == [
            "google_search_call",
            "google_search_result",
            "model_output",
        ]
        assert written[0]["signature"] == "SIG-CALL"
        assert "name" not in written[0]
        assert written[1]["result"] == [{"rendered_content": "<div>x</div>"}]

    def _stream_round_trip(self) -> list[dict[str, Any]]:
        converter = _converter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()
        out: list[dict[str, Any]] = []
        for chunk in _SEARCH_STREAM:
            for event in converter.stream_response_from_provider(chunk, from_ctx):
                chunks = converter.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    out.extend(c for c in chunks if c)
                elif chunks:
                    out.append(chunks)
        return out

    def test_streamed_call_reemits_the_native_step(self):
        starts = [
            c for c in self._stream_round_trip() if c.get("event_type") == "step.start"
        ]
        call_start = starts[0]

        assert call_start["step"]["type"] == "google_search_call"
        assert call_start["step"]["id"] == "c1"
        assert "name" not in call_start["step"]

    def test_streamed_arguments_survive(self):
        deltas = [
            c for c in self._stream_round_trip() if c.get("event_type") == "step.delta"
        ]
        call_delta = deltas[0]

        assert call_delta["delta"]["type"] == "google_search_call"
        assert call_delta["delta"]["arguments"] == {"queries": ["mars rover"]}

    def test_streamed_result_survives(self):
        steps = [
            c["step"]
            for c in self._stream_round_trip()
            if c.get("event_type") == "step.start"
        ]
        result = next(s for s in steps if s["type"] == "google_search_result")

        assert result["call_id"] == "c1"
        assert result["result"] == [{"rendered_content": "<div>x</div>"}]

    def test_streamed_blocks_do_not_leak_internal_metadata(self):
        wire = json.dumps(self._stream_round_trip())

        assert "provider_metadata" not in wire
        assert "intrinsic_kind" not in wire


# ---------------------------------------------------------------------------
# Cross-format -> messages
# ---------------------------------------------------------------------------


class TestCrossFormatToMessages:
    """A server call reaches Anthropic as ``server_tool_use``, never a client one."""

    def test_non_streaming_keeps_the_server_shapes(self):
        ir = _read_response([_SEARCH_CALL, _SEARCH_RESULT, _text_step("hi")])

        out = MessagesConverter().response_to_provider(ir)
        blocks = out["content"]

        assert blocks[0]["type"] == "server_tool_use"
        assert blocks[0]["name"] == "google_search"
        assert blocks[1]["type"] == "google_search_tool_result"
        assert blocks[1]["tool_use_id"] == "c1"
        assert "tool_use" not in [b["type"] for b in blocks]

    def test_streaming_keeps_the_server_shapes(self):
        source = _converter()
        target = MessagesConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()
        emitted: list[dict[str, Any]] = []
        for chunk in _SEARCH_STREAM:
            for event in source.stream_response_from_provider(chunk, from_ctx):
                chunks = target.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    emitted.extend(c for c in chunks if c)
                elif chunks:
                    emitted.append(chunks)

        starts = [
            e["content_block"]["type"]
            for e in emitted
            if e.get("type") == "content_block_start"
        ]
        assert "server_tool_use" in starts
        assert "google_search_tool_result" in starts
        assert "tool_use" not in starts

    def test_code_execution_also_crosses_as_a_server_result(self):
        ir = _read_response([_CODE_CALL, _CODE_RESULT])

        blocks = MessagesConverter().response_to_provider(ir)["content"]

        assert blocks[0]["type"] == "server_tool_use"
        assert blocks[0]["name"] == "code_execution"
        assert blocks[1]["type"] == "code_execution_tool_result"
