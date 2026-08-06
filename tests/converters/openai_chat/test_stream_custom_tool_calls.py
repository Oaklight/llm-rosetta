"""Streaming tool-call deltas with present-but-null union members.

Providers serialise every member of the Chat Completions tool-call union on
each streaming delta and set the inactive ones to ``null`` — including
``type`` on every delta after the first. ``dict.get(key, default)`` returns
``None`` rather than the default for a present-but-null key, which made the
converter dereference ``None`` and abort the stream.

The deltas below are shaped exactly as captured from a live upstream
returning a custom (``apply_patch``) tool call.
"""

from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.openai_chat import OpenAIChatConverter

PATCH_PIECES = ("*** Begin Patch\n", "*** Add File: a.txt\n", "+hi\n", "*** End Patch")
PATCH_TEXT = "".join(PATCH_PIECES)

FIRST_CUSTOM_DELTA = {
    "index": 0,
    "id": "call_abc123",
    "function": None,
    "type": "custom",
    "custom": {"input": "", "name": "apply_patch"},
}
CONTINUATION_CUSTOM_DELTAS = [
    {"index": 0, "id": None, "function": None, "type": None, "custom": {"input": p}}
    for p in PATCH_PIECES
]


def _chunk(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Wrap a single tool-call delta in a Chat Completions stream chunk."""
    return {
        "choices": [
            {"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}
        ]
    }


class TestStreamingCustomToolDeltas:
    """Continuation deltas must survive null union members."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    def test_continuation_delta_with_null_function_does_not_raise(self):
        """A null ``function`` key must not be dereferenced."""
        context = StreamContext()
        self.converter.stream_response_from_provider(
            _chunk(FIRST_CUSTOM_DELTA), context
        )

        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(
                _chunk(CONTINUATION_CUSTOM_DELTAS[0]), context
            ),
        )
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"

    def test_custom_tool_input_is_fully_reassembled(self):
        """Every continuation delta contributes to the accumulated input.

        A bare null-guard would leave ``type`` unresolved, classify the
        continuation deltas as function calls, read an empty ``arguments``
        and silently drop the payload — so assert on the reassembled text,
        not merely on the absence of an exception.
        """
        context = StreamContext()
        events: list[Any] = []
        for delta in [FIRST_CUSTOM_DELTA, *CONTINUATION_CUSTOM_DELTAS]:
            events.extend(
                self.converter.stream_response_from_provider(_chunk(delta), context)
            )

        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert len(starts) == 1
        assert starts[0]["tool_name"] == "apply_patch"
        assert starts[0]["tool_type"] == "custom"

        deltas = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(deltas) == len(CONTINUATION_CUSTOM_DELTAS)
        assert "".join(e["arguments_delta"] for e in deltas) == PATCH_TEXT
        assert context.get_tool_call_args("call_abc123") == PATCH_TEXT

    def test_tool_type_recovered_from_context_when_payload_is_empty(self):
        """A delta carrying neither payload falls back to the registered type."""
        context = StreamContext()
        self.converter.stream_response_from_provider(
            _chunk(FIRST_CUSTOM_DELTA), context
        )

        empty = {"index": 0, "id": None, "function": None, "type": None, "custom": {}}
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(_chunk(empty), context),
        )
        assert events == []
        assert context.get_tool_type("call_abc123") == "custom"

    def test_function_call_deltas_still_work(self):
        """The ordinary function-call path is unaffected."""
        context = StreamContext()
        deltas = [
            {
                "index": 0,
                "id": "call_fn1",
                "type": "function",
                "custom": None,
                "function": {"name": "get_weather", "arguments": ""},
            },
            {
                "index": 0,
                "id": None,
                "type": None,
                "custom": None,
                "function": {"arguments": '{"city":"Chicago"}'},
            },
        ]
        events: list[Any] = []
        for delta in deltas:
            events.extend(
                self.converter.stream_response_from_provider(_chunk(delta), context)
            )

        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert starts[0]["tool_name"] == "get_weather"
        assert starts[0].get("tool_type", "function") == "function"
        assert context.get_tool_call_args("call_fn1") == '{"city":"Chicago"}'

    def test_type_omitted_entirely_still_defaults_to_function(self):
        """Absent ``type``/``custom`` keys keep the legacy function default."""
        context = StreamContext()
        chunk = _chunk(
            {
                "index": 0,
                "id": "call_fn2",
                "function": {"name": "ping", "arguments": "{}"},
            }
        )
        events = cast(
            list[Any], self.converter.stream_response_from_provider(chunk, context)
        )
        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert starts[0]["tool_name"] == "ping"
        assert context.get_tool_type("call_fn2") == "function"

    def test_parallel_custom_and_function_calls_do_not_cross_contaminate(self):
        """Two concurrent calls of different types stay correctly typed."""
        context = StreamContext()
        opening = [
            FIRST_CUSTOM_DELTA,
            {
                "index": 1,
                "id": "call_fn3",
                "type": "function",
                "custom": None,
                "function": {"name": "get_weather", "arguments": ""},
            },
        ]
        continuing = [
            {
                "index": 1,
                "id": None,
                "type": None,
                "custom": None,
                "function": {"arguments": '{"city":"Oak"}'},
            },
            {
                "index": 0,
                "id": None,
                "type": None,
                "function": None,
                "custom": {"input": PATCH_TEXT},
            },
        ]
        for delta in [*opening, *continuing]:
            self.converter.stream_response_from_provider(_chunk(delta), context)

        assert context.get_tool_type("call_abc123") == "custom"
        assert context.get_tool_type("call_fn3") == "function"
        assert context.get_tool_call_args("call_abc123") == PATCH_TEXT
        assert context.get_tool_call_args("call_fn3") == '{"city":"Oak"}'
