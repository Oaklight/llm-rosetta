"""Streamed Anthropic server-tool calls through the IR (#974).

Anthropic executes server tools (web search, code execution) and returns their
calls as ``server_tool_use`` blocks rather than the client-declared
``tool_use``.  The streaming reader used to emit the same ``ToolCallStartEvent``
for both without saying which it was, and the writer hardcoded ``tool_use``, so
a same-format round trip downgraded a server-executed call to a client one.

The reader now marks the event ``tool_type="intrinsic"`` and carries the
canonical kind in ``provider_metadata["intrinsic_kind"]``; the writer mirrors
``MessagesToolOps.ir_tool_call_to_p`` and re-emits ``server_tool_use`` with that
kind as the block name.  The kind is never inferred from the tool name.
"""

import json
from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.types.ir.stream import ToolCallStartEvent

_SERVER_CALL_BLOCK: dict[str, Any] = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {},
}

_CLIENT_CALL_BLOCK: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "get_weather",
    "input": {"city": "SF"},
}


def _call_start(block: dict[str, Any]) -> ToolCallStartEvent:
    """Feed a ``content_block_start`` through the reader, return the call event."""
    converter = MessagesConverter()
    events = converter.stream_response_from_provider(
        {"type": "content_block_start", "index": 1, "content_block": dict(block)}
    )
    return next(e for e in events if e["type"] == "tool_call_start")


class TestReader:
    """The reader says which kind of call the block was."""

    def test_server_tool_use_is_marked_intrinsic(self):
        event = _call_start(_SERVER_CALL_BLOCK)

        assert event["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(event) == "web_search"

    def test_tool_use_is_not_marked_intrinsic(self):
        event = _call_start(_CLIENT_CALL_BLOCK)

        assert event.get("tool_type", "function") == "function"

    def test_event_mirrors_the_non_streaming_part(self):
        """Same carrier as ``p_tool_call_to_ir``: tool_type + intrinsic_kind."""
        part = MessagesToolOps.p_tool_call_to_ir(dict(_SERVER_CALL_BLOCK))
        event = _call_start(_SERVER_CALL_BLOCK)

        assert event["tool_call_id"] == part["tool_call_id"]
        assert event["tool_name"] == part["tool_name"]
        assert event["tool_type"] == part["tool_type"]
        assert get_intrinsic_kind(event) == get_intrinsic_kind(part)

    def test_reader_carries_the_cache_breakpoint(self):
        """``ToolCallPart.cache_hint`` has no counterpart without this (#975 review)."""
        block = dict(_CLIENT_CALL_BLOCK, cache_control={"type": "ephemeral"})

        event = _call_start(block)

        assert event["cache_hint"] == {"type": "ephemeral"}

    def test_reader_carries_a_non_direct_caller(self):
        """Only a non-``direct`` caller is kept, as on the non-streaming path."""
        block = dict(
            _SERVER_CALL_BLOCK,
            caller={"type": "code_execution_20250825", "tool_id": "t"},
        )

        event = _call_start(block)

        assert event["provider_metadata"]["anthropic_caller"]["type"] == (
            "code_execution_20250825"
        )

    def test_reader_drops_the_direct_caller(self):
        block = dict(_SERVER_CALL_BLOCK, caller={"type": "direct"})

        event = _call_start(block)

        assert "anthropic_caller" not in event.get("provider_metadata", {})

    def test_client_call_keeps_its_name(self):
        event = _call_start(_CLIENT_CALL_BLOCK)

        assert event["tool_name"] == "get_weather"

    def test_server_alias_folds_to_canonical_kind(self):
        """A newer code-execution version reports ``bash_code_execution``."""
        block = {
            "type": "server_tool_use",
            "id": "srvtoolu_2",
            "name": "bash_code_execution",
            "input": {},
        }

        event = _call_start(block)

        assert event["tool_name"] == "code_execution"
        assert get_intrinsic_kind(event) == "code_execution"


class TestWriter:
    """The writer mirrors ``ir_tool_call_to_p`` on the call start event."""

    def setup_method(self):
        self.converter = MessagesConverter()

    def _write(self, event: ToolCallStartEvent) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, StreamContext()),
        )

    def _event(self, **overrides: Any) -> ToolCallStartEvent:
        base: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": "srvtoolu_1",
            "tool_name": "web_search",
        }
        base.update(overrides)
        return cast(ToolCallStartEvent, base)

    def test_intrinsic_event_emits_server_tool_use(self):
        out = self._write(
            self._event(
                tool_type="intrinsic",
                provider_metadata={"intrinsic_kind": "web_search"},
            )
        )

        assert out["type"] == "content_block_start"
        assert out["content_block"]["type"] == "server_tool_use"
        assert out["content_block"]["name"] == "web_search"
        assert out["content_block"]["input"] == {}

    def test_function_event_emits_tool_use(self):
        out = self._write(self._event(tool_type="function", tool_name="get_weather"))

        assert out["content_block"]["type"] == "tool_use"
        assert out["content_block"]["name"] == "get_weather"

    def test_context_tracks_the_block_type_it_opened(self):
        """Pins the tracked state rather than leaving it inert (#975 review).

        The other readers tolerate either spelling, so nothing else fails while
        the field keeps the generic ``"tool_use"`` — but the state should name
        the block that was actually opened, and this is what would notice if it
        stopped doing so.
        """
        server_ctx = StreamContext()
        self.converter.stream_response_to_provider(
            self._event(
                tool_type="intrinsic",
                provider_metadata={"intrinsic_kind": "web_search"},
            ),
            server_ctx,
        )
        assert server_ctx.current_block_type == "server_tool_use"

        client_ctx = StreamContext()
        self.converter.stream_response_to_provider(
            self._event(tool_type="function", tool_name="get_weather"), client_ctx
        )
        assert client_ctx.current_block_type == "tool_use"

    def test_writer_restores_the_cache_breakpoint(self):
        out = self._write(
            self._event(tool_type="function", cache_hint={"type": "ephemeral"})
        )

        assert out["content_block"]["cache_control"] == {"type": "ephemeral"}

    def test_writer_restores_a_non_direct_caller_on_a_client_call(self):
        out = self._write(
            self._event(
                tool_type="function",
                tool_name="get_weather",
                provider_metadata={"anthropic_caller": {"type": "code_execution"}},
            )
        )

        assert out["content_block"]["caller"] == {"type": "code_execution"}

    def test_writer_leaves_the_caller_off_a_server_call(self):
        """``ir_tool_call_to_p`` restores it only for ``tool_use``."""
        out = self._write(
            self._event(
                tool_type="intrinsic",
                provider_metadata={
                    "intrinsic_kind": "web_search",
                    "anthropic_caller": {"type": "code_execution"},
                },
            )
        )

        assert out["content_block"]["type"] == "server_tool_use"
        assert "caller" not in out["content_block"]

    def test_missing_tool_type_defaults_to_tool_use(self):
        out = self._write(self._event(tool_name="get_weather"))

        assert out["content_block"]["type"] == "tool_use"

    def test_kind_comes_from_metadata_not_the_tool_name(self):
        """``intrinsic_kind`` wins over a provider-specific spelling."""
        out = self._write(
            self._event(
                tool_name="renamed_server_tool",
                tool_type="intrinsic",
                provider_metadata={"intrinsic_kind": "web_search"},
            )
        )

        assert out["content_block"]["type"] == "server_tool_use"
        assert out["content_block"]["name"] == "web_search"

    def test_unknown_intrinsic_name_is_dropped_not_downgraded(self):
        """A server tool with no canonical kind is neither named nor degraded.

        The kind is read from ``intrinsic_kind`` only, never inferred from the
        tool name (phase 2 removed that fallback, #974), so a missing kind is
        unknown — and unknown is a kind Anthropic has no server tool for.  The
        response leg drops it rather than fabricating a ``server_tool_use`` or
        handing the client a ``tool_use`` it must run (phase 4).
        """
        out = self._write(
            self._event(tool_name="special_search", tool_type="intrinsic")
        )

        assert out == {}

    def test_internal_metadata_does_not_leak(self):
        out = self._write(
            self._event(
                tool_type="intrinsic",
                provider_metadata={"intrinsic_kind": "web_search"},
            )
        )

        wire = json.dumps(out)
        assert "provider_metadata" not in wire
        assert "intrinsic_kind" not in wire


class TestSameFormatRoundTrip:
    """Anthropic -> IR -> Anthropic keeps the call block's type and name."""

    # A captured server-tool call: start block, argument deltas, stop.
    CAPTURE: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": "x"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]

    CLIENT_CAPTURE: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city": "SF"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]

    def _round_trip(self, capture: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converter = MessagesConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()
        out: list[dict[str, Any]] = []
        for chunk in capture:
            for event in converter.stream_response_from_provider(chunk, from_ctx):
                chunks = converter.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    out.extend(c for c in chunks if c)
                elif chunks:
                    out.append(chunks)
        return out

    def test_server_call_reemits_as_server_tool_use(self):
        starts = [
            c
            for c in self._round_trip(self.CAPTURE)
            if c.get("type") == "content_block_start"
        ]

        assert len(starts) == 1
        assert starts[0]["content_block"]["type"] == "server_tool_use"
        assert starts[0]["content_block"]["name"] == "web_search"
        assert starts[0]["content_block"]["input"] == {}

    def test_server_call_input_survives_via_the_delta(self):
        deltas = [
            c
            for c in self._round_trip(self.CAPTURE)
            if c.get("type") == "content_block_delta"
        ]

        assert len(deltas) == 1
        assert deltas[0]["delta"]["type"] == "input_json_delta"
        assert deltas[0]["delta"]["partial_json"] == '{"query": "x"}'

    def test_client_call_still_reemits_as_tool_use(self):
        starts = [
            c
            for c in self._round_trip(self.CLIENT_CAPTURE)
            if c.get("type") == "content_block_start"
        ]

        assert len(starts) == 1
        assert starts[0]["content_block"]["type"] == "tool_use"
        assert starts[0]["content_block"]["name"] == "get_weather"


class TestCrossFormat:
    """A target that cannot render the intrinsic distinction still emits a call."""

    def test_chat_completions_still_emits_a_tool_call(self):
        source = MessagesConverter()
        target = ChatCompletionsConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()

        emitted: list[dict[str, Any]] = []
        for chunk in TestSameFormatRoundTrip.CAPTURE:
            for event in source.stream_response_from_provider(chunk, from_ctx):
                chunks = target.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    emitted.extend(c for c in chunks if c)
                elif chunks:
                    emitted.append(chunks)

        # The call is not dropped; chat_completions has its own tool-call shape.
        assert emitted
