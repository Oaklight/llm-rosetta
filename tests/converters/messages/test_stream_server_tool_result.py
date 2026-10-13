"""Streamed Anthropic server-tool results through the IR (#972).

Anthropic executes server tools (web search, code execution) and returns their
results inside the assistant turn as ``server_tool_use`` + ``*_tool_result``
blocks.  The result block arrives **whole** in a single ``content_block_start``
with no deltas to follow.  Before this, the streaming reader emitted only a
``ContentBlockStartEvent`` for such a block and dropped the payload, so the
result never reached the IR.

The reader now emits a one-shot ``ToolResultEvent``; the Anthropic writer
restores the block from the ``_native`` payload the reader stored, so a
same-format round trip is lossless.  Converters that cannot render the event
register no handler, so it degrades silently.
"""

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.types.ir.stream import ToolResultEvent

_WEB_RESULT_BLOCK: dict[str, Any] = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://example.com",
            "title": "Example",
            "encrypted_content": "enc",
            "page_age": "1 day ago",
        }
    ],
    "caller": {"type": "direct"},
}

_BASH_RESULT_BLOCK: dict[str, Any] = {
    "type": "bash_code_execution_tool_result",
    "tool_use_id": "srvtoolu_2",
    "content": {
        "type": "bash_code_execution_result",
        "stdout": "1\n",
        "stderr": "",
        "return_code": 0,
        "content": [],
    },
}


def _result_events(block: dict[str, Any]) -> list[ToolResultEvent]:
    """Feed a ``content_block_start`` through the reader and return IR events."""
    converter = MessagesConverter()
    return cast(
        list[ToolResultEvent],
        converter.stream_response_from_provider(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": dict(block),
            }
        ),
    )


class TestReader:
    """The Anthropic reader turns a ``*_tool_result`` block into a ToolResultEvent."""

    def test_web_search_result_becomes_one_event(self):
        events = _result_events(_WEB_RESULT_BLOCK)

        assert [e["type"] for e in events] == ["tool_result"]

    def test_event_mirrors_the_non_streaming_part(self):
        (event,) = _result_events(_WEB_RESULT_BLOCK)

        assert event["tool_call_id"] == "srvtoolu_1"
        assert event["result"] == _WEB_RESULT_BLOCK["content"]
        assert event["tool_type"] == "intrinsic"
        assert event["is_error"] is False
        assert event["block_index"] == 1
        assert get_intrinsic_kind(event) == "web_search"

    def test_kind_matches_the_non_streaming_reader(self):
        """Same payload shape as ``p_server_tool_result_to_ir``."""
        part = MessagesToolOps.p_server_tool_result_to_ir(dict(_WEB_RESULT_BLOCK))
        (event,) = _result_events(_WEB_RESULT_BLOCK)

        assert event["tool_call_id"] == part["tool_call_id"]
        assert event["result"] == part["result"]
        assert event["tool_type"] == part["tool_type"]
        assert get_intrinsic_kind(event) == get_intrinsic_kind(part)

    def test_code_execution_result_keeps_the_string_shape(self):
        """A dict-content server result reads back as a JSON string, as the
        non-streaming part does, so ``result`` is not always a list."""
        (event,) = _result_events(_BASH_RESULT_BLOCK)

        assert isinstance(event["result"], str)
        assert json.loads(event["result"])["stdout"] == "1\n"

    @pytest.mark.parametrize(
        "block", [_WEB_RESULT_BLOCK, _BASH_RESULT_BLOCK], ids=["web", "bash"]
    )
    def test_result_alias_folds_to_canonical_kind(self, block):
        (event,) = _result_events(block)

        assert event["type"] == "tool_result"

    def test_bash_alias_folds_to_code_execution(self):
        (event,) = _result_events(_BASH_RESULT_BLOCK)

        assert get_intrinsic_kind(event) == "code_execution"

    def test_other_block_types_do_not_emit_the_event(self):
        events = _result_events({"type": "text", "text": "hi"})

        assert all(e["type"] != "tool_result" for e in events)

    def test_event_is_reexported_from_types_ir(self):
        from llm_rosetta.types.ir import ToolResultEvent as ReExported

        assert ReExported is ToolResultEvent


class TestWriter:
    """The Anthropic writer restores the ``*_tool_result`` block."""

    def setup_method(self):
        self.converter = MessagesConverter()

    def test_writer_replays_the_native_block(self):
        (event,) = _result_events(_WEB_RESULT_BLOCK)

        out = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, StreamContext()),
        )

        assert out["type"] == "content_block_start"
        assert out["index"] == 1
        assert out["content_block"] == _WEB_RESULT_BLOCK

    def test_writer_replays_bash_alias(self):
        (event,) = _result_events(_BASH_RESULT_BLOCK)

        out = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, StreamContext()),
        )

        assert out["content_block"] == _BASH_RESULT_BLOCK

    def test_foreign_native_payload_is_not_replayed(self):
        """A ``_native`` blob from another dialect must not reach the wire.

        The ``base`` check on the stored payload is load-bearing but was
        unpinned: without it the foreign block is copied onto an Anthropic
        payload verbatim, which is the failure the namespace exists to prevent
        (#973 review).
        """
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="s1",
            result=[{"type": "text", "text": "x"}],
            block_index=0,
            provider_metadata={
                "intrinsic_kind": "web_search",
                "_native": {
                    "base": "open_responses",
                    "block": {
                        "type": "web_search_call",
                        "action": {"type": "search", "query": "q"},
                    },
                },
            },
        )

        out = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, StreamContext()),
        )

        assert out["content_block"]["type"] == "web_search_tool_result"
        assert "action" not in out["content_block"]

    def test_fallback_rebuilds_from_ir_fields(self):
        """Without a native payload the block is rebuilt from the IR event."""
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="s1",
            result=[{"type": "text", "text": "x"}],
            is_error=True,
            block_index=3,
            provider_metadata={"intrinsic_kind": "web_search"},
        )

        out = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, StreamContext()),
        )

        assert out["content_block"]["type"] == "web_search_tool_result"
        assert out["content_block"]["tool_use_id"] == "s1"
        # Neither field belongs on a rebuilt server result: Anthropic answers
        # ``400 ... .web_search_tool_result.is_error: Extra inputs are not
        # permitted`` (verified live 2026-10-13), and nothing populates
        # ``anthropic_caller`` — a server result's ``caller`` only survives
        # through the ``_native`` payload.
        assert "is_error" not in out["content_block"]
        assert "caller" not in out["content_block"]
        assert out["index"] == 3


class TestSameFormatRoundTrip:
    """Anthropic -> IR -> Anthropic keeps the result block in place."""

    # The captured assistant turn: call block (index 0), result block (index 1),
    # then text (index 2).
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
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": dict(_WEB_RESULT_BLOCK),
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "~400k"},
        },
        {"type": "content_block_stop", "index": 2},
    ]

    def _round_trip(self) -> list[dict[str, Any]]:
        converter = MessagesConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()
        out: list[dict[str, Any]] = []
        for chunk in self.CAPTURE:
            for event in converter.stream_response_from_provider(chunk, from_ctx):
                chunks = converter.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    out.extend(c for c in chunks if c)
                elif chunks:
                    out.append(chunks)
        return out

    def test_result_block_survives_byte_for_byte(self):
        results = [
            c
            for c in self._round_trip()
            if c.get("content_block", {}).get("type") == "web_search_tool_result"
        ]

        assert len(results) == 1
        assert results[0]["content_block"] == _WEB_RESULT_BLOCK

    def test_result_block_stays_between_the_call_and_the_text(self):
        out = self._round_trip()
        starts = {
            c["index"]: c["content_block"]["type"]
            for c in out
            if c.get("type") == "content_block_start"
        }

        assert starts[0] == "server_tool_use"  # the server-tool call (#974)
        assert starts[1] == "web_search_tool_result"
        assert starts[2] == "text"

    def test_ordering_by_emission(self):
        """The result block is emitted after the call and before the text."""
        types = [
            c["content_block"]["type"]
            for c in self._round_trip()
            if c.get("type") == "content_block_start"
        ]

        assert types.index("web_search_tool_result") < types.index("text")
        assert types.index("web_search_tool_result") > types.index("server_tool_use")


class TestCrossFormatDegrade:
    """A target that cannot render the event drops it silently."""

    SEQ: list[dict[str, Any]] = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": dict(_WEB_RESULT_BLOCK),
        },
        {"type": "content_block_stop", "index": 1},
    ]

    def test_chat_completions_drops_without_raising(self):
        source = MessagesConverter()
        target = ChatCompletionsConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()

        for chunk in self.SEQ:
            for event in source.stream_response_from_provider(chunk, from_ctx):
                chunks = target.stream_response_to_provider(event, to_ctx)
                # Either falsy (dropped) or a list of well-formed chunks.
                if isinstance(chunks, list):
                    assert all(isinstance(c, dict) and c for c in chunks)
                else:
                    assert chunks == {}

    def test_chat_completions_emits_nothing_for_the_result(self):
        source = MessagesConverter()
        target = ChatCompletionsConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()

        emitted: list[dict[str, Any]] = []
        for chunk in self.SEQ:
            for event in source.stream_response_from_provider(chunk, from_ctx):
                chunks = target.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    emitted.extend(chunks)
                elif chunks:
                    emitted.append(chunks)

        assert emitted == []

    def test_no_internal_metadata_leaks_to_the_wire(self):
        """The ``_native`` payload must never reach an emitted chunk.

        Asserted against the same-dialect writer, which does emit chunks: with a
        cross-dialect target everything is dropped, so the assertion would hold
        on an empty list whatever the writer did (#973 review).
        """
        source = MessagesConverter()
        target = MessagesConverter()
        from_ctx = StreamContext()
        to_ctx = StreamContext()

        emitted: list[dict[str, Any]] = []
        for chunk in self.SEQ:
            for event in source.stream_response_from_provider(chunk, from_ctx):
                chunks = target.stream_response_to_provider(event, to_ctx)
                if isinstance(chunks, list):
                    emitted.extend(chunks)
                elif chunks:
                    emitted.append(chunks)

        assert emitted, "the same-dialect writer is expected to emit chunks here"
        assert "_native" not in json.dumps(emitted)
        assert "provider_metadata" not in json.dumps(emitted)


class TestResultBlockCacheHint:
    """A result block's cache breakpoint survives the event (#973 review).

    ``ToolResultPart`` carries ``cache_hint``, so an event that claims to mirror
    it has to as well — otherwise a cross-dialect reader, and the writer's
    fallback rebuild, lose the breakpoint the identical non-streaming block
    keeps.
    """

    BLOCK = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "title": "t", "url": "u"}],
        "cache_control": {"type": "ephemeral"},
    }

    def _event(self):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": self.BLOCK,
        }
        ctx = StreamContext()
        events = MessagesConverter().stream_response_from_provider(chunk, ctx)
        return next(e for e in events if e["type"] == "tool_result")

    def test_reader_carries_the_cache_hint(self):
        assert self._event()["cache_hint"] == {"type": "ephemeral"}

    def test_fallback_writer_restores_cache_control(self):
        event = self._event()
        # keep the kind (so the block is a server result), drop the native blob
        event["provider_metadata"] = {"intrinsic_kind": "web_search"}
        block = MessagesConverter()._ir_tool_result_block(event)
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_fallback_writer_omits_is_error_on_a_server_result(self):
        """Anthropic refuses ``is_error`` on a ``*_tool_result`` block."""
        event = self._event()
        event["provider_metadata"] = {"intrinsic_kind": "web_search"}
        block = MessagesConverter()._ir_tool_result_block(event)
        assert "is_error" not in block
