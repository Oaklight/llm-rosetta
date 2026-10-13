"""Streamed Open Responses server-tool calls through the IR (#985).

The Open Responses format executes server tools (web search, code interpreter)
and returns the call as a native output item (``web_search_call`` /
``code_interpreter_call``) whose payload (``action`` / ``code``) is only known
by the ``done`` event.  The streaming reader used to emit no IR event at all
for these items, and the writer emitted a client ``function_call`` for an
intrinsic ``ToolCallStartEvent``.

The reader now emits a ``ToolCallStartEvent`` marked
``tool_type="intrinsic"`` with the canonical kind in ``provider_metadata`` and
moves the ``done`` payload as a ``ToolCallDeltaEvent``; the writer re-emits the
provider's own server item type.  The kind is never inferred from the tool
name.
"""

import json
from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenResponsesConverter
from llm_rosetta.converters.open_responses.tool_ops import OpenResponsesToolOps
from llm_rosetta.types.ir.stream import IRStreamEvent, ToolCallStartEvent

_WEB_ADDED: dict[str, Any] = {
    "type": "web_search_call",
    "id": "ws_1",
    "status": "in_progress",
}

_WEB_DONE: dict[str, Any] = {
    "type": "web_search_call",
    "id": "ws_1",
    "status": "completed",
    "action": {
        "type": "search",
        "query": "tokyo population",
        "queries": ["tokyo population", "greater tokyo population"],
    },
}

_CI_ADDED: dict[str, Any] = {
    "type": "code_interpreter_call",
    "id": "ci_1",
    "status": "in_progress",
}

_CI_DONE: dict[str, Any] = {
    "type": "code_interpreter_call",
    "id": "ci_1",
    "status": "completed",
    "code": "1 + 1",
    "container_id": "cntr_1",
    "outputs": [{"type": "logs", "logs": "2"}],
}


def _item_chunk(item: dict[str, Any], *, phase: str = "added") -> dict[str, Any]:
    return {
        "type": f"response.output_item.{phase}",
        "output_index": 1,
        "item": dict(item),
    }


def _reader_events(item: dict[str, Any], *, phase: str = "added"):
    converter = OpenResponsesConverter()
    return converter.stream_response_from_provider(
        _item_chunk(item, phase=phase), converter.create_stream_context()
    )


def _call_start(item: dict[str, Any]) -> ToolCallStartEvent:
    events = _reader_events(item)
    return cast(
        ToolCallStartEvent,
        next(e for e in events if e["type"] == "tool_call_start"),
    )


#: Terminal event; the write side flushes the accumulated server items
#: (``output_item.done`` + ``response.completed``) when it sees the finish.
_COMPLETED: dict[str, Any] = {
    "type": "response.completed",
    "response": {"status": "completed", "output": []},
}


def _round_trip(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read *chunks* then write the resulting IR events, keeping order."""
    src = OpenResponsesConverter()
    dst = OpenResponsesConverter()
    from_ctx = src.create_stream_context()
    to_ctx = dst.create_stream_context()
    out: list[dict[str, Any]] = []
    for chunk in [*chunks, _COMPLETED]:
        for event in src.stream_response_from_provider(chunk, from_ctx):
            emitted = dst.stream_response_to_provider(event, to_ctx)
            if isinstance(emitted, list):
                out.extend(c for c in emitted if c)
            elif emitted:
                out.append(emitted)
    return out


class TestReader:
    """The Open Responses reader turns a server item into an intrinsic call."""

    def test_web_search_call_is_marked_intrinsic(self):
        event = _call_start(_WEB_ADDED)

        assert event["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(event) == "web_search"
        assert event["tool_name"] == "web_search"
        assert event["tool_call_id"] == "ws_1"

    def test_code_interpreter_call_is_marked_intrinsic(self):
        event = _call_start(_CI_ADDED)

        assert event["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(event) == "code_interpreter"

    def test_name_is_the_canonical_kind_not_the_wire_type(self):
        # The raw wire type is `web_search_call`; the definition side and
        # `intrinsic_kind` speak `web_search`.
        event = _call_start(_WEB_ADDED)

        assert event["tool_name"] == "web_search"
        assert event["tool_name"] != "web_search_call"

    def test_event_mirrors_the_non_streaming_part(self):
        part = OpenResponsesToolOps.p_tool_call_to_ir(dict(_WEB_DONE))
        event = _call_start(_WEB_DONE)

        assert event["tool_call_id"] == part["tool_call_id"]
        assert event["tool_name"] == part["tool_name"]
        assert event["tool_type"] == part["tool_type"]
        assert get_intrinsic_kind(event) == get_intrinsic_kind(part)

    def test_output_index_is_carried_as_tool_call_index(self):
        event = _call_start(_WEB_ADDED)

        assert event["tool_call_index"] == 1

    def test_done_payload_becomes_a_delta(self):
        converter = OpenResponsesConverter()
        ctx = converter.create_stream_context()
        converter.stream_response_from_provider(_item_chunk(_WEB_ADDED), ctx)
        events = converter.stream_response_from_provider(
            _item_chunk(_WEB_DONE, phase="done"), ctx
        )

        deltas = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(deltas) == 1
        assert deltas[0]["tool_call_id"] == "ws_1"
        payload = json.loads(deltas[0]["arguments_delta"])
        assert payload["query"] == "tokyo population"
        assert payload["queries"] == ["tokyo population", "greater tokyo population"]

    def test_do_not_emit_the_intrinsic_call_as_a_client_call(self):
        events = _reader_events(_WEB_ADDED)

        starts = [e for e in events if e["type"] == "tool_call_start"]
        assert len(starts) == 1
        # A client function call would carry the "function" default.
        assert starts[0].get("tool_type") == "intrinsic"


class TestWriter:
    """The writer re-emits the provider's own server item type."""

    def setup_method(self):
        self.converter = OpenResponsesConverter()

    def _write(self, event: IRStreamEvent) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                event, self.converter.create_stream_context()
            ),
        )

    def _event(self, **overrides: Any) -> ToolCallStartEvent:
        base: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": "ws_1",
            "tool_name": "web_search",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        base.update(overrides)
        return cast(ToolCallStartEvent, base)

    def test_intrinsic_start_emits_a_web_search_item(self):
        out = self._write(self._event())

        assert out["type"] == "response.output_item.added"
        item = out["item"]
        assert item["type"] == "web_search_call"
        assert item["id"] == "ws_1"
        assert "call_id" not in item
        assert "arguments" not in item

    def test_code_interpreter_start_emits_a_code_interpreter_item(self):
        out = self._write(
            self._event(
                tool_call_id="ci_1",
                tool_name="code_interpreter",
                provider_metadata={"intrinsic_kind": "code_interpreter"},
            )
        )

        assert out["item"]["type"] == "code_interpreter_call"
        assert out["item"]["id"] == "ci_1"

    def test_client_function_start_still_emits_a_function_call(self):
        out = self._write(
            self._event(
                tool_type="function",
                tool_name="get_weather",
                provider_metadata=None,
            )
        )

        assert out["item"]["type"] == "function_call"
        assert out["item"]["name"] == "get_weather"

    def test_kind_comes_from_metadata_not_the_name(self):
        out = self._write(
            self._event(
                tool_name="totally_not_web_search",
                provider_metadata={"intrinsic_kind": "web_search"},
            )
        )

        assert out["item"]["type"] == "web_search_call"

    def test_unrenderable_kind_is_dropped_not_downgraded(self):
        # `code_execution` has no Open Responses server item; dropping beats
        # emitting a client `function_call`.
        out = self._write(
            self._event(
                tool_name="code_execution",
                provider_metadata={"intrinsic_kind": "code_execution"},
            )
        )

        assert out == {}

    def test_finish_rebuilds_the_item_with_the_action(self):
        ctx = self.converter.create_stream_context()
        self.converter.stream_response_to_provider(self._event(), ctx)
        self.converter.stream_response_to_provider(
            {
                "type": "tool_call_delta",
                "tool_call_id": "ws_1",
                "arguments_delta": '{"query": "q"}',
            },
            ctx,
        )
        emitted = self.converter.stream_response_to_provider(
            {"type": "finish", "finish_reason": {"reason": "stop"}}, ctx
        )
        assert isinstance(emitted, list)

        done_items = [
            c["item"]
            for c in emitted
            if c.get("type") == "response.output_item.done"
            and c["item"].get("type") == "web_search_call"
        ]
        assert len(done_items) == 1
        assert done_items[0]["id"] == "ws_1"
        assert done_items[0]["status"] == "completed"
        assert done_items[0]["action"] == {"type": "search", "query": "q"}


class TestSameFormatRoundTrip:
    """Open Responses -> IR -> Open Responses keeps the server items."""

    def test_call_item_survives_with_its_action(self):
        out = _round_trip(
            [_item_chunk(_WEB_ADDED), _item_chunk(_WEB_DONE, phase="done")]
        )

        items = [
            c["item"]
            for c in out
            if c.get("type")
            in ("response.output_item.added", "response.output_item.done")
            and c["item"].get("type") == "web_search_call"
        ]
        assert items, "the server call did not survive the round trip"
        final = items[-1]
        assert final["id"] == "ws_1"
        assert final["action"]["query"] == "tokyo population"
        assert final["action"]["queries"] == [
            "tokyo population",
            "greater tokyo population",
        ]

    def test_ordering_is_preserved(self):
        chunks = [
            _item_chunk(_WEB_ADDED),
            _item_chunk(_WEB_DONE, phase="done"),
            _item_chunk(_CI_ADDED),
            _item_chunk(_CI_DONE, phase="done"),
        ]
        out = _round_trip(chunks)

        order = [
            c["item"]["type"]
            for c in out
            if c.get("type") == "response.output_item.added"
            and c["item"].get("type") in ("web_search_call", "code_interpreter_call")
        ]
        assert order == ["web_search_call", "code_interpreter_call"]

    def test_no_client_call_is_ever_emitted(self):
        out = _round_trip(
            [_item_chunk(_WEB_ADDED), _item_chunk(_WEB_DONE, phase="done")]
        )

        for chunk in out:
            item = chunk.get("item") or {}
            assert item.get("type") != "function_call"


class TestCrossFormatDegrade:
    """A foreign target must not turn the server call into a client one."""

    def _to_target(self, target, chunks):
        src = OpenResponsesConverter()
        from_ctx = src.create_stream_context()
        to_ctx = target.create_stream_context()
        emitted: list[dict[str, Any]] = []
        for chunk in chunks:
            for event in src.stream_response_from_provider(chunk, from_ctx):
                out = target.stream_response_to_provider(event, to_ctx)
                if isinstance(out, list):
                    emitted.extend(c for c in out if c)
                elif out:
                    emitted.append(out)
        return emitted

    def test_messages_renders_a_server_call_not_a_client_one(self):
        emitted = self._to_target(
            MessagesConverter(),
            [_item_chunk(_WEB_ADDED), _item_chunk(_WEB_DONE, phase="done")],
        )

        blocks = [
            c["content_block"]["type"]
            for c in emitted
            if c.get("type") == "content_block_start"
        ]
        assert "server_tool_use" in blocks
        assert "tool_use" not in blocks

    def test_google_generate_drops_the_result_event(self):
        # google_generate cannot render a server result; the event is dropped
        # (the call-side handling is #986, tracked separately).
        result_event = {
            "type": "tool_result",
            "tool_call_id": "ci_1",
            "result": "2",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_interpreter"},
        }
        target = GoogleGenerateConverter()

        out = target.stream_response_to_provider(
            cast(Any, result_event), StreamContext()
        )

        assert out == {}
