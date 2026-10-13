"""Streamed Open Responses server-tool results through the IR (#985).

Open Responses carries a server result as its own ``*_call_output`` item
(``code_interpreter_call_output``, …); a web search has no separate output item
— its result rides on the call item's ``action``.  The streaming reader used to
emit only the item's ``ContentBlockStartEvent`` (nothing) for these, and the
writer had no ``tool_result`` handler at all, so a ``ToolResultEvent`` was
dropped and a server result could be re-encoded as a client
``function_call_output``.

The reader now emits a one-shot ``ToolResultEvent`` and the writer re-emits the
provider's own item (verbatim for a same-dialect read, rebuilt otherwise).  A
kind with no output item is dropped silently.
"""

from typing import Any, cast

from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenResponsesConverter
from llm_rosetta.converters.open_responses.tool_ops import OpenResponsesToolOps
from llm_rosetta.types.ir.stream import ToolResultEvent

_CI_RESULT_ITEM: dict[str, Any] = {
    "type": "code_interpreter_call_output",
    "call_id": "ci_1",
    "output": "2",
}

_WEB_RESULT_ITEM: dict[str, Any] = {
    "type": "web_search_call_output",
    "call_id": "ws_1",
    "output": [{"type": "input_text", "text": "a result"}],
}


def _result_events(item: dict[str, Any]) -> list[ToolResultEvent]:
    converter = OpenResponsesConverter()
    return cast(
        list[ToolResultEvent],
        converter.stream_response_from_provider(
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": dict(item),
            },
            converter.create_stream_context(),
        ),
    )


class TestReader:
    """The reader turns a ``*_call_output`` item into a ToolResultEvent."""

    def test_result_item_becomes_one_event(self):
        events = _result_events(_CI_RESULT_ITEM)

        assert [e["type"] for e in events] == ["tool_result"]

    def test_event_mirrors_the_non_streaming_part(self):
        part = OpenResponsesToolOps.p_tool_result_to_ir(dict(_CI_RESULT_ITEM))
        (event,) = _result_events(_CI_RESULT_ITEM)

        assert event["tool_call_id"] == part["tool_call_id"]
        assert event["result"] == part["result"]
        assert event["tool_type"] == part["tool_type"]
        assert get_intrinsic_kind(event) == get_intrinsic_kind(part)

    def test_code_interpreter_kind(self):
        (event,) = _result_events(_CI_RESULT_ITEM)

        assert event["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(event) == "code_interpreter"
        assert event["block_index"] == 2

    def test_web_search_kind(self):
        (event,) = _result_events(_WEB_RESULT_ITEM)

        assert get_intrinsic_kind(event) == "web_search"

    def test_native_item_is_kept_for_replay(self):
        (event,) = _result_events(_CI_RESULT_ITEM)

        native = event["provider_metadata"]["_native"]
        assert native["base"] == "open_responses"
        assert native["item"] == _CI_RESULT_ITEM

    def test_other_item_types_do_not_emit_the_event(self):
        events = _result_events({"type": "web_search_call", "id": "ws_1"})

        assert all(e["type"] != "tool_result" for e in events)


class TestWriter:
    """The writer re-emits the provider's server result item."""

    def setup_method(self):
        self.converter = OpenResponsesConverter()

    def _write(self, event: ToolResultEvent) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                event, self.converter.create_stream_context()
            ),
        )

    def test_same_dialect_replay_is_verbatim(self):
        (event,) = _result_events(_CI_RESULT_ITEM)

        out = self._write(event)

        assert out["type"] == "response.output_item.added"
        assert out["item"] == _CI_RESULT_ITEM
        assert out["output_index"] == 2

    def test_fallback_rebuilds_from_ir_fields(self):
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="ci_1",
            result="2",
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "code_interpreter"},
        )

        out = self._write(event)

        assert out["item"] == {
            "type": "code_interpreter_call_output",
            "call_id": "ci_1",
            "output": "2",
        }

    def test_web_search_result_has_no_output_item_and_is_dropped(self):
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="ws_1",
            result=[{"type": "text", "text": "a result"}],
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "web_search"},
        )

        assert self._write(event) == {}

    def test_unrenderable_kind_is_dropped(self):
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="s1",
            result="x",
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "code_execution"},
        )

        assert self._write(event) == {}

    def test_foreign_native_payload_is_not_replayed(self):
        """A ``_native`` blob from another dialect must not reach the wire."""
        event = ToolResultEvent(
            type="tool_result",
            tool_call_id="ci_1",
            result="2",
            tool_type="intrinsic",
            provider_metadata={
                "intrinsic_kind": "code_interpreter",
                "_native": {
                    "base": "messages",
                    "item": {"type": "code_interpreter_call", "id": "bogus"},
                },
            },
        )

        out = self._write(event)

        assert out["item"]["type"] == "code_interpreter_call_output"
        assert out["item"]["output"] == "2"

    def test_never_emits_a_client_function_call_output(self):
        (event,) = _result_events(_CI_RESULT_ITEM)

        out = self._write(event)

        assert out["item"]["type"] != "function_call_output"
        assert out["item"]["type"] != "custom_tool_call_output"


class TestCrossFormatDegrade:
    """A foreign target renders the server result as a server result or drops it."""

    def _event(self) -> ToolResultEvent:
        return ToolResultEvent(
            type="tool_result",
            tool_call_id="ci_1",
            result=cast(Any, [{"type": "text", "text": "2"}]),
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "code_interpreter"},
        )

    def test_messages_renders_a_server_result(self):
        target = MessagesConverter()
        out = target.stream_response_to_provider(
            self._event(), target.create_stream_context()
        )
        block = cast(dict[str, Any], out)["content_block"]

        assert block["type"].endswith("_tool_result")
        assert block["type"] != "function_call_output"

    def test_google_generate_drops_without_raising(self):
        target = GoogleGenerateConverter()
        out = target.stream_response_to_provider(
            self._event(), target.create_stream_context()
        )

        assert out == {}
