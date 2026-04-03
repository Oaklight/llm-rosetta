"""Tests for openai_responses utility functions."""

from llm_rosetta.converters.base.stream_context import StreamContext
from llm_rosetta.converters.openai_responses._constants import ResponsesEventType
from llm_rosetta.converters.openai_responses.stream_context import (
    OpenAIResponsesStreamContext,
)
from llm_rosetta.converters.openai_responses.utils import (
    build_message_preamble_events,
    resolve_call_id,
)


class TestResolveCallId:
    """Tests for resolve_call_id."""

    def test_returns_call_id_when_present(self):
        chunk = {"call_id": "call_abc", "item_id": "item_xyz"}
        assert resolve_call_id(chunk, None) == "call_abc"

    def test_falls_back_to_item_id_mapping(self):
        ctx = OpenAIResponsesStreamContext()
        ctx.item_id_to_call_id["item_xyz"] = "call_abc"
        chunk = {"item_id": "item_xyz"}
        assert resolve_call_id(chunk, ctx) == "call_abc"

    def test_returns_empty_when_no_call_id_and_no_item_id(self):
        ctx = OpenAIResponsesStreamContext()
        assert resolve_call_id({}, ctx) == ""

    def test_returns_empty_when_context_is_none(self):
        chunk = {"item_id": "item_xyz"}
        assert resolve_call_id(chunk, None) == ""

    def test_returns_empty_when_context_is_base_stream_context(self):
        ctx = StreamContext()
        chunk = {"item_id": "item_xyz"}
        assert resolve_call_id(chunk, ctx) == ""

    def test_returns_empty_when_item_id_not_in_mapping(self):
        ctx = OpenAIResponsesStreamContext()
        chunk = {"item_id": "unknown_item"}
        assert resolve_call_id(chunk, ctx) == ""

    def test_prefers_call_id_over_item_id(self):
        ctx = OpenAIResponsesStreamContext()
        ctx.item_id_to_call_id["item_xyz"] = "call_from_map"
        chunk = {"call_id": "call_direct", "item_id": "item_xyz"}
        assert resolve_call_id(chunk, ctx) == "call_direct"


class TestBuildMessagePreambleEvents:
    """Tests for build_message_preamble_events."""

    def setup_method(self):
        self.ctx = OpenAIResponsesStreamContext()
        self.ctx.response_id = "resp_123"

    def test_returns_two_events(self):
        events = build_message_preamble_events(self.ctx)
        assert len(events) == 2

    def test_first_event_is_output_item_added(self):
        events = build_message_preamble_events(self.ctx)
        assert events[0]["type"] == ResponsesEventType.OUTPUT_ITEM_ADDED
        item = events[0]["item"]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["content"] == []
        assert item["id"] == "msg_resp_123"

    def test_second_event_is_content_part_added(self):
        events = build_message_preamble_events(self.ctx)
        assert events[1]["type"] == ResponsesEventType.CONTENT_PART_ADDED
        assert events[1]["content_index"] == 0
        part = events[1]["part"]
        assert part["type"] == "output_text"
        assert part["text"] == ""

    def test_sets_output_item_emitted(self):
        assert self.ctx.output_item_emitted is False
        build_message_preamble_events(self.ctx)
        assert self.ctx.output_item_emitted is True

    def test_stores_item_id_on_context(self):
        assert self.ctx.item_id == ""
        build_message_preamble_events(self.ctx)
        assert self.ctx.item_id == "msg_resp_123"

    def test_respects_output_index(self):
        events = build_message_preamble_events(self.ctx, output_index=2)
        assert events[0]["output_index"] == 2
        assert events[1]["output_index"] == 2

    def test_default_output_index_is_zero(self):
        events = build_message_preamble_events(self.ctx)
        assert events[0]["output_index"] == 0
        assert events[1]["output_index"] == 0
