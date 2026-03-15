"""
Anthropic Messages API stream converter unit tests.
"""

from typing import Any, cast

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.base.stream_context import StreamContext
from llm_rosetta.types.ir.stream import (
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)


class TestStreamResponseFromProvider:
    """Tests for stream_response_from_provider."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    # --- Text delta ---

    def test_text_delta(self):
        """text_delta in content_block_delta produces TextDeltaEvent."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_text_delta_empty_string(self):
        """Empty text delta still produces an event."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ""},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == ""

    # --- Reasoning delta (thinking) ---

    def test_thinking_delta(self):
        """thinking_delta produces ReasoningDeltaEvent."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me analyze..."},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == "Let me analyze..."

    def test_signature_delta(self):
        """signature_delta produces ReasoningDeltaEvent with signature field."""
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_abc123"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == ""
        assert events[0]["signature"] == "sig_abc123"

    # --- Tool call start ---

    def test_tool_call_start_tool_use(self):
        """content_block_start with tool_use type produces ToolCallStartEvent."""
        event = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {},
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert events[0]["tool_call_id"] == "toolu_abc"
        assert events[0]["tool_name"] == "get_weather"

    def test_tool_call_start_server_tool_use(self):
        """content_block_start with server_tool_use type also produces ToolCallStartEvent."""
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "toolu_srv",
                "name": "web_search",
                "input": {},
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert events[0]["tool_call_id"] == "toolu_srv"
        assert events[0]["tool_name"] == "web_search"

    def test_content_block_start_text_ignored(self):
        """content_block_start with text type produces no events."""
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    # --- Tool call arguments delta ---

    def test_tool_call_arguments_delta(self):
        """input_json_delta produces ToolCallDeltaEvent."""
        event = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"
        assert events[0]["arguments_delta"] == '{"city":'
        assert events[0]["tool_call_id"] == ""  # Anthropic doesn't repeat ID

    # --- Finish event ---

    def test_finish_end_turn(self):
        """message_delta with stop_reason 'end_turn' maps to 'stop'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "stop"

    def test_finish_max_tokens(self):
        """message_delta with stop_reason 'max_tokens' maps to 'length'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "length"

    def test_finish_tool_use(self):
        """message_delta with stop_reason 'tool_use' maps to 'tool_calls'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "tool_calls"

    def test_finish_stop_sequence(self):
        """message_delta with stop_reason 'stop_sequence' maps to 'stop'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "stop_sequence"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "stop"

    # --- Usage event ---

    def test_message_start_usage(self):
        """message_start with usage produces UsageEvent."""
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 0},
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "usage"
        assert events[0]["usage"]["prompt_tokens"] == 25
        assert events[0]["usage"]["completion_tokens"] == 0
        assert events[0]["usage"]["total_tokens"] == 25

    def test_message_delta_with_usage(self):
        """message_delta with usage produces both FinishEvent and UsageEvent."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 0, "output_tokens": 42},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        types = [e["type"] for e in events]
        assert "finish" in types
        assert "usage" in types
        usage_event = [e for e in events if e["type"] == "usage"][0]
        assert usage_event["usage"]["completion_tokens"] == 42

    # --- Ignored events ---

    def test_content_block_stop_ignored(self):
        """content_block_stop produces no events."""
        event = {"type": "content_block_stop", "index": 0}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_message_stop_ignored(self):
        """message_stop produces no events."""
        event = {"type": "message_stop"}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_ping_ignored(self):
        """ping produces no events."""
        event = {"type": "ping"}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    # --- SDK object normalization ---

    def test_normalize_sdk_object(self):
        """SDK objects with model_dump() are normalized."""

        class MockEvent:
            def model_dump(self):
                return {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "sdk"},
                }

        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(
                cast(dict[str, Any], MockEvent())
            ),
        )
        assert len(events) == 1
        assert events[0]["text"] == "sdk"


class TestStreamResponseToProvider:
    """Tests for stream_response_to_provider."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    def test_text_delta(self):
        """TextDeltaEvent → Anthropic content_block_delta."""
        event = cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "text_delta"
        assert result["delta"]["text"] == "Hello"

    def test_reasoning_delta_thinking(self):
        """ReasoningDeltaEvent without signature → thinking_delta."""
        event = cast(
            ReasoningDeltaEvent,
            {"type": "reasoning_delta", "reasoning": "step 1"},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "thinking_delta"
        assert result["delta"]["thinking"] == "step 1"

    def test_reasoning_delta_signature(self):
        """ReasoningDeltaEvent with signature → signature_delta."""
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "",
                "signature": "sig_xyz",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "signature_delta"
        assert result["delta"]["signature"] == "sig_xyz"

    def test_tool_call_start(self):
        """ToolCallStartEvent → Anthropic content_block_start."""
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "toolu_abc",
                "tool_name": "get_weather",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_start"
        assert result["content_block"]["type"] == "tool_use"
        assert result["content_block"]["id"] == "toolu_abc"
        assert result["content_block"]["name"] == "get_weather"
        assert result["content_block"]["input"] == {}

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → Anthropic content_block_delta."""
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "",
                "arguments_delta": '{"city": "NYC"}',
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "input_json_delta"
        assert result["delta"]["partial_json"] == '{"city": "NYC"}'

    def test_finish_event_stop(self):
        """FinishEvent with 'stop' → message_delta with 'end_turn'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "message_delta"
        assert result["delta"]["stop_reason"] == "end_turn"

    def test_finish_event_length(self):
        """FinishEvent with 'length' → message_delta with 'max_tokens'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "length"}},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["delta"]["stop_reason"] == "max_tokens"

    def test_finish_event_tool_calls(self):
        """FinishEvent with 'tool_calls' → message_delta with 'tool_use'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["delta"]["stop_reason"] == "tool_use"

    def test_usage_event(self):
        """UsageEvent → Anthropic message_delta with usage."""
        event = cast(
            UsageEvent,
            {
                "type": "usage",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "message_delta"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = cast(TextDeltaEvent, {"type": "unknown_event"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result == {}


class TestStreamRoundTrip:
    """Round-trip tests: provider → IR → provider."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    def test_text_delta_round_trip(self):
        """Text delta round-trip preserves content."""
        original = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["delta"]["text"] == "Hello"
        assert restored["delta"]["type"] == "text_delta"

    def test_thinking_delta_round_trip(self):
        """Thinking delta round-trip preserves content."""
        original = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "analyzing..."},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["delta"]["thinking"] == "analyzing..."

    def test_signature_delta_round_trip(self):
        """Signature delta round-trip preserves signature."""
        original = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_test"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["delta"]["signature"] == "sig_test"

    def test_tool_call_start_round_trip(self):
        """Tool call start round-trip preserves id and name."""
        original = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search",
                "input": {},
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["content_block"]["id"] == "toolu_abc"
        assert restored["content_block"]["name"] == "search"

    def test_finish_round_trip(self):
        """Finish event round-trip preserves reason mapping."""
        original = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        finish = [e for e in events if e["type"] == "finish"][0]
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(finish)
        )
        assert restored["delta"]["stop_reason"] == "end_turn"


class TestStreamResponseFromProviderWithContext:
    """Tests for stream_response_from_provider with StreamContext."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    # --- StreamStartEvent ---

    def test_message_start_emits_stream_start_with_context(self):
        """message_start emits StreamStartEvent when context is provided."""
        ctx = StreamContext()
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 0},
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        start_events = [e for e in events if e["type"] == "stream_start"]
        assert len(start_events) == 1
        assert start_events[0]["response_id"] == "msg_abc123"
        assert start_events[0]["model"] == "claude-sonnet-4-20250514"

    def test_message_start_updates_context(self):
        """message_start stores metadata in context and marks started."""
        ctx = StreamContext()
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 0},
            },
        }
        self.converter.stream_response_from_provider(event, context=ctx)
        assert ctx.response_id == "msg_abc123"
        assert ctx.model == "claude-sonnet-4-20250514"
        assert ctx.is_started is True

    def test_message_start_emits_stream_start_and_usage(self):
        """message_start with usage emits both StreamStartEvent and UsageEvent."""
        ctx = StreamContext()
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 0},
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        types = [e["type"] for e in events]
        assert "stream_start" in types
        assert "usage" in types
        # StreamStartEvent should come before UsageEvent
        start_idx = types.index("stream_start")
        usage_idx = types.index("usage")
        assert start_idx < usage_idx

    def test_message_start_stream_start_before_usage(self):
        """StreamStartEvent is emitted before UsageEvent from message_start."""
        ctx = StreamContext()
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert events[0]["type"] == "stream_start"
        assert events[1]["type"] == "usage"

    # --- ContentBlockStartEvent ---

    def test_content_block_start_text_with_context(self):
        """content_block_start (text) emits ContentBlockStartEvent with context."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        block_events = [e for e in events if e["type"] == "content_block_start"]
        assert len(block_events) == 1
        assert block_events[0]["block_index"] == 0
        assert block_events[0]["block_type"] == "text"

    def test_content_block_start_thinking_with_context(self):
        """content_block_start (thinking) emits ContentBlockStartEvent with context."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        block_events = [e for e in events if e["type"] == "content_block_start"]
        assert len(block_events) == 1
        assert block_events[0]["block_index"] == 0
        assert block_events[0]["block_type"] == "thinking"

    def test_content_block_start_tool_use_with_context(self):
        """content_block_start (tool_use) emits both ContentBlockStartEvent and ToolCallStartEvent."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {},
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        types = [e["type"] for e in events]
        assert "content_block_start" in types
        assert "tool_call_start" in types
        # ContentBlockStartEvent should come before ToolCallStartEvent
        block_idx = types.index("content_block_start")
        tool_idx = types.index("tool_call_start")
        assert block_idx < tool_idx
        # Verify ContentBlockStartEvent fields
        block_event = [e for e in events if e["type"] == "content_block_start"][0]
        assert block_event["block_index"] == 1
        assert block_event["block_type"] == "tool_use"

    def test_content_block_start_updates_context_block_index(self):
        """content_block_start increments context block index."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        self.converter.stream_response_from_provider(event, context=ctx)
        assert ctx.current_block_index == 0

    # --- ContentBlockEndEvent ---

    def test_content_block_stop_with_context(self):
        """content_block_stop emits ContentBlockEndEvent with context."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {"type": "content_block_stop", "index": 0}
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_end"
        assert events[0]["block_index"] == 0

    def test_content_block_stop_different_index(self):
        """content_block_stop with different index is preserved."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {"type": "content_block_stop", "index": 2}
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert events[0]["block_index"] == 2

    # --- StreamEndEvent ---

    def test_message_stop_emits_stream_end_with_context(self):
        """message_stop emits StreamEndEvent when context is provided."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {"type": "message_stop"}
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "stream_end"
        assert ctx.is_ended is True

    # --- Tool call registration in context ---

    def test_tool_call_registered_in_context(self):
        """Tool call is registered in context when content_block_start (tool_use)."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {},
            },
        }
        self.converter.stream_response_from_provider(event, context=ctx)
        assert ctx.get_tool_name("toolu_abc") == "get_weather"

    def test_tool_call_delta_gets_id_from_context(self):
        """input_json_delta gets tool_call_id from context when available."""
        ctx = StreamContext()
        ctx.mark_started()
        # First register a tool call
        start_event = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {},
            },
        }
        self.converter.stream_response_from_provider(start_event, context=ctx)
        # Then send a delta
        delta_event = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(delta_event, context=ctx),
        )
        delta_events = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(delta_events) == 1
        assert delta_events[0]["tool_call_id"] == "toolu_abc"

    # --- Backward compatibility (no context) ---

    def test_no_context_no_stream_start(self):
        """Without context, message_start does not emit StreamStartEvent."""
        event = {
            "type": "message_start",
            "message": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 25, "output_tokens": 0},
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        types = [e["type"] for e in events]
        assert "stream_start" not in types
        # Usage should still be emitted
        assert "usage" in types

    def test_no_context_no_content_block_events(self):
        """Without context, content_block_start (text) produces no events."""
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_no_context_content_block_stop_ignored(self):
        """Without context, content_block_stop produces no events."""
        event = {"type": "content_block_stop", "index": 0}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_no_context_message_stop_ignored(self):
        """Without context, message_stop produces no events."""
        event = {"type": "message_stop"}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_no_context_tool_call_delta_empty_id(self):
        """Without context, tool_call_delta has empty tool_call_id."""
        event = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        delta_events = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(delta_events) == 1
        assert delta_events[0]["tool_call_id"] == ""


class TestStreamResponseToProviderWithContext:
    """Tests for stream_response_to_provider with StreamContext."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    # --- StreamStartEvent ---

    def test_stream_start_to_message_start(self):
        """StreamStartEvent produces message_start event."""
        event = cast(
            StreamStartEvent,
            {
                "type": "stream_start",
                "response_id": "msg_abc123",
                "model": "claude-sonnet-4-20250514",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "message_start"
        assert result["message"]["id"] == "msg_abc123"
        assert result["message"]["model"] == "claude-sonnet-4-20250514"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == []
        assert result["message"]["stop_reason"] is None
        assert result["message"]["usage"]["input_tokens"] == 0
        assert result["message"]["usage"]["output_tokens"] == 0

    def test_stream_start_updates_context(self):
        """StreamStartEvent stores metadata in context."""
        ctx = StreamContext()
        event = cast(
            StreamStartEvent,
            {
                "type": "stream_start",
                "response_id": "msg_abc123",
                "model": "claude-sonnet-4-20250514",
            },
        )
        self.converter.stream_response_to_provider(event, context=ctx)
        assert ctx.response_id == "msg_abc123"
        assert ctx.model == "claude-sonnet-4-20250514"
        assert ctx.is_started is True

    def test_stream_start_without_context(self):
        """StreamStartEvent works without context."""
        event = cast(
            StreamStartEvent,
            {
                "type": "stream_start",
                "response_id": "msg_abc123",
                "model": "claude-sonnet-4-20250514",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "message_start"
        assert result["message"]["id"] == "msg_abc123"

    # --- StreamEndEvent ---

    def test_stream_end_to_message_stop(self):
        """StreamEndEvent produces message_stop event."""
        event = cast(StreamEndEvent, {"type": "stream_end"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "message_stop"

    def test_stream_end_updates_context(self):
        """StreamEndEvent marks context as ended."""
        ctx = StreamContext()
        ctx.mark_started()
        event = cast(StreamEndEvent, {"type": "stream_end"})
        self.converter.stream_response_to_provider(event, context=ctx)
        assert ctx.is_ended is True

    # --- ContentBlockStartEvent ---

    def test_content_block_start_text(self):
        """ContentBlockStartEvent (text) produces content_block_start."""
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 0,
                "block_type": "text",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_start"
        assert result["index"] == 0
        assert result["content_block"]["type"] == "text"
        assert result["content_block"]["text"] == ""

    def test_content_block_start_thinking(self):
        """ContentBlockStartEvent (thinking) produces content_block_start."""
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 0,
                "block_type": "thinking",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_start"
        assert result["index"] == 0
        assert result["content_block"]["type"] == "thinking"
        assert result["content_block"]["thinking"] == ""

    def test_content_block_start_tool_use_returns_empty(self):
        """ContentBlockStartEvent (tool_use) returns empty dict (handled by ToolCallStartEvent)."""
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 1,
                "block_type": "tool_use",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result == {}

    def test_content_block_start_updates_context(self):
        """ContentBlockStartEvent increments context block index."""
        ctx = StreamContext()
        ctx.mark_started()
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 0,
                "block_type": "text",
            },
        )
        self.converter.stream_response_to_provider(event, context=ctx)
        assert ctx.current_block_index == 0

    # --- ContentBlockEndEvent ---

    def test_content_block_end(self):
        """ContentBlockEndEvent produces content_block_stop."""
        event = cast(
            ContentBlockEndEvent,
            {"type": "content_block_end", "block_index": 0},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "content_block_stop"
        assert result["index"] == 0

    def test_content_block_end_different_index(self):
        """ContentBlockEndEvent with different index is preserved."""
        event = cast(
            ContentBlockEndEvent,
            {"type": "content_block_end", "block_index": 2},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["index"] == 2

    # --- Delta events with context index ---

    def test_text_delta_with_context_has_index(self):
        """TextDeltaEvent includes index when context has block index."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()  # block_index = 0
        event = cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"})
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "content_block_delta"
        assert result["index"] == 0
        assert result["delta"]["text"] == "Hello"

    def test_text_delta_without_context_no_index(self):
        """TextDeltaEvent without context does not include index."""
        event = cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert "index" not in result

    def test_reasoning_delta_with_context_has_index(self):
        """ReasoningDeltaEvent includes index when context has block index."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()  # block_index = 0
        event = cast(
            ReasoningDeltaEvent,
            {"type": "reasoning_delta", "reasoning": "thinking..."},
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["index"] == 0

    def test_signature_delta_with_context_has_index(self):
        """ReasoningDeltaEvent (signature) includes index when context has block index."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()  # block_index = 0
        event = cast(
            ReasoningDeltaEvent,
            {"type": "reasoning_delta", "reasoning": "", "signature": "sig_abc"},
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["index"] == 0

    def test_tool_call_start_with_context_has_index(self):
        """ToolCallStartEvent includes index when context has block index."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()  # block_index = 0
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "toolu_abc",
                "tool_name": "get_weather",
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["index"] == 0

    def test_tool_call_delta_with_context_has_index(self):
        """ToolCallDeltaEvent includes index when context has block index."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()  # block_index = 0
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "",
                "arguments_delta": '{"city":',
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["index"] == 0
