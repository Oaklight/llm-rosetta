"""
Anthropic Messages API stream converter unit tests.
"""

from llmir.converters.anthropic import AnthropicConverter


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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "stop"

    def test_finish_max_tokens(self):
        """message_delta with stop_reason 'max_tokens' maps to 'length'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "length"

    def test_finish_tool_use(self):
        """message_delta with stop_reason 'tool_use' maps to 'tool_calls'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "tool_calls"

    def test_finish_stop_sequence(self):
        """message_delta with stop_reason 'stop_sequence' maps to 'stop'."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "stop_sequence"},
        }
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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
        events = self.converter.stream_response_from_provider(event)
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

        events = self.converter.stream_response_from_provider(MockEvent())
        assert len(events) == 1
        assert events[0]["text"] == "sdk"


class TestStreamResponseToProvider:
    """Tests for stream_response_to_provider."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    def test_text_delta(self):
        """TextDeltaEvent → Anthropic content_block_delta."""
        event = {"type": "text_delta", "text": "Hello"}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "text_delta"
        assert result["delta"]["text"] == "Hello"

    def test_reasoning_delta_thinking(self):
        """ReasoningDeltaEvent without signature → thinking_delta."""
        event = {"type": "reasoning_delta", "reasoning": "step 1"}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "thinking_delta"
        assert result["delta"]["thinking"] == "step 1"

    def test_reasoning_delta_signature(self):
        """ReasoningDeltaEvent with signature → signature_delta."""
        event = {
            "type": "reasoning_delta",
            "reasoning": "",
            "signature": "sig_xyz",
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "signature_delta"
        assert result["delta"]["signature"] == "sig_xyz"

    def test_tool_call_start(self):
        """ToolCallStartEvent → Anthropic content_block_start."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "toolu_abc",
            "tool_name": "get_weather",
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "content_block_start"
        assert result["content_block"]["type"] == "tool_use"
        assert result["content_block"]["id"] == "toolu_abc"
        assert result["content_block"]["name"] == "get_weather"
        assert result["content_block"]["input"] == {}

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → Anthropic content_block_delta."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "",
            "arguments_delta": '{"city": "NYC"}',
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "input_json_delta"
        assert result["delta"]["partial_json"] == '{"city": "NYC"}'

    def test_finish_event_stop(self):
        """FinishEvent with 'stop' → message_delta with 'end_turn'."""
        event = {"type": "finish", "finish_reason": {"reason": "stop"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "message_delta"
        assert result["delta"]["stop_reason"] == "end_turn"

    def test_finish_event_length(self):
        """FinishEvent with 'length' → message_delta with 'max_tokens'."""
        event = {"type": "finish", "finish_reason": {"reason": "length"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["delta"]["stop_reason"] == "max_tokens"

    def test_finish_event_tool_calls(self):
        """FinishEvent with 'tool_calls' → message_delta with 'tool_use'."""
        event = {"type": "finish", "finish_reason": {"reason": "tool_calls"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["delta"]["stop_reason"] == "tool_use"

    def test_usage_event(self):
        """UsageEvent → Anthropic message_delta with usage."""
        event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "message_delta"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = {"type": "unknown_event"}
        result = self.converter.stream_response_to_provider(event)
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
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["delta"]["text"] == "Hello"
        assert restored["delta"]["type"] == "text_delta"

    def test_thinking_delta_round_trip(self):
        """Thinking delta round-trip preserves content."""
        original = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "analyzing..."},
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["delta"]["thinking"] == "analyzing..."

    def test_signature_delta_round_trip(self):
        """Signature delta round-trip preserves signature."""
        original = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_test"},
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
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
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["content_block"]["id"] == "toolu_abc"
        assert restored["content_block"]["name"] == "search"

    def test_finish_round_trip(self):
        """Finish event round-trip preserves reason mapping."""
        original = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        }
        events = self.converter.stream_response_from_provider(original)
        finish = [e for e in events if e["type"] == "finish"][0]
        restored = self.converter.stream_response_to_provider(finish)
        assert restored["delta"]["stop_reason"] == "end_turn"
