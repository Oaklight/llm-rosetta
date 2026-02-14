"""
OpenAI Chat Completions stream converter unit tests.
"""

from llmir.converters.openai_chat import OpenAIChatConverter


class TestStreamResponseFromProvider:
    """Tests for stream_response_from_provider."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    # --- Text delta ---

    def test_text_delta(self):
        """Text content delta is converted to TextDeltaEvent."""
        chunk = {
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"
        assert events[0]["choice_index"] == 0

    def test_text_delta_empty_string(self):
        """Empty string content should still produce a TextDeltaEvent."""
        chunk = {
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == ""

    def test_text_delta_choice_index(self):
        """Choice index is preserved in the event."""
        chunk = {
            "choices": [{"index": 2, "delta": {"content": "Hi"}, "finish_reason": None}]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["choice_index"] == 2

    # --- Reasoning delta ---

    def test_reasoning_content_delta(self):
        """reasoning_content delta is converted to ReasoningDeltaEvent."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Let me think..."},
                    "finish_reason": None,
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == "Let me think..."
        assert events[0]["choice_index"] == 0

    # --- Tool call start ---

    def test_tool_call_start(self):
        """Tool call with id produces ToolCallStartEvent."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "index": 0,
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        start_events = [e for e in events if e["type"] == "tool_call_start"]
        assert len(start_events) == 1
        assert start_events[0]["tool_call_id"] == "call_abc"
        assert start_events[0]["tool_name"] == "get_weather"
        assert start_events[0]["tool_call_index"] == 0
        assert start_events[0]["choice_index"] == 0

    # --- Tool call arguments delta ---

    def test_tool_call_arguments_delta(self):
        """Tool call arguments delta produces ToolCallDeltaEvent."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "function": {"arguments": '{"city":'},
                            }
                        ]
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        delta_events = [e for e in events if e["type"] == "tool_call_delta"]
        assert len(delta_events) == 1
        assert delta_events[0]["arguments_delta"] == '{"city":'
        assert delta_events[0]["tool_call_index"] == 1

    def test_tool_call_start_and_delta_combined(self):
        """Tool call with id and arguments produces both start and delta events."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "index": 0,
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":',
                                },
                            }
                        ]
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        types = [e["type"] for e in events]
        assert "tool_call_start" in types
        assert "tool_call_delta" in types

    def test_multiple_tool_calls_index(self):
        """Multiple tool calls with different indices are handled correctly."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "index": 0,
                                "function": {"name": "func_a", "arguments": ""},
                            },
                            {
                                "id": "call_2",
                                "index": 1,
                                "function": {"name": "func_b", "arguments": ""},
                            },
                        ]
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        start_events = [e for e in events if e["type"] == "tool_call_start"]
        assert len(start_events) == 2
        assert start_events[0]["tool_call_index"] == 0
        assert start_events[1]["tool_call_index"] == 1

    # --- Finish event ---

    def test_finish_stop(self):
        """finish_reason 'stop' produces FinishEvent with reason 'stop'."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "finish"
        assert events[0]["finish_reason"]["reason"] == "stop"

    def test_finish_length(self):
        """finish_reason 'length' maps to 'length'."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["finish_reason"]["reason"] == "length"

    def test_finish_tool_calls(self):
        """finish_reason 'tool_calls' maps to 'tool_calls'."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["finish_reason"]["reason"] == "tool_calls"

    def test_finish_content_filter(self):
        """finish_reason 'content_filter' maps to 'content_filter'."""
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["finish_reason"]["reason"] == "content_filter"

    def test_finish_function_call(self):
        """finish_reason 'function_call' maps to 'tool_calls'."""
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["finish_reason"]["reason"] == "tool_calls"

    # --- Usage event ---

    def test_usage_event(self):
        """Usage in chunk produces UsageEvent."""
        chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "usage"
        assert events[0]["usage"]["prompt_tokens"] == 10
        assert events[0]["usage"]["completion_tokens"] == 5
        assert events[0]["usage"]["total_tokens"] == 15

    # --- Empty / irrelevant events ---

    def test_empty_delta(self):
        """Empty delta with no finish_reason produces no events."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    def test_no_choices(self):
        """Chunk with no choices and no usage produces no events."""
        chunk = {"choices": []}
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    def test_content_none_not_emitted(self):
        """content: None should NOT produce a TextDeltaEvent."""
        chunk = {
            "choices": [{"index": 0, "delta": {"content": None}, "finish_reason": None}]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    # --- SDK object normalization ---

    def test_normalize_sdk_object(self):
        """SDK objects with model_dump() are normalized."""

        class MockChunk:
            def model_dump(self):
                return {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "sdk"},
                            "finish_reason": None,
                        }
                    ]
                }

        events = self.converter.stream_response_from_provider(MockChunk())
        assert len(events) == 1
        assert events[0]["text"] == "sdk"


class TestStreamResponseToProvider:
    """Tests for stream_response_to_provider."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    def test_text_delta(self):
        """TextDeltaEvent → OpenAI chunk."""
        event = {"type": "text_delta", "text": "Hello", "choice_index": 0}
        result = self.converter.stream_response_to_provider(event)
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_reasoning_delta(self):
        """ReasoningDeltaEvent → OpenAI chunk with reasoning_content."""
        event = {
            "type": "reasoning_delta",
            "reasoning": "thinking...",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["choices"][0]["delta"]["reasoning_content"] == "thinking..."

    def test_tool_call_start(self):
        """ToolCallStartEvent → OpenAI chunk."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_1",
            "tool_name": "search",
            "tool_call_index": 0,
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == ""
        assert tc["index"] == 0

    def test_tool_call_start_no_index(self):
        """ToolCallStartEvent without tool_call_index omits index field."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_1",
            "tool_name": "search",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert "index" not in tc

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → OpenAI chunk."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"city":',
            "tool_call_index": 0,
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"city":'
        assert tc["index"] == 0

    def test_tool_call_delta_no_index(self):
        """ToolCallDeltaEvent without tool_call_index omits index field."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"x":1}',
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert "index" not in tc

    def test_finish_event(self):
        """FinishEvent → OpenAI chunk."""
        event = {
            "type": "finish",
            "finish_reason": {"reason": "stop"},
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["delta"] == {}

    def test_usage_event(self):
        """UsageEvent → OpenAI chunk."""
        event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = {"type": "unknown_event"}
        result = self.converter.stream_response_to_provider(event)
        assert result == {}

    def test_default_choice_index(self):
        """Default choice_index is 0 when not specified."""
        event = {"type": "text_delta", "text": "Hi"}
        result = self.converter.stream_response_to_provider(event)
        assert result["choices"][0]["index"] == 0


class TestStreamRoundTrip:
    """Round-trip tests: provider → IR → provider."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    def test_text_delta_round_trip(self):
        """Text delta round-trip preserves content."""
        chunk = {
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["choices"][0]["delta"]["content"] == "Hello"

    def test_reasoning_delta_round_trip(self):
        """Reasoning delta round-trip preserves content."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "step 1"},
                    "finish_reason": None,
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["choices"][0]["delta"]["reasoning_content"] == "step 1"

    def test_finish_round_trip(self):
        """Finish event round-trip preserves reason."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["choices"][0]["finish_reason"] == "stop"

    def test_usage_round_trip(self):
        """Usage event round-trip preserves token counts."""
        chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["usage"]["total_tokens"] == 30
