"""
OpenAI Responses API stream converter unit tests.
"""

from llmir.converters.openai_responses import OpenAIResponsesConverter


class TestStreamResponseFromProvider:
    """Tests for stream_response_from_provider."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    # --- Text delta ---

    def test_text_delta(self):
        """response.output_text.delta produces TextDeltaEvent."""
        event = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "output_index": 0,
            "content_index": 0,
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_text_delta_empty_string(self):
        """Empty text delta still produces an event."""
        event = {
            "type": "response.output_text.delta",
            "delta": "",
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == ""

    # --- Reasoning delta ---

    def test_reasoning_summary_delta(self):
        """response.reasoning_summary_text.delta produces ReasoningDeltaEvent."""
        event = {
            "type": "response.reasoning_summary_text.delta",
            "delta": "Let me think...",
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == "Let me think..."

    # --- Tool call start ---

    def test_tool_call_start_function_call(self):
        """response.output_item.added with function_call produces ToolCallStartEvent."""
        event = {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "",
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert events[0]["tool_call_id"] == "call_abc"
        assert events[0]["tool_name"] == "get_weather"
        assert events[0]["tool_call_index"] == 1

    def test_output_item_added_non_function_call(self):
        """response.output_item.added with non-function_call type produces no events."""
        event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_tool_call_start_no_output_index(self):
        """ToolCallStartEvent without output_index omits tool_call_index."""
        event = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "search",
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert "tool_call_index" not in events[0]

    # --- Tool call arguments delta ---

    def test_tool_call_arguments_delta(self):
        """response.function_call_arguments.delta produces ToolCallDeltaEvent."""
        event = {
            "type": "response.function_call_arguments.delta",
            "call_id": "call_abc",
            "delta": '{"city":',
            "output_index": 1,
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"
        assert events[0]["tool_call_id"] == "call_abc"
        assert events[0]["arguments_delta"] == '{"city":'
        assert events[0]["tool_call_index"] == 1

    def test_tool_call_arguments_delta_no_output_index(self):
        """ToolCallDeltaEvent without output_index omits tool_call_index."""
        event = {
            "type": "response.function_call_arguments.delta",
            "call_id": "call_abc",
            "delta": '{"x":1}',
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert "tool_call_index" not in events[0]

    # --- Response completed ---

    def test_response_completed_stop(self):
        """response.completed with status 'completed' produces FinishEvent with 'stop'."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            },
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "stop"

    def test_response_completed_with_tool_calls(self):
        """response.completed with function_call output sets reason to 'tool_calls'."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": "{}",
                    }
                ],
            },
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "tool_calls"

    def test_response_completed_incomplete_max_tokens(self):
        """response.completed with incomplete status and max_output_tokens reason."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "length"

    def test_response_completed_incomplete_content_filter(self):
        """response.completed with incomplete status and content_filter reason."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
            },
        }
        events = self.converter.stream_response_from_provider(event)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "content_filter"

    def test_response_completed_with_usage(self):
        """response.completed with usage produces both FinishEvent and UsageEvent."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        }
        events = self.converter.stream_response_from_provider(event)
        types = [e["type"] for e in events]
        assert "finish" in types
        assert "usage" in types
        usage_event = [e for e in events if e["type"] == "usage"][0]
        assert usage_event["usage"]["prompt_tokens"] == 10
        assert usage_event["usage"]["completion_tokens"] == 5
        assert usage_event["usage"]["total_tokens"] == 15

    # --- Response failed ---

    def test_response_failed(self):
        """response.failed produces FinishEvent with 'error'."""
        event = {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"message": "Something went wrong"},
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "finish"
        assert events[0]["finish_reason"]["reason"] == "error"

    # --- Ignored events ---

    def test_response_created_ignored(self):
        """response.created produces no events."""
        event = {"type": "response.created", "response": {}}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_response_in_progress_ignored(self):
        """response.in_progress produces no events."""
        event = {"type": "response.in_progress", "response": {}}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_output_item_done_ignored(self):
        """response.output_item.done produces no events."""
        event = {"type": "response.output_item.done", "item": {}}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_content_part_added_ignored(self):
        """response.content_part.added produces no events."""
        event = {"type": "response.content_part.added", "part": {}}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_output_text_done_ignored(self):
        """response.output_text.done produces no events."""
        event = {"type": "response.output_text.done", "text": "final"}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_function_call_arguments_done_ignored(self):
        """response.function_call_arguments.done produces no events."""
        event = {
            "type": "response.function_call_arguments.done",
            "arguments": "{}",
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_unknown_event_ignored(self):
        """Unknown event type produces no events."""
        event = {"type": "some.unknown.event"}
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    # --- SDK object normalization ---

    def test_normalize_sdk_object(self):
        """SDK objects with model_dump() are normalized."""

        class MockEvent:
            def model_dump(self):
                return {
                    "type": "response.output_text.delta",
                    "delta": "sdk",
                }

        events = self.converter.stream_response_from_provider(MockEvent())
        assert len(events) == 1
        assert events[0]["text"] == "sdk"


class TestStreamResponseToProvider:
    """Tests for stream_response_to_provider."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_text_delta(self):
        """TextDeltaEvent → response.output_text.delta."""
        event = {"type": "text_delta", "text": "Hello"}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.output_text.delta"
        assert result["delta"] == "Hello"

    def test_reasoning_delta(self):
        """ReasoningDeltaEvent → response.reasoning_summary_text.delta."""
        event = {"type": "reasoning_delta", "reasoning": "thinking..."}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.reasoning_summary_text.delta"
        assert result["delta"] == "thinking..."

    def test_tool_call_start(self):
        """ToolCallStartEvent → response.output_item.added."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_abc",
            "tool_name": "search",
            "tool_call_index": 1,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.output_item.added"
        assert result["item"]["type"] == "function_call"
        assert result["item"]["call_id"] == "call_abc"
        assert result["item"]["name"] == "search"
        assert result["output_index"] == 1

    def test_tool_call_start_no_index(self):
        """ToolCallStartEvent without tool_call_index omits output_index."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_abc",
            "tool_name": "search",
        }
        result = self.converter.stream_response_to_provider(event)
        assert "output_index" not in result

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → response.function_call_arguments.delta."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_abc",
            "arguments_delta": '{"city":',
            "tool_call_index": 1,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.function_call_arguments.delta"
        assert result["call_id"] == "call_abc"
        assert result["delta"] == '{"city":'
        assert result["output_index"] == 1

    def test_tool_call_delta_no_index(self):
        """ToolCallDeltaEvent without tool_call_index omits output_index."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_abc",
            "arguments_delta": "{}",
        }
        result = self.converter.stream_response_to_provider(event)
        assert "output_index" not in result

    def test_finish_event_stop(self):
        """FinishEvent with 'stop' → response.completed with status 'completed'."""
        event = {"type": "finish", "finish_reason": {"reason": "stop"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.completed"
        assert result["response"]["status"] == "completed"

    def test_finish_event_length(self):
        """FinishEvent with 'length' → response.completed with status 'incomplete'."""
        event = {"type": "finish", "finish_reason": {"reason": "length"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.completed"
        assert result["response"]["status"] == "incomplete"
        assert result["response"]["incomplete_details"]["reason"] == "max_output_tokens"

    def test_finish_event_error(self):
        """FinishEvent with 'error' → response.completed with status 'failed'."""
        event = {"type": "finish", "finish_reason": {"reason": "error"}}
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.completed"
        assert result["response"]["status"] == "failed"

    def test_usage_event(self):
        """UsageEvent → response.completed with usage."""
        event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["type"] == "response.completed"
        assert result["response"]["usage"]["input_tokens"] == 10
        assert result["response"]["usage"]["output_tokens"] == 5
        assert result["response"]["usage"]["total_tokens"] == 15

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = {"type": "unknown_event"}
        result = self.converter.stream_response_to_provider(event)
        assert result == {}


class TestStreamRoundTrip:
    """Round-trip tests: provider → IR → provider."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_text_delta_round_trip(self):
        """Text delta round-trip preserves content."""
        original = {
            "type": "response.output_text.delta",
            "delta": "Hello",
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["type"] == "response.output_text.delta"
        assert restored["delta"] == "Hello"

    def test_reasoning_delta_round_trip(self):
        """Reasoning delta round-trip preserves content."""
        original = {
            "type": "response.reasoning_summary_text.delta",
            "delta": "step 1",
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["type"] == "response.reasoning_summary_text.delta"
        assert restored["delta"] == "step 1"

    def test_tool_call_start_round_trip(self):
        """Tool call start round-trip preserves id and name."""
        original = {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "search",
            },
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["item"]["call_id"] == "call_abc"
        assert restored["item"]["name"] == "search"
        assert restored["output_index"] == 1

    def test_tool_call_delta_round_trip(self):
        """Tool call delta round-trip preserves arguments."""
        original = {
            "type": "response.function_call_arguments.delta",
            "call_id": "call_abc",
            "delta": '{"q": "test"}',
            "output_index": 1,
        }
        events = self.converter.stream_response_from_provider(original)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["call_id"] == "call_abc"
        assert restored["delta"] == '{"q": "test"}'
