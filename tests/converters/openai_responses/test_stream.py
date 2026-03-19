"""
OpenAI Responses API stream converter unit tests.
"""

from typing import Any, cast

from llm_rosetta.converters.base.stream_context import StreamContext
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_text_delta_empty_string(self):
        """Empty text delta still produces an event."""
        event = {
            "type": "response.output_text.delta",
            "delta": "",
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
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
        self.converter = OpenAIResponsesConverter()

    def test_text_delta(self):
        """TextDeltaEvent → response.output_text.delta."""
        event = cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.output_text.delta"
        assert result["delta"] == "Hello"

    def test_reasoning_delta(self):
        """ReasoningDeltaEvent → response.reasoning_summary_text.delta."""
        event = cast(
            ReasoningDeltaEvent,
            {"type": "reasoning_delta", "reasoning": "thinking..."},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.reasoning_summary_text.delta"
        assert result["delta"] == "thinking..."

    def test_tool_call_start(self):
        """ToolCallStartEvent → response.output_item.added."""
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "call_abc",
                "tool_name": "search",
                "tool_call_index": 1,
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.output_item.added"
        assert result["item"]["type"] == "function_call"
        assert result["item"]["call_id"] == "call_abc"
        assert result["item"]["name"] == "search"
        assert result["output_index"] == 1

    def test_tool_call_start_no_index(self):
        """ToolCallStartEvent without tool_call_index omits output_index."""
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "call_abc",
                "tool_name": "search",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert "output_index" not in result

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → response.function_call_arguments.delta."""
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_abc",
                "arguments_delta": '{"city":',
                "tool_call_index": 1,
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.function_call_arguments.delta"
        assert result["call_id"] == "call_abc"
        assert result["delta"] == '{"city":'
        assert result["output_index"] == 1

    def test_tool_call_delta_no_index(self):
        """ToolCallDeltaEvent without tool_call_index omits output_index."""
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_abc",
                "arguments_delta": "{}",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert "output_index" not in result

    def test_finish_event_stop(self):
        """FinishEvent with 'stop' → response.completed with status 'completed'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "completed"

    def test_finish_event_length(self):
        """FinishEvent with 'length' → response.completed with status 'incomplete'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "length"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "incomplete"
        assert (
            completed["response"]["incomplete_details"]["reason"] == "max_output_tokens"
        )

    def test_finish_event_error(self):
        """FinishEvent with 'error' → response.completed with status 'failed'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "error"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "failed"

    def test_usage_event(self):
        """UsageEvent → response.completed with usage."""
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
        assert result["type"] == "response.completed"
        assert result["response"]["usage"]["input_tokens"] == 10
        assert result["response"]["usage"]["output_tokens"] == 5
        assert result["response"]["usage"]["total_tokens"] == 15

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = cast(TextDeltaEvent, {"type": "unknown_event"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
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
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["type"] == "response.output_text.delta"
        assert restored["delta"] == "Hello"

    def test_reasoning_delta_round_trip(self):
        """Reasoning delta round-trip preserves content."""
        original = {
            "type": "response.reasoning_summary_text.delta",
            "delta": "step 1",
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
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
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
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
        events = cast(list[Any], self.converter.stream_response_from_provider(original))
        restored = cast(
            dict[str, Any], self.converter.stream_response_to_provider(events[0])
        )
        assert restored["call_id"] == "call_abc"
        assert restored["delta"] == '{"q": "test"}'


class TestStreamResponseFromProviderWithContext:
    """Tests for stream_response_from_provider with StreamContext."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_response_created_emits_stream_start(self):
        """response.created with context emits StreamStartEvent."""
        ctx = StreamContext()
        event = {
            "type": "response.created",
            "response": {
                "id": "resp_abc123",
                "model": "gpt-4o",
                "created_at": 1700000000,
                "status": "in_progress",
                "output": [],
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "stream_start"
        assert events[0]["response_id"] == "resp_abc123"
        assert events[0]["model"] == "gpt-4o"
        assert events[0]["created"] == 1700000000
        assert ctx.response_id == "resp_abc123"
        assert ctx.model == "gpt-4o"
        assert ctx.is_started is True

    def test_response_created_without_context_no_events(self):
        """response.created without context produces no events (backward compat)."""
        event = {
            "type": "response.created",
            "response": {
                "id": "resp_abc123",
                "model": "gpt-4o",
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_response_completed_emits_stream_end(self):
        """response.completed with context emits StreamEndEvent after other events."""
        ctx = StreamContext()
        ctx.mark_started()
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
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        types = [e["type"] for e in events]
        assert "finish" in types
        assert "usage" in types
        assert "stream_end" in types
        # StreamEndEvent must be last
        assert types[-1] == "stream_end"
        assert ctx.is_ended is True

    def test_response_failed_emits_stream_end(self):
        """response.failed with context emits StreamEndEvent after FinishEvent."""
        ctx = StreamContext()
        ctx.mark_started()
        event = {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"message": "Something went wrong"},
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        types = [e["type"] for e in events]
        assert types == ["finish", "stream_end"]
        assert events[0]["finish_reason"]["reason"] == "error"
        assert ctx.is_ended is True

    def test_output_item_added_function_call_registers_tool(self):
        """response.output_item.added (function_call) registers tool in context."""
        ctx = StreamContext()
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
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert ctx.get_tool_name("call_abc") == "get_weather"

    def test_output_item_added_message_emits_content_block_start(self):
        """response.output_item.added (message) with context emits ContentBlockStartEvent."""
        ctx = StreamContext()
        event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_start"
        assert events[0]["block_type"] == "text"
        assert events[0]["block_index"] == 0

    def test_output_item_added_message_without_context_no_events(self):
        """response.output_item.added (message) without context produces no events."""
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

    def test_content_part_added_emits_content_block_start(self):
        """response.content_part.added with context emits ContentBlockStartEvent."""
        ctx = StreamContext()
        event = {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": ""},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_start"
        assert events[0]["block_type"] == "text"
        assert events[0]["block_index"] == 0

    def test_content_part_added_summary_text(self):
        """response.content_part.added with summary_text maps to thinking block type."""
        ctx = StreamContext()
        event = {
            "type": "response.content_part.added",
            "part": {"type": "summary_text", "text": ""},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["block_type"] == "thinking"

    def test_content_part_added_without_context_no_events(self):
        """response.content_part.added without context produces no events."""
        event = {
            "type": "response.content_part.added",
            "part": {"type": "output_text", "text": ""},
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_content_part_done_emits_content_block_end(self):
        """response.content_part.done with context emits ContentBlockEndEvent."""
        ctx = StreamContext()
        ctx.next_block_index()  # set to 0
        event = {
            "type": "response.content_part.done",
            "part": {"type": "output_text", "text": "Hello"},
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_end"
        assert events[0]["block_index"] == 0

    def test_content_part_done_without_context_no_events(self):
        """response.content_part.done without context produces no events."""
        event = {
            "type": "response.content_part.done",
            "part": {"type": "output_text", "text": "Hello"},
        }
        events = self.converter.stream_response_from_provider(event)
        assert events == []

    def test_output_item_done_message_emits_content_block_end(self):
        """response.output_item.done (message) with context emits ContentBlockEndEvent."""
        ctx = StreamContext()
        ctx.next_block_index()  # set to 0
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_end"
        assert events[0]["block_index"] == 0

    def test_text_delta_unchanged_with_context(self):
        """Text delta behavior is unchanged when context is provided."""
        ctx = StreamContext()
        event = {
            "type": "response.output_text.delta",
            "delta": "Hello",
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_response_completed_without_context_no_stream_end(self):
        """response.completed without context does not emit StreamEndEvent."""
        event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [],
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(event))
        types = [e["type"] for e in events]
        assert "stream_end" not in types
        assert "finish" in types


class TestStreamResponseToProviderWithContext:
    """Tests for stream_response_to_provider with StreamContext."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_stream_start_event(self):
        """StreamStartEvent → response.created."""
        ctx = StreamContext()
        event = cast(
            StreamStartEvent,
            {
                "type": "stream_start",
                "response_id": "resp_abc123",
                "model": "gpt-4o",
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "response.created"
        assert result["response"]["id"] == "resp_abc123"
        assert result["response"]["model"] == "gpt-4o"
        assert result["response"]["status"] == "in_progress"
        assert result["response"]["output"] == []
        assert ctx.response_id == "resp_abc123"
        assert ctx.model == "gpt-4o"
        assert ctx.is_started is True

    def test_stream_start_without_context(self):
        """StreamStartEvent without context still produces response.created."""
        event = cast(
            StreamStartEvent,
            {
                "type": "stream_start",
                "response_id": "resp_abc123",
                "model": "gpt-4o",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.created"
        assert result["response"]["id"] == "resp_abc123"

    def test_stream_end_event(self):
        """StreamEndEvent → empty dict."""
        ctx = StreamContext()
        ctx.mark_started()
        event = cast(StreamEndEvent, {"type": "stream_end"})
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result == {}
        assert ctx.is_ended is True

    def test_stream_end_without_context(self):
        """StreamEndEvent without context → empty dict."""
        event = cast(StreamEndEvent, {"type": "stream_end"})
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result == {}

    def test_content_block_start_text(self):
        """ContentBlockStartEvent (text) → response.content_part.added."""
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 0,
                "block_type": "text",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.content_part.added"
        assert result["part"]["type"] == "output_text"
        assert result["part"]["text"] == ""

    def test_content_block_start_non_text(self):
        """ContentBlockStartEvent (non-text) → empty dict."""
        event = cast(
            ContentBlockStartEvent,
            {
                "type": "content_block_start",
                "block_index": 0,
                "block_type": "thinking",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result == {}

    def test_content_block_end(self):
        """ContentBlockEndEvent → response.content_part.done."""
        event = cast(
            ContentBlockEndEvent,
            {
                "type": "content_block_end",
                "block_index": 0,
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["type"] == "response.content_part.done"
        assert result["part"]["type"] == "output_text"

    def test_usage_with_context_no_duplicate_completed(self):
        """UsageEvent with context stores usage, returns empty dict (no duplicate)."""
        ctx = StreamContext()
        ctx.mark_started()
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
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result == {}
        assert ctx.pending_usage is not None
        assert ctx.pending_usage["prompt_tokens"] == 10
        assert ctx.pending_usage["completion_tokens"] == 5

    def test_usage_without_context_backward_compat(self):
        """UsageEvent without context produces response.completed (backward compat)."""
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
        assert result["type"] == "response.completed"
        assert result["response"]["usage"]["input_tokens"] == 10

    def test_finish_with_context_merges_usage(self):
        """FinishEvent with context merges pending usage into response.completed."""
        ctx = StreamContext()
        ctx.mark_started()
        ctx.pending_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "completed"
        assert completed["response"]["usage"]["input_tokens"] == 10
        assert completed["response"]["usage"]["output_tokens"] == 5
        assert completed["response"]["usage"]["total_tokens"] == 15

    def test_finish_with_context_no_pending_usage(self):
        """FinishEvent with context but no pending usage omits usage field."""
        ctx = StreamContext()
        ctx.mark_started()
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "completed"
        assert "usage" not in completed["response"]

    def test_finish_without_context_backward_compat(self):
        """FinishEvent without context produces response.completed (backward compat)."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "completed"

    def test_no_duplicate_response_completed_with_context(self):
        """With context, UsageEvent + FinishEvent produce only one response.completed."""
        ctx = StreamContext()
        ctx.mark_started()

        # First: UsageEvent → stored in context, returns empty
        usage_event = cast(
            UsageEvent,
            {
                "type": "usage",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )
        usage_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(usage_event, context=ctx),
        )
        assert usage_result == {}

        # Second: FinishEvent → response.completed with merged usage
        finish_event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        finish_results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(finish_event, context=ctx),
        )
        finish_completed = next(
            r for r in finish_results if r["type"] == "response.completed"
        )
        assert finish_completed["response"]["usage"]["input_tokens"] == 20
        assert finish_completed["response"]["usage"]["output_tokens"] == 10

    def test_full_stream_sequence_with_context(self):
        """Full stream sequence produces correct events with no duplicates."""
        ctx = StreamContext()

        # 1. StreamStartEvent
        start_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(
                    StreamStartEvent,
                    {
                        "type": "stream_start",
                        "response_id": "resp_123",
                        "model": "gpt-4o",
                    },
                ),
                context=ctx,
            ),
        )
        assert start_result["type"] == "response.created"

        # 2. ContentBlockStartEvent
        block_start_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(
                    ContentBlockStartEvent,
                    {
                        "type": "content_block_start",
                        "block_index": 0,
                        "block_type": "text",
                    },
                ),
                context=ctx,
            ),
        )
        assert block_start_result["type"] == "response.content_part.added"

        # 3. TextDeltaEvent (first delta with context returns a list:
        #    output_item.added, content_part.added, then the delta itself)
        text_results = self.converter.stream_response_to_provider(
            cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"}),
            context=ctx,
        )
        # May be a list (first delta with context) or a single dict
        if isinstance(text_results, list):
            text_delta = next(
                r for r in text_results if r["type"] == "response.output_text.delta"
            )
        else:
            text_delta = text_results
        assert text_delta["type"] == "response.output_text.delta"

        # 4. ContentBlockEndEvent
        block_end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(
                    ContentBlockEndEvent,
                    {"type": "content_block_end", "block_index": 0},
                ),
                context=ctx,
            ),
        )
        assert block_end_result["type"] == "response.content_part.done"

        # 5. UsageEvent → stored, no output
        usage_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(
                    UsageEvent,
                    {
                        "type": "usage",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                ),
                context=ctx,
            ),
        )
        assert usage_result == {}

        # 6. FinishEvent → response.completed with merged usage
        finish_results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(
                cast(
                    FinishEvent,
                    {"type": "finish", "finish_reason": {"reason": "stop"}},
                ),
                context=ctx,
            ),
        )
        finish_completed = next(
            r for r in finish_results if r["type"] == "response.completed"
        )
        assert finish_completed["response"]["usage"]["input_tokens"] == 10

        # 7. StreamEndEvent → empty
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result == {}

        # Verify: only ONE response.completed was produced in the entire sequence
        all_results: list[Any] = [
            start_result,
            block_start_result,
            block_end_result,
            usage_result,
            end_result,
        ]
        # Flatten list results (text_results and finish_results may be lists)
        if isinstance(text_results, list):
            all_results.extend(text_results)
        else:
            all_results.append(text_results)
        all_results.extend(finish_results)
        completed_count = sum(
            1
            for r in all_results
            if isinstance(r, dict) and r.get("type") == "response.completed"
        )
        assert completed_count == 1
