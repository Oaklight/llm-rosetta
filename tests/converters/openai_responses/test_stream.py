"""
OpenAI Responses API stream converter unit tests.
"""

from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.converters.openai_responses.stream_context import (
    OpenAIResponsesStreamContext,
)
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

    def test_response_in_progress_passthrough(self):
        """response.in_progress is preserved as a passthrough event."""
        event = {"type": "response.in_progress", "response": {}}
        events = self.converter.stream_response_from_provider(event)
        assert events == [
            {
                "type": "provider_passthrough",
                "provider": "openai_responses",
                "payload": event,
            }
        ]

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
        """ReasoningDeltaEvent → response.reasoning_summary_text.delta with lifecycle."""
        event = cast(
            ReasoningDeltaEvent,
            {"type": "reasoning_delta", "reasoning": "thinking..."},
        )
        result = self.converter.stream_response_to_provider(event)
        assert isinstance(result, list)
        # Without context: just the delta event
        delta = result[-1]
        assert delta["type"] == "response.reasoning_summary_text.delta"
        assert delta["delta"] == "thinking..."

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
        assert result["item"]["id"] == "call_abc"
        assert result["item"]["type"] == "function_call"
        assert result["item"]["call_id"] == "call_abc"
        assert result["item"]["name"] == "search"
        assert result["output_index"] == 1

    def test_tool_call_start_no_index(self):
        """ToolCallStartEvent without tool_call_index defaults output_index to 0."""
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "call_abc",
                "tool_name": "search",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["item"]["id"] == "call_abc"
        assert result["output_index"] == 0

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
        assert result["item_id"] == "call_abc"
        assert result["delta"] == '{"city":'
        assert result["output_index"] == 1

    def test_tool_call_delta_no_index(self):
        """ToolCallDeltaEvent without tool_call_index defaults output_index to 0."""
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_abc",
                "arguments_delta": "{}",
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["item_id"] == "call_abc"
        assert result["output_index"] == 0

    def test_tool_call_delta_empty_id_resolved_by_index(self):
        """ToolCallDeltaEvent with empty tool_call_id resolved via context index."""
        ctx = OpenAIResponsesStreamContext()
        # Simulate a prior tool_call_start that registered the call
        ctx.register_tool_call("call_abc", "get_weather")
        ctx.register_tool_call_item("call_abc", "call_abc")

        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "",
                "arguments_delta": '{"city":"Beijing"}',
                "tool_call_index": 0,
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "response.function_call_arguments.delta"
        assert result["item_id"] == "call_abc"
        assert result["delta"] == '{"city":"Beijing"}'
        # Verify args were accumulated under the resolved call_id
        assert ctx.get_tool_call_args("call_abc") == '{"city":"Beijing"}'

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

    def test_finish_event_content_filter(self):
        """FinishEvent with 'content_filter' → response.completed with status 'incomplete'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "content_filter"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "incomplete"
        assert completed["response"]["incomplete_details"] == {
            "reason": "content_filter"
        }

    def test_finish_event_tool_calls(self):
        """FinishEvent with 'tool_calls' → response.completed with status 'completed'."""
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
        )
        results = cast(
            list[dict[str, Any]], self.converter.stream_response_to_provider(event)
        )
        completed = next(r for r in results if r["type"] == "response.completed")
        assert completed["response"]["status"] == "completed"

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
        result = self.converter.stream_response_to_provider(events[0])
        assert isinstance(result, list)
        delta = result[-1]
        assert delta["type"] == "response.reasoning_summary_text.delta"
        assert delta["delta"] == "step 1"

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
        assert restored["item"]["id"] == "call_abc"
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
        assert restored["item_id"] == "call_abc"
        assert restored["delta"] == '{"q": "test"}'


class TestStreamResponseFromProviderWithContext:
    """Tests for stream_response_from_provider with StreamContext."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_response_created_emits_stream_start(self):
        """response.created with context emits StreamStartEvent."""
        ctx = OpenAIResponsesStreamContext()
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
        # IR stores the stem (prefix stripped)
        assert events[0]["response_id"] == "abc123"
        assert events[0]["model"] == "gpt-4o"
        assert events[0]["created"] == 1700000000
        assert ctx.response_id == "abc123"
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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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

    def test_output_item_added_message_emits_no_events(self):
        """response.output_item.added (message) with context produces no IR events.

        The actual content block is signaled by response.content_part.added.
        """
        ctx = OpenAIResponsesStreamContext()
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
        assert len(events) == 0

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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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

    def test_output_item_done_message_no_events(self):
        """response.output_item.done (message) with context produces no IR events.

        The actual content block end is signaled by response.content_part.done.
        """
        ctx = OpenAIResponsesStreamContext()
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
        assert len(events) == 0

    def test_text_delta_unchanged_with_context(self):
        """Text delta behavior is unchanged when context is provided."""
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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
        ctx = OpenAIResponsesStreamContext()
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

    def test_finish_with_context_defers_response_completed(self):
        """FinishEvent with context defers response.completed to StreamEndEvent."""
        ctx = OpenAIResponsesStreamContext()
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
        # FinishEvent should NOT emit response.completed with context
        assert not any(r.get("type") == "response.completed" for r in results)
        # But it should store the response in context for later
        assert ctx.pending_response is not None
        assert ctx.pending_response["usage"]["input_tokens"] == 10

        # StreamEndEvent emits the deferred response.completed
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result["type"] == "response.completed"
        assert end_result["response"]["status"] == "completed"
        assert end_result["response"]["usage"]["input_tokens"] == 10
        assert end_result["response"]["usage"]["output_tokens"] == 5
        assert end_result["response"]["usage"]["total_tokens"] == 15

    def test_finish_with_context_no_pending_usage(self):
        """FinishEvent with context but no pending usage omits usage field."""
        ctx = OpenAIResponsesStreamContext()
        ctx.mark_started()
        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        # FinishEvent defers response.completed
        assert not any(r.get("type") == "response.completed" for r in results)
        assert ctx.pending_response is not None
        assert "usage" not in ctx.pending_response

        # StreamEndEvent emits without usage
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result["type"] == "response.completed"
        assert end_result["response"]["status"] == "completed"
        assert "usage" not in end_result["response"]

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
        """With context, UsageEvent + FinishEvent + StreamEndEvent produce one response.completed."""
        ctx = OpenAIResponsesStreamContext()
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

        # Second: FinishEvent → deferred, no response.completed yet
        finish_event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        finish_results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(finish_event, context=ctx),
        )
        assert not any(r.get("type") == "response.completed" for r in finish_results)
        assert ctx.pending_response is not None
        assert ctx.pending_response["usage"]["input_tokens"] == 20

        # Third: StreamEndEvent → emits deferred response.completed
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result["type"] == "response.completed"
        assert end_result["response"]["usage"]["input_tokens"] == 20
        assert end_result["response"]["usage"]["output_tokens"] == 10

    def test_full_stream_sequence_with_context(self):
        """Full stream sequence produces correct events with no duplicates."""
        ctx = OpenAIResponsesStreamContext()

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

        # 2. ContentBlockStartEvent — with context, first text block emits
        # both output_item.added and content_part.added as a list.
        block_start_results = self.converter.stream_response_to_provider(
            cast(
                ContentBlockStartEvent,
                {
                    "type": "content_block_start",
                    "block_index": 0,
                    "block_type": "text",
                },
            ),
            context=ctx,
        )
        assert isinstance(block_start_results, list)
        assert len(block_start_results) == 2
        assert block_start_results[0]["type"] == "response.output_item.added"
        assert block_start_results[1]["type"] == "response.content_part.added"

        # 3. TextDeltaEvent — output_item already emitted by ContentBlockStart,
        #    so this should return a single delta (not a list).
        text_results = self.converter.stream_response_to_provider(
            cast(TextDeltaEvent, {"type": "text_delta", "text": "Hello"}),
            context=ctx,
        )
        if isinstance(text_results, list):
            text_delta = next(
                r for r in text_results if r["type"] == "response.output_text.delta"
            )
        else:
            text_delta = text_results
        assert text_delta["type"] == "response.output_text.delta"

        # 4. ContentBlockEndEvent — returns output_text.done + content_part.done
        block_end_results = self.converter.stream_response_to_provider(
            cast(
                ContentBlockEndEvent,
                {"type": "content_block_end", "block_index": 0},
            ),
            context=ctx,
        )
        assert isinstance(block_end_results, list)
        assert len(block_end_results) == 2
        assert block_end_results[0]["type"] == "response.output_text.done"
        assert block_end_results[1]["type"] == "response.content_part.done"

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

        # 6. FinishEvent → deferred, no response.completed yet
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
        assert not any(r.get("type") == "response.completed" for r in finish_results)
        assert ctx.pending_response is not None

        # 7. StreamEndEvent → emits deferred response.completed
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result["type"] == "response.completed"
        assert end_result["response"]["usage"]["input_tokens"] == 10

        # Verify: only ONE response.completed was produced in the entire sequence
        all_results: list[Any] = [
            start_result,
            usage_result,
            end_result,
        ]
        # Flatten list results
        if isinstance(block_start_results, list):
            all_results.extend(block_start_results)
        else:
            all_results.append(block_start_results)
        if isinstance(block_end_results, list):
            all_results.extend(block_end_results)
        else:
            all_results.append(block_end_results)
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

    def test_cross_chunk_usage_after_finish(self):
        """UsageEvent arriving after FinishEvent (OpenAI Chat pattern) is merged."""
        ctx = OpenAIResponsesStreamContext()
        ctx.mark_started()

        # 1. FinishEvent arrives first (no pending_usage yet)
        finish_event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "stop"}},
        )
        finish_results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(finish_event, context=ctx),
        )
        assert not any(r.get("type") == "response.completed" for r in finish_results)
        assert ctx.pending_response is not None
        assert "usage" not in ctx.pending_response

        # 2. UsageEvent arrives in a separate chunk
        usage_event = cast(
            UsageEvent,
            {
                "type": "usage",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                },
            },
        )
        usage_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(usage_event, context=ctx),
        )
        assert usage_result == {}

        # 3. StreamEndEvent merges the late usage into deferred response
        end_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(
                cast(StreamEndEvent, {"type": "stream_end"}),
                context=ctx,
            ),
        )
        assert end_result["type"] == "response.completed"
        assert end_result["response"]["usage"]["input_tokens"] == 50
        assert end_result["response"]["usage"]["output_tokens"] == 25
        assert end_result["response"]["usage"]["total_tokens"] == 75


class TestCustomToolCallStreaming:
    """Tests for custom_tool_call streaming events."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    # --- From provider ---

    def test_custom_tool_call_output_item_added(self):
        """response.output_item.added with custom_tool_call produces ToolCallStartEvent."""
        ctx = OpenAIResponsesStreamContext()
        event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "ctc_001",
                "type": "custom_tool_call",
                "call_id": "call_custom_1",
                "name": "my_tool",
                "input": "",
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert events[0]["tool_call_id"] == "call_custom_1"
        assert events[0]["tool_name"] == "my_tool"
        assert events[0]["tool_type"] == "custom"
        assert events[0]["tool_call_index"] == 0
        # Context should register the custom tool type
        assert ctx.get_tool_name("call_custom_1") == "my_tool"
        assert ctx.get_tool_type("call_custom_1") == "custom"
        assert ctx.get_tool_call_item_id("call_custom_1") == "ctc_001"

    def test_custom_tool_call_input_delta(self):
        """response.custom_tool_call_input.delta produces ToolCallDeltaEvent."""
        ctx = OpenAIResponsesStreamContext()
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        event = {
            "type": "response.custom_tool_call_input.delta",
            "call_id": "call_custom_1",
            "delta": "hello ",
            "output_index": 0,
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"
        assert events[0]["tool_call_id"] == "call_custom_1"
        assert events[0]["arguments_delta"] == "hello "
        assert ctx.get_tool_call_args("call_custom_1") == "hello "

    def test_custom_tool_call_input_done(self):
        """response.custom_tool_call_input.done stores final input in context."""
        ctx = OpenAIResponsesStreamContext()
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        ctx.append_tool_call_args("call_custom_1", "hello ")
        event = {
            "type": "response.custom_tool_call_input.done",
            "call_id": "call_custom_1",
            "input": "hello world",
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        # Done event produces no IR events
        assert len(events) == 0
        # But the final input is stored in context
        assert ctx.get_tool_call_args("call_custom_1") == "hello world"

    def test_custom_tool_call_output_item_done(self):
        """response.output_item.done with custom_tool_call stores input in context."""
        ctx = OpenAIResponsesStreamContext()
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "custom_tool_call",
                "call_id": "call_custom_1",
                "name": "my_tool",
                "input": "final input text",
            },
        }
        self.converter.stream_response_from_provider(event, context=ctx)
        assert ctx.get_tool_call_args("call_custom_1") == "final input text"

    # --- To provider ---

    def test_tool_call_start_custom_to_p(self):
        """ToolCallStartEvent with tool_type='custom' → custom_tool_call item."""
        ctx = OpenAIResponsesStreamContext()
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "call_custom_1",
                "tool_name": "my_tool",
                "tool_type": "custom",
                "tool_call_index": 0,
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "response.output_item.added"
        assert result["item"]["type"] == "custom_tool_call"
        assert result["item"]["call_id"] == "call_custom_1"
        assert result["item"]["name"] == "my_tool"
        assert result["item"]["input"] == ""
        assert result["item"]["status"] == "in_progress"
        # Context should have registered the custom type
        assert ctx.get_tool_type("call_custom_1") == "custom"

    def test_tool_call_delta_custom_to_p(self):
        """ToolCallDeltaEvent for custom tool → custom_tool_call_input.delta."""
        ctx = OpenAIResponsesStreamContext()
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        ctx.register_tool_call_item("call_custom_1", "call_custom_1")
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_custom_1",
                "arguments_delta": "hello ",
                "tool_call_index": 0,
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "response.custom_tool_call_input.delta"
        assert result["delta"] == "hello "

    def test_tool_call_delta_function_to_p(self):
        """ToolCallDeltaEvent for function tool → function_call_arguments.delta."""
        ctx = OpenAIResponsesStreamContext()
        ctx.register_tool_call("call_fn_1", "get_weather", "function")
        ctx.register_tool_call_item("call_fn_1", "call_fn_1")
        event = cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_fn_1",
                "arguments_delta": '{"city":',
            },
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["type"] == "response.function_call_arguments.delta"

    def test_finish_with_custom_tool_call(self):
        """FinishEvent emits custom_tool_call done events and output items."""
        ctx = OpenAIResponsesStreamContext()
        ctx.mark_started()
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        ctx.register_tool_call_item("call_custom_1", "ctc_001")
        ctx.set_tool_call_args("call_custom_1", "hello world")

        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
        )
        results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event, context=ctx),
        )

        # Should have custom_tool_call_input.done and output_item.done
        done_events = [
            r
            for r in results
            if r.get("type") == "response.custom_tool_call_input.done"
        ]
        assert len(done_events) == 1
        assert done_events[0]["input"] == "hello world"
        assert done_events[0]["item_id"] == "ctc_001"

        item_done_events = [
            r for r in results if r.get("type") == "response.output_item.done"
        ]
        assert len(item_done_events) == 1
        assert item_done_events[0]["item"]["type"] == "custom_tool_call"
        assert item_done_events[0]["item"]["input"] == "hello world"
        assert item_done_events[0]["item"]["name"] == "my_tool"
        assert item_done_events[0]["item"]["status"] == "completed"

        # Deferred response should have custom_tool_call output
        assert ctx.pending_response is not None
        output = ctx.pending_response["output"]
        assert len(output) == 1
        assert output[0]["type"] == "custom_tool_call"
        assert output[0]["input"] == "hello world"

    def test_finish_with_mixed_tool_calls(self):
        """FinishEvent with both function and custom tool calls emits correct types."""
        ctx = OpenAIResponsesStreamContext()
        ctx.mark_started()
        ctx.register_tool_call("call_fn_1", "get_weather", "function")
        ctx.register_tool_call_item("call_fn_1", "fc_001")
        ctx.set_tool_call_args("call_fn_1", '{"city": "NYC"}')
        ctx.register_tool_call("call_custom_1", "my_tool", "custom")
        ctx.register_tool_call_item("call_custom_1", "ctc_001")
        ctx.set_tool_call_args("call_custom_1", "do something")

        event = cast(
            FinishEvent,
            {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
        )
        results = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event, context=ctx),
        )

        # Function call done events
        fn_done = [
            r
            for r in results
            if r.get("type") == "response.function_call_arguments.done"
        ]
        assert len(fn_done) == 1
        assert fn_done[0]["arguments"] == '{"city": "NYC"}'

        # Custom tool call done events
        custom_done = [
            r
            for r in results
            if r.get("type") == "response.custom_tool_call_input.done"
        ]
        assert len(custom_done) == 1
        assert custom_done[0]["input"] == "do something"

        # Output items
        item_done = [r for r in results if r.get("type") == "response.output_item.done"]
        assert len(item_done) == 2
        assert item_done[0]["item"]["type"] == "function_call"
        assert item_done[1]["item"]["type"] == "custom_tool_call"

    # --- Round-trip ---

    def test_custom_tool_call_stream_round_trip(self):
        """Custom tool call round-trips through streaming: provider → IR → provider."""
        ctx_from = OpenAIResponsesStreamContext()
        ctx_to = OpenAIResponsesStreamContext()

        # 1. output_item.added
        added_event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "ctc_001",
                "type": "custom_tool_call",
                "call_id": "call_custom_1",
                "name": "my_tool",
                "input": "",
            },
        }
        ir_events = cast(
            list[Any],
            self.converter.stream_response_from_provider(added_event, context=ctx_from),
        )
        assert len(ir_events) == 1
        restored = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(ir_events[0], context=ctx_to),
        )
        assert restored["item"]["type"] == "custom_tool_call"
        assert restored["item"]["name"] == "my_tool"
        assert restored["item"]["input"] == ""

        # 2. custom_tool_call_input.delta
        delta_event = {
            "type": "response.custom_tool_call_input.delta",
            "call_id": "call_custom_1",
            "delta": "search query",
            "output_index": 0,
        }
        ir_events = cast(
            list[Any],
            self.converter.stream_response_from_provider(delta_event, context=ctx_from),
        )
        assert len(ir_events) == 1
        restored = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(ir_events[0], context=ctx_to),
        )
        assert restored["type"] == "response.custom_tool_call_input.delta"
        assert restored["delta"] == "search query"


class TestFunctionCallItemIdPreservation:
    """Tests for preserving Responses function_call item id (fc_ prefix)
    through streaming round-trip, distinct from call_id (call_ prefix).

    Covers issue #401.
    """

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    # --- from_p: ToolCallStartEvent carries responses_item_id ---

    def test_from_p_carries_item_id_in_provider_metadata(self):
        """output_item.added with distinct id/call_id stores responses_item_id."""
        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_abc123",
                "call_id": "call_xyz789",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "tool_call_start"
        assert event["tool_call_id"] == "call_xyz789"
        pm = event.get("provider_metadata", {})
        assert pm.get("responses_item_id") == "fc_abc123"

    def test_from_p_no_metadata_when_ids_equal(self):
        """output_item.added where id == call_id does not set provider_metadata."""
        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_abc",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "",
                "status": "in_progress",
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert "provider_metadata" not in events[0]

    def test_from_p_no_metadata_when_id_empty(self):
        """output_item.added with empty id does not set provider_metadata."""
        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "",
                "call_id": "call_abc",
                "name": "search",
                "arguments": "",
                "status": "in_progress",
            },
        }
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert "provider_metadata" not in events[0]

    # --- to_p: restores fc_ item id from provider_metadata ---

    def test_to_p_restores_fc_item_id(self):
        """ToolCallStartEvent with provider_metadata restores fc_ item id."""
        ctx = OpenAIResponsesStreamContext()
        ctx.response_id = "resp_001"
        ctx.model = "gpt-4"
        ctx.mark_started()

        event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="call_xyz789",
            tool_name="get_weather",
            provider_metadata={"responses_item_id": "fc_abc123"},
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["item"]["id"] == "fc_abc123"
        assert result["item"]["call_id"] == "call_xyz789"

    def test_to_p_fallback_without_metadata(self):
        """ToolCallStartEvent without provider_metadata uses call_id as id."""
        ctx = OpenAIResponsesStreamContext()
        ctx.response_id = "resp_001"
        ctx.model = "gpt-4"
        ctx.mark_started()

        event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="call_xyz789",
            tool_name="get_weather",
        )
        result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(event, context=ctx),
        )
        assert result["item"]["id"] == "call_xyz789"
        assert result["item"]["call_id"] == "call_xyz789"

    # --- Full round-trip ---

    def test_round_trip_preserves_fc_item_id(self):
        """Full streaming round-trip preserves distinct fc_ item id."""
        ctx_from = OpenAIResponsesStreamContext()
        ctx_to = OpenAIResponsesStreamContext()
        ctx_to.response_id = "resp_001"
        ctx_to.model = "gpt-4"
        ctx_to.mark_started()

        added_chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_abc123",
                "call_id": "call_xyz789",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }
        ir_events = cast(
            list[Any],
            self.converter.stream_response_from_provider(added_chunk, context=ctx_from),
        )
        restored = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(ir_events[0], context=ctx_to),
        )
        assert restored["item"]["id"] == "fc_abc123"
        assert restored["item"]["call_id"] == "call_xyz789"

    def test_done_and_completed_use_fc_item_id(self):
        """Done and completed events use the fc_ item id from start event."""
        ctx = OpenAIResponsesStreamContext()
        ctx.response_id = "resp_001"
        ctx.model = "gpt-4"
        ctx.mark_started()

        # Register tool call with distinct ids via start event
        start_event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="call_xyz789",
            tool_name="get_weather",
            provider_metadata={"responses_item_id": "fc_abc123"},
        )
        self.converter.stream_response_to_provider(start_event, context=ctx)

        # Delta should use fc_ item_id
        delta_event = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id="call_xyz789",
            arguments_delta='{"city": "London"}',
        )
        delta_result = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(delta_event, context=ctx),
        )
        assert delta_result["item_id"] == "fc_abc123"

        # Finish triggers done events
        finish_event = FinishEvent(
            type="finish",
            finish_reason={"reason": "tool_calls"},
        )
        results = self.converter.stream_response_to_provider(finish_event, context=ctx)
        result_list = results if isinstance(results, list) else [results]

        # output_item.done should have fc_ id
        done_events = [
            e
            for e in result_list
            if isinstance(e, dict) and e.get("type") == "response.output_item.done"
        ]
        assert len(done_events) == 1
        assert done_events[0]["item"]["id"] == "fc_abc123"
        assert done_events[0]["item"]["call_id"] == "call_xyz789"

        # response.completed output should have fc_ id
        assert ctx.pending_response is not None
        output = ctx.pending_response.get("output", [])
        fc_items = [o for o in output if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["id"] == "fc_abc123"
        assert fc_items[0]["call_id"] == "call_xyz789"


class TestUnifiedOutputIndex:
    """Tests for unified output_index management (#418)."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def _run_stream(self, ir_events):
        """Run a sequence of IR events through the converter and collect output."""
        ctx = StreamContext()
        ctx.response_id = "resp_test"
        ctx.model = "test-model"
        results = []
        for ev in ir_events:
            out = self.converter.stream_response_to_provider(ev, context=ctx)
            if isinstance(out, list):
                results.extend(out)
            elif isinstance(out, dict) and out:
                results.append(out)
        return results

    def _get_output_indices(self, events):
        """Extract (type, output_index) from output_item.added events."""
        return [
            (ev["item"]["type"], ev["output_index"])
            for ev in events
            if ev.get("type") == "response.output_item.added"
        ]

    def test_message_only(self):
        """[message] → output_index [0]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "text_delta", "text": "hi"},
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [("message", 0)]

    def test_reasoning_then_message(self):
        """[reasoning, message] → output_index [0, 1]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "think"},
                {"type": "text_delta", "text": "hi"},
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [("reasoning", 0), ("message", 1)]

    def test_reasoning_then_tool_call(self):
        """[reasoning, fc] → output_index [0, 1]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "think"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_1",
                    "tool_name": "search",
                    "tool_call_index": 0,
                },
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [("reasoning", 0), ("function_call", 1)]

    def test_reasoning_then_parallel_tools(self):
        """[reasoning, fc, fc, fc] → output_index [0, 1, 2, 3]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "think"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_1",
                    "tool_name": "fn1",
                    "tool_call_index": 0,
                },
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_2",
                    "tool_name": "fn2",
                    "tool_call_index": 1,
                },
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_3",
                    "tool_name": "fn3",
                    "tool_call_index": 2,
                },
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [
            ("reasoning", 0),
            ("function_call", 1),
            ("function_call", 2),
            ("function_call", 3),
        ]

    def test_reasoning_message_then_tools(self):
        """[reasoning, message, fc, fc] → output_index [0, 1, 2, 3]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "think"},
                {"type": "text_delta", "text": "hi"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_1",
                    "tool_name": "fn1",
                    "tool_call_index": 0,
                },
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_2",
                    "tool_name": "fn2",
                    "tool_call_index": 1,
                },
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [
            ("reasoning", 0),
            ("message", 1),
            ("function_call", 2),
            ("function_call", 3),
        ]

    def test_no_reasoning_backward_compat(self):
        """[message, fc, fc] → output_index [0, 1, 2] (no reasoning)."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "text_delta", "text": "hi"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_1",
                    "tool_name": "fn1",
                    "tool_call_index": 0,
                },
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_2",
                    "tool_name": "fn2",
                    "tool_call_index": 1,
                },
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [
            ("message", 0),
            ("function_call", 1),
            ("function_call", 2),
        ]

    def test_tools_only_no_message(self):
        """[fc, fc] → output_index [0, 1]."""
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_1",
                    "tool_name": "fn1",
                    "tool_call_index": 0,
                },
                {
                    "type": "tool_call_start",
                    "tool_call_id": "call_2",
                    "tool_name": "fn2",
                    "tool_call_index": 1,
                },
            ]
        )
        indices = self._get_output_indices(events)
        assert indices == [
            ("function_call", 0),
            ("function_call", 1),
        ]


class TestReasoningStreamingLifecycle:
    """Tests for reasoning streaming lifecycle events (#408)."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def _run_stream(self, ir_events):
        ctx = StreamContext()
        ctx.response_id = "resp_test"
        ctx.model = "test-model"
        results = []
        for ev in ir_events:
            out = self.converter.stream_response_to_provider(ev, context=ctx)
            if isinstance(out, list):
                results.extend(out)
            elif isinstance(out, dict) and out:
                results.append(out)
        return results

    def test_reasoning_delta_has_lifecycle_preamble(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "step 1"},
            ]
        )
        types = [e.get("type") for e in events]
        assert "response.output_item.added" in types
        assert "response.reasoning_summary.part.added" in types
        assert "response.reasoning_summary_text.delta" in types

    def test_reasoning_item_added_has_id_and_status(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "step 1"},
            ]
        )
        added = [e for e in events if e.get("type") == "response.output_item.added"]
        assert len(added) == 1
        item = added[0]["item"]
        assert item["type"] == "reasoning"
        assert item["id"].startswith("rs_")
        assert item["status"] == "in_progress"

    def test_reasoning_delta_has_required_fields(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "step 1"},
            ]
        )
        deltas = [
            e
            for e in events
            if e.get("type") == "response.reasoning_summary_text.delta"
        ]
        assert len(deltas) == 1
        d = deltas[0]
        assert "item_id" in d
        assert d["item_id"].startswith("rs_")
        assert "output_index" in d
        assert d["summary_index"] == 0
        assert d["delta"] == "step 1"

    def test_reasoning_done_events_at_finish(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "step 1"},
                {"type": "text_delta", "text": "result"},
                {
                    "type": "finish",
                    "finish_reason": {"reason": "stop"},
                },
            ]
        )
        types = [e.get("type") for e in events]
        assert "response.reasoning_summary_text.done" in types
        assert "response.reasoning_summary.part.done" in types
        # output_item.done should appear for both reasoning and message
        done_events = [
            e for e in events if e.get("type") == "response.output_item.done"
        ]
        done_types = [e["item"]["type"] for e in done_events]
        assert "reasoning" in done_types
        assert "message" in done_types

    def test_reasoning_in_response_completed_output(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "thinking"},
                {"type": "text_delta", "text": "answer"},
                {
                    "type": "finish",
                    "finish_reason": {"reason": "stop"},
                },
                {"type": "stream_end"},
            ]
        )
        completed = [e for e in events if e.get("type") == "response.completed"]
        assert len(completed) == 1
        output = completed[0]["response"]["output"]
        output_types = [item["type"] for item in output]
        assert "reasoning" in output_types
        assert "message" in output_types

    def test_second_reasoning_delta_no_duplicate_preamble(self):
        events = self._run_stream(
            [
                {"type": "stream_start", "response_id": "resp_1", "model": "m"},
                {"type": "reasoning_delta", "reasoning": "step 1"},
                {"type": "reasoning_delta", "reasoning": "step 2"},
            ]
        )
        added = [e for e in events if e.get("type") == "response.output_item.added"]
        assert len(added) == 1
        deltas = [
            e
            for e in events
            if e.get("type") == "response.reasoning_summary_text.delta"
        ]
        assert len(deltas) == 2
        assert deltas[0]["item_id"] == deltas[1]["item_id"]


class TestStreamingResponseIdPrefix:
    """Tests for response ID prefix handling in streaming path."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_from_provider_strips_prefix_in_stream_start(self):
        """stream_response_from_provider strips resp_ prefix."""
        ctx = OpenAIResponsesStreamContext()
        event = {
            "type": "response.created",
            "response": {
                "id": "resp_stream_test_123",
                "model": "gpt-4o",
                "created_at": 1700000000,
            },
        }
        events = cast(
            list[Any],
            self.converter.stream_response_from_provider(event, context=ctx),
        )
        assert events[0]["response_id"] == "stream_test_123"
        assert ctx.response_id == "stream_test_123"

    def test_to_provider_adds_prefix_in_response_created(self):
        """stream_response_to_provider adds resp_ prefix."""
        ctx = OpenAIResponsesStreamContext()
        ir_event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "abc123",
            "model": "gpt-4o",
            "created": 1700000000,
        }
        result = self.converter.stream_response_to_provider(ir_event, context=ctx)
        assert result["response"]["id"] == "resp_abc123"
        # Context stores the stem (without prefix)
        assert ctx.response_id == "abc123"

    def test_finish_response_has_prefixed_id(self):
        """response.completed event includes prefixed response ID."""
        ctx = OpenAIResponsesStreamContext()
        # Simulate stream start
        start_event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "finish_test",
            "model": "gpt-4o",
            "created": 1700000000,
        }
        self.converter.stream_response_to_provider(start_event, context=ctx)

        # Send a text delta so there's content
        delta_event: TextDeltaEvent = {"type": "text_delta", "text": "Hello"}
        self.converter.stream_response_to_provider(delta_event, context=ctx)

        # Finish
        finish_event: FinishEvent = {
            "type": "finish",
            "finish_reason": {"reason": "stop"},
        }
        self.converter.stream_response_to_provider(
            finish_event, context=ctx
        )

        # stream_end triggers the deferred response.completed
        end_event: StreamEndEvent = {"type": "stream_end"}
        end_result = self.converter.stream_response_to_provider(
            end_event, context=ctx
        )
        # Find response.completed event
        completed = None
        for r in [end_result] if isinstance(end_result, dict) else end_result:
            if isinstance(r, dict) and r.get("type") == "response.completed":
                completed = r
                break
        assert completed is not None
        assert completed["response"]["id"] == "resp_finish_test"

    def test_finish_response_has_completed_at(self):
        """response.completed event includes completed_at timestamp."""
        ctx = OpenAIResponsesStreamContext()
        start_event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "ts_test",
            "model": "gpt-4o",
            "created": 1700000000,
        }
        self.converter.stream_response_to_provider(start_event, context=ctx)
        delta_event: TextDeltaEvent = {"type": "text_delta", "text": "Hi"}
        self.converter.stream_response_to_provider(delta_event, context=ctx)
        finish_event: FinishEvent = {
            "type": "finish",
            "finish_reason": {"reason": "stop"},
        }
        self.converter.stream_response_to_provider(
            finish_event, context=ctx
        )

        # stream_end triggers the deferred response.completed
        end_event: StreamEndEvent = {"type": "stream_end"}
        end_result = self.converter.stream_response_to_provider(
            end_event, context=ctx
        )
        completed = None
        for r in [end_result] if isinstance(end_result, dict) else end_result:
            if isinstance(r, dict) and r.get("type") == "response.completed":
                completed = r
                break
        assert completed is not None
        assert "completed_at" in completed["response"]
        assert isinstance(completed["response"]["completed_at"], int)
        assert completed["response"]["completed_at"] > 0
