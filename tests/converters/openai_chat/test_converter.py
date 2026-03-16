"""
OpenAI Chat Converter integration tests (non-stream + stream).
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.types.ir import (
    FinishEvent,
    IRRequest,
    IRResponse,
    Message,
    TextDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)


class TestOpenAIChatConverter:
    """Integration tests for OpenAIChatConverter."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    # ==================== request_to_provider ====================

    def test_request_to_provider_basic(self):
        """Test basic IRRequest -> OpenAI request."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
                ],
            },
        )
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["model"] == "gpt-4o"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"

    def test_request_to_provider_with_system_instruction(self):
        """Test IRRequest with system_instruction."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
                ],
                "system_instruction": "You are helpful.",
            },
        )
        result, _ = self.converter.request_to_provider(ir_request)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."
        assert result["messages"][1]["role"] == "user"

    def test_request_to_provider_full(self):
        """Test full IRRequest with all config options."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
                ],
                "system_instruction": "Be helpful.",
                "generation": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "stop_sequences": ["\n", "END"],
                },
                "response_format": {"type": "json_object"},
                "reasoning": {"effort": "medium"},
                "stream": {"enabled": True, "include_usage": True},
                "cache": {"key": "test-cache", "retention": "24h"},
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                        "required_parameters": [],
                        "metadata": {},
                    }
                ],
                "tool_choice": {"mode": "auto", "tool_name": ""},
                "tool_config": {"disable_parallel": True},
            },
        )
        result, warnings = self.converter.request_to_provider(ir_request)

        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 0.7
        assert result["max_completion_tokens"] == 100
        assert result["stop"] == ["\n", "END"]
        assert result["response_format"] == {"type": "json_object"}
        assert result["reasoning_effort"] == "medium"
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}
        assert result["prompt_cache_key"] == "test-cache"
        assert result["prompt_cache_retention"] == "24h"
        assert len(result["tools"]) == 1
        assert result["tool_choice"] == "auto"
        assert result["parallel_tool_calls"] is False

    def test_request_to_provider_extensions(self):
        """Test provider_extensions pass-through."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
                ],
                "provider_extensions": {"user": "test-user", "store": True},
            },
        )
        result, _ = self.converter.request_to_provider(ir_request)
        assert result["user"] == "test-user"
        assert result["store"] is True

    # ==================== request_from_provider ====================

    def test_request_from_provider_basic(self):
        """Test basic OpenAI request -> IRRequest."""
        provider_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = self.converter.request_from_provider(provider_request)
        assert result["model"] == "gpt-4o"
        assert result["system_instruction"] == "Be helpful"
        messages = list(result["messages"])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_request_from_provider_full(self):
        """Test full OpenAI request -> IRRequest."""
        provider_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_completion_tokens": 200,
            "stop": "END",
            "reasoning_effort": "high",
            "stream": True,
            "stream_options": {"include_usage": True},
            "prompt_cache_key": "k1",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        result = self.converter.request_from_provider(provider_request)
        assert result["generation"]["temperature"] == 0.5
        assert result["generation"]["max_tokens"] == 200
        assert result["reasoning"]["effort"] == "high"
        assert result["stream"]["enabled"] is True
        assert result["stream"]["include_usage"] is True
        assert result["cache"]["key"] == "k1"
        tools = list(result["tools"])
        assert len(tools) == 1
        assert result["tool_choice"]["mode"] == "any"
        assert result["tool_config"]["disable_parallel"] is True

    # ==================== response_from_provider ====================

    def test_response_from_provider(self):
        """Test OpenAI response -> IRResponse."""
        provider_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "system_fingerprint": "fp_abc",
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "response"
        assert result["model"] == "gpt-4o"
        assert len(result["choices"]) == 1
        assert list(result["choices"][0]["message"]["content"])[0]["text"] == "Hello!"
        assert result["choices"][0]["finish_reason"]["reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["system_fingerprint"] == "fp_abc"

    def test_response_from_provider_with_tool_calls(self):
        """Test response with tool calls."""
        provider_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        choice = result["choices"][0]
        assert choice["finish_reason"]["reason"] == "tool_calls"
        tc = list(choice["message"]["content"])[0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"

    def test_response_from_provider_with_usage_details(self):
        """Test response with detailed usage statistics."""
        provider_response = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["cache_read_tokens"] == 5
        assert result["usage"]["reasoning_tokens"] == 8

    # ==================== response_to_provider ====================

    def test_response_to_provider(self):
        """Test IRResponse -> OpenAI response."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-1",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello!"}],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    # ==================== messages_to_provider / messages_from_provider ====================

    def test_messages_to_provider(self):
        """Test messages_to_provider delegation."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )
        result, warnings = self.converter.messages_to_provider(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Hi"

    def test_messages_from_provider(self):
        """Test messages_from_provider delegation."""
        provider_msgs = [{"role": "user", "content": "Hello"}]
        result = self.converter.messages_from_provider(provider_msgs)
        assert len(result) == 1
        msg = cast(Any, result[0])
        assert list(msg["content"])[0]["text"] == "Hello"

    # ==================== _normalize ====================

    def test_normalize_dict(self):
        """Test _normalize with dict input."""
        data = {"key": "value"}
        assert OpenAIChatConverter._normalize(data) is data

    def test_normalize_pydantic(self):
        """Test _normalize with Pydantic-like object."""

        class MockModel:
            def model_dump(self):
                return {"model": "gpt-4o"}

        result = OpenAIChatConverter._normalize(MockModel())
        assert result == {"model": "gpt-4o"}

    def test_normalize_invalid(self):
        """Test _normalize raises on unsupported type."""
        with pytest.raises(TypeError, match="Cannot normalize"):
            OpenAIChatConverter._normalize(42)

    # ==================== Stream ====================

    def test_stream_response_from_provider_text_delta(self):
        """Test stream chunk with text delta."""
        chunk = {
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_stream_response_from_provider_tool_call_start(self):
        """Test stream chunk with tool call start."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert any(e["type"] == "tool_call_start" for e in events)
        start = [e for e in events if e["type"] == "tool_call_start"][0]
        assert start["tool_call_id"] == "call_1"
        assert start["tool_name"] == "get_weather"

    def test_stream_response_from_provider_tool_call_delta(self):
        """Test stream chunk with tool call arguments delta."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"function": {"arguments": '{"city":'}}]},
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert any(e["type"] == "tool_call_delta" for e in events)

    def test_stream_response_from_provider_finish(self):
        """Test stream chunk with finish reason."""
        chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "finish"
        assert events[0]["finish_reason"]["reason"] == "stop"

    def test_stream_response_from_provider_usage(self):
        """Test stream chunk with usage."""
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
        assert events[0]["usage"]["total_tokens"] == 15

    def test_stream_response_to_provider_text_delta(self):
        """Test IR text_delta -> OpenAI chunk."""
        event = cast(
            TextDeltaEvent,
            {"type": "text_delta", "text": "Hi", "choice_index": 0},
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["choices"][0]["delta"]["content"] == "Hi"

    def test_stream_response_to_provider_tool_call_start(self):
        """Test IR tool_call_start -> OpenAI chunk."""
        event = cast(
            ToolCallStartEvent,
            {
                "type": "tool_call_start",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "choice_index": 0,
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "search"

    def test_stream_response_to_provider_finish(self):
        """Test IR finish -> OpenAI chunk."""
        event = cast(
            FinishEvent,
            {
                "type": "finish",
                "finish_reason": {"reason": "stop"},
                "choice_index": 0,
            },
        )
        result = cast(dict[str, Any], self.converter.stream_response_to_provider(event))
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_stream_response_to_provider_usage(self):
        """Test IR usage -> OpenAI chunk."""
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
        assert result["usage"]["total_tokens"] == 15


class TestOpenAIChatConverterFullRoundTrip:
    """Full round-trip conversion tests."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    def test_request_round_trip(self):
        """Test IRRequest -> OpenAI -> IRRequest round-trip."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
                ],
                "system_instruction": "Be helpful.",
                "generation": {"temperature": 0.7, "max_tokens": 100},
                "tools": [
                    {
                        "type": "function",
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                        "required_parameters": [],
                        "metadata": {},
                    }
                ],
                "tool_choice": {"mode": "auto", "tool_name": ""},
            },
        )
        provider, _ = self.converter.request_to_provider(ir_request)
        restored = self.converter.request_from_provider(provider)

        assert restored["model"] == "gpt-4o"
        assert restored["system_instruction"] == "Be helpful."
        assert restored["generation"]["temperature"] == 0.7
        assert restored["generation"]["max_tokens"] == 100
        tools = list(restored["tools"])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_response_round_trip(self):
        """Test OpenAI response -> IR -> OpenAI round-trip."""
        provider_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        ir_response = self.converter.response_from_provider(provider_response)
        restored = self.converter.response_to_provider(ir_response)

        assert restored["id"] == "chatcmpl-123"
        assert restored["object"] == "chat.completion"
        assert restored["model"] == "gpt-4o"
        assert restored["choices"][0]["message"]["content"] == "Hello!"
        assert restored["choices"][0]["finish_reason"] == "stop"
        assert restored["usage"]["total_tokens"] == 15

    def test_stream_event_round_trip(self):
        """Test stream event round-trip: OpenAI chunk -> IR events -> OpenAI chunks."""
        original_chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]
        }
        events = self.converter.stream_response_from_provider(original_chunk)
        assert len(events) == 1

        restored = cast(
            dict[str, Any],
            self.converter.stream_response_to_provider(events[0]),
        )
        assert restored["choices"][0]["delta"]["content"] == "Hi"
