"""
OpenAI Responses Converter integration tests.
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir import (
    IRRequest,
    IRResponse,
    Message,
)


class TestOpenAIResponsesConverter:
    """Integration tests for OpenAIResponsesConverter."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    # ==================== request_to_provider ====================

    def test_request_to_provider_basic(self):
        """Test basic IRRequest -> OpenAI Responses request."""
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
        assert len(result["input"]) == 1
        assert result["input"][0]["type"] == "message"
        assert result["input"][0]["role"] == "user"

    def test_request_to_provider_with_system_instruction_string(self):
        """Test IRRequest with string system_instruction -> instructions."""
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
        assert result["instructions"] == "You are helpful."

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
        assert result["instructions"] == "Be helpful."
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100
        assert result["text"] == {"type": "json_object"}
        assert result["reasoning"] == {"effort": "medium"}
        assert result["stream"] is True
        # Responses API does NOT support stream_options (Chat-only field)
        assert "stream_options" not in result
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
        """Test basic OpenAI Responses request -> IRRequest."""
        provider_request = {
            "model": "gpt-4o",
            "instructions": "Be helpful",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        result = self.converter.request_from_provider(provider_request)
        assert result["model"] == "gpt-4o"
        assert result["system_instruction"] == "Be helpful"
        messages = list(result["messages"])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_request_from_provider_full(self):
        """Test full OpenAI Responses request -> IRRequest."""
        provider_request = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
            "temperature": 0.5,
            "max_output_tokens": 200,
            "reasoning": {"effort": "high"},
            "stream": True,
            "stream_options": {"include_usage": True},
            "prompt_cache_key": "k1",
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {},
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

    def test_request_from_provider_with_text_format(self):
        """Test text field -> response_format."""
        provider_request = {
            "model": "gpt-4o",
            "input": [],
            "text": {"type": "json_object"},
        }
        result = self.converter.request_from_provider(provider_request)
        assert result["response_format"]["type"] == "json_object"

    def test_request_from_provider_codex_custom_tool_passes_validation(self):
        """End-to-end: a Codex ``"custom"`` apply_patch tool passes IR validation.

        Regression test for the IR validator added in v0.3.0 (commit 5dc1b94)
        rejecting ``tools[i].type == "custom"`` because IR's ToolDefinition
        Literal only allows ``"function"`` and ``"mcp"``.  The source
        converter must downgrade unknown provider tool types to ``"function"``
        so the request reaches the target converter.
        """
        provider_request = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "patch this"}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run a shell command",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified-diff style patch.",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: ...",
                    },
                },
            ],
        }
        # Must not raise — pre-fix this raised ValidationError with
        # "Expected one of ('function', 'mcp') at 'tools[1].type', got 'custom'".
        result = self.converter.request_from_provider(provider_request)
        tools = list(result["tools"])
        assert len(tools) == 2
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "shell"
        assert tools[1]["type"] == "function"
        assert tools[1]["name"] == "apply_patch"
        assert tools[1]["metadata"]["provider_type"] == "custom"

    def test_request_from_provider_malformed_tool_raises_with_context(self):
        """Test that malformed tools raise clear errors with tool type/name context."""
        provider_request = {
            "model": "gpt-4o",
            "input": [],
            "tools": [42],  # non-dict tool triggers conversion error
        }
        with pytest.raises(ValueError, match=r"Unsupported tool"):
            self.converter.request_from_provider(provider_request)

    # ==================== response_from_provider ====================

    def test_response_from_provider_basic(self):
        """Test OpenAI Responses response -> IRResponse."""
        provider_response = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello! How can I help?",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["id"] == "resp_123"
        assert result["object"] == "response"
        assert result["model"] == "gpt-4o"
        assert len(result["choices"]) == 1
        assert list(result["choices"][0]["message"]["content"])[0]["text"] == (  # type: ignore
            "Hello! How can I help?"
        )
        assert result["choices"][0]["finish_reason"]["reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_response_from_provider_with_tool_calls(self):
        """Test response with tool calls."""
        provider_response = {
            "id": "resp_456",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        choice = result["choices"][0]
        tc = list(choice["message"]["content"])[0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"
        assert tc["tool_call_id"] == "call_1"

    def test_response_from_provider_with_reasoning(self):
        """Test response with reasoning content."""
        provider_response = {
            "id": "resp_789",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "content": "Let me think step by step...",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer is 42."}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert len(result["choices"]) == 1
        content = list(result["choices"][0]["message"]["content"])
        # Should contain reasoning + text
        assert any(p["type"] == "reasoning" for p in content)
        assert any(p["type"] == "text" for p in content)
        assert result["usage"]["reasoning_tokens"] == 5

    def test_response_from_provider_incomplete_status(self):
        """Test response with incomplete status -> length finish reason."""
        provider_response = {
            "id": "resp_inc",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Partial..."}],
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["choices"][0]["finish_reason"]["reason"] == "length"

    def test_response_from_provider_content_filter(self):
        """Test response with content_filter incomplete reason."""
        provider_response = {
            "id": "resp_cf",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "..."}],
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["choices"][0]["finish_reason"]["reason"] == "content_filter"

    def test_response_from_provider_failed_status(self):
        """Test response with failed status -> error finish reason."""
        provider_response = {
            "id": "resp_fail",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "failed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Error"}],
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["choices"][0]["finish_reason"]["reason"] == "error"

    def test_response_from_provider_cancelled_status(self):
        """Test response with cancelled status."""
        provider_response = {
            "id": "resp_cancel",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "cancelled",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Cancelled"}],
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["choices"][0]["finish_reason"]["reason"] == "cancelled"

    def test_response_from_provider_with_usage_details(self):
        """Test response with detailed usage statistics."""
        provider_response = {
            "id": "resp_usage",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 8},
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["cache_read_tokens"] == 5
        assert result["usage"]["reasoning_tokens"] == 8

    def test_response_from_provider_with_service_tier(self):
        """Test response with service_tier."""
        provider_response = {
            "id": "resp_st",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                }
            ],
            "service_tier": "default",
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["service_tier"] == "default"

    def test_response_from_provider_with_service_tier_none(self):
        """Test response with service_tier=None does not break validation."""
        provider_response = {
            "id": "resp_st_none",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                }
            ],
            "service_tier": None,
        }
        result = self.converter.response_from_provider(provider_response)
        assert "service_tier" not in result

    # ==================== response_to_provider ====================

    def test_response_to_provider_basic(self):
        """Test IRResponse -> OpenAI Responses response."""
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
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["type"] == "output_text"
        assert result["output"][0]["content"][0]["text"] == "Hello!"

    def test_response_to_provider_with_tool_calls(self):
        """Test IRResponse with tool calls -> provider response."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-2",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_call",
                                    "tool_call_id": "call_1",
                                    "tool_name": "get_weather",
                                    "tool_input": {"city": "NYC"},
                                }
                            ],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "function_call"

    def test_response_to_provider_length_finish_reason(self):
        """Test IRResponse with length finish reason -> incomplete status."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-3",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Partial"}],
                        },
                        "finish_reason": {"reason": "length"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"

    def test_response_to_provider_content_filter_finish_reason(self):
        """Test IRResponse with content_filter finish reason -> incomplete status."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-cf",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Filtered"}],
                        },
                        "finish_reason": {"reason": "content_filter"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "content_filter"

    def test_response_to_provider_error_finish_reason(self):
        """Test IRResponse with error finish reason -> failed status."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-4",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Error"}],
                        },
                        "finish_reason": {"reason": "error"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["status"] == "failed"

    def test_response_to_provider_with_usage(self):
        """Test IRResponse with usage -> provider usage format."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-5",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hi"}],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cache_read_tokens": 5,
                    "reasoning_tokens": 8,
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30
        assert result["usage"]["input_tokens_details"]["cached_tokens"] == 5
        assert result["usage"]["output_tokens_details"]["reasoning_tokens"] == 8

    def test_response_to_provider_with_reasoning(self):
        """Test IRResponse with reasoning -> provider reasoning item."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp-6",
                "object": "response",
                "created": 1000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "reasoning", "reasoning": "Thinking..."},
                                {"type": "text", "text": "Answer"},
                            ],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        output = result["output"]
        # Should have reasoning item + message item
        types = [item["type"] for item in output]
        assert "reasoning" in types
        assert "message" in types

    # ==================== messages_to_provider / messages_from_provider ====================

    def test_messages_to_provider(self):
        """Test messages_to_provider delegation."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )
        result, warnings = self.converter.messages_to_provider(messages)
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"

    def test_messages_from_provider(self):
        """Test messages_from_provider delegation."""
        provider_msgs = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        result = self.converter.messages_from_provider(provider_msgs)
        assert len(result) == 1
        msg = cast(Any, result[0])
        assert list(msg["content"])[0]["text"] == "Hello"

    # ==================== _normalize ====================

    def test_normalize_dict(self):
        """Test _normalize with dict input."""
        data = {"key": "value"}
        assert OpenAIResponsesConverter._normalize(data) is data

    def test_normalize_pydantic(self):
        """Test _normalize with Pydantic-like object."""

        class MockModel:
            def model_dump(self):
                return {"model": "gpt-4o"}

        result = OpenAIResponsesConverter._normalize(MockModel())
        assert result == {"model": "gpt-4o"}

    def test_normalize_to_dict(self):
        """Test _normalize with to_dict method."""

        class MockObj:
            def to_dict(self):
                return {"key": "val"}

        result = OpenAIResponsesConverter._normalize(MockObj())
        assert result == {"key": "val"}

    def test_normalize_invalid(self):
        """Test _normalize raises on unsupported type."""
        with pytest.raises(TypeError, match="Cannot normalize"):
            OpenAIResponsesConverter._normalize(42)

    # ==================== to_provider (backward compat) ====================

    def test_to_provider_with_ir_request(self):
        """Test to_provider with IRRequest dict."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                ],
                "generation": {"temperature": 0.7, "max_tokens": 100},
            },
        )
        result, warnings = self.converter.to_provider(ir_request)
        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100

    def test_to_provider_with_message_list(self):
        """Test to_provider with plain message list."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        )
        result, warnings = self.converter.to_provider(messages)
        assert "input" in result
        assert len(result["input"]) == 1

    def test_to_provider_with_tools(self):
        """Test to_provider with tools parameter."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        )
        tools = [
            {
                "type": "function",
                "name": "test",
                "description": "Test",
                "parameters": {},
            }
        ]
        result, warnings = self.converter.to_provider(messages, tools=tools)
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_to_provider_with_tool_choice(self):
        """Test to_provider with tool_choice parameter."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        )
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"mode": "auto"}
        )
        assert result["tool_choice"] == "auto"

    # ==================== validate_ir_input ====================

    def test_validate_ir_input(self):
        """Test validate_ir_input delegation."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )
        errors = self.converter.validate_ir_input(messages)
        assert errors == []


class TestOpenAIResponsesConverterFullRoundTrip:
    """Full round-trip conversion tests."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_request_round_trip(self):
        """Test IRRequest -> OpenAI Responses -> IRRequest round-trip."""
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
        """Test OpenAI Responses response -> IR -> OpenAI Responses round-trip."""
        provider_response = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        ir_response = self.converter.response_from_provider(provider_response)
        restored = self.converter.response_to_provider(ir_response)

        assert restored["id"] == "resp_123"
        assert restored["object"] == "response"
        assert restored["model"] == "gpt-4o"
        assert restored["status"] == "completed"
        assert len(restored["output"]) == 1
        assert restored["output"][0]["content"][0]["text"] == "Hello!"
        assert restored["usage"]["total_tokens"] == 15

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with tool calls."""
        # Turn 1: User asks
        ir_request_1 = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What's the weather?"}],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        )
        provider_req_1, _ = self.converter.request_to_provider(ir_request_1)
        assert "input" in provider_req_1
        assert "tools" in provider_req_1

        # Simulate API response with tool call
        provider_resp_1 = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                }
            ],
        }
        ir_resp_1 = self.converter.response_from_provider(provider_resp_1)
        assistant_msg = ir_resp_1["choices"][0]["message"]
        assert list(assistant_msg["content"])[0]["type"] == "tool_call"

        # Turn 2: Send tool result
        ir_request_2 = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What's the weather?"}],
                    },
                    assistant_msg,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": "call_abc",
                                "result": "Sunny, 25C",
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        )
        provider_req_2, _ = self.converter.request_to_provider(ir_request_2)
        # Should have user message + function_call + function_call_output
        assert len(provider_req_2["input"]) >= 3

    def test_system_message_in_messages_round_trip(self):
        """Test system message in messages list round-trip."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are helpful."}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}],
                    },
                ],
            },
        )
        provider, _ = self.converter.request_to_provider(ir_request)
        restored = self.converter.request_from_provider(provider)

        # System message should be preserved in messages
        messages = list(restored["messages"])
        assert len(messages) >= 2
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
