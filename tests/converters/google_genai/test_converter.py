"""
Google GenAI Converter integration tests.

Tests the top-level GoogleGenAIConverter with full request/response conversion.
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.google_genai import GoogleConverter, GoogleGenAIConverter
from llm_rosetta.types.ir import (
    IRRequest,
    IRResponse,
    Message,
    ToolDefinition,
)


class TestGoogleGenAIConverter:
    """Integration tests for GoogleGenAIConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = GoogleGenAIConverter()

    # ==================== request_to_provider ====================

    def test_simple_request(self):
        """Test simple request conversion."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["model"] == "gemini-2.0-flash"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello!"

    def test_request_with_system_instruction_string(self):
        """Test request with string system instruction."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "system_instruction": "You are a helpful assistant.",
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert "system_instruction" in result
        assert result["system_instruction"]["parts"][0]["text"] == (
            "You are a helpful assistant."
        )

    def test_request_with_system_instruction_parts(self):
        """Test request with parts-based system instruction."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "system_instruction": [
                {"type": "text", "text": "Be helpful."},
                {"type": "text", "text": "Be concise."},
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert len(result["system_instruction"]["parts"]) == 2

    def test_request_with_system_message_in_messages(self):
        """Test system message in messages list is extracted."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert "system_instruction" in result
        assert result["system_instruction"]["parts"][0]["text"] == "Be helpful."
        # System message should not be in contents list
        assert all(c["role"] != "system" for c in result["contents"])

    def test_request_with_generation_config(self):
        """Test request with generation config."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "generation": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
                "stop_sequences": ["\n\n"],
            },
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["temperature"] == 0.7
        assert config["max_output_tokens"] == 1024
        assert config["top_p"] == 0.9
        assert config["top_k"] == 50
        assert config["stop_sequences"] == ["\n\n"]

    def test_request_with_tools(self):
        """Test request with tools."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]}
            ],
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
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert len(config["tools"]) == 1
        assert config["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    def test_request_with_tool_choice(self):
        """Test request with tool choice."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]}
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"mode": "tool", "tool_name": "get_weather"},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["tool_config"]["function_calling_config"]["mode"] == "ANY"
        assert config["tool_config"]["function_calling_config"][
            "allowed_function_names"
        ] == ["get_weather"]

    def test_request_with_response_format(self):
        """Test request with response format."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "JSON please"}]}
            ],
            "response_format": {"type": "json_object"},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["response_mime_type"] == "application/json"

    def test_request_with_reasoning(self):
        """Test request with reasoning config."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash-thinking",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Think!"}]}
            ],
            "reasoning": {"budget_tokens": 4096},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["thinking_config"]["thinking_budget"] == 4096

    def test_request_with_stream(self):
        """Test request with stream config."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "stream": {"enabled": True},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["stream"] is True

    def test_request_with_cache(self):
        """Test request with cache config."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "cache": {"key": "cached-content-123"},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["cached_content"] == "cached-content-123"

    def test_request_with_provider_extensions(self):
        """Test request with provider extensions pass-through."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "provider_extensions": {"safety_settings": [{"category": "HARM"}]},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        config = result["config"]
        assert config["safety_settings"] == [{"category": "HARM"}]

    # ==================== request_from_provider ====================

    def test_request_from_provider_basic(self):
        """Test basic request from provider."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "config": {},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["model"] == "gemini-2.0-flash"
        messages = list(ir_request["messages"])
        assert len(messages) == 1
        assert list(messages[0]["content"])[0]["text"] == "Hello"

    def test_request_from_provider_with_system_instruction_string(self):
        """Test request from provider with string system instruction."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "system_instruction": "You are helpful.",
            "config": {},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["system_instruction"] == "You are helpful."

    def test_request_from_provider_with_system_instruction_dict(self):
        """Test request from provider with dict system instruction."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "system_instruction": {
                "role": "user",
                "parts": [{"text": "Be helpful."}],
            },
            "config": {},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["system_instruction"] == [
            {"type": "text", "text": "Be helpful."}
        ]

    def test_request_from_provider_with_tools(self):
        """Test request from provider with tools."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": "search",
                                "description": "Search",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ]
                    }
                ]
            },
        }
        ir_request = self.converter.request_from_provider(provider_request)
        tools = list(ir_request["tools"])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_request_from_provider_with_tool_config(self):
        """Test request from provider with tool config."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {"tool_config": {"function_calling_config": {"mode": "AUTO"}}},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["tool_choice"]["mode"] == "auto"

    def test_request_from_provider_with_generation_config(self):
        """Test request from provider with generation config fields."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {
                "temperature": 0.5,
                "max_output_tokens": 512,
                "top_p": 0.8,
            },
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["generation"]["temperature"] == 0.5
        assert ir_request["generation"]["max_tokens"] == 512
        assert ir_request["generation"]["top_p"] == 0.8

    def test_request_from_provider_with_response_format(self):
        """Test request from provider with response format."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {"response_mime_type": "application/json"},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["response_format"]["type"] == "json_object"

    def test_request_from_provider_with_thinking_config(self):
        """Test request from provider with thinking config."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {"thinking_config": {"thinking_budget": 8192}},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["reasoning"]["budget_tokens"] == 8192

    def test_request_from_provider_pydantic(self):
        """Test request from provider with Pydantic model."""

        class MockPydanticModel:
            def model_dump(self):
                return {
                    "model": "gemini-2.0-flash",
                    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                    "config": {},
                }

        ir_request = self.converter.request_from_provider(
            cast(dict[str, Any], MockPydanticModel())
        )
        assert ir_request["model"] == "gemini-2.0-flash"

    # ==================== response_from_provider ====================

    def test_response_from_provider_basic(self):
        """Test basic response from provider."""
        provider_response = {
            "response_id": "resp-123",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello! How can I help?"}],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 15,
                "candidates_token_count": 25,
                "total_token_count": 40,
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["id"] == "resp-123"
        assert result["object"] == "response"
        assert result["model"] == "gemini-2.0-flash"
        assert len(result["choices"]) == 1

        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert list(choice["message"]["content"])[0]["text"] == "Hello! How can I help?"
        assert choice["finish_reason"]["reason"] == "stop"

        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 25
        assert result["usage"]["total_tokens"] == 40

    def test_response_from_provider_with_tool_call(self):
        """Test response with tool call."""
        provider_response = {
            "response_id": "resp-tool",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "SF"},
                                }
                            }
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        choice = result["choices"][0]
        tc = list(choice["message"]["content"])[0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"

    def test_response_from_provider_with_reasoning(self):
        """Test response with reasoning (thought) parts."""
        provider_response = {
            "response_id": "resp-think",
            "model_version": "gemini-2.0-flash-thinking",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "Let me think..."},
                            {"text": "The answer is 42."},
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        choice = result["choices"][0]
        content = list(choice["message"]["content"])
        assert content[0]["type"] == "reasoning"
        assert content[0]["reasoning"] == "Let me think..."
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "The answer is 42."

    def test_response_from_provider_with_thoughts_token_count(self):
        """Test response with thoughts_token_count in usage."""
        provider_response = {
            "response_id": "resp-usage",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hi"}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 20,
                "total_token_count": 30,
                "thoughts_token_count": 100,
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["reasoning_tokens"] == 100

    def test_response_from_provider_finish_reasons(self):
        """Test various finish reason mappings."""
        reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "MALFORMED_FUNCTION_CALL": "error",
            "OTHER": "error",
        }
        for google_reason, ir_reason in reason_map.items():
            provider_response = {
                "response_id": f"resp-{google_reason}",
                "model_version": "gemini-2.0-flash",
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "Hi"}]},
                        "finish_reason": google_reason,
                    }
                ],
            }
            result = self.converter.response_from_provider(provider_response)
            assert result["choices"][0]["finish_reason"]["reason"] == ir_reason, (
                f"Failed for {google_reason}"
            )

    def test_response_from_provider_pydantic(self):
        """Test response from provider with Pydantic model."""

        class MockResponse:
            def model_dump(self):
                return {
                    "response_id": "resp-pydantic",
                    "model_version": "gemini-2.0-flash",
                    "candidates": [
                        {
                            "index": 0,
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Hi"}],
                            },
                            "finish_reason": "STOP",
                        }
                    ],
                }

        result = self.converter.response_from_provider(
            cast(dict[str, Any], MockResponse())
        )
        assert result["id"] == "resp-pydantic"

    def test_response_from_provider_no_content(self):
        """Test response with candidate but no content."""
        provider_response = {
            "response_id": "resp-empty",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "finish_reason": "SAFETY",
                }
            ],
        }
        result = self.converter.response_from_provider(provider_response)
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"

    # ==================== response_to_provider ====================

    def test_response_to_provider_basic(self):
        """Test basic response to provider."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_123",
                "object": "response",
                "created": 1700000000,
                "model": "gemini-2.0-flash",
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
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["responseId"] == "resp_123"
        assert result["modelVersion"] == "gemini-2.0-flash"
        assert len(result["candidates"]) == 1
        candidate = result["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert candidate["content"]["parts"][0]["text"] == "Hello!"
        assert candidate["finishReason"] == "STOP"
        assert result["usageMetadata"]["promptTokenCount"] == 10
        assert result["usageMetadata"]["candidatesTokenCount"] == 20
        assert result["usageMetadata"]["totalTokenCount"] == 30

    def test_response_to_provider_with_tool_calls(self):
        """Test response to provider with tool calls."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_tc",
                "object": "response",
                "created": 1700000000,
                "model": "gemini-2.0-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_call",
                                    "tool_call_id": "call_123",
                                    "tool_name": "search",
                                    "tool_input": {"q": "test"},
                                    "tool_type": "function",
                                }
                            ],
                        },
                        "finish_reason": {"reason": "tool_calls"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        candidate = result["candidates"][0]
        assert candidate["finishReason"] == "STOP"
        assert "functionCall" in candidate["content"]["parts"][0]
        assert candidate["content"]["parts"][0]["functionCall"]["name"] == "search"

    def test_response_to_provider_with_reasoning(self):
        """Test response to provider with reasoning parts."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_think",
                "object": "response",
                "created": 1700000000,
                "model": "gemini-2.0-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "reasoning", "reasoning": "Thinking..."},
                                {"type": "text", "text": "Answer."},
                            ],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            },
        )
        result = self.converter.response_to_provider(ir_response)
        parts = result["candidates"][0]["content"]["parts"]
        assert parts[0]["thought"] is True
        assert parts[0]["text"] == "Thinking..."
        assert parts[1]["text"] == "Answer."

    def test_response_to_provider_with_reasoning_tokens(self):
        """Test response to provider with reasoning tokens in usage."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_rt",
                "object": "response",
                "created": 1700000000,
                "model": "gemini-2.0-flash",
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
                    "reasoning_tokens": 50,
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        assert result["usageMetadata"]["thoughtsTokenCount"] == 50

    def test_response_to_provider_finish_reasons(self):
        """Test finish reason mapping to provider."""
        reason_map = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "error": "OTHER",
        }
        for ir_reason, google_reason in reason_map.items():
            ir_response = cast(
                IRResponse,
                {
                    "id": f"resp_{ir_reason}",
                    "object": "response",
                    "created": 1700000000,
                    "model": "gemini-2.0-flash",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Hi"}],
                            },
                            "finish_reason": {"reason": ir_reason},
                        }
                    ],
                },
            )
            result = self.converter.response_to_provider(ir_response)
            assert result["candidates"][0]["finishReason"] == google_reason, (
                f"Failed for {ir_reason}"
            )

    # ==================== messages_to_provider / messages_from_provider ====================

    def test_messages_to_provider(self):
        """Test messages_to_provider delegates to message_ops."""
        messages = cast(
            list[Message],
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        )
        result, warnings = self.converter.messages_to_provider(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"

    def test_messages_from_provider(self):
        """Test messages_from_provider delegates to message_ops."""
        provider_messages = [
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        result = self.converter.messages_from_provider(provider_messages)
        assert len(result) == 1
        msg = cast(Any, result[0])
        assert msg["role"] == "user"
        assert list(msg["content"])[0]["text"] == "Hello"

    # ==================== _normalize ====================

    def test_normalize_dict(self):
        """Test _normalize with dict input."""
        data = {"key": "value"}
        assert GoogleGenAIConverter._normalize(data) is data

    def test_normalize_pydantic(self):
        """Test _normalize with Pydantic-like object."""

        class MockModel:
            def model_dump(self):
                return {"model": "gemini-2.0-flash"}

        result = GoogleGenAIConverter._normalize(MockModel())
        assert result == {"model": "gemini-2.0-flash"}

    def test_normalize_tuple(self):
        """Test _normalize with tuple input (unwraps first element)."""
        data = ({"key": "value"}, "extra")
        result = GoogleGenAIConverter._normalize(data)
        assert result == {"key": "value"}

    def test_normalize_to_dict(self):
        """Test _normalize with to_dict method."""

        class MockObj:
            def to_dict(self):
                return {"key": "value"}

        result = GoogleGenAIConverter._normalize(MockObj())
        assert result == {"key": "value"}

    def test_normalize_invalid(self):
        """Test _normalize raises on unsupported type."""
        with pytest.raises(TypeError, match="Cannot normalize"):
            GoogleGenAIConverter._normalize(42)

    # ==================== Backward Compatibility ====================

    def test_build_config_with_tools(self):
        """Test build_config with tools."""
        tools = cast(
            list[ToolDefinition],
            [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        config = self.converter.build_config(tools=tools)
        assert config is not None
        assert len(config["tools"]) == 1

    def test_build_config_with_tool_choice(self):
        """Test build_config with tool choice."""
        config = self.converter.build_config(tool_choice={"mode": "auto"})
        assert config is not None
        assert config["tool_config"]["function_calling_config"]["mode"] == "AUTO"

    def test_build_config_empty(self):
        """Test build_config with no args returns None."""
        config = self.converter.build_config()
        assert config is None

    def test_to_provider_with_ir_input(self):
        """Test to_provider with IRInput (message list)."""
        ir_input = cast(
            list[Message],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        )
        result, warnings = self.converter.to_provider(ir_input)
        assert "system_instruction" in result
        assert result["system_instruction"]["parts"][0]["text"] == "Be helpful."
        assert len(result["contents"]) == 1

    def test_to_provider_with_ir_request(self):
        """Test to_provider with IRRequest (dict with messages key)."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                ],
            },
        )
        result, warnings = self.converter.to_provider(ir_request)
        assert result["model"] == "gemini-2.0-flash"

    def test_to_provider_with_tools(self):
        """Test to_provider with tools parameter."""
        ir_input = cast(
            list[Message],
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        )
        tools = cast(
            list[ToolDefinition],
            [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        result, warnings = self.converter.to_provider(ir_input, tools=tools)
        assert len(result["tools"]) == 1

    # ==================== GoogleConverter Alias ====================

    def test_google_converter_alias(self):
        """Test GoogleConverter is an alias for GoogleGenAIConverter."""
        assert GoogleConverter is GoogleGenAIConverter

    def test_google_converter_alias_works(self):
        """Test GoogleConverter alias can be instantiated and used."""
        converter = GoogleConverter()
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
        }
        result, warnings = converter.request_to_provider(ir_request)
        assert result["model"] == "gemini-2.0-flash"


class TestGoogleGenAIConverterFullRoundTrip:
    """Full round-trip conversion tests."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_request_round_trip(self):
        """Test IRRequest -> Google -> IRRequest round-trip."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
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
            "tool_choice": {"mode": "auto"},
        }
        provider, _ = self.converter.request_to_provider(ir_request)
        restored = self.converter.request_from_provider(provider)

        assert restored["model"] == "gemini-2.0-flash"
        assert restored["generation"]["temperature"] == 0.7
        assert restored["generation"]["max_tokens"] == 100
        tools = list(restored["tools"])
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_response_round_trip(self):
        """Test Google response -> IR -> Google round-trip."""
        provider_response = {
            "response_id": "resp-rt",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello!"}],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }
        ir_response = self.converter.response_from_provider(provider_response)
        restored = self.converter.response_to_provider(ir_response)

        assert restored["responseId"] == "resp-rt"
        assert restored["modelVersion"] == "gemini-2.0-flash"
        assert restored["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"
        assert restored["candidates"][0]["finishReason"] == "STOP"
        assert restored["usageMetadata"]["totalTokenCount"] == 15

    def test_message_round_trip(self):
        """Test message round-trip: IR -> Google -> IR."""
        original = cast(
            list[Message],
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            ],
        )
        provider, _ = self.converter.messages_to_provider(original)
        restored = self.converter.messages_from_provider(provider)

        assert len(restored) == 2
        r0 = cast(Any, restored[0])
        r1 = cast(Any, restored[1])
        assert r0["role"] == "user"
        assert list(r0["content"])[0]["text"] == "Hello"
        assert r1["role"] == "assistant"
        assert list(r1["content"])[0]["text"] == "Hi!"
