"""
OpenAI Chat Completions Full Request/Response Conversion Tests
"""

import pytest
from llmir.converters.openai_chat import OpenAIChatConverter
from llmir.types.ir_request import IRRequest
from llmir.types.ir_response import Message, TextPart

class TestOpenAIChatFullConversion:
    """OpenAI Chat full conversion test class"""

    def setup_method(self):
        """Set up tests"""
        self.converter = OpenAIChatConverter()

    def test_full_request_conversion(self):
        """Test full IRRequest to OpenAI conversion"""
        ir_request: IRRequest = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}]
                }
            ],
            "system_instruction": "You are a helpful assistant.",
            "generation": {
                "temperature": 0.7,
                "max_tokens": 100,
                "stop_sequences": ["\n", "END"],
            },
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "medium"},
            "stream": {"enabled": True, "include_usage": True},
            "cache": {"key": "test-cache", "retention": "24h"}
        }

        result, warnings = self.converter.to_provider(ir_request)

        assert result["model"] == "gpt-4o"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Hello!"
        
        assert result["temperature"] == 0.7
        assert result["max_completion_tokens"] == 100
        assert result["stop"] == ["\n", "END"]
        assert result["response_format"] == {"type": "json_object"}
        assert result["reasoning_effort"] == "medium"
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}
        assert result["prompt_cache_key"] == "test-cache"
        assert result["prompt_cache_retention"] == "24h"

    def test_full_response_conversion(self):
        """Test full OpenAI response to IRResponse conversion"""
        provider_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!",
                    },
                    "finish_reason": "stop",
                    "logprobs": {"content": []}
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {
                    "reasoning_tokens": 5
                },
                "prompt_tokens_details": {
                    "cached_tokens": 2
                }
            },
            "system_fingerprint": "fp_4470629fc2"
        }

        result = self.converter.from_provider(provider_response)

        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "response"
        assert result["created"] == 1677652288
        assert result["model"] == "gpt-4o"
        assert len(result["choices"]) == 1
        
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"][0]["text"] == "Hello there!"
        assert choice["finish_reason"]["reason"] == "stop"
        assert choice["logprobs"] == {"content": []}
        
        usage = result["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert usage["reasoning_tokens"] == 5
        assert usage["cache_read_tokens"] == 2
        assert result["system_fingerprint"] == "fp_4470629fc2"

    def test_request_with_tools(self):
        """Test request with tools and tool choice"""
        ir_request: IRRequest = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather?"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            ],
            "tool_choice": {"mode": "tool", "tool_name": "get_weather"},
            "tool_config": {"disable_parallel": True}
        }

        result, _ = self.converter.to_provider(ir_request)

        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
        assert result["parallel_tool_calls"] is False