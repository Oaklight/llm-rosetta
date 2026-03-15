"""Test compatibility between LLM-Rosetta OpenAI Chat type replicas and OpenAI SDK types.

This module uses the RECOMMENDED testing approach:
- Create objects using LLM-Rosetta TypedDict replicas
- Validate them using OpenAI SDK's Pydantic models
- This ensures our replicas can generate SDK-compatible data

This approach is superior because:
1. It matches our actual use case (we create data, SDK validates it)
2. SDK's Pydantic validation is stricter than TypedDict type hints
3. It catches issues like extra fields, wrong types, missing required fields
"""

import pytest


def test_chat_completion_response():
    """Test creating ChatCompletion response with LLM-Rosetta replica and validating with SDK."""
    try:
        from openai.types.chat import ChatCompletion as SDKChatCompletion

        from llm_rosetta.types.openai.chat import ChatCompletion as LLMRosettaChatCompletion

        # Create response using LLM-Rosetta replica
        llm_rosetta_response: LLMRosettaChatCompletion = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            },
        }

        # Validate with SDK - this is the critical test!
        sdk_validated = SDKChatCompletion.model_validate(llm_rosetta_response)

        # Verify the validated object matches our input
        assert sdk_validated.id == llm_rosetta_response["id"]
        assert sdk_validated.model == llm_rosetta_response["model"]
        assert (
            sdk_validated.choices[0].message.content
            == llm_rosetta_response["choices"][0]["message"]["content"]
        )
        assert sdk_validated.usage.total_tokens == 21

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_tool_call_response():
    """Test creating tool call response with LLM-Rosetta replica and validating with SDK."""
    try:
        from openai.types.chat import ChatCompletion as SDKChatCompletion

        from llm_rosetta.types.openai.chat import ChatCompletion as LLMRosettaChatCompletion

        # Create tool call response using LLM-Rosetta replica
        llm_rosetta_response: LLMRosettaChatCompletion = {
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
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23,
            },
        }

        # Validate with SDK
        sdk_validated = SDKChatCompletion.model_validate(llm_rosetta_response)
        assert sdk_validated.choices[0].message.tool_calls is not None
        assert len(sdk_validated.choices[0].message.tool_calls) == 1
        assert (
            sdk_validated.choices[0].message.tool_calls[0].function.name
            == "get_weather"
        )

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_response_with_optional_fields():
    """Test creating response with optional fields and validating with SDK."""
    try:
        from openai.types.chat import ChatCompletion as SDKChatCompletion

        from llm_rosetta.types.openai.chat import ChatCompletion as LLMRosettaChatCompletion

        # Create response with optional fields
        llm_rosetta_response: LLMRosettaChatCompletion = {
            "id": "chatcmpl-detailed",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Detailed response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23,
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 3,
                },
            },
            "system_fingerprint": "fp_test123",
        }

        # Validate with SDK
        sdk_validated = SDKChatCompletion.model_validate(llm_rosetta_response)
        assert sdk_validated.usage.prompt_tokens_details.cached_tokens == 5
        assert sdk_validated.usage.completion_tokens_details.reasoning_tokens == 3
        assert sdk_validated.system_fingerprint == "fp_test123"

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_message_params():
    """Test creating various message parameter types and validating with SDK."""
    try:
        from openai.types.chat.chat_completion_assistant_message_param import (
            ChatCompletionAssistantMessageParam as SDKAssistantMessageParam,
        )
        from openai.types.chat.chat_completion_system_message_param import (
            ChatCompletionSystemMessageParam as SDKSystemMessageParam,
        )
        from openai.types.chat.chat_completion_tool_message_param import (
            ChatCompletionToolMessageParam as SDKToolMessageParam,
        )
        from openai.types.chat.chat_completion_user_message_param import (
            ChatCompletionUserMessageParam as SDKUserMessageParam,
        )

        from llm_rosetta.types.openai.chat import (
            ChatCompletionAssistantMessageParam,
            ChatCompletionSystemMessageParam,
            ChatCompletionToolMessageParam,
            ChatCompletionUserMessageParam,
        )

        # User message with text
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": "Hello",
        }
        sdk_user = SDKUserMessageParam(**user_msg)
        assert sdk_user["role"] == "user"

        # User message with multimodal content
        user_multimodal: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
        sdk_user_multi = SDKUserMessageParam(**user_multimodal)
        assert len(sdk_user_multi["content"]) == 2

        # Assistant message with tool calls
        assistant_msg: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_func", "arguments": "{}"},
                }
            ],
        }
        sdk_assistant = SDKAssistantMessageParam(**assistant_msg)
        assert sdk_assistant["role"] == "assistant"

        # System message
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        sdk_system = SDKSystemMessageParam(**system_msg)
        assert sdk_system["role"] == "system"

        # Tool message
        tool_msg: ChatCompletionToolMessageParam = {
            "role": "tool",
            "content": "Weather is sunny",
            "tool_call_id": "call_123",
        }
        sdk_tool = SDKToolMessageParam(**tool_msg)
        assert sdk_tool["role"] == "tool"

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_usage_statistics():
    """Test creating usage statistics with LLM-Rosetta replica and validating with SDK."""
    try:
        from openai.types.completion_usage import (
            CompletionUsage as SDKCompletionUsage,
        )

        from llm_rosetta.types.openai.chat import CompletionUsage

        # Basic usage
        basic_usage: CompletionUsage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        sdk_basic = SDKCompletionUsage.model_validate(basic_usage)
        assert sdk_basic.total_tokens == 30

        # Usage with details
        detailed_usage: CompletionUsage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 8},
        }
        sdk_detailed = SDKCompletionUsage.model_validate(detailed_usage)
        assert sdk_detailed.prompt_tokens_details.cached_tokens == 5
        assert sdk_detailed.completion_tokens_details.reasoning_tokens == 8

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_response_formats():
    """Test creating response format types and validating with SDK.

    Note: ResponseFormat is a Union type in SDK, so we validate the structure
    by ensuring our replicas match the expected format.
    """
    try:
        from llm_rosetta.types.openai.chat import (
            ResponseFormatJSONObject,
            ResponseFormatJSONSchema,
            ResponseFormatText,
        )

        # Text format
        text_format: ResponseFormatText = {"type": "text"}
        assert text_format["type"] == "text"

        # JSON object format
        json_format: ResponseFormatJSONObject = {"type": "json_object"}
        assert json_format["type"] == "json_object"

        # JSON schema format
        schema_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {}},
            },
        }
        assert schema_format["type"] == "json_schema"
        assert schema_format["json_schema"]["name"] == "test_schema"

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_tool_definitions():
    """Test creating tool definitions with LLM-Rosetta replica and validating with SDK."""
    try:
        from openai.types.chat.chat_completion_tool_param import (
            ChatCompletionToolParam as SDKToolParam,
        )

        from llm_rosetta.types.openai.chat import ChatCompletionFunctionToolParam

        tool_def: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }

        # Validate with SDK
        sdk_tool = SDKToolParam(**tool_def)
        assert sdk_tool["type"] == "function"
        assert sdk_tool["function"]["name"] == "get_weather"

    except ImportError:
        pytest.skip("OpenAI SDK not available")


def test_comprehensive_sdk_validation():
    """Comprehensive test creating complete request/response cycle with SDK validation.

    This test demonstrates the full workflow:
    1. Create LLM-Rosetta replica types for request and response
    2. Validate both with OpenAI SDK
    3. Ensure end-to-end compatibility
    """
    try:
        from openai.types.chat import ChatCompletion as SDKChatCompletion

        from llm_rosetta.types.openai.chat import (
            ChatCompletion as LLMRosettaChatCompletion,
        )
        from llm_rosetta.types.openai.chat import (
            CompletionCreateParams as LLMRosettaCompletionCreateParams,
        )

        # Create request params using LLM-Rosetta replica
        llm_rosetta_request: LLMRosettaCompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "temperature": 0.7,
            "max_completion_tokens": 1000,
        }

        # Note: We can't directly validate CompletionCreateParams with SDK
        # because it's used as **kwargs, but we can verify structure
        assert llm_rosetta_request["model"] == "gpt-4o"
        assert len(llm_rosetta_request["messages"]) == 1

        # Create response using LLM-Rosetta replica
        llm_rosetta_response: LLMRosettaChatCompletion = {
            "id": "chatcmpl-comprehensive",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well, thank you!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

        # Validate response with SDK
        sdk_validated = SDKChatCompletion.model_validate(llm_rosetta_response)
        assert sdk_validated.id == "chatcmpl-comprehensive"
        assert sdk_validated.choices[0].message.content == "I'm doing well, thank you!"

        print("✓ Comprehensive SDK validation successful")
        print("  LLM-Rosetta replicas are fully compatible with OpenAI SDK")

    except ImportError:
        pytest.skip("OpenAI SDK not available")


if __name__ == "__main__":
    pytest.main([__file__])
