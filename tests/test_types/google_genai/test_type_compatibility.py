"""Test compatibility between LLM-Rosetta Google GenAI type replicas and Google SDK types.

This module tests:
- All TypedDict replicas can be correctly instantiated
- Required and optional fields work as expected
- If the Google GenAI SDK is available, SDK objects' dict representations
  are compatible with our TypedDict definitions

Reference: tests/test_types/openai/chat/test_type_compatibility.py
"""

from typing import Any, cast

import pytest

from llm_rosetta.types.google import (
    Blob,
    Candidate,
    Citation,
    CitationMetadata,
    CodeExecutionResult,
    CodeExecutionResultPart,
    Content,
    ExecutableCode,
    ExecutableCodePart,
    FileData,
    FileDataPart,
    FunctionCall,
    FunctionCallPart,
    FunctionDeclaration,
    FunctionResponse,
    FunctionResponsePart,
    GenerateContentConfig,
    GenerateContentRequest,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    GenerateContentResponseUsageMetadata,
    GroundingAttribution,
    InlineDataPart,
    ModalityTokenCount,
    Part,
    SafetyRating,
    SafetySetting,
    Schema,
    TextPart,
    ThinkingConfig,
    Tool,
)

# ============================================================================
# Content type instantiation tests
# ============================================================================


class TestContentTypes:
    """Test content and part type instantiation."""

    def test_text_part(self):
        """Test creating a TextPart."""
        part: TextPart = {"text": "Hello, world!"}
        assert part["text"] == "Hello, world!"

    def test_inline_data_part(self):
        """Test creating an InlineDataPart with Blob."""
        blob: Blob = {
            "mime_type": "image/png",
            "data": b"fake_image_data",
        }
        part: InlineDataPart = {"inline_data": blob}
        assert part["inline_data"]["mime_type"] == "image/png"

    def test_file_data_part(self):
        """Test creating a FileDataPart."""
        file_data: FileData = {
            "mime_type": "image/jpeg",
            "file_uri": "gs://bucket/image.jpg",
        }
        part: FileDataPart = {"file_data": file_data}
        assert part["file_data"]["file_uri"] == "gs://bucket/image.jpg"

    def test_function_call_part(self):
        """Test creating a FunctionCallPart."""
        func_call: FunctionCall = {
            "name": "get_weather",
            "args": {"location": "NYC"},
        }
        part: FunctionCallPart = {"function_call": func_call}
        assert part["function_call"]["name"] == "get_weather"

    def test_function_call_with_id(self):
        """Test creating a FunctionCall with id field."""
        func_call: FunctionCall = {
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "NYC"},
        }
        assert func_call["id"] == "call_123"

    def test_function_response_part(self):
        """Test creating a FunctionResponsePart."""
        func_response: FunctionResponse = {
            "name": "get_weather",
            "response": {"temperature": 72},
        }
        part: FunctionResponsePart = {"function_response": func_response}
        assert part["function_response"]["name"] == "get_weather"

    def test_executable_code_part(self):
        """Test creating an ExecutableCodePart."""
        exec_code: ExecutableCode = {
            "language": "PYTHON",
            "code": "print('hello')",
        }
        part: ExecutableCodePart = {"executable_code": exec_code}
        assert part["executable_code"]["language"] == "PYTHON"

    def test_code_execution_result_part(self):
        """Test creating a CodeExecutionResultPart."""
        code_result: CodeExecutionResult = {
            "outcome": "OUTCOME_OK",
            "output": "hello\n",
        }
        part: CodeExecutionResultPart = {"code_execution_result": code_result}
        assert part["code_execution_result"]["outcome"] == "OUTCOME_OK"

    def test_unified_part(self):
        """Test creating a unified Part with text field."""
        part: Part = {"text": "Hello, world!"}
        assert part["text"] == "Hello, world!"

    def test_unified_part_with_function_call(self):
        """Test creating a unified Part with function_call field."""
        part: Part = {
            "function_call": {
                "name": "get_weather",
                "args": {"location": "NYC"},
            }
        }
        assert part["function_call"]["name"] == "get_weather"

    def test_unified_part_with_thought(self):
        """Test creating a unified Part with thought fields."""
        part: Part = {
            "text": "Let me think about this...",
            "thought": True,
            "thought_signature": "sig_abc123",
        }
        assert part["thought"] is True
        assert part["thought_signature"] == "sig_abc123"

    def test_content(self):
        """Test creating a Content object."""
        content: Content = {
            "parts": [{"text": "Hello"}, {"text": "World"}],
            "role": "user",
        }
        assert content["role"] == "user"
        assert len(content["parts"]) == 2

    def test_content_model_role(self):
        """Test creating Content with model role."""
        content: Content = {
            "parts": [{"text": "I can help with that."}],
            "role": "model",
        }
        assert content["role"] == "model"

    def test_content_minimal(self):
        """Test creating Content with minimal fields."""
        content: Content = {
            "parts": [{"text": "Hello"}],
        }
        assert "role" not in content

    def test_blob(self):
        """Test creating a Blob."""
        blob: Blob = {
            "data": b"raw_bytes",
            "mime_type": "application/octet-stream",
            "display_name": "test_file",
        }
        assert blob["mime_type"] == "application/octet-stream"
        assert blob["display_name"] == "test_file"


# ============================================================================
# Request type instantiation tests
# ============================================================================


class TestRequestTypes:
    """Test request parameter type instantiation."""

    def test_schema(self):
        """Test creating a Schema."""
        schema: Schema = {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "age": {"type": "INTEGER"},
            },
            "required": ["name"],
        }
        assert schema["type"] == "OBJECT"
        assert "name" in schema["properties"]

    def test_function_declaration(self):
        """Test creating a FunctionDeclaration."""
        func_decl: FunctionDeclaration = {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING"},
                },
                "required": ["location"],
            },
        }
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get the current weather"

    def test_function_declaration_with_json_schema(self):
        """Test creating a FunctionDeclaration with parameters_json_schema."""
        func_decl: FunctionDeclaration = {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
        assert func_decl["parameters_json_schema"]["type"] == "object"

    def test_tool(self):
        """Test creating a Tool."""
        tool: Tool = {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "location": {"type": "STRING"},
                        },
                    },
                }
            ],
        }
        assert len(tool["function_declarations"]) == 1

    def test_tool_with_code_execution(self):
        """Test creating a Tool with code_execution."""
        tool: Tool = {
            "code_execution": {},
        }
        assert tool["code_execution"] == {}

    def test_safety_setting(self):
        """Test creating a SafetySetting."""
        safety: SafetySetting = {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        }
        assert safety["category"] == "HARM_CATEGORY_HARASSMENT"

    def test_thinking_config(self):
        """Test creating a ThinkingConfig."""
        thinking: ThinkingConfig = {
            "include_thoughts": True,
            "thinking_budget": 1024,
        }
        assert thinking["include_thoughts"] is True
        assert thinking["thinking_budget"] == 1024

    def test_thinking_config_disabled(self):
        """Test creating a disabled ThinkingConfig."""
        thinking: ThinkingConfig = {
            "thinking_budget": 0,
        }
        assert thinking["thinking_budget"] == 0

    def test_generate_content_config(self):
        """Test creating a GenerateContentConfig."""
        config: GenerateContentConfig = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40.0,
            "max_output_tokens": 1024,
            "candidate_count": 1,
            "stop_sequences": ["END"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "seed": 42,
            "response_logprobs": True,
            "logprobs": 5,
            "response_mime_type": "application/json",
            "response_schema": {"type": "OBJECT"},
            "response_modalities": ["TEXT"],
        }
        assert config["temperature"] == 0.7
        assert config["max_output_tokens"] == 1024

    def test_generate_content_config_with_tools(self):
        """Test creating a GenerateContentConfig with tools and safety."""
        config: GenerateContentConfig = {
            "temperature": 0.5,
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "search",
                            "description": "Search the web",
                        }
                    ]
                }
            ],
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                }
            ],
            "thinking_config": {
                "include_thoughts": True,
                "thinking_budget": 2048,
            },
        }
        assert len(config["tools"]) == 1
        assert len(config["safety_settings"]) == 1

    def test_generate_content_config_with_system_instruction(self):
        """Test GenerateContentConfig with system_instruction."""
        config: GenerateContentConfig = {
            "system_instruction": {
                "parts": [{"text": "You are a helpful assistant."}],
                "role": "user",
            },
            "temperature": 0.7,
        }
        assert config["system_instruction"]["parts"][0]["text"] == (
            "You are a helpful assistant."
        )

    def test_generate_content_request_with_string(self):
        """Test creating a GenerateContentRequest with string contents."""
        request: GenerateContentRequest = {
            "model": "gemini-2.0-flash",
            "contents": "Hello, world!",
        }
        assert request["model"] == "gemini-2.0-flash"
        assert request["contents"] == "Hello, world!"

    def test_generate_content_request_with_content_list(self):
        """Test creating a GenerateContentRequest with content list."""
        request: GenerateContentRequest = {
            "model": "gemini-2.0-flash",
            "contents": [
                {
                    "parts": [{"text": "What is the weather?"}],
                    "role": "user",
                }
            ],
            "config": {
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        }
        assert isinstance(request["contents"], list)
        assert request["config"]["temperature"] == 0.7


# ============================================================================
# Response type instantiation tests
# ============================================================================


class TestResponseTypes:
    """Test response type instantiation."""

    def test_safety_rating(self):
        """Test creating a SafetyRating."""
        rating: SafetyRating = {
            "category": "HARM_CATEGORY_HARASSMENT",
            "probability": "LOW",
            "blocked": False,
        }
        assert rating["probability"] == "LOW"
        assert rating["blocked"] is False

    def test_safety_rating_with_scores(self):
        """Test creating a SafetyRating with score fields."""
        rating: SafetyRating = {
            "category": "HARM_CATEGORY_HARASSMENT",
            "probability": "LOW",
            "probability_score": 0.1,
            "severity": "HARM_SEVERITY_LOW",
            "severity_score": 0.05,
        }
        assert rating["probability_score"] == 0.1

    def test_citation(self):
        """Test creating a Citation."""
        citation: Citation = {
            "start_index": 0,
            "end_index": 10,
            "uri": "https://example.com",
            "title": "Example",
            "license": "MIT",
        }
        assert citation["uri"] == "https://example.com"

    def test_citation_metadata(self):
        """Test creating a CitationMetadata."""
        metadata: CitationMetadata = {
            "citations": [
                {
                    "start_index": 0,
                    "end_index": 10,
                    "uri": "https://example.com",
                }
            ],
        }
        assert len(metadata["citations"]) == 1

    def test_grounding_attribution(self):
        """Test creating a GroundingAttribution."""
        grounding: GroundingAttribution = {
            "source_id": "source_123",
            "content": {"text": "Source content"},
        }
        assert grounding["source_id"] == "source_123"

    def test_modality_token_count(self):
        """Test creating a ModalityTokenCount."""
        token_count: ModalityTokenCount = {
            "modality": "TEXT",
            "token_count": 100,
        }
        assert token_count["token_count"] == 100

    def test_candidate(self):
        """Test creating a Candidate."""
        candidate: Candidate = {
            "content": {
                "parts": [{"text": "Response text"}],
                "role": "model",
            },
            "finish_reason": "STOP",
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability": "NEGLIGIBLE",
                }
            ],
            "token_count": 50,
            "index": 0,
        }
        assert candidate["finish_reason"] == "STOP"
        assert candidate["content"]["role"] == "model"

    def test_candidate_with_function_call(self):
        """Test creating a Candidate with function call in content."""
        candidate: Candidate = {
            "content": {
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    }
                ],
                "role": "model",
            },
            "finish_reason": "STOP",
            "index": 0,
        }
        assert candidate["content"] is not None
        fc = candidate["content"]["parts"][0]
        assert fc["function_call"] is not None
        assert fc["function_call"]["name"] == "get_weather"

    def test_candidate_with_citation_metadata(self):
        """Test creating a Candidate with citation metadata."""
        candidate: Candidate = {
            "content": {
                "parts": [{"text": "According to sources..."}],
                "role": "model",
            },
            "citation_metadata": {
                "citations": [
                    {
                        "start_index": 0,
                        "end_index": 25,
                        "uri": "https://example.com",
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
        assert len(candidate["citation_metadata"]["citations"]) == 1

    def test_usage_metadata(self):
        """Test creating a GenerateContentResponseUsageMetadata."""
        usage: GenerateContentResponseUsageMetadata = {
            "prompt_token_count": 100,
            "candidates_token_count": 200,
            "total_token_count": 300,
            "cached_content_token_count": 50,
            "thoughts_token_count": 25,
            "tool_use_prompt_token_count": 10,
            "prompt_tokens_details": [{"modality": "TEXT", "token_count": 100}],
            "candidates_tokens_details": [{"modality": "TEXT", "token_count": 200}],
            "cache_tokens_details": None,
            "tool_use_prompt_tokens_details": None,
            "traffic_type": "ON_DEMAND",
        }
        assert usage["total_token_count"] == 300
        assert usage["thoughts_token_count"] == 25

    def test_prompt_feedback(self):
        """Test creating a GenerateContentResponsePromptFeedback."""
        feedback: GenerateContentResponsePromptFeedback = {
            "block_reason": None,
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability": "NEGLIGIBLE",
                }
            ],
        }
        assert feedback["safety_ratings"] is not None

    def test_prompt_feedback_blocked(self):
        """Test creating a blocked GenerateContentResponsePromptFeedback."""
        feedback: GenerateContentResponsePromptFeedback = {
            "block_reason": "SAFETY",
            "block_reason_message": "Content was blocked due to safety.",
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "HIGH",
                    "blocked": True,
                }
            ],
        }
        assert feedback["block_reason"] == "SAFETY"

    def test_generate_content_response(self):
        """Test creating a complete GenerateContentResponse."""
        response: GenerateContentResponse = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help?"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                    "safety_ratings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        }
                    ],
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 8,
                "total_token_count": 18,
            },
            "model_version": "gemini-2.0-flash-001",
            "response_id": "resp_123",
        }
        assert response["model_version"] == "gemini-2.0-flash-001"
        assert len(response["candidates"]) == 1
        assert response["usage_metadata"]["total_token_count"] == 18

    def test_generate_content_response_minimal(self):
        """Test creating a minimal GenerateContentResponse."""
        response: GenerateContentResponse = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hi"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
        assert len(response["candidates"]) == 1

    def test_generate_content_response_with_prompt_feedback(self):
        """Test creating a GenerateContentResponse with prompt feedback."""
        response: GenerateContentResponse = {
            "candidates": None,
            "prompt_feedback": {
                "block_reason": "SAFETY",
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "HIGH",
                        "blocked": True,
                    }
                ],
            },
        }
        assert response["prompt_feedback"]["block_reason"] == "SAFETY"


# ============================================================================
# __init__ module exports test
# ============================================================================


class TestGoogleInit:
    """Test the __init__ module exports."""

    def test_all_exports(self):
        """Test that all expected types are exported from __init__."""
        from llm_rosetta.types.google import __all__

        expected = [
            # Content types
            "Blob",
            "CodeExecutionResult",
            "CodeExecutionResultPart",
            "Content",
            "ExecutableCode",
            "ExecutableCodePart",
            "FileData",
            "FileDataPart",
            "FunctionCall",
            "FunctionCallPart",
            "FunctionResponse",
            "FunctionResponsePart",
            "InlineDataPart",
            "Part",
            "PartUnion",
            "TextPart",
            # Request types
            "FunctionDeclaration",
            "GenerateContentConfig",
            "GenerateContentRequest",
            "SafetySetting",
            "Schema",
            "ThinkingConfig",
            "Tool",
            # Response types
            "Candidate",
            "Citation",
            "CitationMetadata",
            "FinishReason",
            "GenerateContentResponse",
            "GenerateContentResponsePromptFeedback",
            "GenerateContentResponseUsageMetadata",
            "GroundingAttribution",
            "ModalityTokenCount",
            "SafetyRating",
        ]
        assert set(__all__) == set(expected)


# ============================================================================
# SDK compatibility tests (only run if SDK is available)
# ============================================================================


class TestSDKCompatibility:
    """Test compatibility with Google GenAI SDK types.

    These tests only run if the google-genai SDK is installed.
    They verify that SDK objects' dict representations are compatible
    with our TypedDict definitions.
    """

    def test_sdk_content_to_dict(self):
        """Test that SDK Content.model_dump() is compatible with our Content."""
        try:
            from google.genai import types as sdk_types

            sdk_content = sdk_types.Content(
                parts=[sdk_types.Part(text="Hello from SDK")],
                role="user",
            )
            content_dict = sdk_content.model_dump(exclude_none=True)

            # Verify it matches our Content TypedDict structure
            content: Content = content_dict
            assert content["role"] == "user"
            assert len(content["parts"]) == 1
            assert content["parts"][0]["text"] == "Hello from SDK"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_part_text_to_dict(self):
        """Test that SDK Part with text is compatible with our Part."""
        try:
            from google.genai import types as sdk_types

            sdk_part = sdk_types.Part(text="Hello")
            part_dict = sdk_part.model_dump(exclude_none=True)

            part: Part = part_dict
            assert part["text"] == "Hello"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_part_function_call_to_dict(self):
        """Test that SDK Part with function_call is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_part = sdk_types.Part(
                function_call=sdk_types.FunctionCall(
                    name="get_weather",
                    args={"location": "NYC"},
                )
            )
            part_dict = sdk_part.model_dump(exclude_none=True)

            part: Part = part_dict
            assert part["function_call"]["name"] == "get_weather"
            assert part["function_call"]["args"]["location"] == "NYC"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_function_declaration_to_dict(self):
        """Test that SDK FunctionDeclaration is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_func = sdk_types.FunctionDeclaration(
                name="get_weather",
                description="Get the current weather",
                parameters=sdk_types.Schema(
                    type=cast(Any, "OBJECT"),
                    properties={
                        "location": sdk_types.Schema(type=cast(Any, "STRING")),
                    },
                    required=["location"],
                ),
            )
            func_dict = sdk_func.model_dump(exclude_none=True)

            func_decl: FunctionDeclaration = func_dict
            assert func_decl["name"] == "get_weather"
            assert func_decl["description"] == "Get the current weather"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_tool_to_dict(self):
        """Test that SDK Tool is compatible with our Tool."""
        try:
            from google.genai import types as sdk_types

            sdk_tool = sdk_types.Tool(
                function_declarations=[
                    sdk_types.FunctionDeclaration(
                        name="search",
                        description="Search the web",
                    )
                ]
            )
            tool_dict = sdk_tool.model_dump(exclude_none=True)

            tool: Tool = tool_dict
            assert len(tool["function_declarations"]) == 1
            assert tool["function_declarations"][0]["name"] == "search"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_safety_setting_to_dict(self):
        """Test that SDK SafetySetting is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_safety = sdk_types.SafetySetting(
                category=cast(Any, "HARM_CATEGORY_HARASSMENT"),
                threshold=cast(Any, "BLOCK_MEDIUM_AND_ABOVE"),
            )
            safety_dict = sdk_safety.model_dump(exclude_none=True)

            safety: SafetySetting = safety_dict
            assert safety["category"] == "HARM_CATEGORY_HARASSMENT"
            assert safety["threshold"] == "BLOCK_MEDIUM_AND_ABOVE"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_thinking_config_to_dict(self):
        """Test that SDK ThinkingConfig is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_thinking = sdk_types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=2048,
            )
            thinking_dict = sdk_thinking.model_dump(exclude_none=True)

            thinking: ThinkingConfig = thinking_dict
            assert thinking["include_thoughts"] is True
            assert thinking["thinking_budget"] == 2048

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_generate_content_response_to_dict(self):
        """Test that SDK GenerateContentResponse is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_response = sdk_types.GenerateContentResponse(
                candidates=[
                    sdk_types.Candidate(
                        content=sdk_types.Content(
                            parts=[sdk_types.Part(text="Hello!")],
                            role="model",
                        ),
                        finish_reason=cast(Any, "STOP"),
                        index=0,
                    )
                ],
                usage_metadata=sdk_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
                model_version="gemini-2.0-flash-001",
                response_id="resp_test",
            )
            response_dict = sdk_response.model_dump(exclude_none=True)

            response: GenerateContentResponse = response_dict
            assert len(response["candidates"]) == 1
            assert response["candidates"][0]["finish_reason"] == "STOP"
            assert response["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"
            assert response["usage_metadata"]["total_token_count"] == 15
            assert response["model_version"] == "gemini-2.0-flash-001"

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_candidate_with_safety_ratings(self):
        """Test that SDK Candidate with safety ratings is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_candidate = sdk_types.Candidate(
                content=sdk_types.Content(
                    parts=[sdk_types.Part(text="Safe response")],
                    role="model",
                ),
                finish_reason=cast(Any, "STOP"),
                safety_ratings=[
                    sdk_types.SafetyRating(
                        category=cast(Any, "HARM_CATEGORY_HARASSMENT"),
                        probability=cast(Any, "NEGLIGIBLE"),
                        blocked=False,
                    )
                ],
                index=0,
            )
            candidate_dict = sdk_candidate.model_dump(exclude_none=True)

            candidate: Candidate = candidate_dict
            assert len(candidate["safety_ratings"]) == 1
            assert (
                candidate["safety_ratings"][0]["category"] == "HARM_CATEGORY_HARASSMENT"
            )

        except ImportError:
            pytest.skip("Google GenAI SDK not available")

    def test_sdk_usage_metadata_to_dict(self):
        """Test that SDK UsageMetadata is compatible."""
        try:
            from google.genai import types as sdk_types

            sdk_usage = sdk_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=200,
                total_token_count=300,
                cached_content_token_count=50,
                thoughts_token_count=25,
            )
            usage_dict = sdk_usage.model_dump(exclude_none=True)

            usage: GenerateContentResponseUsageMetadata = usage_dict
            assert usage["prompt_token_count"] == 100
            assert usage["total_token_count"] == 300
            assert usage["thoughts_token_count"] == 25

        except ImportError:
            pytest.skip("Google GenAI SDK not available")


if __name__ == "__main__":
    pytest.main([__file__])
