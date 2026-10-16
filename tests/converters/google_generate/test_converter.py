"""
Google GenAI Converter integration tests.

Tests the top-level GoogleGenerateConverter with full request/response conversion.
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.base.helpers.citations import citation_url
from llm_rosetta.converters.google_generate import (
    GoogleConverter,
    GoogleGenerateConverter,
)
from llm_rosetta.types.ir import (
    IRRequest,
    IRResponse,
    Message,
)


class TestGoogleGenerateConverter:
    """Integration tests for GoogleGenerateConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = GoogleGenerateConverter()

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
            "system_instruction": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        # The SDK shape nests it in ``config`` — that is where
        # ``google.genai``'s GenerateContentConfig declares it (#936).
        assert "system_instruction" not in result
        assert result["config"]["system_instruction"]["parts"][0]["text"] == (
            "You are a helpful assistant."
        )

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
        assert (
            result["config"]["system_instruction"]["parts"][0]["text"] == "Be helpful."
        )
        # System message should not be in contents list
        assert all(c["role"] != "system" for c in result["contents"])

    def test_sdk_shape_is_exactly_generate_content_kwargs(self):
        """The SDK dict must be splattable into ``generate_content(**body)`` (#936).

        ``Models.generate_content()`` accepts only ``model``, ``contents`` and
        ``config``.  A top-level ``system_instruction`` raised ``TypeError`` on
        every google-genai version tested (1.0.0 → 2.29.0), which is why the
        instruction now lives in ``config``.
        """
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "system_instruction": [{"type": "text", "text": "Be helpful."}],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert set(result) == {"model", "contents", "config"}
        assert "system_instruction" not in result

    def test_rest_shape_keeps_top_level_system_instruction(self):
        """The REST wire form is unchanged by #936: top-level, and no ``config``."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "system_instruction": [{"type": "text", "text": "Be helpful."}],
        }
        result, warnings = self.converter.request_to_provider(
            ir_request, output_format="rest"
        )
        assert "config" not in result
        assert result["system_instruction"]["parts"][0]["text"] == "Be helpful."

    def test_both_placements_resolve_the_same_way(self):
        """Reading back and lifting to REST agree when a dict carries both (#936).

        The converter never writes both, but a hand-built request can.  Since
        ``config`` is the shape's authoritative location, it wins in both
        directions — otherwise the same dict would yield one instruction via
        ``request_from_provider`` and another on the REST wire.
        """
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {
                "system_instruction": {"role": "user", "parts": [{"text": "CONFIG"}]}
            },
            "system_instruction": {"role": "user", "parts": [{"text": "TOPLEVEL"}]},
        }

        restored = self.converter.request_from_provider(provider_request)
        assert restored["system_instruction"] == [{"type": "text", "text": "CONFIG"}]

        rest = GoogleGenerateConverter._to_rest_body(provider_request)
        assert rest["system_instruction"]["parts"][0]["text"] == "CONFIG"

    @pytest.mark.parametrize("output_format", ["sdk", "rest"])
    def test_system_instruction_round_trips(self, output_format: str):
        """Both shapes read their system instruction back into IR (#936).

        The pipeline's fidelity shadow check builds a bare ``ConversionContext``
        so a google target always gets the SDK shape — if
        ``request_from_provider`` could not read it out of ``config``, that
        check would report a silent loss.
        """
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "system_instruction": [{"type": "text", "text": "Be helpful."}],
        }
        provider_request, warnings = self.converter.request_to_provider(
            ir_request, output_format=output_format
        )
        restored = self.converter.request_from_provider(provider_request)
        assert restored["system_instruction"] == [
            {"type": "text", "text": "Be helpful."}
        ]

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

    @pytest.mark.parametrize("bad_format", ["REST", "Rest", "restful", "", "sdk "])
    def test_request_rejects_unknown_output_format(self, bad_format):
        """An unrecognised ``output_format`` must not fall through to the SDK body.

        The converter selects REST by exact match and otherwise emits the SDK
        shape, which the Generative Language REST endpoint rejects with HTTP
        400 ``Unknown name "config"`` (#882).  Rejecting the value here keeps a
        typo from reaching the upstream call.
        """
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        with pytest.raises(ValueError, match="output_format"):
            self.converter.request_to_provider(ir_request, output_format=bad_format)

    @pytest.mark.parametrize(
        ("output_format", "present", "absent"),
        [
            ("rest", "generationConfig", "config"),
            ("sdk", "config", "generationConfig"),
        ],
    )
    def test_valid_output_format_selects_expected_body(
        self, output_format: str, present: str, absent: str
    ):
        """Both supported values still reach their own body shape past the guard."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "generation": {"temperature": 0.7, "max_tokens": 1024},
        }
        result, warnings = self.converter.request_to_provider(
            ir_request, output_format=output_format
        )
        assert present in result
        assert absent not in result

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
        assert list(messages[0]["content"])[0]["text"] == "Hello"  # type: ignore

    def test_request_from_provider_with_system_instruction_string(self):
        """Test request from provider with string system instruction."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "system_instruction": "You are helpful.",
            "config": {},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["system_instruction"] == [
            {"type": "text", "text": "You are helpful."}
        ]

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

    def test_request_from_provider_with_system_instruction_camelcase(self):
        """Test request from provider with camelCase systemInstruction (REST API)."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["system_instruction"] == [
            {"type": "text", "text": "Be concise."}
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

    def test_request_from_provider_malformed_tool_raises_with_context(self):
        """Test that malformed tools raise clear errors with tool type/name context."""
        provider_request = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "config": {
                "tools": [42],  # non-dict tool triggers conversion error
            },
        }
        with pytest.raises(ValueError, match=r"Unsupported tool"):
            self.converter.request_from_provider(provider_request)

    # ==================== response_from_provider ====================

    def test_response_from_provider_grounding_metadata(self):
        """Candidate grounding metadata becomes IR URL citations.

        The citation is a first-class part with its own flat fields; Google's
        search-level metadata (queries, rendered entry point) is not a citation
        and rides the message's ``provider_metadata``.
        """
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "According to the source."}],
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com", "title": "Example"}}
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"startIndex": 0, "endIndex": 23},
                                "groundingChunkIndices": [0],
                            }
                        ],
                        "webSearchQueries": ["population of Tokyo"],
                        "searchEntryPoint": {"renderedContent": "<style>x</style>"},
                    },
                }
            ]
        }
        result = self.converter.response_from_provider(provider_response)
        message = result["choices"][0]["message"]
        assert message["content"][1] == {
            "type": "citation",
            "citation_type": "url_citation",
            "url": "https://example.com",
            "title": "Example",
            "start_index": 0,
            "end_index": 23,
            # Which wrapper the source came from, so a write back is faithful.
            "provider_metadata": {"google": {"_chunk_type": "web"}},
        }
        assert message["provider_metadata"]["google"]["web_search_queries"] == [
            "population of Tokyo"
        ]
        assert message["provider_metadata"]["google"]["search_entry_point"] == {
            "rendered_content": "<style>x</style>"
        }

    def test_grounding_search_metadata_is_not_repeated_per_citation(self):
        """The search metadata belongs to the turn, not to each source."""
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Answer"}],
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://a.example", "title": "A"}},
                            {"web": {"uri": "https://b.example", "title": "B"}},
                        ],
                        "webSearchQueries": ["q"],
                    },
                }
            ]
        }
        result = self.converter.response_from_provider(provider_response)
        message = result["choices"][0]["message"]
        citations = [
            part for part in message["content"] if part.get("type") == "citation"
        ]
        assert [citation_url(c) for c in citations] == [
            "https://a.example",
            "https://b.example",
        ]
        # The turn's search metadata is not repeated on the sources; only the
        # chunk wrapper each source came from is kept.
        assert all(
            set(c.get("provider_metadata", {}).get("google", {})) == {"_chunk_type"}
            for c in citations
        )
        assert message["provider_metadata"]["google"]["web_search_queries"] == ["q"]

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
        assert list(choice["message"]["content"])[0]["text"] == "Hello! How can I help?"  # ty: ignore[invalid-key]
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

    def test_response_from_provider_modality_token_details(self):
        """Test Google list[ModalityTokenCount] → IR dict[str, int]."""
        provider_response = {
            "response_id": "resp-modality",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hi"}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 50,
                "candidates_token_count": 20,
                "total_token_count": 70,
                "prompt_tokens_details": [
                    {"modality": "TEXT", "token_count": 40},
                    {"modality": "IMAGE", "token_count": 10},
                ],
                "candidates_tokens_details": [
                    {"modality": "TEXT", "token_count": 20},
                ],
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["prompt_tokens_details"] == {
            "text_tokens": 40,
            "image_tokens": 10,
        }
        assert result["usage"]["completion_tokens_details"] == {
            "text_tokens": 20,
        }

    def test_response_from_provider_modality_camelcase(self):
        """Test camelCase modality token details (REST API format)."""
        provider_response = {
            "responseId": "resp-camel",
            "modelVersion": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hi"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 10,
                "totalTokenCount": 40,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 30},
                ],
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["prompt_tokens_details"] == {"text_tokens": 30}

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

    def test_response_to_provider_modality_token_details(self):
        """Test IR dict[str, int] → Google list[ModalityTokenCount]."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_mod",
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
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                    "prompt_tokens_details": {
                        "text_tokens": 40,
                        "image_tokens": 10,
                    },
                    "completion_tokens_details": {
                        "text_tokens": 20,
                    },
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        prompt_details = result["usageMetadata"]["promptTokensDetails"]
        assert isinstance(prompt_details, list)
        assert len(prompt_details) == 2
        # Convert to dict for easier assertion (order may vary)
        detail_map = {d["modality"]: d["tokenCount"] for d in prompt_details}
        assert detail_map["TEXT"] == 40
        assert detail_map["IMAGE"] == 10

        comp_details = result["usageMetadata"]["candidatesTokensDetails"]
        assert isinstance(comp_details, list)
        assert len(comp_details) == 1
        assert comp_details[0]["modality"] == "TEXT"
        assert comp_details[0]["tokenCount"] == 20

    def test_response_to_provider_modality_filters_non_standard(self):
        """Non-standard modality keys (cached, reasoning) excluded from details array."""
        ir_response = cast(
            IRResponse,
            {
                "id": "resp_filter",
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
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "reasoning_tokens": 30,
                    "cache_read_tokens": 20,
                    "prompt_tokens_details": {
                        "text_tokens": 80,
                        "cached_tokens": 20,
                    },
                    "completion_tokens_details": {
                        "text_tokens": 20,
                        "reasoning_tokens": 30,
                    },
                },
            },
        )
        result = self.converter.response_to_provider(ir_response)
        usage = result["usageMetadata"]
        # Dedicated fields should be present
        assert usage["thoughtsTokenCount"] == 30
        assert usage["cachedContentTokenCount"] == 20
        # Details arrays should only have valid modalities
        prompt_details = usage["promptTokensDetails"]
        modalities = {d["modality"] for d in prompt_details}
        assert "TEXT" in modalities
        assert "CACHED" not in modalities
        comp_details = usage["candidatesTokensDetails"]
        modalities = {d["modality"] for d in comp_details}
        assert "TEXT" in modalities
        assert "REASONING" not in modalities

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
        assert GoogleGenerateConverter._normalize(data) is data

    def test_normalize_pydantic(self):
        """Test _normalize with Pydantic-like object."""

        class MockModel:
            def model_dump(self):
                return {"model": "gemini-2.0-flash"}

        result = GoogleGenerateConverter._normalize(MockModel())
        assert result == {"model": "gemini-2.0-flash"}

    def test_normalize_tuple(self):
        """Test _normalize with tuple input (unwraps first element)."""
        data = ({"key": "value"}, "extra")
        result = GoogleGenerateConverter._normalize(data)
        assert result == {"key": "value"}

    def test_normalize_to_dict(self):
        """Test _normalize with to_dict method."""

        class MockObj:
            def to_dict(self):
                return {"key": "value"}

        result = GoogleGenerateConverter._normalize(MockObj())
        assert result == {"key": "value"}

    def test_normalize_invalid(self):
        """Test _normalize raises on unsupported type."""
        with pytest.raises(TypeError, match="Cannot normalize"):
            GoogleGenerateConverter._normalize(42)

    # ==================== request_to_provider / messages_to_provider ====================

    def test_messages_to_provider_with_system(self):
        """Test messages_to_provider with system + user messages."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        )
        result, warnings = self.converter.messages_to_provider(messages)
        # messages_to_provider returns the converted message list
        assert len(result) >= 1

    def test_request_to_provider_basic_ir_request(self):
        """Test request_to_provider with a simple IRRequest."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
                ],
            },
        )
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["model"] == "gemini-2.0-flash"

    def test_request_to_provider_with_tools(self):
        """Test request_to_provider with tools."""
        ir_request = cast(
            IRRequest,
            {
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )
        result, warnings = self.converter.request_to_provider(ir_request)
        assert len(result["config"]["tools"]) == 1

    # ==================== GoogleConverter Alias ====================

    def test_google_converter_alias(self):
        """Test GoogleConverter is an alias for GoogleGenerateConverter."""
        assert GoogleConverter is GoogleGenerateConverter

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


class TestGoogleGenerateConverterFullRoundTrip:
    """Full round-trip conversion tests."""

    def setup_method(self):
        self.converter = GoogleGenerateConverter()

    def test_request_round_trip(self):
        """Test IRRequest -> Google -> IRRequest round-trip."""
        ir_request: IRRequest = {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "system_instruction": [{"type": "text", "text": "Be helpful."}],
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

    def test_response_round_trip_with_modality_details(self):
        """Test modality token details survive Google → IR → Google round-trip."""
        provider_response = {
            "response_id": "resp-mod-rt",
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hi"}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 50,
                "candidates_token_count": 20,
                "total_token_count": 70,
                "prompt_tokens_details": [
                    {"modality": "TEXT", "token_count": 40},
                    {"modality": "IMAGE", "token_count": 10},
                ],
            },
        }
        ir_response = self.converter.response_from_provider(provider_response)
        restored = self.converter.response_to_provider(ir_response)

        details = restored["usageMetadata"]["promptTokensDetails"]
        assert isinstance(details, list)
        detail_map = {d["modality"]: d["tokenCount"] for d in details}
        assert detail_map["TEXT"] == 40
        assert detail_map["IMAGE"] == 10

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


class TestGooglePromptBlockHandling:
    """Tests for promptFeedback.blockReason and missing finishReason values (#433)."""

    def setup_method(self):
        from llm_rosetta.converters.google_generate import GoogleGenerateConverter

        self.converter = GoogleGenerateConverter()

    def test_prompt_block_produces_content_filter(self):
        """promptFeedback.blockReason with no candidates → content_filter finish."""
        response = {
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
            "usageMetadata": {"promptTokenCount": 14, "totalTokenCount": 14},
        }
        ir = self.converter.response_from_provider(response)
        assert len(ir["choices"]) == 1
        assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"
        assert ir["choices"][0]["message"]["content"] == []

    def test_safety_finish_reason(self):
        """finishReason=SAFETY → content_filter."""
        response = {
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "content": {"role": "model", "parts": []},
                }
            ],
        }
        ir = self.converter.response_from_provider(response)
        assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"

    def test_blocklist_finish_reason(self):
        """finishReason=BLOCKLIST → content_filter."""
        response = {
            "candidates": [
                {
                    "finishReason": "BLOCKLIST",
                    "content": {"role": "model", "parts": [{"text": ""}]},
                }
            ],
        }
        ir = self.converter.response_from_provider(response)
        assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"

    def test_spii_finish_reason(self):
        """finishReason=SPII → content_filter."""
        response = {
            "candidates": [
                {
                    "finishReason": "SPII",
                    "content": {"role": "model", "parts": []},
                }
            ],
        }
        ir = self.converter.response_from_provider(response)
        assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"

    def test_streaming_prompt_block(self):
        """Streaming chunk with promptFeedback.blockReason → finish + stream_end."""
        from llm_rosetta.converters.base.context import StreamContext

        ctx = StreamContext()
        ctx.mark_started()
        ctx.response_id = "test"
        ctx.model = "gemini"

        chunk = {
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
            "usageMetadata": {"promptTokenCount": 14, "totalTokenCount": 14},
        }
        events = self.converter.stream_response_from_provider(chunk, context=ctx)
        types = [e["type"] for e in events]
        assert "finish" in types
        assert "stream_end" in types
        finish = [e for e in events if e["type"] == "finish"][0]
        assert finish["finish_reason"]["reason"] == "content_filter"

    def test_refusal_wire_body_is_clean(self):
        """IR refusal → Google response carries no internal carrier (#941).

        Google has no native refusal type, so the converter emits the refusal
        text plus an internal ``_llm_rosetta_provider_metadata`` marker for round-trip
        recovery.  That marker must not reach the wire — Google rejects an
        unknown field — so ``response_to_provider`` strips it.
        """
        ir = {
            "id": "test",
            "object": "response",
            "created": 1700000000,
            "model": "gemini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "I cannot help."}],
                    },
                    "finish_reason": {"reason": "content_filter"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
        google_resp = self.converter.response_to_provider(ir)  # ty: ignore[invalid-argument-type]
        parts = google_resp["candidates"][0]["content"]["parts"]
        assert len(parts) == 1
        assert parts[0]["text"] == "I cannot help."
        assert "_llm_rosetta_provider_metadata" not in parts[0]

        # Reading the clean wire body back yields text: the refusal marker only
        # survives the ops-level round trip (see test_refusal_ops_round_trip).
        ir2 = self.converter.response_from_provider(google_resp)
        content = ir2["choices"][0]["message"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_refusal_ops_round_trip(self):
        """The ops-level refusal round trip still recovers the refusal (#941)."""
        from llm_rosetta.converters.google_generate.content_ops import (
            GoogleGenerateContentOps,
        )

        part = GoogleGenerateContentOps.ir_refusal_to_p(
            cast(Any, {"type": "refusal", "refusal": "I cannot help."})
        )
        assert part["_llm_rosetta_provider_metadata"]["is_refusal"] is True

        restored = GoogleGenerateContentOps.p_part_to_ir(part)
        assert len(restored) == 1
        assert restored[0]["type"] == "refusal"
        assert restored[0]["refusal"] == "I cannot help."

    def test_rest_thinking_config_round_trip(self):
        """REST generationConfig.thinkingConfig survives request round-trip (regression for #584)."""
        rest_request = {
            "contents": [{"role": "user", "parts": [{"text": "What is 2+2?"}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "low",
                    "includeThoughts": True,
                }
            },
        }
        ir = self.converter.request_from_provider(rest_request)
        assert ir["reasoning"]["effort"] == "low"
        assert ir["reasoning"]["include_thoughts"] is True
        target, _ = self.converter.request_to_provider(ir)
        config = target.get("config", {})
        assert config["thinking_config"]["thinking_level"] == "low"
        assert config["thinking_config"]["include_thoughts"] is True

    def test_rest_thinking_config_not_lost_in_pipeline(self):
        """Pipeline Google→Google preserves thinkingConfig from REST format (regression for #584)."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("google", "google")
        rest_request = {
            "contents": [{"role": "user", "parts": [{"text": "test"}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "high",
                    "includeThoughts": True,
                }
            },
        }
        target = pipeline.convert_request(rest_request)
        gen_config = target.get("generationConfig", {})
        assert "thinkingConfig" in gen_config
        assert gen_config["thinkingConfig"]["thinkingLevel"] == "high"
        assert gen_config["thinkingConfig"]["includeThoughts"] is True

    def test_to_rest_body_preserves_thinking_config(self):
        """Test _to_rest_body lifts thinking_config to generationConfig.thinkingConfig."""
        sdk_request = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {
                "temperature": 0.7,
                "thinking_config": {
                    "thinking_level": "high",
                    "include_thoughts": True,
                },
            },
        }
        rest_body = self.converter._to_rest_body(sdk_request)
        assert rest_body["generationConfig"]["temperature"] == 0.7
        assert rest_body["generationConfig"]["thinkingConfig"] == {
            "thinkingLevel": "high",
            "includeThoughts": True,
        }
