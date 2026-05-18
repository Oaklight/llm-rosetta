"""Tests for argo_anthropic shim transforms.

Covers both the request-side ``_normalize_thinking`` and the response-side
``_normalize_openai_response`` transforms.
"""

from __future__ import annotations

import json

from llm_rosetta.shims.providers.argo_anthropic.transforms import (
    _normalize_openai_response,
    _normalize_thinking,
)


# ---------------------------------------------------------------------------
# _normalize_thinking (to_transform, request-side)
# ---------------------------------------------------------------------------


class TestNormalizeThinking:
    def test_no_thinking_passthrough(self):
        body: dict = {"model": "claudeopus46", "max_tokens": 1024}
        result = _normalize_thinking(body)
        assert result is body
        assert "thinking" not in result

    def test_enabled_unchanged(self):
        body: dict = {
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 512},
        }
        result = _normalize_thinking(body)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 512

    def test_adaptive_converted_to_enabled(self):
        body: dict = {"max_tokens": 2048, "thinking": {"type": "adaptive"}}
        result = _normalize_thinking(body)
        assert result["thinking"]["type"] == "enabled"
        assert "budget_tokens" in result["thinking"]
        # budget = max(1024, int(2048 * 0.8)) = 1638
        assert result["thinking"]["budget_tokens"] == 1638

    def test_adaptive_budget_less_than_max_tokens(self):
        body: dict = {"max_tokens": 2000, "thinking": {"type": "adaptive"}}
        result = _normalize_thinking(body)
        budget = result["thinking"]["budget_tokens"]
        assert budget < result["max_tokens"], "budget_tokens must be < max_tokens"

    def test_adaptive_small_max_tokens_bumps_max_tokens(self):
        """When max_tokens is very small, max_tokens is bumped to keep invariant."""
        body: dict = {"max_tokens": 100, "thinking": {"type": "adaptive"}}
        result = _normalize_thinking(body)
        budget = result["thinking"]["budget_tokens"]
        assert budget >= 1024
        assert budget < result["max_tokens"]

    def test_adaptive_existing_budget_tokens_preserved(self):
        """If budget_tokens is already set, do not overwrite it."""
        body: dict = {
            "max_tokens": 4096,
            "thinking": {"type": "adaptive", "budget_tokens": 999},
        }
        result = _normalize_thinking(body)
        assert result["thinking"]["budget_tokens"] == 999

    def test_non_dict_thinking_passthrough(self):
        body: dict = {"thinking": "enabled"}
        result = _normalize_thinking(body)
        assert result["thinking"] == "enabled"


# ---------------------------------------------------------------------------
# _normalize_openai_response (from_transform, response-side)
# ---------------------------------------------------------------------------


class TestNormalizeOpenAIResponse:
    # --- pass-through cases ---

    def test_anthropic_format_passthrough(self):
        """Responses that already have 'content' are returned unchanged."""
        body = {
            "id": "msg_001",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
            "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        result = _normalize_openai_response(body)
        assert result is body

    def test_type_message_passthrough(self):
        """Responses with 'type: message' are returned unchanged."""
        body = {"type": "message", "role": "assistant"}
        result = _normalize_openai_response(body)
        assert result is body

    def test_empty_choices_passthrough(self):
        body = {"id": "x", "choices": []}
        result = _normalize_openai_response(body)
        assert result is body

    def test_no_choices_passthrough(self):
        body = {"id": "x", "model": "some-model"}
        result = _normalize_openai_response(body)
        assert result is body

    def test_bad_choice_type_passthrough(self):
        body = {"choices": ["not-a-dict"]}
        result = _normalize_openai_response(body)
        assert result is body

    def test_no_message_in_choice_passthrough(self):
        body = {"choices": [{"index": 0, "finish_reason": "stop"}]}
        result = _normalize_openai_response(body)
        assert result is body

    # --- conversion cases ---

    def test_plain_text_response(self):
        """Simple OpenAI Chat format with text content is converted correctly."""
        body = {
            "id": "chatcmpl-001",
            "object": "chat.completion",
            "model": "claudeopus46",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        result = _normalize_openai_response(body)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 4
        # OpenAI-specific keys should be gone
        assert "choices" not in result
        assert "object" not in result
        # Non-format keys preserved
        assert result["id"] == "chatcmpl-001"
        assert result["model"] == "claudeopus46"

    def test_finish_reason_length_maps_to_max_tokens(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "..."},
                    "finish_reason": "length",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert result["stop_reason"] == "max_tokens"

    def test_finish_reason_tool_calls_maps_to_tool_use(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None, "tool_calls": []},
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert result["stop_reason"] == "tool_use"

    def test_finish_reason_unknown_defaults_to_end_turn(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "some_unknown_reason",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert result["stop_reason"] == "end_turn"

    def test_null_content_produces_no_text_block(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert result["content"] == []

    def test_tool_calls_converted_to_tool_use_blocks(self):
        tool_input = {"location": "Chicago"}
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps(tool_input),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_abc"
        assert block["name"] == "get_weather"
        assert block["input"] == tool_input

    def test_tool_calls_with_bad_json_arguments(self):
        """Malformed tool arguments fall back to empty dict."""
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "function": {
                                    "name": "bad_tool",
                                    "arguments": "not-valid-json{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = _normalize_openai_response(body)
        block = result["content"][0]
        assert block["input"] == {}

    def test_mixed_text_and_tool_calls(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q": "hello"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Let me check that."}
        assert result["content"][1]["type"] == "tool_use"

    def test_usage_missing_defaults_to_zero(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = _normalize_openai_response(body)
        assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}

    def test_original_body_not_mutated(self):
        """The original body dict should not be mutated."""
        body = {
            "id": "chatcmpl-x",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        original_choices = body["choices"]
        _normalize_openai_response(body)
        # Original still has choices
        assert body["choices"] is original_choices

    def test_real_argo_claudeopus46_response(self):
        """Reproduce the exact format seen in the lambda5 502 errors.

        Argo returns OpenAI Chat format from ``/v1/messages`` for claudeopus46.
        After normalization the Anthropic converter must be able to parse it.
        """
        body = {
            "id": "chatcmpl-argo46",
            "object": "chat.completion",
            "created": 1779058858,
            "model": "claudeopus46",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you?",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 9,
                "total_tokens": 22,
            },
        }
        result = _normalize_openai_response(body)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [
            {"type": "text", "text": "Hello! How can I assist you?"}
        ]
        assert result["stop_reason"] == "end_turn"
        assert result["model"] == "claudeopus46"

        # Verify the Anthropic converter can now parse this
        from llm_rosetta.converters.anthropic import AnthropicConverter

        conv = AnthropicConverter()
        ir = conv.response_from_provider(result)
        choices = ir.get("choices", [])
        assert choices
        msg = choices[0].get("message", {})
        parts = msg.get("content", [])
        assert any(p.get("type") == "text" for p in parts)
