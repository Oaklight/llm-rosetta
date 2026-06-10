"""Tests for provider-specific reasoning transforms (#185).

Covers MiniMax, OpenRouter reasoning field normalization transforms.
"""

from __future__ import annotations

from llm_rosetta.shims.providers.minimax.openai_chat.transforms import (
    _inject_reasoning_split,
    _parse_think_tags,
)
from llm_rosetta.shims.providers.openrouter.transforms import (
    _rename_reasoning_field,
)


# ===========================================================================
# MiniMax: _inject_reasoning_split
# ===========================================================================


class TestInjectReasoningSplit:
    def test_injects_when_thinking_present(self):
        body = {"thinking": {"type": "adaptive"}, "messages": []}
        result = _inject_reasoning_split(body)
        assert result["reasoning_split"] is True

    def test_no_inject_without_thinking(self):
        body = {"messages": []}
        result = _inject_reasoning_split(body)
        assert "reasoning_split" not in result

    def test_no_overwrite_existing_reasoning_split(self):
        body = {"thinking": {"type": "adaptive"}, "reasoning_split": False}
        result = _inject_reasoning_split(body)
        assert result["reasoning_split"] is False


# ===========================================================================
# MiniMax: _parse_think_tags
# ===========================================================================


class TestParseThinkTags:
    def test_single_think_tag(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Let me reason.</think>The answer is 42.",
                    }
                }
            ]
        }
        result = _parse_think_tags(body)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me reason."
        assert msg["content"] == "The answer is 42."

    def test_multiple_think_tags(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Step 1.</think>Middle text.<think>Step 2.</think>Final.",
                    }
                }
            ]
        }
        result = _parse_think_tags(body)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Step 1.\nStep 2."
        assert msg["content"] == "Middle text.Final."

    def test_no_think_tags(self):
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "Just a normal response."}}
            ]
        }
        result = _parse_think_tags(body)
        msg = result["choices"][0]["message"]
        assert "reasoning_content" not in msg
        assert msg["content"] == "Just a normal response."

    def test_skip_when_reasoning_content_exists(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Should not parse.</think>Answer.",
                        "reasoning_content": "Already split.",
                    }
                }
            ]
        }
        result = _parse_think_tags(body)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Already split."
        assert "<think>" in msg["content"]

    def test_no_choices(self):
        body = {"error": "something"}
        result = _parse_think_tags(body)
        assert result == body


# ===========================================================================
# OpenRouter: _rename_reasoning_field
# ===========================================================================


class TestRenameReasoningField:
    def test_renames_reasoning_to_reasoning_content(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning": "Let me think...",
                    }
                }
            ]
        }
        result = _rename_reasoning_field(body)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me think..."
        assert "reasoning" not in msg

    def test_no_rename_when_reasoning_content_exists(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning": "Original.",
                        "reasoning_content": "Already there.",
                    }
                }
            ]
        }
        result = _rename_reasoning_field(body)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "Already there."
        assert msg["reasoning"] == "Original."

    def test_no_reasoning_field(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": "No reasoning."}}]
        }
        result = _rename_reasoning_field(body)
        msg = result["choices"][0]["message"]
        assert "reasoning_content" not in msg

    def test_no_choices(self):
        body = {"error": "something"}
        result = _rename_reasoning_field(body)
        assert result == body


# ===========================================================================
# OpenAI Chat: reasoning metadata round-trip
# ===========================================================================


class TestOpenAIChatReasoningMetadataRoundTrip:
    """Test reasoning_details and encrypted_content survive IR round-trip."""

    def test_reasoning_details_preserved(self):
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        conv = OpenAIChatConverter()
        resp = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning_content": "Let me think...",
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "text": "Let me think...",
                                "format": "anthropic-claude-v1",
                                "index": 0,
                                "signature": "sig_abc123",
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        ir = conv.response_from_provider(resp)
        out = conv.response_to_provider(ir)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me think..."
        assert "reasoning_details" in msg
        assert msg["reasoning_details"][0]["signature"] == "sig_abc123"

    def test_encrypted_content_preserved(self):
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        conv = OpenAIChatConverter()
        resp = {
            "id": "c2",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning_content": "Thinking...",
                        "encrypted_content": "gAAAAABq_encrypted_data_here",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        ir = conv.response_from_provider(resp)
        out = conv.response_to_provider(ir)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "Thinking..."
        assert msg["encrypted_content"] == "gAAAAABq_encrypted_data_here"

    def test_no_metadata_when_absent(self):
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        conv = OpenAIChatConverter()
        resp = {
            "id": "c3",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning_content": "Plain reasoning.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        ir = conv.response_from_provider(resp)
        out = conv.response_to_provider(ir)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "Plain reasoning."
        assert "reasoning_details" not in msg
        assert "encrypted_content" not in msg
