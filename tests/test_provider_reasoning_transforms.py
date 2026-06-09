"""Tests for provider-specific reasoning transforms (#185).

Covers MiniMax, OpenRouter reasoning field normalization transforms.
"""

from __future__ import annotations

from llm_rosetta.shims.providers.minimax.transforms import (
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
