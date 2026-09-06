"""Cross-format round-trip tests for Google Interactions converter.

Tests A→IR→B→IR→A fidelity for request and response conversion
across all supported provider pairs involving google_interactions.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta import get_converter_for_provider


# ── Helpers ────────────────────────────────────────────────────────


def roundtrip_request(
    body: dict[str, Any],
    src: str,
    via: str,
) -> dict[str, Any]:
    """A→IR→B→IR→A request round-trip."""
    src_conv = get_converter_for_provider(src)
    via_conv = get_converter_for_provider(via)

    ir = src_conv.request_from_provider(body)
    mid, _ = via_conv.request_to_provider(ir)
    ir2 = via_conv.request_from_provider(mid)
    back, _ = src_conv.request_to_provider(ir2)
    return back


def roundtrip_response(
    body: dict[str, Any],
    src: str,
    via: str,
) -> dict[str, Any]:
    """A→IR→B→IR→A response round-trip."""
    src_conv = get_converter_for_provider(src)
    via_conv = get_converter_for_provider(via)

    ir = src_conv.response_from_provider(body)
    mid = via_conv.response_to_provider(ir)
    ir2 = via_conv.response_from_provider(mid)
    back = src_conv.response_to_provider(ir2)
    return back


# ── Fixtures ───────────────────────────────────────────────────────

INTERACTIONS_REQUEST_SIMPLE = {
    "model": "gemini-3.5-flash",
    "input": "Say hello.",
}

INTERACTIONS_REQUEST_MULTI_TURN = {
    "model": "gemini-3.5-flash",
    "input": [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "My name is Alice."}],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "Hello Alice!"}]},
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "What is my name?"}],
        },
    ],
}

INTERACTIONS_REQUEST_WITH_TOOLS = {
    "model": "gemini-3.5-flash",
    "input": "What is the weather?",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
}

INTERACTIONS_RESPONSE_SIMPLE = {
    "id": "int_123",
    "object": "interaction",
    "model": "gemini-3.5-flash",
    "status": "completed",
    "created": "2025-12-04T15:01:45Z",
    "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "Hello!"}]},
    ],
    "usage": {
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "total_tokens": 15,
    },
}

INTERACTIONS_RESPONSE_WITH_THINKING = {
    "id": "int_456",
    "object": "interaction",
    "model": "gemini-3.5-flash",
    "status": "completed",
    "created": "2025-12-04T15:01:45Z",
    "steps": [
        {
            "type": "thought",
            "signature": "sig_abc",
            "summary": [{"type": "text", "text": "Thinking..."}],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "42"}]},
    ],
    "usage": {
        "total_input_tokens": 20,
        "total_output_tokens": 10,
        "total_tokens": 80,
        "total_thought_tokens": 50,
    },
}

INTERACTIONS_RESPONSE_TOOL_CALL = {
    "id": "int_789",
    "object": "interaction",
    "model": "gemini-3.5-flash",
    "status": "requires_action",
    "created": "2025-12-04T15:01:45Z",
    "steps": [
        {
            "type": "function_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": {"location": "SF"},
        },
    ],
}

VIA_PROVIDERS = ["openai_chat", "anthropic", "google_generate"]


# ── Request round-trip tests ───────────────────────────────────────


class TestRequestRoundtrip:
    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_simple_request(self, via: str):
        result = roundtrip_request(
            INTERACTIONS_REQUEST_SIMPLE, "google_interactions", via
        )
        assert result["model"] == "gemini-3.5-flash"
        if isinstance(result["input"], str):
            assert "hello" in result["input"].lower()
        else:
            user_steps = [s for s in result["input"] if s.get("type") == "user_input"]
            assert len(user_steps) >= 1

    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_multi_turn_request(self, via: str):
        result = roundtrip_request(
            INTERACTIONS_REQUEST_MULTI_TURN, "google_interactions", via
        )
        assert result["model"] == "gemini-3.5-flash"
        inp = result["input"]
        assert isinstance(inp, list)
        types = [s.get("type") for s in inp]
        assert "user_input" in types

    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_tools_preserved(self, via: str):
        result = roundtrip_request(
            INTERACTIONS_REQUEST_WITH_TOOLS, "google_interactions", via
        )
        assert "tools" in result
        assert any(t.get("name") == "get_weather" for t in result["tools"])


# ── Response round-trip tests ──────────────────────────────────────


class TestResponseRoundtrip:
    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_simple_response(self, via: str):
        result = roundtrip_response(
            INTERACTIONS_RESPONSE_SIMPLE, "google_interactions", via
        )
        assert result["status"] == "completed"
        model_outputs = [
            s for s in result.get("steps", []) if s.get("type") == "model_output"
        ]
        assert len(model_outputs) >= 1
        assert any(
            "Hello" in c.get("text", "")
            for s in model_outputs
            for c in s.get("content", [])
        )

    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_tool_call_response(self, via: str):
        result = roundtrip_response(
            INTERACTIONS_RESPONSE_TOOL_CALL, "google_interactions", via
        )
        assert result["status"] == "requires_action"
        fc_steps = [
            s for s in result.get("steps", []) if s.get("type") == "function_call"
        ]
        assert len(fc_steps) >= 1
        assert fc_steps[0]["name"] == "get_weather"

    @pytest.mark.parametrize("via", VIA_PROVIDERS)
    def test_usage_preserved(self, via: str):
        result = roundtrip_response(
            INTERACTIONS_RESPONSE_SIMPLE, "google_interactions", via
        )
        usage = result.get("usage", {})
        assert usage.get("total_input_tokens", 0) > 0
        assert usage.get("total_output_tokens", 0) > 0


# ── Reverse direction: other formats → Interactions → back ─────────


OPENAI_CHAT_REQUEST = {
    "model": "gemini-3.5-flash",
    "messages": [
        {"role": "user", "content": "Hello!"},
    ],
}

OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1733324505,
    "model": "gemini-3.5-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class TestReverseRoundtrip:
    def test_openai_chat_via_interactions_request(self):
        result = roundtrip_request(
            OPENAI_CHAT_REQUEST, "openai_chat", "google_interactions"
        )
        assert "messages" in result
        user_msgs = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_msgs) >= 1

    def test_openai_chat_via_interactions_response(self):
        result = roundtrip_response(
            OPENAI_CHAT_RESPONSE, "openai_chat", "google_interactions"
        )
        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["choices"][0]["finish_reason"] == "stop"
