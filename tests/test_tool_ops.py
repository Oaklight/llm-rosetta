"""Tests for the tool_ops convenience API."""

import pytest

from llm_rosetta import tool_ops
from llm_rosetta.types.ir.tools import ToolDefinition

# Shared fixture: a minimal IR tool definition
IR_TOOL: ToolDefinition = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


# ==================== to_* shortcuts ====================


class TestToProvider:
    """Test IR → provider conversion shortcuts."""

    def test_to_openai_chat(self):
        result = tool_ops.to_openai_chat(IR_TOOL)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert "parameters" in result["function"]

    def test_to_openai_responses(self):
        result = tool_ops.to_openai_responses(IR_TOOL)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert "parameters" in result

    def test_to_anthropic(self):
        result = tool_ops.to_anthropic(IR_TOOL)
        assert result["name"] == "get_weather"
        assert "input_schema" in result

    def test_to_google_genai(self):
        result = tool_ops.to_google_genai(IR_TOOL)
        # Google wraps in function_declarations
        assert "function_declarations" in result
        decl = result["function_declarations"][0]
        assert decl["name"] == "get_weather"


# ==================== from_* shortcuts ====================


class TestFromProvider:
    """Test provider → IR conversion shortcuts."""

    def test_from_openai_chat(self):
        provider_tool = tool_ops.to_openai_chat(IR_TOOL)
        recovered = tool_ops.from_openai_chat(provider_tool)
        assert recovered is not None
        assert recovered["name"] == "get_weather"
        assert recovered["type"] == "function"

    def test_from_openai_responses(self):
        provider_tool = tool_ops.to_openai_responses(IR_TOOL)
        recovered = tool_ops.from_openai_responses(provider_tool)
        assert recovered is not None
        assert recovered["name"] == "get_weather"

    def test_from_anthropic(self):
        provider_tool = tool_ops.to_anthropic(IR_TOOL)
        recovered = tool_ops.from_anthropic(provider_tool)
        assert recovered is not None
        assert recovered["name"] == "get_weather"

    def test_from_google_genai(self):
        provider_tool = tool_ops.to_google_genai(IR_TOOL)
        recovered = tool_ops.from_google_genai(provider_tool)
        # Google may return a list of ToolDefinitions
        if isinstance(recovered, list):
            assert len(recovered) >= 1
            assert recovered[0]["name"] == "get_weather"
        else:
            assert recovered is not None
            assert recovered["name"] == "get_weather"


# ==================== Unified dispatch ====================


class TestUnifiedDispatch:
    """Test to_provider / from_provider dispatch."""

    @pytest.mark.parametrize(
        "provider",
        ["openai_chat", "openai_responses", "anthropic", "google"],
    )
    def test_to_provider_canonical(self, provider: str):
        result = tool_ops.to_provider(IR_TOOL, provider=provider)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("openai-chat", "openai_chat"),
            ("openai-responses", "openai_responses"),
            ("google-genai", "google"),
        ],
    )
    def test_to_provider_aliases(self, alias: str, canonical: str):
        result_alias = tool_ops.to_provider(IR_TOOL, provider=alias)  # ty: ignore[invalid-argument-type]
        result_canonical = tool_ops.to_provider(IR_TOOL, provider=canonical)  # ty: ignore[invalid-argument-type]
        assert result_alias == result_canonical

    def test_to_provider_invalid(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            tool_ops.to_provider(IR_TOOL, provider="not_a_provider")  # ty: ignore[invalid-argument-type]

    def test_from_provider_invalid(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            tool_ops.from_provider({}, provider="not_a_provider")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        "provider",
        ["openai_chat", "openai_responses", "anthropic"],
    )
    def test_round_trip(self, provider: str):
        """to_provider then from_provider should recover the tool name."""
        provider_tool = tool_ops.to_provider(IR_TOOL, provider=provider)  # ty: ignore[invalid-argument-type]
        recovered = tool_ops.from_provider(provider_tool, provider=provider)  # ty: ignore[invalid-argument-type]
        assert recovered is not None
        if isinstance(recovered, list):
            assert recovered[0]["name"] == "get_weather"
        else:
            assert recovered["name"] == "get_weather"

    def test_round_trip_google(self):
        """Google round-trip (may return list)."""
        provider_tool = tool_ops.to_provider(IR_TOOL, provider="google")
        recovered = tool_ops.from_provider(provider_tool, provider="google")
        if isinstance(recovered, list):
            assert any(t["name"] == "get_weather" for t in recovered)
        else:
            assert recovered is not None
            assert recovered["name"] == "get_weather"
