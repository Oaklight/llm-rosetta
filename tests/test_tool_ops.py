"""Tests for the tool_ops convenience API."""

import pytest

from llm_rosetta import tool_ops
from llm_rosetta.types.ir.tools import ToolDefinition

# Shared fixtures
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

IR_CALL = {
    "type": "tool_call",
    "tool_call_id": "call_123",
    "tool_name": "get_weather",
    "tool_input": {"city": "London"},
}

IR_RESULT = {
    "type": "tool_result",
    "tool_call_id": "call_123",
    "result": "Sunny, 22°C",
}

ALL_PROVIDERS = [
    "messages",
    "google_generate",
    "google_interactions",
    "chat_completions",
    "open_responses",
]

# Deprecated spellings still accepted by the convenience API; each maps to the
# canonical name it resolves to.  All are removed in 0.16.0 (#919 / #921).
DEPRECATED_ALIASES = [
    ("openai_chat", "chat_completions"),
    ("openai-chat", "chat_completions"),
    ("chat-completions", "chat_completions"),
    ("openai_responses", "open_responses"),
    ("openai-responses", "open_responses"),
    ("openai--open_responses", "open_responses"),
    ("open-responses", "open_responses"),
    ("google", "google_generate"),
    ("google-genai", "google_generate"),
    ("google-generate", "google_generate"),
    ("google-interactions", "google_interactions"),
]


# ==================== to_* shortcuts ====================


class TestToProvider:
    """Test IR → provider conversion shortcuts."""

    def test_to_chat_completions(self):
        result = tool_ops.to_chat_completions(IR_TOOL)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert "parameters" in result["function"]

    def test_to_open_responses(self):
        result = tool_ops.to_open_responses(IR_TOOL)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert "parameters" in result

    def test_deprecated_to_aliases_warn_and_match(self):
        with pytest.warns(DeprecationWarning):
            legacy_chat = tool_ops.to_openai_chat(IR_TOOL)
        with pytest.warns(DeprecationWarning):
            legacy_responses = tool_ops.to_openai_responses(IR_TOOL)
        assert legacy_chat == tool_ops.to_chat_completions(IR_TOOL)
        assert legacy_responses == tool_ops.to_open_responses(IR_TOOL)

    def test_to_messages(self):
        result = tool_ops.to_messages(IR_TOOL)
        assert result["name"] == "get_weather"
        assert "input_schema" in result

    def test_to_google_generate(self):
        result = tool_ops.to_google_generate(IR_TOOL)
        assert "function_declarations" in result
        decl = result["function_declarations"][0]
        assert decl["name"] == "get_weather"

    def test_to_google_interactions(self):
        result = tool_ops.to_google_interactions(IR_TOOL)
        assert isinstance(result, dict)
        assert result["name"] == "get_weather"


# ==================== from_* shortcuts ====================


class TestFromProvider:
    """Test provider → IR conversion shortcuts."""

    def test_from_chat_completions(self):
        provider_tool = tool_ops.to_chat_completions(IR_TOOL)
        recovered = tool_ops.from_chat_completions(provider_tool)
        assert recovered is not None
        assert recovered["name"] == "get_weather"
        assert recovered["type"] == "function"

    def test_from_open_responses(self):
        provider_tool = tool_ops.to_open_responses(IR_TOOL)
        recovered = tool_ops.from_open_responses(provider_tool)
        assert recovered is not None
        assert isinstance(recovered, dict)
        assert recovered["name"] == "get_weather"

    def test_deprecated_from_aliases_warn_and_match(self):
        with pytest.warns(DeprecationWarning):
            legacy_chat = tool_ops.from_openai_chat(
                tool_ops.to_chat_completions(IR_TOOL)
            )
        with pytest.warns(DeprecationWarning):
            legacy_responses = tool_ops.from_openai_responses(
                tool_ops.to_open_responses(IR_TOOL)
            )
        assert legacy_chat == tool_ops.from_chat_completions(
            tool_ops.to_chat_completions(IR_TOOL)
        )
        assert legacy_responses == tool_ops.from_open_responses(
            tool_ops.to_open_responses(IR_TOOL)
        )

    def test_from_messages(self):
        provider_tool = tool_ops.to_messages(IR_TOOL)
        recovered = tool_ops.from_messages(provider_tool)
        assert recovered is not None
        assert recovered["name"] == "get_weather"

    def test_from_google_generate(self):
        provider_tool = tool_ops.to_google_generate(IR_TOOL)
        recovered = tool_ops.from_google_generate(provider_tool)
        if isinstance(recovered, list):
            assert len(recovered) >= 1
            assert recovered[0]["name"] == "get_weather"
        else:
            assert recovered is not None
            assert recovered["name"] == "get_weather"

    def test_from_google_interactions(self):
        provider_tool = tool_ops.to_google_interactions(IR_TOOL)
        recovered = tool_ops.from_google_interactions(provider_tool)
        if isinstance(recovered, list):
            assert len(recovered) >= 1
            assert recovered[0]["name"] == "get_weather"
        else:
            assert recovered is not None
            assert recovered["name"] == "get_weather"


# ==================== Unified definition dispatch ====================


class TestUnifiedDispatch:
    """Test to_provider / from_provider dispatch."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_to_provider_canonical(self, provider: str):
        result = tool_ops.to_provider(IR_TOOL, provider=provider)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, dict)

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_canonical_names_do_not_warn(self, provider: str):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = tool_ops.to_provider(IR_TOOL, provider=provider)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, dict)

    @pytest.mark.parametrize("alias,canonical", DEPRECATED_ALIASES)
    def test_to_provider_aliases(self, alias: str, canonical: str):
        with pytest.warns(DeprecationWarning):
            result_alias = tool_ops.to_provider(IR_TOOL, provider=alias)  # ty: ignore[invalid-argument-type]
        result_canonical = tool_ops.to_provider(IR_TOOL, provider=canonical)  # ty: ignore[invalid-argument-type]
        assert result_alias == result_canonical

    def test_to_provider_invalid(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            tool_ops.to_provider(IR_TOOL, provider="not_a_provider")  # ty: ignore[invalid-argument-type]

    def test_from_provider_invalid(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            tool_ops.from_provider({}, provider="not_a_provider")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_round_trip(self, provider: str):
        """to_provider then from_provider should recover the tool name."""
        provider_tool = tool_ops.to_provider(IR_TOOL, provider=provider)  # ty: ignore[invalid-argument-type]
        recovered = tool_ops.from_provider(provider_tool, provider=provider)  # ty: ignore[invalid-argument-type]
        if isinstance(recovered, list):
            assert any(t["name"] == "get_weather" for t in recovered)
        else:
            assert recovered is not None
            assert recovered["name"] == "get_weather"


# ==================== Choice dispatch ====================


class TestChoiceDispatch:
    """Test tool choice conversion dispatch."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_choice_auto(self, provider: str):
        ir_choice = {"mode": "auto"}
        result = tool_ops.choice_to_provider(ir_choice, provider=provider)  # ty: ignore[invalid-argument-type]
        assert result is not None

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_choice_none(self, provider: str):
        ir_choice = {"mode": "none"}
        result = tool_ops.choice_to_provider(ir_choice, provider=provider)  # ty: ignore[invalid-argument-type]
        assert result is not None

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_choice_round_trip(self, provider: str):
        ir_choice = {"mode": "auto"}
        provider_choice = tool_ops.choice_to_provider(ir_choice, provider=provider)  # ty: ignore[invalid-argument-type]
        recovered = tool_ops.choice_from_provider(provider_choice, provider=provider)  # ty: ignore[invalid-argument-type]
        assert recovered["mode"] == "auto"


# ==================== Call dispatch ====================


class TestCallDispatch:
    """Test tool call conversion dispatch."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_call_to_provider(self, provider: str):
        result = tool_ops.call_to_provider(IR_CALL, provider=provider)  # ty: ignore[invalid-argument-type]
        assert result is not None

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_call_round_trip(self, provider: str):
        provider_call = tool_ops.call_to_provider(IR_CALL, provider=provider)  # ty: ignore[invalid-argument-type]
        recovered = tool_ops.call_from_provider(provider_call, provider=provider)  # ty: ignore[invalid-argument-type]
        assert recovered["tool_name"] == "get_weather"


# ==================== Result dispatch ====================


class TestResultDispatch:
    """Test tool result conversion dispatch."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_result_to_provider(self, provider: str):
        result = tool_ops.result_to_provider(IR_RESULT, provider=provider)  # ty: ignore[invalid-argument-type]
        assert result is not None

    @pytest.mark.parametrize(
        "provider",
        ["chat_completions", "open_responses", "messages", "google_interactions"],
    )
    def test_result_round_trip(self, provider: str):
        provider_result = tool_ops.result_to_provider(IR_RESULT, provider=provider)  # ty: ignore[invalid-argument-type]
        recovered = tool_ops.result_from_provider(provider_result, provider=provider)  # ty: ignore[invalid-argument-type]
        assert recovered["tool_call_id"] == "call_123"


# ==================== Config dispatch ====================


class TestConfigDispatch:
    """Test tool config conversion dispatch."""

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_config_to_provider(self, provider: str):
        ir_config = {"tool_choice": "auto"}
        result = tool_ops.config_to_provider(ir_config, provider=provider)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, dict)

    @pytest.mark.parametrize("provider", ALL_PROVIDERS)
    def test_config_from_provider(self, provider: str):
        provider_config = {"tool_choice": "auto"}
        result = tool_ops.config_from_provider(provider_config, provider=provider)  # ty: ignore[invalid-argument-type]
        assert isinstance(result, dict)
