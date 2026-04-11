"""
Anthropic ConfigOps unit tests.
"""

from typing import cast

import pytest

from llm_rosetta.converters.anthropic.config_ops import AnthropicConfigOps
from llm_rosetta.types.ir import GenerationConfig, ReasoningConfig


class TestAnthropicConfigOps:
    """Unit tests for AnthropicConfigOps."""

    # ==================== Generation Config ====================

    def test_ir_generation_config_basic(self):
        """Test basic generation config conversion."""
        ir_config = cast(
            GenerationConfig,
            {
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
            },
        )
        result = AnthropicConfigOps.ir_generation_config_to_p(ir_config)
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1024
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50

    def test_ir_generation_config_default_max_tokens(self):
        """Test default max_tokens when not specified."""
        result = AnthropicConfigOps.ir_generation_config_to_p({})
        assert result["max_tokens"] == 4096

    def test_ir_generation_config_temperature_clamped(self):
        """Test temperature is clamped to 1.0 max."""
        ir_config = cast(GenerationConfig, {"temperature": 1.5})
        result = AnthropicConfigOps.ir_generation_config_to_p(ir_config)
        assert result["temperature"] == 1.0

    def test_ir_generation_config_stop_sequences(self):
        """Test stop_sequences conversion."""
        ir_config = cast(GenerationConfig, {"stop_sequences": ["\n\nHuman:", "END"]})
        result = AnthropicConfigOps.ir_generation_config_to_p(ir_config)
        assert result["stop_sequences"] == ["\n\nHuman:", "END"]

    def test_ir_generation_config_unsupported_fields(self):
        """Test unsupported fields produce warnings."""
        ir_config = cast(
            GenerationConfig,
            {
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "seed": 42,
            },
        )
        with pytest.warns(UserWarning):
            result = AnthropicConfigOps.ir_generation_config_to_p(ir_config)
        # Unsupported fields should not be in result
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert "seed" not in result

    def test_p_generation_config_to_ir(self):
        """Test Anthropic generation params → IR GenerationConfig."""
        provider = {
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "stop_sequences": ["STOP"],
        }
        result = AnthropicConfigOps.p_generation_config_to_ir(provider)
        assert result["max_tokens"] == 2048
        assert result["temperature"] == 0.8
        assert result["top_p"] == 0.95
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["STOP"]

    def test_p_generation_config_to_ir_empty(self):
        """Test empty provider config → empty IR."""
        result = AnthropicConfigOps.p_generation_config_to_ir({})
        assert result == {}

    def test_p_generation_config_to_ir_non_dict(self):
        """Test non-dict input → empty IR."""
        result = AnthropicConfigOps.p_generation_config_to_ir("invalid")
        assert result == {}

    # ==================== Response Format ====================

    def test_ir_response_format_warning(self):
        """Test response format produces warning."""
        with pytest.warns(UserWarning, match="does not support response_format"):
            result = AnthropicConfigOps.ir_response_format_to_p({"type": "json_object"})
        assert result == {}

    def test_p_response_format_to_ir(self):
        """Test Anthropic response format → empty IR."""
        result = AnthropicConfigOps.p_response_format_to_ir({})
        assert result == {}

    # ==================== Stream Config ====================

    def test_ir_stream_config_enabled(self):
        """Test stream enabled → Anthropic stream param."""
        result = AnthropicConfigOps.ir_stream_config_to_p({"enabled": True})
        assert result["stream"] is True

    def test_ir_stream_config_disabled(self):
        """Test stream disabled → Anthropic stream param."""
        result = AnthropicConfigOps.ir_stream_config_to_p({"enabled": False})
        assert result["stream"] is False

    def test_ir_stream_config_include_usage_warning(self):
        """Test include_usage produces warning."""
        with pytest.warns(UserWarning, match="always includes usage"):
            result = AnthropicConfigOps.ir_stream_config_to_p(
                {"enabled": True, "include_usage": True}
            )
        assert result["stream"] is True

    def test_p_stream_config_to_ir(self):
        """Test Anthropic stream param → IR StreamConfig."""
        result = AnthropicConfigOps.p_stream_config_to_ir({"stream": True})
        assert result["enabled"] is True

    def test_p_stream_config_to_ir_empty(self):
        """Test empty stream config → empty IR."""
        result = AnthropicConfigOps.p_stream_config_to_ir({})
        assert result == {}

    # ==================== Reasoning Config ====================

    def test_ir_reasoning_config_enabled_with_budget(self):
        """Test mode: enabled + budget_tokens → 'enabled' type."""
        ir_reasoning = cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 2048})
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir_reasoning)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 2048

    def test_ir_reasoning_config_enabled_without_budget(self):
        """Test mode: enabled without budget_tokens → 'adaptive' fallback.

        'enabled' requires budget_tokens; when absent, fall back to
        'adaptive' to produce a valid Anthropic request.
        """
        ir_reasoning = cast(ReasoningConfig, {"mode": "enabled"})
        with pytest.warns(UserWarning, match="budget_tokens"):
            result = AnthropicConfigOps.ir_reasoning_config_to_p(ir_reasoning)
        assert result["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in result["thinking"]

    def test_ir_reasoning_config_disabled(self):
        """Test mode: disabled → Anthropic thinking param."""
        ir_reasoning = cast(ReasoningConfig, {"mode": "disabled"})
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir_reasoning)
        assert result["thinking"]["type"] == "disabled"

    def test_ir_reasoning_config_auto(self):
        """Test mode: auto → adaptive thinking."""
        ir_reasoning = cast(ReasoningConfig, {"mode": "auto"})
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir_reasoning)
        assert result["thinking"]["type"] == "adaptive"

    def test_ir_reasoning_config_auto_with_effort(self):
        """Test mode: auto + effort → adaptive + output_config.effort."""
        ir_reasoning = cast(ReasoningConfig, {"mode": "auto", "effort": "high"})
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir_reasoning)
        assert result["thinking"]["type"] == "adaptive"
        assert result["output_config"]["effort"] == "high"
        assert "effort" not in result["thinking"]

    def test_ir_reasoning_config_effort_adaptive(self):
        """Test effort without mode → adaptive thinking + output_config."""
        result = AnthropicConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "high"})
        )
        assert result["thinking"]["type"] == "adaptive"
        assert result["output_config"]["effort"] == "high"

    def test_ir_reasoning_config_effort_max(self):
        """Test 'max' effort maps directly to output_config."""
        result = AnthropicConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "max"})
        )
        assert result["thinking"]["type"] == "adaptive"
        assert result["output_config"]["effort"] == "max"

    def test_ir_reasoning_config_effort_minimal_warning(self):
        """Test 'minimal' effort downgraded to 'low' with warning."""
        with pytest.warns(UserWarning, match="minimal"):
            result = AnthropicConfigOps.ir_reasoning_config_to_p(
                cast(ReasoningConfig, {"effort": "minimal"})
            )
        assert result["output_config"]["effort"] == "low"

    def test_ir_reasoning_config_effort_with_budget(self):
        """Test effort + budget_tokens combined."""
        result = AnthropicConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "medium", "budget_tokens": 4096})
        )
        assert result["thinking"]["type"] == "adaptive"
        assert result["thinking"]["budget_tokens"] == 4096
        assert result["output_config"]["effort"] == "medium"

    def test_ir_reasoning_config_budget_only(self):
        """Test budget_tokens without mode or effort → 'enabled' type."""
        result = AnthropicConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"budget_tokens": 4096})
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 4096
        assert "output_config" not in result

    def test_p_reasoning_config_to_ir(self):
        """Test Anthropic thinking → IR ReasoningConfig."""
        provider = {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        result = AnthropicConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "enabled"
        assert result["budget_tokens"] == 4096

    def test_p_reasoning_config_to_ir_adaptive(self):
        """Test adaptive thinking + output_config.effort → IR."""
        provider = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        result = AnthropicConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "auto"
        assert result["effort"] == "high"

    def test_p_reasoning_config_to_ir_adaptive_no_effort(self):
        """Test adaptive thinking without effort → IR mode: auto only."""
        provider = {"thinking": {"type": "adaptive"}}
        result = AnthropicConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "auto"
        assert "effort" not in result
        assert "budget_tokens" not in result

    def test_p_reasoning_config_to_ir_empty(self):
        """Test empty reasoning config → empty IR."""
        result = AnthropicConfigOps.p_reasoning_config_to_ir({})
        assert result == {}

    def test_reasoning_config_roundtrip_adaptive_no_effort(self):
        """Test round-trip: adaptive (no effort) → IR → adaptive.

        Regression test for argo-proxy#502: force_conversion caused
        {"type": "adaptive"} to become {"type": "enabled"} (without
        budget_tokens), which Anthropic API rejects.
        """
        # Anthropic → IR
        provider_input = {"thinking": {"type": "adaptive"}}
        ir = AnthropicConfigOps.p_reasoning_config_to_ir(provider_input)
        assert ir["mode"] == "auto"
        # IR → Anthropic
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir)
        assert result["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in result["thinking"]

    def test_reasoning_config_roundtrip_adaptive_with_effort(self):
        """Test round-trip: adaptive + effort → IR → adaptive + effort."""
        provider_input = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        ir = AnthropicConfigOps.p_reasoning_config_to_ir(provider_input)
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir)
        assert result["thinking"]["type"] == "adaptive"
        assert result["output_config"]["effort"] == "high"

    def test_reasoning_config_roundtrip_enabled_with_budget(self):
        """Test round-trip: enabled + budget → IR → enabled + budget."""
        provider_input = {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        ir = AnthropicConfigOps.p_reasoning_config_to_ir(provider_input)
        result = AnthropicConfigOps.ir_reasoning_config_to_p(ir)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 4096

    # ==================== Cache Config ====================

    def test_ir_cache_config_warning(self):
        """Test cache config produces warning."""
        with pytest.warns(UserWarning, match="block-level"):
            result = AnthropicConfigOps.ir_cache_config_to_p({"key": "test"})
        assert result == {}

    def test_p_cache_config_to_ir(self):
        """Test Anthropic cache → empty IR."""
        result = AnthropicConfigOps.p_cache_config_to_ir({})
        assert result == {}
