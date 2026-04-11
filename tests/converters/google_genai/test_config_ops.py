"""
Google GenAI ConfigOps unit tests.
"""

from typing import cast

import pytest

from llm_rosetta.converters.google_genai.config_ops import GoogleGenAIConfigOps
from llm_rosetta.types.ir import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)


class TestGoogleGenAIConfigOps:
    """Unit tests for GoogleGenAIConfigOps."""

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
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(ir_config)
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 1024
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50

    def test_ir_generation_config_stop_sequences(self):
        """Test stop_sequences conversion."""
        ir_config = cast(GenerationConfig, {"stop_sequences": ["\n\n", "END"]})
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(ir_config)
        assert result["stop_sequences"] == ["\n\n", "END"]

    def test_ir_generation_config_penalties(self):
        """Test frequency_penalty and presence_penalty conversion."""
        ir_config = cast(
            GenerationConfig,
            {
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
            },
        )
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(ir_config)
        assert result["frequency_penalty"] == 0.5
        assert result["presence_penalty"] == 0.3

    def test_ir_generation_config_seed(self):
        """Test seed conversion."""
        ir_config = cast(GenerationConfig, {"seed": 42})
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(ir_config)
        assert result["seed"] == 42

    def test_ir_generation_config_n_to_candidate_count(self):
        """Test n → candidate_count mapping."""
        ir_config = cast(GenerationConfig, {"n": 2})
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(ir_config)
        assert result["candidate_count"] == 2

    def test_ir_generation_config_empty(self):
        """Test empty generation config."""
        result = GoogleGenAIConfigOps.ir_generation_config_to_p(
            cast(GenerationConfig, {})
        )
        assert result == {}

    def test_ir_generation_config_logit_bias_warning(self):
        """Test logit_bias produces warning."""
        with pytest.warns(UserWarning, match="logit_bias"):
            GoogleGenAIConfigOps.ir_generation_config_to_p(
                cast(GenerationConfig, {"logit_bias": {1: 0.5}})
            )

    def test_ir_generation_config_logprobs_warning(self):
        """Test logprobs produces warning."""
        with pytest.warns(UserWarning, match="logprobs"):
            GoogleGenAIConfigOps.ir_generation_config_to_p(
                cast(GenerationConfig, {"logprobs": True})
            )

    def test_p_generation_config_to_ir(self):
        """Test Google generation params → IR GenerationConfig."""
        provider = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
            "stop_sequences": ["STOP"],
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "seed": 123,
        }
        result = GoogleGenAIConfigOps.p_generation_config_to_ir(provider)
        assert result["temperature"] == 0.8
        assert result["top_p"] == 0.95
        assert result["top_k"] == 40
        assert result["max_tokens"] == 2048
        assert result["stop_sequences"] == ["STOP"]
        assert result["frequency_penalty"] == 0.2
        assert result["presence_penalty"] == 0.1
        assert result["seed"] == 123

    def test_p_generation_config_to_ir_candidate_count(self):
        """Test provider candidate_count → IR n."""
        provider = {"candidate_count": 3}
        result = GoogleGenAIConfigOps.p_generation_config_to_ir(provider)
        assert result["n"] == 3

    def test_p_generation_config_to_ir_empty(self):
        """Test empty provider config → empty IR."""
        result = GoogleGenAIConfigOps.p_generation_config_to_ir({})
        assert result == {}

    def test_p_generation_config_to_ir_non_dict(self):
        """Test non-dict input → empty IR."""
        result = GoogleGenAIConfigOps.p_generation_config_to_ir("invalid")
        assert result == {}

    # ==================== Response Format ====================

    def test_ir_response_format_json_object(self):
        """Test json_object → response_mime_type."""
        result = GoogleGenAIConfigOps.ir_response_format_to_p(
            cast(ResponseFormatConfig, {"type": "json_object"})
        )
        assert result["response_mime_type"] == "application/json"
        assert "response_schema" not in result

    def test_ir_response_format_json_schema(self):
        """Test json_schema → response_mime_type + response_schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = GoogleGenAIConfigOps.ir_response_format_to_p(
            cast(ResponseFormatConfig, {"type": "json_schema", "json_schema": schema})
        )
        assert result["response_mime_type"] == "application/json"
        assert result["response_schema"] == schema

    def test_ir_response_format_text(self):
        """Test text format → empty result."""
        result = GoogleGenAIConfigOps.ir_response_format_to_p(
            cast(ResponseFormatConfig, {"type": "text"})
        )
        assert result == {}

    def test_ir_response_format_default(self):
        """Test default (no type) → empty result."""
        result = GoogleGenAIConfigOps.ir_response_format_to_p(
            cast(ResponseFormatConfig, {})
        )
        assert result == {}

    def test_p_response_format_json_object(self):
        """Test response_mime_type application/json → json_object."""
        result = GoogleGenAIConfigOps.p_response_format_to_ir(
            {"response_mime_type": "application/json"}
        )
        assert result["type"] == "json_object"

    def test_p_response_format_json_schema(self):
        """Test response_mime_type + response_schema → json_schema."""
        schema = {"type": "object", "properties": {}}
        result = GoogleGenAIConfigOps.p_response_format_to_ir(
            {"response_mime_type": "application/json", "response_schema": schema}
        )
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_p_response_format_empty(self):
        """Test empty provider format → empty IR."""
        result = GoogleGenAIConfigOps.p_response_format_to_ir({})
        assert result == {}

    def test_p_response_format_non_dict(self):
        """Test non-dict input → empty IR."""
        result = GoogleGenAIConfigOps.p_response_format_to_ir("invalid")
        assert result == {}

    def test_response_format_round_trip_json_object(self):
        """Test json_object round-trip."""
        original = cast(ResponseFormatConfig, {"type": "json_object"})
        provider = GoogleGenAIConfigOps.ir_response_format_to_p(original)
        restored = GoogleGenAIConfigOps.p_response_format_to_ir(provider)
        assert restored["type"] == "json_object"

    def test_response_format_round_trip_json_schema(self):
        """Test json_schema round-trip."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        original = cast(
            ResponseFormatConfig, {"type": "json_schema", "json_schema": schema}
        )
        provider = GoogleGenAIConfigOps.ir_response_format_to_p(original)
        restored = GoogleGenAIConfigOps.p_response_format_to_ir(provider)
        assert restored["type"] == "json_schema"
        assert restored["json_schema"] == schema

    # ==================== Stream Config ====================

    def test_ir_stream_config_enabled(self):
        """Test stream enabled → Google stream param."""
        result = GoogleGenAIConfigOps.ir_stream_config_to_p(
            cast(StreamConfig, {"enabled": True})
        )
        assert result["stream"] is True

    def test_ir_stream_config_disabled(self):
        """Test stream disabled → Google stream param."""
        result = GoogleGenAIConfigOps.ir_stream_config_to_p(
            cast(StreamConfig, {"enabled": False})
        )
        assert result["stream"] is False

    def test_ir_stream_config_empty(self):
        """Test empty stream config → empty result."""
        result = GoogleGenAIConfigOps.ir_stream_config_to_p(cast(StreamConfig, {}))
        assert result == {}

    def test_p_stream_config_to_ir(self):
        """Test Google stream param → IR StreamConfig."""
        result = GoogleGenAIConfigOps.p_stream_config_to_ir({"stream": True})
        assert result["enabled"] is True

    def test_p_stream_config_to_ir_empty(self):
        """Test empty stream config → empty IR."""
        result = GoogleGenAIConfigOps.p_stream_config_to_ir({})
        assert result == {}

    def test_p_stream_config_to_ir_non_dict(self):
        """Test non-dict input → empty IR."""
        result = GoogleGenAIConfigOps.p_stream_config_to_ir("invalid")
        assert result == {}

    # ==================== Reasoning Config ====================

    def test_ir_reasoning_config_budget_tokens(self):
        """Test reasoning budget_tokens → thinking_config."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"budget_tokens": 4096})
        )
        assert result["thinking_config"]["thinking_budget"] == 4096

    def test_ir_reasoning_config_effort(self):
        """Test effort → thinking_level."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "high"})
        )
        assert result["thinking_config"]["thinking_level"] == "high"

    def test_ir_reasoning_config_effort_minimal(self):
        """Test 'minimal' effort maps directly to Google."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "minimal"})
        )
        assert result["thinking_config"]["thinking_level"] == "minimal"

    def test_ir_reasoning_config_effort_max_warning(self):
        """Test 'max' effort downgraded to 'high' with warning."""
        with pytest.warns(UserWarning, match="max"):
            result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
                cast(ReasoningConfig, {"effort": "max"})
            )
        assert result["thinking_config"]["thinking_level"] == "high"

    def test_ir_reasoning_config_effort_with_budget(self):
        """Test effort + budget_tokens combined."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"effort": "medium", "budget_tokens": 4096})
        )
        assert result["thinking_config"]["thinking_level"] == "medium"
        assert result["thinking_config"]["thinking_budget"] == 4096

    def test_ir_reasoning_config_empty(self):
        """Test empty reasoning config → empty result."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {})
        )
        assert result == {}

    def test_ir_reasoning_config_mode_disabled(self):
        """Test mode: disabled → thinking_budget: 0."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"mode": "disabled"})
        )
        assert result["thinking_config"]["thinking_budget"] == 0

    def test_ir_reasoning_config_mode_auto(self):
        """Test mode: auto → thinking_budget: -1."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"mode": "auto"})
        )
        assert result["thinking_config"]["thinking_budget"] == -1

    def test_ir_reasoning_config_mode_auto_with_effort(self):
        """Test mode: auto + effort → thinking_budget: -1 + thinking_level."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"mode": "auto", "effort": "high"})
        )
        assert result["thinking_config"]["thinking_budget"] == -1
        assert result["thinking_config"]["thinking_level"] == "high"

    def test_ir_reasoning_config_mode_enabled_with_budget(self):
        """Test mode: enabled + budget → uses budget_tokens."""
        result = GoogleGenAIConfigOps.ir_reasoning_config_to_p(
            cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 4096})
        )
        assert result["thinking_config"]["thinking_budget"] == 4096

    def test_p_reasoning_config_to_ir(self):
        """Test Google thinking_config → IR ReasoningConfig."""
        provider = {"thinking_config": {"thinking_budget": 8192}}
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "enabled"
        assert result["budget_tokens"] == 8192

    def test_p_reasoning_config_to_ir_budget_zero(self):
        """Test thinking_budget: 0 → mode: disabled."""
        provider = {"thinking_config": {"thinking_budget": 0}}
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "disabled"

    def test_p_reasoning_config_to_ir_budget_negative(self):
        """Test thinking_budget: -1 → mode: auto."""
        provider = {"thinking_config": {"thinking_budget": -1}}
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert result["mode"] == "auto"

    def test_p_reasoning_config_to_ir_with_level(self):
        """Test thinking_level → IR effort + mode: auto."""
        provider = {"thinking_config": {"thinking_level": "low"}}
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert result["effort"] == "low"
        assert result["mode"] == "auto"

    def test_p_reasoning_config_to_ir_empty(self):
        """Test empty reasoning config → empty IR."""
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir({})
        assert result == {}

    def test_p_reasoning_config_to_ir_non_dict(self):
        """Test non-dict input → empty IR."""
        result = GoogleGenAIConfigOps.p_reasoning_config_to_ir("invalid")
        assert result == {}

    def test_reasoning_config_round_trip(self):
        """Test reasoning config round-trip: enabled + budget."""
        original = cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 2048})
        provider = GoogleGenAIConfigOps.ir_reasoning_config_to_p(original)
        restored = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert restored["mode"] == "enabled"
        assert restored["budget_tokens"] == 2048

    def test_reasoning_config_effort_round_trip(self):
        """Test effort round-trip."""
        original = cast(ReasoningConfig, {"effort": "high"})
        provider = GoogleGenAIConfigOps.ir_reasoning_config_to_p(original)
        restored = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert restored["effort"] == "high"

    def test_reasoning_config_roundtrip_auto(self):
        """Test round-trip: auto → thinking_budget: -1 → auto."""
        original = cast(ReasoningConfig, {"mode": "auto"})
        provider = GoogleGenAIConfigOps.ir_reasoning_config_to_p(original)
        assert provider["thinking_config"]["thinking_budget"] == -1
        restored = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert restored["mode"] == "auto"

    def test_reasoning_config_roundtrip_disabled(self):
        """Test round-trip: disabled → thinking_budget: 0 → disabled."""
        original = cast(ReasoningConfig, {"mode": "disabled"})
        provider = GoogleGenAIConfigOps.ir_reasoning_config_to_p(original)
        assert provider["thinking_config"]["thinking_budget"] == 0
        restored = GoogleGenAIConfigOps.p_reasoning_config_to_ir(provider)
        assert restored["mode"] == "disabled"

    # ==================== Cache Config ====================

    def test_ir_cache_config_key(self):
        """Test cache key → cached_content."""
        result = GoogleGenAIConfigOps.ir_cache_config_to_p(
            cast(CacheConfig, {"key": "cache-resource-123"})
        )
        assert result["cached_content"] == "cache-resource-123"

    def test_ir_cache_config_retention_warning(self):
        """Test cache retention produces warning."""
        with pytest.warns(UserWarning, match="retention"):
            result = GoogleGenAIConfigOps.ir_cache_config_to_p(
                cast(CacheConfig, {"key": "cache-1", "retention": "24h"})
            )
        assert result["cached_content"] == "cache-1"

    def test_ir_cache_config_empty(self):
        """Test empty cache config → empty result."""
        result = GoogleGenAIConfigOps.ir_cache_config_to_p(cast(CacheConfig, {}))
        assert result == {}

    def test_p_cache_config_to_ir(self):
        """Test Google cached_content → IR CacheConfig."""
        result = GoogleGenAIConfigOps.p_cache_config_to_ir(
            {"cached_content": "cache-resource-456"}
        )
        assert result["key"] == "cache-resource-456"

    def test_p_cache_config_to_ir_empty(self):
        """Test empty cache config → empty IR."""
        result = GoogleGenAIConfigOps.p_cache_config_to_ir({})
        assert result == {}

    def test_p_cache_config_to_ir_non_dict(self):
        """Test non-dict input → empty IR."""
        result = GoogleGenAIConfigOps.p_cache_config_to_ir("invalid")
        assert result == {}

    def test_cache_config_round_trip(self):
        """Test cache config round-trip."""
        original = cast(CacheConfig, {"key": "my-cache"})
        provider = GoogleGenAIConfigOps.ir_cache_config_to_p(original)
        restored = GoogleGenAIConfigOps.p_cache_config_to_ir(provider)
        assert restored["key"] == "my-cache"
