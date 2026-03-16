"""
OpenAI Chat ConfigOps unit tests.
"""

import pytest

from typing import cast

from llm_rosetta.converters.openai_chat.config_ops import OpenAIChatConfigOps
from llm_rosetta.types.ir import CacheConfig, GenerationConfig


class TestOpenAIChatConfigOps:
    """Unit tests for OpenAIChatConfigOps."""

    # ==================== Generation Config ====================

    def test_ir_generation_config_to_p_direct_fields(self):
        """Test direct mapping fields."""
        ir_config = cast(
            GenerationConfig,
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "seed": 42,
                "logprobs": True,
                "top_logprobs": 5,
                "n": 2,
            },
        )
        result = OpenAIChatConfigOps.ir_generation_config_to_p(ir_config)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["frequency_penalty"] == 0.5
        assert result["presence_penalty"] == 0.3
        assert result["seed"] == 42
        assert result["logprobs"] is True
        assert result["top_logprobs"] == 5
        assert result["n"] == 2

    def test_ir_generation_config_max_tokens(self):
        """Test max_tokens → max_completion_tokens."""
        result = OpenAIChatConfigOps.ir_generation_config_to_p({"max_tokens": 100})
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_ir_generation_config_stop_sequences(self):
        """Test stop_sequences → stop."""
        # Single stop sequence
        result = OpenAIChatConfigOps.ir_generation_config_to_p(
            {"stop_sequences": ["END"]}
        )
        assert result["stop"] == "END"

        # Multiple stop sequences
        result = OpenAIChatConfigOps.ir_generation_config_to_p(
            {"stop_sequences": ["\n", "END"]}
        )
        assert result["stop"] == ["\n", "END"]

    def test_ir_generation_config_top_k_warning(self):
        """Test top_k produces warning."""
        with pytest.warns(UserWarning, match="top_k"):
            OpenAIChatConfigOps.ir_generation_config_to_p({"top_k": 40})

    def test_p_generation_config_to_ir(self):
        """Test OpenAI generation params → IR GenerationConfig."""
        provider = {
            "temperature": 0.5,
            "max_completion_tokens": 200,
            "stop": ["\n"],
            "seed": 123,
        }
        result = OpenAIChatConfigOps.p_generation_config_to_ir(provider)
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["stop_sequences"] == ["\n"]
        assert result["seed"] == 123

    def test_p_generation_config_legacy_max_tokens(self):
        """Test legacy max_tokens field."""
        result = OpenAIChatConfigOps.p_generation_config_to_ir({"max_tokens": 50})
        assert result["max_tokens"] == 50

    def test_p_generation_config_stop_string(self):
        """Test stop as string → stop_sequences list."""
        result = OpenAIChatConfigOps.p_generation_config_to_ir({"stop": "END"})
        assert result["stop_sequences"] == ["END"]

    def test_generation_config_round_trip(self):
        """Test generation config round-trip."""
        original = cast(
            GenerationConfig,
            {"temperature": 0.8, "max_tokens": 150, "stop_sequences": ["X"]},
        )
        provider = OpenAIChatConfigOps.ir_generation_config_to_p(original)
        # Build a provider-like dict for reverse conversion
        restored = OpenAIChatConfigOps.p_generation_config_to_ir(provider)
        assert restored["temperature"] == 0.8
        assert restored["max_tokens"] == 150

    # ==================== Response Format ====================

    def test_ir_response_format_text(self):
        """Test text response format."""
        result = OpenAIChatConfigOps.ir_response_format_to_p({"type": "text"})
        assert result["response_format"] == {"type": "text"}

    def test_ir_response_format_json_object(self):
        """Test json_object response format."""
        result = OpenAIChatConfigOps.ir_response_format_to_p({"type": "json_object"})
        assert result["response_format"] == {"type": "json_object"}

    def test_ir_response_format_json_schema(self):
        """Test json_schema response format."""
        schema = {"name": "test", "schema": {"type": "object"}}
        result = OpenAIChatConfigOps.ir_response_format_to_p(
            {"type": "json_schema", "json_schema": schema}
        )
        assert result["response_format"]["type"] == "json_schema"
        assert result["response_format"]["json_schema"] == schema

    def test_p_response_format_to_ir(self):
        """Test OpenAI response_format → IR."""
        result = OpenAIChatConfigOps.p_response_format_to_ir({"type": "json_object"})
        assert result["type"] == "json_object"

    # ==================== Stream Config ====================

    def test_ir_stream_config_to_p(self):
        """Test IR StreamConfig → OpenAI stream params."""
        result = OpenAIChatConfigOps.ir_stream_config_to_p(
            {"enabled": True, "include_usage": True}
        )
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}

    def test_ir_stream_config_disabled(self):
        """Test disabled stream."""
        result = OpenAIChatConfigOps.ir_stream_config_to_p({"enabled": False})
        assert result["stream"] is False
        assert "stream_options" not in result

    def test_p_stream_config_to_ir(self):
        """Test OpenAI stream params → IR StreamConfig."""
        result = OpenAIChatConfigOps.p_stream_config_to_ir(
            {"stream": True, "stream_options": {"include_usage": True}}
        )
        assert result["enabled"] is True
        assert result["include_usage"] is True

    # ==================== Reasoning Config ====================

    def test_ir_reasoning_config_to_p(self):
        """Test IR ReasoningConfig → OpenAI reasoning_effort."""
        result = OpenAIChatConfigOps.ir_reasoning_config_to_p({"effort": "high"})
        assert result["reasoning_effort"] == "high"

    def test_ir_reasoning_config_budget_warning(self):
        """Test budget_tokens produces warning."""
        with pytest.warns(UserWarning, match="budget_tokens"):
            OpenAIChatConfigOps.ir_reasoning_config_to_p({"budget_tokens": 1000})

    def test_p_reasoning_config_to_ir(self):
        """Test OpenAI reasoning_effort → IR ReasoningConfig."""
        result = OpenAIChatConfigOps.p_reasoning_config_to_ir(
            {"reasoning_effort": "medium"}
        )
        assert result["effort"] == "medium"

    # ==================== Cache Config ====================

    def test_ir_cache_config_to_p(self):
        """Test IR CacheConfig → OpenAI cache params."""
        result = OpenAIChatConfigOps.ir_cache_config_to_p(
            {"key": "test-key", "retention": "24h"}
        )
        assert result["prompt_cache_key"] == "test-key"
        assert result["prompt_cache_retention"] == "24h"

    def test_p_cache_config_to_ir(self):
        """Test OpenAI cache params → IR CacheConfig."""
        result = OpenAIChatConfigOps.p_cache_config_to_ir(
            {"prompt_cache_key": "k1", "prompt_cache_retention": "in-memory"}
        )
        assert result["key"] == "k1"
        assert result["retention"] == "in-memory"

    def test_cache_config_round_trip(self):
        """Test cache config round-trip."""
        original = cast(CacheConfig, {"key": "my-key", "retention": "24h"})
        provider = OpenAIChatConfigOps.ir_cache_config_to_p(original)
        restored = OpenAIChatConfigOps.p_cache_config_to_ir(provider)
        assert restored["key"] == original["key"]
        assert restored["retention"] == original["retention"]
