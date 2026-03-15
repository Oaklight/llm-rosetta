"""
OpenAI Responses ConfigOps unit tests.
"""

import pytest

from llm_rosetta.converters.openai_responses.config_ops import OpenAIResponsesConfigOps


class TestOpenAIResponsesConfigOps:
    """Unit tests for OpenAIResponsesConfigOps."""

    # ==================== Generation Config ====================

    def test_ir_generation_config_to_p_direct_fields(self):
        """Test direct mapping fields."""
        ir_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_logprobs": 5,
        }
        result = OpenAIResponsesConfigOps.ir_generation_config_to_p(ir_config)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["top_logprobs"] == 5

    def test_ir_generation_config_max_tokens(self):
        """Test max_tokens → max_output_tokens."""
        result = OpenAIResponsesConfigOps.ir_generation_config_to_p({"max_tokens": 100})
        assert result["max_output_tokens"] == 100
        assert "max_tokens" not in result

    def test_ir_generation_config_truncation(self):
        """Test truncation pass-through."""
        result = OpenAIResponsesConfigOps.ir_generation_config_to_p(
            {"truncation": "auto"}
        )
        assert result["truncation"] == "auto"

    def test_ir_generation_config_top_k_warning(self):
        """Test top_k produces warning."""
        with pytest.warns(UserWarning, match="top_k"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p({"top_k": 40})

    def test_ir_generation_config_frequency_penalty_warning(self):
        """Test frequency_penalty produces warning."""
        with pytest.warns(UserWarning, match="frequency_penalty"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p(
                {"frequency_penalty": 0.5}
            )

    def test_ir_generation_config_presence_penalty_warning(self):
        """Test presence_penalty produces warning."""
        with pytest.warns(UserWarning, match="presence_penalty"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p(
                {"presence_penalty": 0.3}
            )

    def test_ir_generation_config_logit_bias_warning(self):
        """Test logit_bias produces warning."""
        with pytest.warns(UserWarning, match="logit_bias"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p(
                {"logit_bias": {100: 10}}
            )

    def test_ir_generation_config_seed_warning(self):
        """Test seed produces warning."""
        with pytest.warns(UserWarning, match="seed"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p({"seed": 42})

    def test_ir_generation_config_n_warning(self):
        """Test n produces warning."""
        with pytest.warns(UserWarning, match="\\bn\\b"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p({"n": 2})

    def test_ir_generation_config_stop_sequences_warning(self):
        """Test stop_sequences produces warning."""
        with pytest.warns(UserWarning, match="stop_sequences"):
            OpenAIResponsesConfigOps.ir_generation_config_to_p(
                {"stop_sequences": ["END"]}
            )

    def test_ir_generation_config_empty(self):
        """Test empty config returns empty dict."""
        result = OpenAIResponsesConfigOps.ir_generation_config_to_p({})
        assert result == {}

    def test_p_generation_config_to_ir(self):
        """Test OpenAI Responses generation params → IR GenerationConfig."""
        provider = {
            "temperature": 0.5,
            "max_output_tokens": 200,
            "top_p": 0.9,
            "top_logprobs": 3,
            "truncation": "auto",
        }
        result = OpenAIResponsesConfigOps.p_generation_config_to_ir(provider)
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["top_logprobs"] == 3
        assert result["truncation"] == "auto"

    def test_p_generation_config_to_ir_non_dict(self):
        """Test non-dict input returns empty dict."""
        result = OpenAIResponsesConfigOps.p_generation_config_to_ir("not a dict")
        assert result == {}

    def test_p_generation_config_to_ir_empty(self):
        """Test empty dict returns empty dict."""
        result = OpenAIResponsesConfigOps.p_generation_config_to_ir({})
        assert result == {}

    def test_generation_config_round_trip(self):
        """Test generation config round-trip."""
        original = {"temperature": 0.8, "max_tokens": 150, "top_p": 0.95}
        provider = OpenAIResponsesConfigOps.ir_generation_config_to_p(original)
        restored = OpenAIResponsesConfigOps.p_generation_config_to_ir(provider)
        assert restored["temperature"] == 0.8
        assert restored["max_tokens"] == 150
        assert restored["top_p"] == 0.95

    # ==================== Response Format ====================

    def test_ir_response_format_text(self):
        """Test text response format → text field."""
        result = OpenAIResponsesConfigOps.ir_response_format_to_p({"type": "text"})
        assert result["text"] == {"type": "text"}

    def test_ir_response_format_json_object(self):
        """Test json_object response format → text field."""
        result = OpenAIResponsesConfigOps.ir_response_format_to_p(
            {"type": "json_object"}
        )
        assert result["text"] == {"type": "json_object"}

    def test_ir_response_format_json_schema(self):
        """Test json_schema response format → text field with schema."""
        schema = {"name": "test", "schema": {"type": "object"}}
        result = OpenAIResponsesConfigOps.ir_response_format_to_p(
            {"type": "json_schema", "json_schema": schema}
        )
        assert result["text"]["type"] == "json_schema"
        assert result["text"]["json_schema"] == schema

    def test_ir_response_format_unknown(self):
        """Test unknown format type returns empty dict."""
        result = OpenAIResponsesConfigOps.ir_response_format_to_p(
            {"type": "unknown_type"}
        )
        assert result == {}

    def test_p_response_format_to_ir(self):
        """Test OpenAI Responses text format → IR."""
        result = OpenAIResponsesConfigOps.p_response_format_to_ir(
            {"type": "json_object"}
        )
        assert result["type"] == "json_object"

    def test_p_response_format_to_ir_json_schema(self):
        """Test json_schema text format → IR with schema."""
        schema = {"name": "test", "schema": {"type": "object"}}
        result = OpenAIResponsesConfigOps.p_response_format_to_ir(
            {"type": "json_schema", "json_schema": schema}
        )
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_p_response_format_to_ir_non_dict(self):
        """Test non-dict input returns empty dict."""
        result = OpenAIResponsesConfigOps.p_response_format_to_ir("not a dict")
        assert result == {}

    def test_response_format_round_trip(self):
        """Test response format round-trip."""
        original = {"type": "json_object"}
        provider = OpenAIResponsesConfigOps.ir_response_format_to_p(original)
        restored = OpenAIResponsesConfigOps.p_response_format_to_ir(provider["text"])
        assert restored["type"] == "json_object"

    # ==================== Stream Config ====================

    def test_ir_stream_config_to_p(self):
        """Test IR StreamConfig → OpenAI Responses stream params."""
        result = OpenAIResponsesConfigOps.ir_stream_config_to_p(
            {"enabled": True, "include_usage": True}
        )
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}

    def test_ir_stream_config_disabled(self):
        """Test disabled stream."""
        result = OpenAIResponsesConfigOps.ir_stream_config_to_p({"enabled": False})
        assert result["stream"] is False
        assert "stream_options" not in result

    def test_ir_stream_config_no_usage(self):
        """Test stream enabled without include_usage."""
        result = OpenAIResponsesConfigOps.ir_stream_config_to_p({"enabled": True})
        assert result["stream"] is True
        assert "stream_options" not in result

    def test_p_stream_config_to_ir(self):
        """Test OpenAI Responses stream params → IR StreamConfig."""
        result = OpenAIResponsesConfigOps.p_stream_config_to_ir(
            {"stream": True, "stream_options": {"include_usage": True}}
        )
        assert result["enabled"] is True
        assert result["include_usage"] is True

    def test_p_stream_config_to_ir_non_dict(self):
        """Test non-dict input returns empty dict."""
        result = OpenAIResponsesConfigOps.p_stream_config_to_ir("not a dict")
        assert result == {}

    def test_stream_config_round_trip(self):
        """Test stream config round-trip."""
        original = {"enabled": True, "include_usage": True}
        provider = OpenAIResponsesConfigOps.ir_stream_config_to_p(original)
        restored = OpenAIResponsesConfigOps.p_stream_config_to_ir(provider)
        assert restored["enabled"] is True
        assert restored["include_usage"] is True

    # ==================== Reasoning Config ====================

    def test_ir_reasoning_config_to_p(self):
        """Test IR ReasoningConfig → OpenAI Responses reasoning object."""
        result = OpenAIResponsesConfigOps.ir_reasoning_config_to_p({"effort": "high"})
        assert result["reasoning"] == {"effort": "high"}

    def test_ir_reasoning_config_with_type(self):
        """Test reasoning config with type field."""
        result = OpenAIResponsesConfigOps.ir_reasoning_config_to_p(
            {"type": "enabled", "effort": "medium"}
        )
        assert result["reasoning"]["type"] == "enabled"
        assert result["reasoning"]["effort"] == "medium"

    def test_ir_reasoning_config_budget_warning(self):
        """Test budget_tokens produces warning."""
        with pytest.warns(UserWarning, match="budget_tokens"):
            OpenAIResponsesConfigOps.ir_reasoning_config_to_p({"budget_tokens": 1000})

    def test_ir_reasoning_config_empty(self):
        """Test empty reasoning config returns empty dict."""
        result = OpenAIResponsesConfigOps.ir_reasoning_config_to_p({})
        assert result == {}

    def test_p_reasoning_config_to_ir(self):
        """Test OpenAI Responses reasoning → IR ReasoningConfig."""
        result = OpenAIResponsesConfigOps.p_reasoning_config_to_ir(
            {"reasoning": {"effort": "medium", "type": "enabled"}}
        )
        assert result["effort"] == "medium"
        assert result["type"] == "enabled"

    def test_p_reasoning_config_to_ir_direct(self):
        """Test direct reasoning object (without nesting)."""
        result = OpenAIResponsesConfigOps.p_reasoning_config_to_ir({"effort": "low"})
        assert result["effort"] == "low"

    def test_p_reasoning_config_to_ir_non_dict(self):
        """Test non-dict input returns empty dict."""
        result = OpenAIResponsesConfigOps.p_reasoning_config_to_ir("not a dict")
        assert result == {}

    def test_reasoning_config_round_trip(self):
        """Test reasoning config round-trip."""
        original = {"effort": "high"}
        provider = OpenAIResponsesConfigOps.ir_reasoning_config_to_p(original)
        restored = OpenAIResponsesConfigOps.p_reasoning_config_to_ir(provider)
        assert restored["effort"] == "high"

    # ==================== Cache Config ====================

    def test_ir_cache_config_to_p(self):
        """Test IR CacheConfig → OpenAI Responses cache params."""
        result = OpenAIResponsesConfigOps.ir_cache_config_to_p(
            {"key": "test-key", "retention": "24h"}
        )
        assert result["prompt_cache_key"] == "test-key"
        assert result["prompt_cache_retention"] == "24h"

    def test_ir_cache_config_partial(self):
        """Test partial cache config."""
        result = OpenAIResponsesConfigOps.ir_cache_config_to_p({"key": "k1"})
        assert result["prompt_cache_key"] == "k1"
        assert "prompt_cache_retention" not in result

    def test_p_cache_config_to_ir(self):
        """Test OpenAI Responses cache params → IR CacheConfig."""
        result = OpenAIResponsesConfigOps.p_cache_config_to_ir(
            {"prompt_cache_key": "k1", "prompt_cache_retention": "in-memory"}
        )
        assert result["key"] == "k1"
        assert result["retention"] == "in-memory"

    def test_p_cache_config_to_ir_non_dict(self):
        """Test non-dict input returns empty dict."""
        result = OpenAIResponsesConfigOps.p_cache_config_to_ir("not a dict")
        assert result == {}

    def test_cache_config_round_trip(self):
        """Test cache config round-trip."""
        original = {"key": "my-key", "retention": "24h"}
        provider = OpenAIResponsesConfigOps.ir_cache_config_to_p(original)
        restored = OpenAIResponsesConfigOps.p_cache_config_to_ir(provider)
        assert restored["key"] == original["key"]
        assert restored["retention"] == original["retention"]
