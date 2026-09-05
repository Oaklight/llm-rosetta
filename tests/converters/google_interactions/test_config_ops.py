"""Tests for Google Interactions config_ops."""

from llm_rosetta.converters.google_interactions.config_ops import (
    GoogleInteractionsConfigOps,
)


class TestGenerationConfig:
    def test_ir_to_provider(self):
        ir = {"max_tokens": 1024, "stop_sequences": ["END"], "seed": 42}
        result = GoogleInteractionsConfigOps.ir_generation_config_to_p(ir)  # ty: ignore
        assert result["max_output_tokens"] == 1024
        assert result["stop_sequences"] == ["END"]
        assert result["seed"] == 42

    def test_provider_to_ir(self):
        p = {"max_output_tokens": 2048, "stop_sequences": ["STOP"], "seed": 7}
        result = GoogleInteractionsConfigOps.p_generation_config_to_ir(p)
        assert result["max_tokens"] == 2048
        assert result["stop_sequences"] == ["STOP"]
        assert result["seed"] == 7

    def test_empty_config(self):
        result = GoogleInteractionsConfigOps.ir_generation_config_to_p({})
        assert result == {}

    def test_roundtrip(self):
        ir = {"max_tokens": 512, "seed": 99}
        p = GoogleInteractionsConfigOps.ir_generation_config_to_p(ir)  # ty: ignore
        back = GoogleInteractionsConfigOps.p_generation_config_to_ir(p)
        assert back["max_tokens"] == 512
        assert back["seed"] == 99


class TestReasoningConfig:
    def test_thinking_level_to_ir(self):
        for level, expected in [
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
        ]:
            p = {"thinking_level": level}
            result = GoogleInteractionsConfigOps.p_reasoning_to_ir(p)
            assert result["effort"] == expected
            assert result["mode"] == "enabled"

    def test_ir_effort_to_thinking_level(self):
        for effort, expected in [
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ]:
            ir = {"effort": effort}
            result = GoogleInteractionsConfigOps.ir_reasoning_to_p(ir)  # ty: ignore
            assert result["thinking_level"] == expected

    def test_thinking_summaries(self):
        p = {"thinking_summaries": "auto"}
        result = GoogleInteractionsConfigOps.p_reasoning_to_ir(p)
        assert result["summary"] == "auto"

        p = {"thinking_summaries": "none"}
        result = GoogleInteractionsConfigOps.p_reasoning_to_ir(p)
        assert result["summary"] == "none"

    def test_ir_summary_to_provider(self):
        ir = {"summary": "auto"}
        result = GoogleInteractionsConfigOps.ir_reasoning_to_p(ir)  # ty: ignore
        assert result["thinking_summaries"] == "auto"

    def test_roundtrip_reasoning(self):
        ir = {"effort": "medium", "summary": "auto"}
        p = GoogleInteractionsConfigOps.ir_reasoning_to_p(ir)  # ty: ignore
        back = GoogleInteractionsConfigOps.p_reasoning_to_ir(p)
        assert back["effort"] == "medium"
        assert back["summary"] == "auto"


class TestResponseFormat:
    def test_json_object(self):
        ir = {"type": "json_object"}
        result = GoogleInteractionsConfigOps.ir_response_format_to_p(ir)  # ty: ignore
        assert result["mime_type"] == "application/json"

    def test_json_schema(self):
        ir = {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        result = GoogleInteractionsConfigOps.ir_response_format_to_p(ir)  # ty: ignore
        assert result["mime_type"] == "application/json"
        assert "response_schema" in result

    def test_text_format(self):
        ir = {"type": "text"}
        result = GoogleInteractionsConfigOps.ir_response_format_to_p(ir)  # ty: ignore
        assert result is None

    def test_provider_json_to_ir(self):
        p = {"mime_type": "application/json"}
        result = GoogleInteractionsConfigOps.p_response_format_to_ir(p)
        assert result["type"] == "json_object"

    def test_provider_json_schema_to_ir(self):
        p = {
            "mime_type": "application/json",
            "response_schema": {"type": "object"},
        }
        result = GoogleInteractionsConfigOps.p_response_format_to_ir(p)
        assert result["type"] == "json_schema"
        assert result["json_schema"] == {"type": "object"}
