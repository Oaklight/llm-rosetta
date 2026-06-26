"""Tests for ConversionPipeline.profile timing data."""

from llm_rosetta.pipeline import ConversionPipeline


class TestPipelineProfile:
    """Verify that pipeline.profile is populated after conversion."""

    def _make_simple_request(self) -> dict:
        """Create a minimal OpenAI chat request."""
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }

    def _make_simple_response(self) -> dict:
        """Create a minimal Anthropic response."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "test-model",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def test_profile_empty_before_conversion(self):
        pipeline = ConversionPipeline("openai_chat", "anthropic")
        assert pipeline.profile == {}

    def test_profile_populated_after_convert_request(self):
        pipeline = ConversionPipeline("openai_chat", "anthropic")
        pipeline.convert_request(self._make_simple_request())

        p = pipeline.profile
        assert "source_to_ir_ms" in p
        assert "ir_transforms_ms" in p
        assert "ir_to_target_ms" in p
        assert "body_transforms_ms" in p
        assert "request_conversion_ms" in p

        # All values should be non-negative floats
        for key, val in p.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"
            assert val >= 0, f"{key} is negative: {val}"

        # Total should be >= sum of parts (due to overhead)
        parts_sum = (
            p["source_to_ir_ms"]
            + p["ir_transforms_ms"]
            + p["ir_to_target_ms"]
            + p["body_transforms_ms"]
        )
        assert p["request_conversion_ms"] >= parts_sum * 0.9  # allow small float error

    def test_profile_populated_after_convert_response(self):
        pipeline = ConversionPipeline("openai_chat", "anthropic")
        pipeline.convert_request(self._make_simple_request())
        pipeline.convert_response(self._make_simple_response())

        p = pipeline.profile
        assert "response_from_target_ms" in p
        assert "response_to_source_ms" in p
        assert "response_conversion_ms" in p

        # Response conversion total should be >= sum of parts
        parts_sum = p["response_from_target_ms"] + p["response_to_source_ms"]
        assert p["response_conversion_ms"] >= parts_sum * 0.9

    def test_profile_has_all_keys_after_full_roundtrip(self):
        pipeline = ConversionPipeline("openai_chat", "anthropic")
        pipeline.convert_request(self._make_simple_request())
        pipeline.convert_response(self._make_simple_response())

        expected_keys = {
            "source_to_ir_ms",
            "ir_transforms_ms",
            "ir_to_target_ms",
            "body_transforms_ms",
            "request_conversion_ms",
            "response_from_target_ms",
            "response_to_source_ms",
            "response_conversion_ms",
        }
        assert expected_keys == set(pipeline.profile.keys())

    def test_same_format_conversion(self):
        """Profile is populated even for same-format conversion."""
        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(self._make_simple_request())

        p = pipeline.profile
        assert "request_conversion_ms" in p
        assert p["request_conversion_ms"] >= 0
