"""Tests for ConversionPipeline.profile timing data."""

from llm_rosetta.pipeline import ConversionPipeline

# Every profile value is rounded to this many milliseconds, independently.
_PROFILE_QUANTUM_MS = 0.01


def _assert_total_covers_parts(total: float, *parts: float) -> None:
    """Assert *total* is not below the sum of *parts*, allowing for rounding.

    The parts are nested inside the timed region the total measures, so
    the true total is always the larger.  Each value is rounded on its
    own, though, which lets a sum of *n* parts drift up to *n* half-quanta
    above its true value while the total drifts half a quantum below —
    a gap that owes nothing to how long the work actually took.  Hence an
    absolute allowance rather than a percentage: these timings are small
    enough that 10% of the sum is less than one rounding step.
    """
    slack = _PROFILE_QUANTUM_MS / 2 * (len(parts) + 1)
    assert total >= sum(parts) - slack, (
        f"total {total}ms is below the sum of {parts} by more than the "
        f"{slack}ms these roundings can account for"
    )


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
        _assert_total_covers_parts(
            p["request_conversion_ms"],
            p["source_to_ir_ms"],
            p["ir_transforms_ms"],
            p["ir_to_target_ms"],
            p["body_transforms_ms"],
        )

    def test_profile_populated_after_convert_response(self):
        pipeline = ConversionPipeline("openai_chat", "anthropic")
        pipeline.convert_request(self._make_simple_request())
        pipeline.convert_response(self._make_simple_response())

        p = pipeline.profile
        assert "response_from_target_ms" in p
        assert "response_to_source_ms" in p
        assert "response_conversion_ms" in p

        # Response conversion total should be >= sum of parts
        _assert_total_covers_parts(
            p["response_conversion_ms"],
            p["response_from_target_ms"],
            p["response_to_source_ms"],
        )

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
