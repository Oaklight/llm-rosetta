"""Tests for ConversionPipeline passthrough mode and FidelityChecker."""

from __future__ import annotations

from llm_rosetta.fidelity import FidelityChecker
from llm_rosetta.pipeline import ConversionPipeline


# ============================================================================
# Passthrough mode tests
# ============================================================================


class TestPassthroughMode:
    def test_same_format_passthrough_returns_body_identity(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", force_conversion=False
        )
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = pipeline.convert_request(body)
        assert result is body

    def test_same_format_response_passthrough(self) -> None:
        pipeline = ConversionPipeline("anthropic", "anthropic", force_conversion=False)
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        pipeline.convert_request(body)

        response = {
            "id": "msg_123",
            "type": "message",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        result = pipeline.convert_response(response)
        assert result is response

    def test_force_conversion_true_does_roundtrip(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", force_conversion=True
        )
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = pipeline.convert_request(body)
        # Round-trip produces a new dict (not the same object)
        assert result is not body
        assert result["model"] == "gpt-4o"

    def test_different_format_always_converts(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "anthropic", force_conversion=False
        )
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = pipeline.convert_request(body)
        # Cross-format produces Anthropic-format output
        assert "max_tokens" in result or "messages" in result
        assert result is not body

    def test_passthrough_stream_processor(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", force_conversion=False
        )
        pipeline.convert_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        processor = pipeline.create_stream_processor()
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        result = processor.process_chunk(chunk)
        assert result == [chunk]

    def test_passthrough_profile(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", force_conversion=False
        )
        pipeline.convert_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert "request_conversion_ms" in pipeline.profile

    def test_passthrough_ir_request_empty(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", force_conversion=False
        )
        pipeline.convert_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert pipeline.ir_request == {}


# ============================================================================
# Converter caching tests
# ============================================================================


class TestConverterCaching:
    def test_same_provider_returns_same_instance(self) -> None:
        from llm_rosetta.auto_detect import get_converter_for_provider

        c1 = get_converter_for_provider("openai_chat")
        c2 = get_converter_for_provider("openai_chat")
        assert c1 is c2

    def test_shim_and_base_share_instance(self) -> None:
        from llm_rosetta.auto_detect import _converter_cache

        # After getting via a base name, cache should have it
        from llm_rosetta.auto_detect import get_converter_for_provider

        get_converter_for_provider("anthropic")
        assert "anthropic" in _converter_cache


# ============================================================================
# FidelityChecker tests
# ============================================================================


class TestFidelityCheckerCritical:
    def setup_method(self) -> None:
        self.checker = FidelityChecker(mode="critical")

    def test_identical_bodies_no_diff(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert self.checker.compare_request(body, body) == []

    def test_missing_field_detected(self) -> None:
        original = {
            "model": "gpt-4o",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        roundtripped = {"model": "gpt-4o"}
        diffs = self.checker.compare_request(original, roundtripped)
        assert any(d.kind == "missing" for d in diffs)

    def test_changed_field_detected(self) -> None:
        original = {"model": "gpt-4o", "stream": True}
        roundtripped = {"model": "gpt-4o", "stream": False}
        diffs = self.checker.compare_request(original, roundtripped)
        assert any(d.kind == "changed" and "stream" in d.path for d in diffs)

    def test_response_stop_reason_diff(self) -> None:
        original = {"stop_reason": "end_turn", "content": [{"type": "text"}]}
        roundtripped = {"stop_reason": "stop", "content": [{"type": "text"}]}
        diffs = self.checker.compare_response(original, roundtripped)
        assert any("stop_reason" in d.path for d in diffs)


class TestFidelityCheckerFull:
    def setup_method(self) -> None:
        self.checker = FidelityChecker(mode="full")

    def test_identical_bodies(self) -> None:
        body = {"a": 1, "b": [2, 3]}
        assert self.checker.compare_request(body, body) == []

    def test_nested_diff_has_full_path(self) -> None:
        original = {"messages": [{"role": "user", "content": [{"type": "text"}]}]}
        roundtripped = {"messages": [{"role": "user", "content": [{"type": "image"}]}]}
        diffs = self.checker.compare_request(original, roundtripped)
        assert len(diffs) == 1
        assert "messages[0].content[0].type" in diffs[0].path

    def test_added_field(self) -> None:
        original = {"a": 1}
        roundtripped = {"a": 1, "b": 2}
        diffs = self.checker.compare_request(original, roundtripped)
        assert any(d.kind == "added" and d.path == "b" for d in diffs)

    def test_type_change(self) -> None:
        original = {"x": "hello"}
        roundtripped = {"x": 123}
        diffs = self.checker.compare_request(original, roundtripped)
        assert any(d.kind == "type_changed" for d in diffs)

    def test_list_length_diff(self) -> None:
        original = {"items": [1, 2]}
        roundtripped = {"items": [1, 2, 3]}
        diffs = self.checker.compare_request(original, roundtripped)
        assert any(d.kind == "added" and "[2]" in d.path for d in diffs)


# ============================================================================
# Fidelity integration in pipeline
# ============================================================================


class TestFidelityIntegration:
    def test_fidelity_check_runs_in_passthrough(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat",
            "openai_chat",
            force_conversion=False,
            fidelity_mode="critical",
        )
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = pipeline.convert_request(body)
        # Body is still passthrough (identity)
        assert result is body

    def test_fidelity_mode_none_skips_check(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat",
            "openai_chat",
            force_conversion=False,
            fidelity_mode=None,
        )
        assert pipeline._fidelity is None

    def test_fidelity_mode_only_on_passthrough(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat",
            "anthropic",
            force_conversion=False,
            fidelity_mode="critical",
        )
        # Different formats → not passthrough → no fidelity checker
        assert pipeline._fidelity is None

    def test_fidelity_full_mode_in_passthrough(self) -> None:
        pipeline = ConversionPipeline(
            "openai_chat",
            "openai_chat",
            force_conversion=False,
            fidelity_mode="full",
        )
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = pipeline.convert_request(body)
        assert result is body
        assert pipeline._fidelity is not None
