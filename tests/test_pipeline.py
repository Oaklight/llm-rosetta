"""Tests for llm_rosetta.pipeline and llm_rosetta.capabilities.

Note: This file imports private helpers (resolve_shim, _apply_config_reasoning_override)
directly from llm_rosetta.capabilities for unit-testing internal logic.
"""

import copy
from typing import Any

import pytest

from llm_rosetta.capabilities import (
    _apply_config_reasoning_override,
    enforce_reasoning,
)
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.pipeline import apply_ir_transforms
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    register_shim,
    resolve_shim,
    unregister_shim,
)
from llm_rosetta.shims.transforms import (
    strip_non_vision_images,
    truncate_images as truncate_images_transform,
    unwind_parallel_tool_calls as unwind_transform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REASONING_CAP = ReasoningCapability(
    disabled="omit",
    effort_field="reasoning_effort",
    effort_map={"low": "low", "medium": "medium", "high": "high"},
)

_MODEL_REASONING_CAP = ReasoningCapability(
    disabled="thinking_disabled",
    effort_field="output_config.effort",
    thinking_type="enabled",
    effort_map={"low": "low", "high": "high"},
    budget_tokens_default_ratio=0.8,
)


def _make_shim(**kwargs: Any) -> ProviderShim:
    """Create a ProviderShim with sensible defaults, overridable via kwargs."""
    defaults: dict[str, Any] = dict(name="test-shim", base="openai_chat")
    defaults.update(kwargs)
    return ProviderShim(**defaults)


@pytest.fixture(autouse=True)
def _register_cleanup():
    """Ensure test shims are cleaned up after each test."""
    yield
    for name in ("test-shim", "test-shim-img", "test-shim-unwind"):
        unregister_shim(name)


def _simple_ir_request(n_messages: int = 1, n_images: int = 0) -> dict[str, Any]:
    """Build a minimal IR request dict for testing."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"message {i}"} for i in range(n_messages)
    ]
    for i in range(n_images):
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": f"img{i}",
                    "media_type": "image/png",
                },
            }
        )
    return {
        "messages": [{"role": "user", "content": content}],
        "tools": [],
    }


# ---------------------------------------------------------------------------
# resolve_shim
# ---------------------------------------------------------------------------


class TestResolveShim:
    def test_none(self):
        assert resolve_shim(None) is None

    def test_provider_shim_instance(self):
        shim = _make_shim()
        assert resolve_shim(shim) is shim

    def test_registered_name(self):
        shim = _make_shim()
        register_shim(shim)
        assert resolve_shim("test-shim") is shim

    def test_unknown_name(self):
        assert resolve_shim("nonexistent-shim") is None


# ---------------------------------------------------------------------------
# enforce_reasoning
# ---------------------------------------------------------------------------


class TestEnforceReasoning:
    def test_none_shim_is_noop(self):
        ctx = ConversionContext()
        enforce_reasoning(ctx, None)
        assert "reasoning_cap" not in ctx.options

    def test_shim_without_reasoning_is_noop(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=None)
        enforce_reasoning(ctx, shim)
        assert "reasoning_cap" not in ctx.options

    def test_provider_level_reasoning(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=_REASONING_CAP)
        enforce_reasoning(ctx, shim)
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_model_level_override(self):
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        enforce_reasoning(ctx, shim, model="gpt-4")
        assert ctx.options["reasoning_cap"] is _MODEL_REASONING_CAP

    def test_model_not_in_overrides_falls_back(self):
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        enforce_reasoning(ctx, shim, model="gpt-3.5")
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_config_override_highest_priority(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=_REASONING_CAP)
        enforce_reasoning(ctx, shim, config_override={"thinking_type": "adaptive"})
        cap = ctx.options["reasoning_cap"]
        assert cap.thinking_type == "adaptive"
        # Other fields inherited from base
        assert cap.disabled == _REASONING_CAP.disabled
        assert cap.effort_field == _REASONING_CAP.effort_field

    def test_config_override_on_model_override(self):
        """Config override should apply on top of model-level override."""
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        enforce_reasoning(
            ctx, shim, model="gpt-4", config_override={"disabled": "block"}
        )
        cap = ctx.options["reasoning_cap"]
        assert cap.disabled == "block"
        # Rest inherited from model-level
        assert cap.thinking_type == _MODEL_REASONING_CAP.thinking_type

    def test_accepts_registered_name(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=_REASONING_CAP)
        register_shim(shim)
        enforce_reasoning(ctx, "test-shim")
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_unknown_name_is_noop(self):
        ctx = ConversionContext()
        enforce_reasoning(ctx, "nonexistent")
        assert "reasoning_cap" not in ctx.options


# ---------------------------------------------------------------------------
# apply_ir_transforms
# ---------------------------------------------------------------------------


class TestApplyIrTransforms:
    def test_none_shim_passthrough(self):
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)
        result = apply_ir_transforms(ir, None)
        assert result == original

    def test_shim_no_features_passthrough(self):
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)
        shim = _make_shim()
        result = apply_ir_transforms(ir, shim)
        assert result == original

    def test_strip_non_vision_when_no_vision_cap(self):
        """Images should be stripped when model lacks vision capability."""
        ir = _simple_ir_request(n_images=3)
        shim = _make_shim(ir_transforms=(strip_non_vision_images(),))
        result = apply_ir_transforms(
            ir, shim, model_capabilities=["text"], upstream_model="deepseek-chat"
        )
        # Images should be replaced with text placeholders
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) == 0

    def test_no_strip_when_vision_cap(self):
        """Images should NOT be stripped when model has vision capability."""
        ir = _simple_ir_request(n_images=3)
        original = copy.deepcopy(ir)
        shim = _make_shim(ir_transforms=(strip_non_vision_images(),))
        result = apply_ir_transforms(
            ir, shim, model_capabilities=["text", "vision"], upstream_model="gpt-4o"
        )
        assert result == original

    def test_no_strip_when_caps_none(self):
        """Images should NOT be stripped when model_capabilities is None."""
        ir = _simple_ir_request(n_images=3)
        original = copy.deepcopy(ir)
        shim = _make_shim(ir_transforms=(strip_non_vision_images(),))
        result = apply_ir_transforms(ir, shim, model_capabilities=None)
        assert result == original

    def test_image_limit_enforced(self):
        """Shim with max_images should truncate excess images."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(
            name="test-shim-img", ir_transforms=(truncate_images_transform(2),)
        )
        result = apply_ir_transforms(ir, shim)
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) <= 2

    def test_image_limit_pattern_match(self):
        """Image limit should only fire when model matches pattern."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(
            name="test-shim-img",
            ir_transforms=(truncate_images_transform(2, pattern="^gpt"),),
        )
        # Matching model
        result = apply_ir_transforms(copy.deepcopy(ir), shim, upstream_model="gpt-4o")
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) <= 2

    def test_image_limit_pattern_no_match(self):
        """Image limit should NOT fire when model doesn't match pattern."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(
            name="test-shim-img",
            ir_transforms=(truncate_images_transform(2, pattern="^gpt"),),
        )
        result = apply_ir_transforms(
            copy.deepcopy(ir), shim, upstream_model="gemini-pro"
        )
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) == 5  # untouched

    def test_unwind_parallel_tool_calls(self):
        """Shim with unwind should split parallel tool calls."""
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "tool_name": "fn_a",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "tool_name": "fn_b",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "a",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_2",
                            "result": "b",
                        },
                    ],
                },
            ],
            "tools": [],
        }
        shim = _make_shim(name="test-shim-unwind", ir_transforms=(unwind_transform(),))
        result = apply_ir_transforms(ir, shim)
        # After unwind: user + (assistant+tool) + (assistant+tool) = 5
        assert len(result["messages"]) == 5

    def test_unwind_pattern_no_match(self):
        """Unwind should NOT fire when model doesn't match pattern."""
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "tool_name": "fn_a",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "tool_name": "fn_b",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "a",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_2",
                            "result": "b",
                        },
                    ],
                },
            ],
            "tools": [],
        }
        shim = _make_shim(
            name="test-shim-unwind",
            ir_transforms=(unwind_transform(pattern="^gemini"),),
        )
        result = apply_ir_transforms(ir, shim, upstream_model="gpt-4o")
        assert len(result["messages"]) == 4  # untouched

    def test_accepts_registered_name(self):
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(
            name="test-shim-img", ir_transforms=(truncate_images_transform(2),)
        )
        register_shim(shim)
        result = apply_ir_transforms(ir, "test-shim-img")
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) <= 2


# ---------------------------------------------------------------------------
# _apply_config_reasoning_override
# ---------------------------------------------------------------------------


class TestApplyConfigReasoningOverride:
    def test_partial_override(self):
        result = _apply_config_reasoning_override(
            _REASONING_CAP, {"thinking_type": "adaptive"}
        )
        assert result.thinking_type == "adaptive"
        assert result.disabled == _REASONING_CAP.disabled
        assert result.effort_field == _REASONING_CAP.effort_field
        assert result.effort_map == _REASONING_CAP.effort_map

    def test_full_override(self):
        override = {
            "disabled": "block",
            "effort_field": "custom_effort",
            "max_effort": "high",
            "thinking_type": "enabled",
            "unsigned_reasoning_blocks": "drop",
            "effort_map": {"a": "b"},
            "budget_tokens_default_ratio": 0.5,
        }
        result = _apply_config_reasoning_override(_REASONING_CAP, override)
        assert result.disabled == "block"
        assert result.effort_field == "custom_effort"
        assert result.thinking_type == "enabled"
        assert result.budget_tokens_default_ratio == 0.5

    def test_empty_override_preserves_base(self):
        result = _apply_config_reasoning_override(_REASONING_CAP, {})
        assert result.disabled == _REASONING_CAP.disabled
        assert result.effort_field == _REASONING_CAP.effort_field
        assert result.effort_map == _REASONING_CAP.effort_map


# ---------------------------------------------------------------------------
# ConversionPipeline
# ---------------------------------------------------------------------------


class TestConversionPipeline:
    """Tests for the high-level ConversionPipeline class."""

    def test_convert_request_openai_to_openai(self):
        """Same-format round-trip produces valid target body."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        target = pipeline.convert_request(body)
        assert "messages" in target
        assert target["model"] == "gpt-4"

    def test_convert_response_openai_to_openai(self):
        """Response round-trip produces valid source response."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        response = pipeline.convert_response(
            {
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            }
        )
        assert "choices" in response

    def test_convert_request_raises_conversion_error(self):
        """Completely invalid body should raise ConversionError with phase info."""
        from llm_rosetta.pipeline import ConversionError, ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        with pytest.raises(ConversionError) as exc_info:
            # messages must be iterable — passing an int triggers a parse error
            pipeline.convert_request({"model": "gpt-4", "messages": 123})
        assert exc_info.value.phase == "source_to_ir"

    def test_convert_request_twice_raises(self):
        """Calling convert_request twice raises RuntimeError (one-shot)."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        with pytest.raises(RuntimeError, match="one-shot"):
            pipeline.convert_request(
                {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
            )

    def test_convert_response_before_request_raises(self):
        """Calling convert_response before convert_request raises RuntimeError."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        with pytest.raises(RuntimeError):
            pipeline.convert_response({"choices": []})

    def test_on_ir_ready_callback_request(self):
        """on_ir_ready callback fires after source→IR, before shim transforms."""
        from llm_rosetta.pipeline import ConversionPipeline

        captured: list[dict[str, Any]] = []
        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            on_ir_ready=lambda ir: captured.append(ir),
        )
        assert len(captured) == 1
        assert "messages" in captured[0]

    def test_on_ir_ready_callback_response(self):
        """on_ir_ready callback fires after target→IR in convert_response."""
        from llm_rosetta.pipeline import ConversionPipeline

        captured: list[dict[str, Any]] = []
        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        pipeline.convert_response(
            {
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
            on_ir_ready=lambda ir: captured.append(ir),
        )
        assert len(captured) == 1

    def test_context_available_after_convert_request(self):
        """Pipeline context should be accessible after convert_request."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        with pytest.raises(RuntimeError):
            _ = pipeline.context
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        ctx = pipeline.context
        assert ctx.options.get("metadata_mode") == "preserve"

    def test_ir_request_available_after_convert_request(self):
        """Pipeline ir_request should be accessible after convert_request."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        with pytest.raises(RuntimeError):
            _ = pipeline.ir_request
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        ir = pipeline.ir_request
        assert "messages" in ir

    def test_no_shim_passthrough(self):
        """Pipeline without shim still works — no transforms applied."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat", None)
        target = pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert "messages" in target

    def test_cross_format_openai_to_anthropic(self):
        """Cross-format conversion produces valid target body."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "anthropic")
        target = pipeline.convert_request(
            {"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}
        )
        # Anthropic format should have "messages" with different structure
        assert "messages" in target

    def test_create_stream_processor(self):
        """StreamProcessor should be creatable after convert_request."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        processor = pipeline.create_stream_processor()
        assert processor is not None

    def test_create_stream_processor_before_request_raises(self):
        """create_stream_processor before convert_request raises RuntimeError."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        with pytest.raises(RuntimeError):
            pipeline.create_stream_processor()

    def test_stream_processor_on_ir_event_callback(self):
        """StreamProcessor on_ir_event callback fires for each IR event."""
        from llm_rosetta.pipeline import ConversionPipeline

        captured: list[dict[str, Any]] = []
        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        processor = pipeline.create_stream_processor(
            on_ir_event=lambda ev: captured.append(ev)
        )
        # Feed a simple streaming chunk
        chunk = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {"content": "hello"}, "index": 0}],
        }
        events = processor.process_chunk(chunk)
        # Should produce source events and fire callback for IR events
        assert isinstance(events, list)
        # Callback should have been called at least once if IR events were produced
        if events:
            assert len(captured) > 0
