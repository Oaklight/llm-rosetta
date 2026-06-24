"""Tests for llm_rosetta.shims.pipeline — unified shim entry points."""

import copy

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims.pipeline import (
    _apply_config_reasoning_override,
    _resolve_shim,
    apply_shim_to_ir,
    setup_shim_context,
)
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    register_shim,
    unregister_shim,
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


def _make_shim(**kwargs):
    """Create a ProviderShim with sensible defaults, overridable via kwargs."""
    defaults = dict(name="test-shim", base="openai_chat")
    defaults.update(kwargs)
    return ProviderShim(**defaults)


@pytest.fixture(autouse=True)
def _register_cleanup():
    """Ensure test shims are cleaned up after each test."""
    yield
    for name in ("test-shim", "test-shim-img", "test-shim-unwind"):
        unregister_shim(name)


def _simple_ir_request(n_messages=1, n_images=0):
    """Build a minimal IR request dict for testing."""
    content = [{"type": "text", "text": f"message {i}"} for i in range(n_messages)]
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
# _resolve_shim
# ---------------------------------------------------------------------------


class TestResolveShim:
    def test_none(self):
        assert _resolve_shim(None) is None

    def test_provider_shim_instance(self):
        shim = _make_shim()
        assert _resolve_shim(shim) is shim

    def test_registered_name(self):
        shim = _make_shim()
        register_shim(shim)
        assert _resolve_shim("test-shim") is shim

    def test_unknown_name(self):
        assert _resolve_shim("nonexistent-shim") is None


# ---------------------------------------------------------------------------
# setup_shim_context
# ---------------------------------------------------------------------------


class TestSetupShimContext:
    def test_none_shim_is_noop(self):
        ctx = ConversionContext()
        setup_shim_context(ctx, None)
        assert "reasoning_cap" not in ctx.options

    def test_shim_without_reasoning_is_noop(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=None)
        setup_shim_context(ctx, shim)
        assert "reasoning_cap" not in ctx.options

    def test_provider_level_reasoning(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=_REASONING_CAP)
        setup_shim_context(ctx, shim)
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_model_level_override(self):
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        setup_shim_context(ctx, shim, model="gpt-4")
        assert ctx.options["reasoning_cap"] is _MODEL_REASONING_CAP

    def test_model_not_in_overrides_falls_back(self):
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        setup_shim_context(ctx, shim, model="gpt-3.5")
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_config_override_highest_priority(self):
        ctx = ConversionContext()
        shim = _make_shim(reasoning=_REASONING_CAP)
        setup_shim_context(ctx, shim, config_override={"thinking_type": "adaptive"})
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
        setup_shim_context(
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
        setup_shim_context(ctx, "test-shim")
        assert ctx.options["reasoning_cap"] is _REASONING_CAP

    def test_unknown_name_is_noop(self):
        ctx = ConversionContext()
        setup_shim_context(ctx, "nonexistent")
        assert "reasoning_cap" not in ctx.options


# ---------------------------------------------------------------------------
# apply_shim_to_ir
# ---------------------------------------------------------------------------


class TestApplyShimToIr:
    def test_none_shim_passthrough(self):
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)
        result = apply_shim_to_ir(ir, None)
        assert result == original

    def test_shim_no_features_passthrough(self):
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)
        shim = _make_shim()
        result = apply_shim_to_ir(ir, shim)
        assert result == original

    def test_strip_non_vision_when_no_vision_cap(self):
        """Images should be stripped when model lacks vision capability."""
        ir = _simple_ir_request(n_images=3)
        result = apply_shim_to_ir(
            ir, None, model_capabilities=["text"], upstream_model="deepseek-chat"
        )
        # Images should be replaced with text placeholders
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) == 0

    def test_no_strip_when_vision_cap(self):
        """Images should NOT be stripped when model has vision capability."""
        ir = _simple_ir_request(n_images=3)
        original = copy.deepcopy(ir)
        result = apply_shim_to_ir(
            ir, None, model_capabilities=["text", "vision"], upstream_model="gpt-4o"
        )
        assert result == original

    def test_no_strip_when_caps_none(self):
        """Images should NOT be stripped when model_capabilities is None."""
        ir = _simple_ir_request(n_images=3)
        original = copy.deepcopy(ir)
        result = apply_shim_to_ir(ir, None, model_capabilities=None)
        assert result == original

    def test_image_limit_enforced(self):
        """Shim with max_images should truncate excess images."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(name="test-shim-img", max_images=2)
        result = apply_shim_to_ir(ir, shim)
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) <= 2

    def test_image_limit_pattern_match(self):
        """Image limit should only fire when model matches pattern."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(name="test-shim-img", max_images=2, max_images_pattern="^gpt")
        # Matching model
        result = apply_shim_to_ir(copy.deepcopy(ir), shim, upstream_model="gpt-4o")
        content = result["messages"][0]["content"]
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) <= 2

    def test_image_limit_pattern_no_match(self):
        """Image limit should NOT fire when model doesn't match pattern."""
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(name="test-shim-img", max_images=2, max_images_pattern="^gpt")
        result = apply_shim_to_ir(copy.deepcopy(ir), shim, upstream_model="gemini-pro")
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
        shim = _make_shim(name="test-shim-unwind", unwind_parallel_tool_calls=True)
        result = apply_shim_to_ir(ir, shim)
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
            unwind_parallel_tool_calls=True,
            unwind_parallel_tool_calls_pattern="^gemini",
        )
        result = apply_shim_to_ir(ir, shim, upstream_model="gpt-4o")
        assert len(result["messages"]) == 4  # untouched

    def test_accepts_registered_name(self):
        ir = _simple_ir_request(n_images=5)
        shim = _make_shim(name="test-shim-img", max_images=2)
        register_shim(shim)
        result = apply_shim_to_ir(ir, "test-shim-img")
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
