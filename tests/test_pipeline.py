"""Tests for llm_rosetta.pipeline and llm_rosetta.capabilities.

Note: This file imports private helpers (resolve_shim, _apply_config_reasoning_override)
directly for unit-testing internal logic.  The private one comes from its own module
rather than the package, so the package's public surface stays the public surface.
"""

import copy
from typing import Any, get_args, get_type_hints

import pytest

from llm_rosetta.capabilities import (
    enforce_reasoning,
    strip_reasoning_for_non_reasoning,
)
from llm_rosetta.capabilities.enforcement import _apply_config_reasoning_override
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.pipeline import apply_ir_transforms
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    register_shim,
    resolve_shim,
    unregister_shim,
)
from llm_rosetta.transforms import (
    auto_cache_breakpoints,
    strip_non_vision_images,
    truncate_images as truncate_images_transform,
    unwind_parallel_tool_calls as unwind_transform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REASONING_CAP = ReasoningCapability(
    effort_field="reasoning_effort",
    effort_range=("low", "high"),
)

_MODEL_REASONING_CAP = ReasoningCapability(
    thinking_modes={"auto": "adaptive", "enabled": "enabled", "disabled": "disabled"},
    effort_field="output_config.effort",
    effort_range=("low", "high"),
    budget_ratio=0.8,
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
    for name in ("test-shim", "test-shim-img", "test-shim-unwind", "test-shim-cache"):
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
        override_modes = {"auto": "adaptive", "disabled": "disabled"}
        enforce_reasoning(ctx, shim, config_override={"thinking_modes": override_modes})
        cap = ctx.options["reasoning_cap"]
        assert cap.thinking_modes == override_modes
        assert cap.effort_field == _REASONING_CAP.effort_field

    def test_config_override_on_model_override(self):
        """Config override should apply on top of model-level override."""
        ctx = ConversionContext()
        shim = _make_shim(
            reasoning=_REASONING_CAP,
            model_reasoning={"gpt-4": _MODEL_REASONING_CAP},
        )
        enforce_reasoning(
            ctx, shim, model="gpt-4", config_override={"budget_ratio": 0.5}
        )
        cap = ctx.options["reasoning_cap"]
        assert cap.budget_ratio == 0.5
        assert cap.thinking_modes == _MODEL_REASONING_CAP.thinking_modes

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
# strip_reasoning_for_non_reasoning
# ---------------------------------------------------------------------------


class TestStripReasoningForNonReasoning:
    def test_strips_when_no_reasoning_cap(self):
        """Reasoning config stripped when model lacks 'reasoning' capability."""
        ir = _simple_ir_request()
        ir["reasoning"] = {"mode": "auto", "effort": "high"}
        result = strip_reasoning_for_non_reasoning(ir, model_capabilities=["text"])
        assert "reasoning" not in result

    def test_keeps_when_reasoning_cap_present(self):
        """Reasoning config kept when model has 'reasoning' capability."""
        ir = _simple_ir_request()
        ir["reasoning"] = {"mode": "auto", "effort": "high"}
        result = strip_reasoning_for_non_reasoning(
            ir, model_capabilities=["text", "reasoning"]
        )
        assert result["reasoning"]["mode"] == "auto"
        assert result["reasoning"]["effort"] == "high"

    def test_noop_when_caps_none(self):
        """No-op when model_capabilities is None (unknown)."""
        ir = _simple_ir_request()
        ir["reasoning"] = {"mode": "enabled"}
        result = strip_reasoning_for_non_reasoning(ir, model_capabilities=None)
        assert result["reasoning"]["mode"] == "enabled"

    def test_noop_when_no_reasoning_in_request(self):
        """No-op when request has no reasoning config."""
        ir = _simple_ir_request()
        result = strip_reasoning_for_non_reasoning(ir, model_capabilities=["text"])
        assert "reasoning" not in result


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

    def test_auto_cache_breakpoints_injects(self):
        """auto_cache_breakpoints should inject cache_hint on IR parts."""
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                {"role": "user", "content": [{"type": "text", "text": "q2"}]},
            ],
            "system_instruction": [{"type": "text", "text": "Be helpful."}],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                }
            ],
        }
        shim = _make_shim(
            name="test-shim-cache", ir_transforms=(auto_cache_breakpoints(),)
        )
        result = apply_ir_transforms(ir, shim)
        # Should have cache_hint on: tool, system, 2 user messages = 4
        hints = 0
        if result["tools"][-1].get("cache_hint"):
            hints += 1
        if result["system_instruction"][-1].get("cache_hint"):
            hints += 1
        for msg in result["messages"]:
            if msg.get("role") == "user":
                for part in msg.get("content", []):
                    if part.get("cache_hint"):
                        hints += 1
        assert hints == 4

    def test_auto_cache_breakpoints_noop_when_hints_exist(self):
        """auto_cache_breakpoints should not inject when hints already present."""
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q1"}]},
            ],
            "system_instruction": [
                {"type": "text", "text": "sys", "cache_hint": {"type": "ephemeral"}}
            ],
            "tools": [],
        }
        original = copy.deepcopy(ir)
        shim = _make_shim(
            name="test-shim-cache", ir_transforms=(auto_cache_breakpoints(),)
        )
        result = apply_ir_transforms(ir, shim)
        assert result == original

    def test_auto_cache_breakpoints_repr(self):
        t = auto_cache_breakpoints()
        assert "auto_cache_breakpoints()" in repr(t)
        t2 = auto_cache_breakpoints(mode="fill_gaps")
        assert "fill_gaps" in repr(t2)


# ---------------------------------------------------------------------------
# _apply_config_reasoning_override
# ---------------------------------------------------------------------------


class TestApplyConfigReasoningOverride:
    def test_partial_override(self):
        modes = {"auto": "adaptive", "disabled": "disabled"}
        result = _apply_config_reasoning_override(
            _REASONING_CAP, {"thinking_modes": modes}
        )
        assert result.thinking_modes == modes
        assert result.effort_field == _REASONING_CAP.effort_field
        assert result.effort_range == _REASONING_CAP.effort_range

    def test_full_override(self):
        override = {
            "thinking_modes": {"enabled": "enabled"},
            "thinking_default": "enabled",
            "effort_field": "custom_effort",
            "effort_range": ["low", "high"],
            "budget_ratio": 0.5,
            "visibility_modes": {"auto": "auto"},
            "unsigned_blocks": "preserve",
        }
        result = _apply_config_reasoning_override(_REASONING_CAP, override)
        assert result.effort_field == "custom_effort"
        assert result.thinking_modes == {"enabled": "enabled"}
        assert result.budget_ratio == 0.5
        assert result.effort_range == ("low", "high")

    def test_empty_override_preserves_base(self):
        result = _apply_config_reasoning_override(_REASONING_CAP, {})
        assert result.effort_field == _REASONING_CAP.effort_field
        assert result.effort_range == _REASONING_CAP.effort_range
        assert result.thinking_modes == _REASONING_CAP.thinking_modes


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

    @pytest.mark.parametrize(
        "target_provider",
        ["google", "google_generate", "asksage--google_generate"],
    )
    def test_google_target_emits_rest_body(self, target_provider):
        """Every google provider name selects the REST body shape (#882).

        SDK-shaped output (``config``) is rejected by the Generative
        Language REST endpoint, so the canonical ``google_generate``
        name — and any shim name resolving to it — must apply the same
        REST transform as the ``google`` shim name.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("messages", target_provider)
        target = pipeline.convert_request(
            {
                "model": "gemini-2.0-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            }
        )
        assert "generationConfig" in target
        assert "config" not in target

    def test_google_generate_sdk_override_still_honoured(self):
        """Explicit ``google_output_format='sdk'`` bypasses the REST default."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline(
            "messages", "google_generate", google_output_format="sdk"
        )
        target = pipeline.convert_request(
            {
                "model": "gemini-2.0-flash",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            }
        )
        assert "config" in target
        assert "generationConfig" not in target

    @pytest.mark.parametrize("bad_format", ["REST", "Rest", "restful", "", "sdk "])
    def test_google_output_format_rejects_unknown_values(self, bad_format):
        """A mistyped ``google_output_format`` fails here, not upstream (#901).

        The converter selects REST only on an exact ``"rest"`` match and
        falls through to the SDK body shape for anything else — a shape
        the Generative Language REST endpoint rejects with HTTP 400
        ``Unknown name "config"``.  An unrecognised value must therefore
        be rejected at construction rather than silently mis-selecting.

        The value is routed through a ``str`` on purpose: a caller whose
        value comes from config is exactly the case the ``Literal``
        annotation cannot protect.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        value: str = bad_format
        with pytest.raises(ValueError, match="google_output_format"):
            ConversionPipeline(
                "messages",
                "google_generate",
                # ty cannot protect this call — that is the case under test.
                google_output_format=value,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize(
        "target_provider",
        ["chat_completions", "messages", "google_interactions"],
    )
    def test_google_output_format_inert_for_non_google_target(self, target_provider):
        """The value is validated only where it is read (#901 review).

        ``google_output_format`` is consulted only for a Google Generate
        target, so a config-driven caller that always carries the key must
        not fail on a route that ignores it.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        value: str = "REST"
        ConversionPipeline(
            "messages",
            target_provider,
            # ty cannot protect this call — that is the case under test.
            google_output_format=value,  # ty: ignore[invalid-argument-type]
        )

    def test_google_output_format_literal_tracks_converter_constant(self):
        """The annotation and the runtime guard must not drift apart (#901).

        ``google_output_format`` is annotated ``Literal["rest", "sdk"]``
        while the accepted set is owned by ``GOOGLE_OUTPUT_FORMATS`` in the
        converter.  A ``Literal`` cannot be derived from a constant, so pin
        the two in lockstep here.
        """
        from llm_rosetta.converters.google_generate import GOOGLE_OUTPUT_FORMATS
        from llm_rosetta.pipeline import ConversionPipeline

        hints = get_type_hints(ConversionPipeline.__init__)
        assert set(get_args(hints["google_output_format"])) == set(
            GOOGLE_OUTPUT_FORMATS
        )

    def test_chat_reasoning_request_omits_unproven_native_item(self):
        """Chat reasoning is not portable as a native Responses input item."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gpt-5",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "Thinking",
                    },
                    {"role": "user", "content": "Follow-up"},
                ],
            }
        )
        assert [item["type"] for item in target["input"]] == [
            "message",
            "message",
            "message",
        ]
        assert [item["role"] for item in target["input"]] == [
            "user",
            "assistant",
            "user",
        ]
        assistant = target["input"][1]
        assert assistant["content"] == [{"type": "output_text", "text": "Answer"}]
        visible_text = [
            part["text"]
            for item in target["input"]
            for part in item.get("content", [])
            if "text" in part
        ]
        assert "Thinking" not in visible_text
        assert not any(
            str(item.get("id", "")).startswith("rs_") for item in target["input"]
        )
        assert "store" not in target

    def test_chat_function_call_request_keeps_completed_status(self):
        """Reasoning status handling does not alter function-call request items."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_status",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_status",
                        "content": "result",
                    },
                ],
            }
        )
        function_call = next(
            item for item in target["input"] if item["type"] == "function_call"
        )
        assert function_call["status"] == "completed"

    def test_responses_reasoning_request_preserves_fields_without_status(self):
        """Request conversion preserves reasoning metadata except output-only status."""
        from llm_rosetta.pipeline import ConversionPipeline

        summary = [{"type": "summary_text", "text": "Thinking"}]
        pipeline = ConversionPipeline("openai_responses", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gpt-5",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_original",
                        "summary": summary,
                        "encrypted_content": "opaque-signature",
                        "status": "completed",
                    }
                ],
            }
        )
        assert [item["type"] for item in target["input"]] == ["reasoning"]
        reasoning = target["input"][0]
        assert reasoning["id"] == "rs_original"
        assert reasoning["summary"] == summary
        assert reasoning["encrypted_content"] == "opaque-signature"
        assert "status" not in reasoning

    def test_responses_reasoning_request_preserves_explicit_empty_summary(self):
        """Responses provenance fields survive request round-trip exactly."""
        from llm_rosetta.pipeline import ConversionPipeline

        raw_content = [{"type": "reasoning_text", "text": "Raw reasoning"}]
        pipeline = ConversionPipeline("openai_responses", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gpt-5",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_empty_summary",
                        "summary": [],
                        "content": raw_content,
                        "encrypted_content": "opaque-signature",
                        "status": "completed",
                    }
                ],
            }
        )
        assert target["input"] == [
            {
                "type": "reasoning",
                "id": "rs_empty_summary",
                "summary": [],
                "content": raw_content,
                "encrypted_content": "opaque-signature",
            }
        ]

    def test_anthropic_reasoning_request_omits_unproven_native_item(self):
        """Anthropic thinking is not portable as a native Responses input item."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("messages", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Let me reason about this",
                                "signature": "sig_abc",
                            },
                            {"type": "text", "text": "Answer"},
                        ],
                    },
                    {"role": "user", "content": "Follow-up"},
                ],
            }
        )
        assert [item["type"] for item in target["input"]] == [
            "message",
            "message",
            "message",
        ]
        assistant = target["input"][1]
        assert assistant["content"] == [{"type": "output_text", "text": "Answer"}]
        assert not any(
            str(item.get("id", "")).startswith("rs_") for item in target["input"]
        )

    def test_google_reasoning_request_omits_unproven_native_item(self):
        """Google thought part is not portable as a native Responses input item."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("google", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gemini-2.5-flash",
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Question"}],
                    },
                    {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "Internal reasoning"},
                            {"text": "Answer"},
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [{"text": "Follow-up"}],
                    },
                ],
            }
        )
        assert [item["type"] for item in target["input"]] == [
            "message",
            "message",
            "message",
        ]
        assistant = target["input"][1]
        assert assistant["content"] == [{"type": "output_text", "text": "Answer"}]
        assert not any(
            str(item.get("id", "")).startswith("rs_") for item in target["input"]
        )

    def test_chat_tool_list_content_to_responses(self):
        """Chat tool list content converts to Responses input blocks."""
        from llm_rosetta.pipeline import ConversionPipeline

        data_url = "data:image/png;base64,aW1hZ2U="
        pipeline = ConversionPipeline("openai_chat", "openai_responses")
        target = pipeline.convert_request(
            {
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_multimodal",
                                "type": "function",
                                "function": {"name": "inspect", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_multimodal",
                        "content": [
                            {"type": "text", "text": "Screenshot result"},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                        ],
                    },
                ],
            }
        )

        tool_output = next(
            item for item in target["input"] if item["type"] == "function_call_output"
        )
        assert tool_output["type"] == "function_call_output"
        assert tool_output["call_id"] == "call_multimodal"
        assert tool_output["output"] == [
            {"type": "input_text", "text": "Screenshot result"},
            {"type": "input_image", "image_url": data_url, "detail": "high"},
        ]
        assert all(block["type"] != "image_url" for block in tool_output["output"])

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

        pipeline = ConversionPipeline("openai_chat", "messages")
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

    def test_pipeline_sets_response_id_prefix_on_context(self):
        """Pipeline sets response_id_prefix from shim on context."""
        from llm_rosetta.pipeline import ConversionPipeline

        # #909: the ``resp_`` prefix is a shim trait, so the OpenAI-profile
        # shim must be named explicitly (the neutral base has no prefix).
        pipeline = ConversionPipeline(
            "openai_chat", "open_responses", target_shim="openai_responses"
        )
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        # Verify the pipeline resolved the target prefix
        assert pipeline._target_id_prefix == "resp_"

    def test_pipeline_response_uses_shim_prefix(self):
        """Pipeline response conversion applies shim-driven prefix."""
        from llm_rosetta.pipeline import ConversionPipeline

        # openai_responses → openai_responses round-trip through pipeline
        pipeline = ConversionPipeline("openai_responses", "openai_responses")
        body = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "hi"}],
        }
        pipeline.convert_request(body)
        upstream_response = {
            "id": "resp_pipeline_test_123",
            "object": "response",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
        }
        result = pipeline.convert_response(upstream_response)
        # Source converter adds resp_ prefix back
        assert result["id"] == "resp_pipeline_test_123"

    def test_chat_tool_list_content_round_trip(self):
        """Chat tool list content survives A→IR→A round-trip via packing."""
        from llm_rosetta.pipeline import ConversionPipeline

        data_url = "data:image/png;base64,aW1hZ2U="
        pipeline = ConversionPipeline("openai_chat", "openai_chat")
        target = pipeline.convert_request(
            {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_rt",
                                "type": "function",
                                "function": {"name": "screenshot", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_rt",
                        "content": [
                            {"type": "text", "text": "Screenshot captured"},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                        ],
                    },
                ],
            }
        )

        # Chat packing moves image to synthetic user message
        tool_msg = [m for m in target["messages"] if m.get("role") == "tool"][0]
        content = tool_msg["content"]
        # Tool message retains text only (image packed out)
        assert isinstance(content, list)
        assert content == [{"type": "text", "text": "Screenshot captured"}]

        # Image appears in the synthetic user message
        user_msgs = [m for m in target["messages"] if m.get("role") == "user"]
        synthetic = user_msgs[-1]
        assert isinstance(synthetic["content"], list)
        assert any(p.get("type") == "image_url" for p in synthetic["content"])

    def test_implied_target_applies_the_reasoning_config(self):
        """A shim-spelled ``target_provider`` reaches the reasoning gate.

        The pipeline resolves the target shim once, so naming the shim as the
        target *provider* gives the same traits as passing ``target_shim=``.
        This gate read the raw argument, so with no explicit shim the target's
        reasoning config was silently dropped.

        Each of the four readers has its own case below — one equality assertion
        only turns red for whichever reader happens to differ first.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
        }
        implied = ConversionPipeline(
            "chat_completions", "deepseek--chat_completions"
        ).convert_request(dict(body))
        explicit = ConversionPipeline(
            "chat_completions",
            "deepseek--chat_completions",
            target_shim=resolve_shim("deepseek--chat_completions"),
        ).convert_request(dict(body))
        assert implied.get("thinking") == {"type": "disabled"}
        assert implied == explicit

    def test_implied_target_applies_ir_transforms(self):
        """A late system message is hoisted for a shim-spelled target too.

        Pins the IR-transforms site on its own — the equality case above only
        turns red for the reasoning gate.  Every chat-completions shim carries
        ``hoist_late_system_messages()``, so the request body is reshaped here,
        not just a capability field.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "LATE"},
                {"role": "assistant", "content": "ok"},
            ],
        }
        out = ConversionPipeline(
            "chat_completions", "deepseek--chat_completions"
        ).convert_request(copy.deepcopy(body))
        explicit = ConversionPipeline(
            "chat_completions",
            "deepseek--chat_completions",
            target_shim=resolve_shim("deepseek--chat_completions"),
        ).convert_request(copy.deepcopy(body))
        assert not any(m.get("role") == "system" for m in out["messages"])
        assert any("<system>" in str(m.get("content")) for m in out["messages"])
        assert out == explicit

    def test_implied_target_applies_the_custom_tool_policy(self):
        """A custom tool survives a shim-spelled target that declares support.

        Pins the custom-tools site on its own.  ``alcf--sophia`` declares
        ``custom_tools: true``, so the implied form must keep the tool rather
        than downgrade it to a function with a synthesised ``input`` schema.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "custom", "name": "x", "format": {"type": "text"}}],
        }
        out = ConversionPipeline("open_responses", "alcf--sophia").convert_request(
            copy.deepcopy(body)
        )
        explicit = ConversionPipeline(
            "open_responses",
            "alcf--sophia",
            target_shim=resolve_shim("alcf--sophia"),
        ).convert_request(copy.deepcopy(body))
        assert out["tools"] == [
            {"type": "custom", "custom": {"name": "x", "format": {"type": "text"}}}
        ]
        assert out == explicit

    def test_implied_target_applies_body_transforms(self):
        """The target shim's body transforms run for an implied target too.

        Pins the fourth reader on its own.  This is the site a user notices most:
        ``deepseek--chat_completions`` declares
        ``strip_fields('n', 'logit_bias', 'seed', …)`` as a post-IR body
        transform, so for an implied target those fields used to reach the
        upstream intact while the explicit form stripped them.
        """
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 7,
            "frequency_penalty": 0.5,
            "n": 2,
        }
        out = ConversionPipeline(
            "chat_completions", "deepseek--chat_completions"
        ).convert_request(copy.deepcopy(body))
        explicit = ConversionPipeline(
            "chat_completions",
            "deepseek--chat_completions",
            target_shim=resolve_shim("deepseek--chat_completions"),
        ).convert_request(copy.deepcopy(body))
        assert "seed" not in out and "frequency_penalty" not in out and "n" not in out
        assert out == explicit


class TestStreamProcessorIterEvents:
    """Tests for StreamProcessor.iter_events (#872)."""

    def _make_processor(self, source="openai_chat", target="messages"):
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline(source, target)
        pipeline.convert_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        return pipeline.create_stream_processor()

    def test_iter_events_yields_all_events(self):
        """iter_events yields source-format events for all chunks."""
        processor = self._make_processor("openai_chat", "openai_chat")
        chunks = [
            {
                "id": "chatcmpl-abc",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "created": 1700000000,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}}
                ],
            },
            {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        events = list(processor.iter_events(chunks))
        assert len(events) > 0
        # Should have content and finish events
        has_content = any(
            c.get("choices", [{}])[0].get("delta", {}).get("content") == "Hi"
            for c in events
            if c.get("choices")
        )
        assert has_content

    def test_iter_events_flushes_deferred_stream_end(self):
        """iter_events auto-flushes deferred StreamEndEvent for Anthropic output."""
        processor = self._make_processor("messages", "openai_chat")
        chunks = [
            {
                "id": "chatcmpl-abc",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "created": 1700000000,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}}
                ],
            },
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
            # Finish without usage
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        events = list(processor.iter_events(chunks))
        types = [e.get("type") for e in events]
        assert "message_delta" in types, "message_delta missing — #838 regression"
        assert "message_stop" in types, "message_stop missing — #838 regression"

    def test_iter_events_empty_input(self):
        """iter_events with empty iterable returns no events."""
        processor = self._make_processor("openai_chat", "openai_chat")
        events = list(processor.iter_events([]))
        assert events == []


# ---------------------------------------------------------------------------
# Intrinsic IR transforms (converter._INTRINSIC_IR_TRANSFORMS)
# ---------------------------------------------------------------------------


class TestIntrinsicIrTransforms:
    """Verify that converter-intrinsic IR transforms are applied by the pipeline."""

    def test_intrinsic_without_shim(self):
        """Intrinsic transforms run even when shim is None."""
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "system", "content": [{"type": "text", "text": "late sys"}]},
                {"role": "user", "content": [{"type": "text", "text": "bye"}]},
            ],
        }

        class _FakeConverter:
            from llm_rosetta.transforms import hoist_late_system_messages

            _INTRINSIC_IR_TRANSFORMS = (hoist_late_system_messages(),)

        result = apply_ir_transforms(ir, shim=None, converter=_FakeConverter())
        # The late system message should have been rewritten to role: "user"
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_intrinsic_and_shim_compose(self):
        """Intrinsic transforms run first, then shim transforms."""
        call_order: list[str] = []

        def _intrinsic_fn(body, ctx):
            call_order.append("intrinsic")
            return body

        def _shim_fn(body, ctx):
            call_order.append("shim")
            return body

        from llm_rosetta.transforms import _NamedIRTransform

        intrinsic_t = _NamedIRTransform(_intrinsic_fn, "intrinsic()")
        shim_t = _NamedIRTransform(_shim_fn, "shim()")

        class _FakeConverter:
            _INTRINSIC_IR_TRANSFORMS = (intrinsic_t,)

        shim = _make_shim(ir_transforms=(shim_t,))
        ir = _simple_ir_request()
        apply_ir_transforms(ir, shim, converter=_FakeConverter())
        assert call_order == ["intrinsic", "shim"]

    def test_intrinsic_dedup_with_shim(self):
        """Same transform in intrinsic and shim runs only once."""
        call_count = 0

        def _counting_fn(body, ctx):
            nonlocal call_count
            call_count += 1
            return body

        from llm_rosetta.transforms import _NamedIRTransform

        t = _NamedIRTransform(_counting_fn, "dedup_test()")

        class _FakeConverter:
            _INTRINSIC_IR_TRANSFORMS = (t,)

        # Shim has a transform with the same repr
        shim_t = _NamedIRTransform(_counting_fn, "dedup_test()")
        shim = _make_shim(ir_transforms=(shim_t,))
        ir = _simple_ir_request()
        apply_ir_transforms(ir, shim, converter=_FakeConverter())
        assert call_count == 1

    def test_no_converter_no_shim_is_noop(self):
        """Both converter=None and shim=None → no-op (backward compat)."""
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)
        result = apply_ir_transforms(ir, None, converter=None)
        assert result == original

    def test_converter_without_intrinsic_attr(self):
        """Converter without _INTRINSIC_IR_TRANSFORMS attr → no-op."""
        ir = _simple_ir_request()
        original = copy.deepcopy(ir)

        class _PlainConverter:
            pass

        result = apply_ir_transforms(ir, None, converter=_PlainConverter())  # ty: ignore[invalid-argument-type]
        assert result == original

    def test_openai_chat_has_no_intrinsic_transforms(self):
        """OpenAI Chat converter should not declare intrinsic transforms."""
        from llm_rosetta.converters.chat_completions import ChatCompletionsConverter

        assert ChatCompletionsConverter._INTRINSIC_IR_TRANSFORMS == ()

    def test_openai_responses_has_no_intrinsic_transforms(self):
        """OpenAI Responses converter should not declare intrinsic transforms."""
        from llm_rosetta.converters.open_responses import OpenAIResponsesConverter

        assert OpenAIResponsesConverter._INTRINSIC_IR_TRANSFORMS == ()

    def test_anthropic_has_intrinsic_transforms(self):
        """Anthropic converter declares hoist + cache breakpoints."""
        from llm_rosetta.converters.messages import MessagesConverter

        names = [repr(t) for t in MessagesConverter._INTRINSIC_IR_TRANSFORMS]
        assert "hoist_late_system_messages()" in names
        assert "auto_cache_breakpoints()" in names

    def test_google_generate_has_intrinsic_transforms(self):
        """Google Generate converter declares hoist."""
        from llm_rosetta.converters.google_generate import GoogleGenerateConverter

        names = [repr(t) for t in GoogleGenerateConverter._INTRINSIC_IR_TRANSFORMS]
        assert "hoist_late_system_messages()" in names

    def test_google_interactions_has_intrinsic_transforms(self):
        """Google Interactions converter declares hoist."""
        from llm_rosetta.converters.google_interactions import (
            GoogleInteractionsConverter,
        )

        names = [repr(t) for t in GoogleInteractionsConverter._INTRINSIC_IR_TRANSFORMS]
        assert "hoist_late_system_messages()" in names


class TestDeepSeekReasoningContentBackfill:
    """DeepSeek's thinking mode rejects a history that ends on a tool result
    unless every assistant tool-call message carries ``reasoning_content``.

    Reproduces the shape a cross-format agent harness produces when it drops
    the model's reasoning for a turn: a Messages history whose last turn is a
    bare ``tool_use`` (no ``thinking``), converted to Chat Completions.  The
    deepseek shim backfills an empty value; a bare conversion does not.
    """

    @pytest.fixture(autouse=True)
    def _load_shims(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    @staticmethod
    def _source_body():
        return {
            "model": "deepseek-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "read it"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should read the file.",
                            "signature": "",
                        },
                        {
                            "type": "tool_use",
                            "id": "c1",
                            "name": "read",
                            "input": {"path": "/x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c1",
                            "content": "data",
                        }
                    ],
                },
                # This turn's thinking was dropped by the client.
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "c2",
                            "name": "read",
                            "input": {"path": "/y"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c2",
                            "content": "more",
                        }
                    ],
                },
            ],
        }

    @staticmethod
    def _callers(out):
        return [
            m
            for m in out["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]

    def test_deepseek_shim_backfills_missing_reasoning_content(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "messages",
            "chat_completions",
            target_shim="deepseek--chat_completions",
        ).convert_request(self._source_body())

        callers = self._callers(out)
        assert len(callers) == 2
        assert callers[0]["reasoning_content"] == "I should read the file."
        assert callers[1]["reasoning_content"] == ""
        assert out["messages"][-1]["role"] == "tool"

    def test_bare_conversion_leaves_field_absent(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("messages", "chat_completions").convert_request(
            self._source_body()
        )

        callers = self._callers(out)
        assert callers[0]["reasoning_content"] == "I should read the file."
        assert "reasoning_content" not in callers[1]


class TestUnsignedBlocksPolicy:
    """Signature-requiring Messages targets withhold unsigned reasoning (#954).

    A history whose reasoning arrived from a signature-less format cannot be
    replayed to Anthropic: the API rejects any thinking block without a
    signature.  The shim's ``unsigned_blocks: preserve`` policy drops the block
    from the wire and keeps it in metadata; the default ``as_is`` sends it.

    Which targets need the policy is per provider, probed live 2026-10-13
    (anthropic, openrouter, the opencode Messages lane and AskSage reject;
    MiniMax's lane accepts).
    """

    @pytest.fixture(autouse=True)
    def _load_shims(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    @staticmethod
    def _source_body():
        """A Chat history whose assistant turn carries reasoning, no signature."""
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "reasoning_content": "think hard",
                    "content": "Hello",
                },
                {"role": "user", "content": "bye"},
            ],
        }

    @staticmethod
    def _thinking_blocks(out):
        return [
            block
            for message in out["messages"]
            if message.get("role") == "assistant"
            for block in message.get("content", [])
            if block.get("type") == "thinking"
        ]

    @pytest.mark.parametrize(
        "shim_name",
        [
            "anthropic--messages",
            "openrouter--messages",
            "opencode--messages",
            "asksage--messages",
        ],
    )
    def test_signature_requiring_shim_withholds_unsigned_reasoning(self, shim_name):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions", "messages", target_shim=shim_name
        ).convert_request(self._source_body())

        assert self._thinking_blocks(out) == []
        assert out["messages"][1]["content"] == [{"type": "text", "text": "Hello"}]

    def test_minimax_shim_keeps_it_on_the_wire(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions", "messages", target_shim="minimax--messages"
        ).convert_request(self._source_body())

        assert len(self._thinking_blocks(out)) == 1

    def test_bare_conversion_sends_it_unsigned(self):
        """Without a shim the default policy is ``as_is``."""
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("chat_completions", "messages").convert_request(
            self._source_body()
        )

        blocks = self._thinking_blocks(out)
        assert len(blocks) == 1
        assert "signature" not in blocks[0]


class TestMessagesShimReasoningDefaults:
    """Adding a shim ``reasoning:`` section must not reset the cap (#954).

    ``opencode--messages`` and ``asksage--messages`` had no section before, so
    they inherited the Messages base cap.  Adding one to carry the unsigned
    policy made the omitted fields fall back to the loader defaults, which sent
    the effort out as a flat ``reasoning_effort`` on an Anthropic payload.
    """

    @pytest.fixture(autouse=True)
    def _load_shims(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    @staticmethod
    def _body():
        return {
            "model": "m",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}],
        }

    @pytest.mark.parametrize(
        "shim_name",
        ["opencode--messages", "asksage--messages"],
    )
    def test_effort_still_goes_out_as_output_config(self, shim_name):
        """MiniMax is not in this list: it has no effort parameter (#961)."""
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions", "messages", target_shim=shim_name
        ).convert_request(self._body())

        assert out["thinking"] == {"type": "adaptive"}
        assert out["output_config"] == {"effort": "high"}
        assert "reasoning_effort" not in out


class TestShimEffortLadders:
    """Each lane clamps or drops the effort its server requires (#961).

    ALCF answers 500 for anything outside vLLM's low/medium/high, MiniMax's
    Messages lane has no effort parameter at all, and MiniMax's Chat lane
    documents low…max.  The shim declarations have to land on exactly those.
    """

    @pytest.fixture(autouse=True)
    def _load_shims(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    @staticmethod
    def _body(effort):
        return {
            "model": "m",
            "reasoning_effort": effort,
            "messages": [{"role": "user", "content": "hi"}],
        }

    @pytest.mark.parametrize(
        ("effort", "sent"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_alcf_clamps_to_the_vllm_ladder(self, effort, sent):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions", "chat_completions", target_shim="alcf--sophia"
        ).convert_request(self._body(effort))

        assert out["reasoning_effort"] == sent

    def test_minimax_messages_sends_no_effort_at_all(self):
        """Thinking is supported there; effort is not — not even as a stray field."""
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions", "messages", target_shim="minimax--messages"
        ).convert_request(self._body("max"))

        assert "reasoning_effort" not in out
        assert "output_config" not in out
        assert out["thinking"] == {"type": "adaptive"}

    @pytest.mark.parametrize(("effort", "sent"), [("minimal", "low"), ("max", "max")])
    def test_minimax_chat_clamps_to_the_documented_ladder(self, effort, sent):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "chat_completions",
            "chat_completions",
            target_shim="minimax--chat_completions",
        ).convert_request(self._body(effort))

        assert out["reasoning_effort"] == sent


class TestReasoningRenameWiring:
    """The rename only helps if it is wired, and if it covers both holders.

    The unit tests call ``_rename_reasoning_field`` directly, so emptying a
    shim's ``pre_ir_transforms`` used to leave the suite green while the lane
    lost its reasoning again — both the ``alcf--sophia`` and the
    ``openrouter--chat_completions`` shim carry this rename, so both are driven
    through the response leg here (#969 review).
    """

    @pytest.fixture(autouse=True)
    def _load_shims(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    @staticmethod
    def _pipeline(shim_name):
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline(
            "messages", "chat_completions", target_shim=shim_name
        )
        pipeline.convert_request(
            {
                "model": "openai/gpt-oss-120b",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        return pipeline

    @staticmethod
    def _chunk(delta, finish_reason=None):
        return {
            "id": "chatcmpl-alcf",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    @pytest.mark.parametrize(
        "shim_name", ["alcf--sophia", "openrouter--chat_completions"]
    )
    def test_non_streaming_response_keeps_the_lane_reasoning(self, shim_name):
        out = self._pipeline(shim_name).convert_response(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "555",
                            "reasoning": "15*37 = 555",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

        assert [b["type"] for b in out["content"]] == ["thinking", "text"]
        assert out["content"][0]["thinking"] == "15*37 = 555"

    @pytest.mark.parametrize(
        "shim_name", ["alcf--sophia", "openrouter--chat_completions"]
    )
    def test_streaming_delta_keeps_the_lane_reasoning(self, shim_name):
        processor = self._pipeline(shim_name).create_stream_processor()

        events = []
        for chunk in (
            self._chunk({"role": "assistant", "reasoning": "let me think"}),
            self._chunk({"content": "555"}),
            self._chunk({}, finish_reason="stop"),
        ):
            events.extend(processor.process_chunk(chunk))

        starts = [
            e["content_block"]["type"]
            for e in events
            if e.get("type") == "content_block_start"
        ]
        assert starts == ["thinking", "text"], starts
        thinking = "".join(
            e["delta"]["thinking"]
            for e in events
            if e.get("type") == "content_block_delta"
            and e["delta"]["type"] == "thinking_delta"
        )
        assert thinking == "let me think"
