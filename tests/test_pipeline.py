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

        pipeline = ConversionPipeline("anthropic", "openai_responses")
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

    def test_pipeline_sets_response_id_prefix_on_context(self):
        """Pipeline sets response_id_prefix from shim on context."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_chat", "openai_responses")
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
