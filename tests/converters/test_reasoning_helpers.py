"""Tests for shim-driven reasoning helpers.

Test matrix:
- Input normalisation: none → disabled, effort values pass through
- OpenAI (Chat+Responses): disabled → omit (no thinking_modes), effort clamped
- Anthropic: disabled → thinking.type=disabled, effort clamped to [low, max]
- Google: disabled → thinkingBudget=0, effort clamped
- DeepSeek-style: thinking_modes with enabled/disabled
- Custom shim override
"""

from __future__ import annotations

from typing import Any, cast

from llm_rosetta.converters.base.helpers.reasoning import (
    DEFAULT_REASONING_CAPS,
    apply_reasoning_config,
    normalize_reasoning_input,
)
from llm_rosetta.shims.provider_shim import ReasoningCapability
from llm_rosetta.types.ir.reasoning import ReasoningConfig


# ── Input normalisation ────────────────────────────────────────────────────


class TestNormalizeReasoningInput:
    def test_none_becomes_disabled(self):
        result = normalize_reasoning_input(cast(ReasoningConfig, {"effort": "none"}))
        assert result["mode"] == "disabled"
        assert "effort" not in result

    def test_effort_values_pass_through(self):
        for level in ("minimal", "low", "medium", "high", "xhigh", "max"):
            result = normalize_reasoning_input(cast(ReasoningConfig, {"effort": level}))
            assert result["effort"] == level

    def test_none_preserves_other_fields(self):
        result = normalize_reasoning_input(
            cast(ReasoningConfig, {"effort": "none", "budget_tokens": 4096})
        )
        assert result["mode"] == "disabled"
        assert result["budget_tokens"] == 4096
        assert "effort" not in result

    def test_empty_passes_through(self):
        result = normalize_reasoning_input(cast(ReasoningConfig, {}))
        assert result == {}

    def test_does_not_mutate_original(self):
        original: dict[str, Any] = {"effort": "xhigh"}
        normalize_reasoning_input(cast(ReasoningConfig, original))
        assert original["effort"] == "xhigh"


# ── OpenAI Chat shim ──────────────────────────────────────────────────────


class TestOpenAIChatShim:
    """OpenAI Chat: no thinking_modes, effort clamped to [minimal, high]."""

    cap = DEFAULT_REASONING_CAPS["openai_chat"]

    def test_disabled_omits_all(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result == {}

    def test_effort_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_effort_minimal(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "minimal"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "minimal"

    def test_effort_xhigh_clamped_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_effort_max_clamped_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_mode_auto_no_thinking_for_standard_openai(self):
        """Standard OpenAI: mode=auto does NOT produce a thinking block."""
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "auto"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert "thinking" not in result

    def test_mode_enabled_no_thinking_for_standard_openai(self):
        """Standard OpenAI: mode=enabled does NOT produce a thinking block."""
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 2048}),
            self.cap,
            converter_type="openai_chat",
        )
        assert "thinking" not in result


# ── OpenAI Chat thinking-capable shim ─────────────────────────────────────


class TestOpenAIChatThinkingCapable:
    """OpenAI Chat with thinking_modes: mode/budget → thinking block."""

    cap = ReasoningCapability(
        thinking_modes={
            "auto": "adaptive",
            "enabled": "enabled",
            "disabled": "disabled",
        },
        effort_field="reasoning_effort",
        effort_range=("minimal", "high"),
    )

    def test_mode_auto_maps_to_adaptive(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "auto"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_mode_enabled_with_budget(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 2048}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 2048

    def test_mode_disabled_emits_thinking_disabled(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "disabled"


# ── OpenAI Responses shim ─────────────────────────────────────────────────


class TestOpenAIResponsesShim:
    """OpenAI Responses: disabled → omit, effort in reasoning object."""

    cap = DEFAULT_REASONING_CAPS["openai_responses"]

    def test_disabled_omits_all(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result == {}

    def test_effort_in_reasoning_object(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "medium"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result["reasoning"]["effort"] == "medium"

    def test_xhigh_clamped_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result["reasoning"]["effort"] == "high"

    def test_max_clamped_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result["reasoning"]["effort"] == "high"


# ── Anthropic shim ────────────────────────────────────────────────────────


class TestAnthropicShim:
    """Anthropic: disabled → thinking.type=disabled, effort clamped to [low, max]."""

    cap = DEFAULT_REASONING_CAPS["anthropic"]

    def test_disabled_emits_thinking_disabled(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["thinking"]["type"] == "disabled"

    def test_minimal_clamped_to_low(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "minimal"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["output_config"]["effort"] == "low"

    def test_xhigh_passes_through(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["output_config"]["effort"] == "xhigh"

    def test_max_passes_through(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["output_config"]["effort"] == "max"

    def test_high_passes_through(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["output_config"]["effort"] == "high"


# ── Google shim ───────────────────────────────────────────────────────────


class TestGoogleShim:
    """Google: disabled → thinkingBudget=0, effort → thinking_level."""

    cap = DEFAULT_REASONING_CAPS["google"]

    def test_disabled_emits_budget_zero(self):
        """Google disabled → thinking_budget=0."""
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_budget"] == 0

    def test_effort_emits_thinking_level(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high"}),
            self.cap,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_level"] == "high"

    def test_budget_still_works(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"budget_tokens": 8192}),
            self.cap,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_budget"] == 8192


# ── DeepSeek-style shim ──────────────────────────────────────────────────


class TestDeepSeekShim:
    """DeepSeek: thinking_modes with enabled/disabled only (no auto)."""

    cap = ReasoningCapability(
        thinking_modes={"enabled": "enabled", "disabled": "disabled"},
        effort_field="reasoning_effort",
        effort_range=("low", "max"),
    )

    def test_disabled_emits_thinking_disabled(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "disabled"

    def test_auto_not_supported_no_thinking(self):
        """DeepSeek has no 'auto' in thinking_modes → no thinking block."""
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "auto"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert "thinking" not in result

    def test_enabled_produces_thinking(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 4096}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 4096


# ── Summary / visibility cross-format ─────────────────────────────────────


class TestSummaryIncludeThoughtsCrossFormat:
    def test_summary_forwarded_to_google(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "summary": "auto"}),
            DEFAULT_REASONING_CAPS["google"],
            converter_type="google",
        )
        assert result["thinking_config"]["include_thoughts"] is True

    def test_summary_forwarded_to_responses(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "summary": "detailed"}),
            DEFAULT_REASONING_CAPS["openai_responses"],
            converter_type="openai_responses",
        )
        assert result["reasoning"]["summary"] == "detailed"

    def test_summary_forwarded_to_openai_chat(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "summary": "detailed"}),
            DEFAULT_REASONING_CAPS["openai_chat"],
            converter_type="openai_chat",
        )
        assert result["reasoning"]["summary"] == "detailed"

    def test_summary_to_anthropic_display_summarized(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "summary": "concise"}),
            DEFAULT_REASONING_CAPS["anthropic"],
            converter_type="anthropic",
        )
        assert result["thinking"]["display"] == "summarized"

    def test_summary_none_to_anthropic_display_omitted(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "summary": "none"}),
            DEFAULT_REASONING_CAPS["anthropic"],
            converter_type="anthropic",
        )
        assert result["thinking"]["display"] == "omitted"

    def test_include_thoughts_true_to_openai_chat(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "include_thoughts": True}),
            DEFAULT_REASONING_CAPS["openai_chat"],
            converter_type="openai_chat",
        )
        assert result["reasoning"]["summary"] == "auto"

    def test_include_thoughts_true_to_anthropic(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "include_thoughts": True}),
            DEFAULT_REASONING_CAPS["anthropic"],
            converter_type="anthropic",
        )
        assert result["thinking"]["display"] == "summarized"

    def test_include_thoughts_false_to_anthropic(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "include_thoughts": False}),
            DEFAULT_REASONING_CAPS["anthropic"],
            converter_type="anthropic",
        )
        assert result["thinking"]["display"] == "omitted"

    def test_include_thoughts_true_to_responses_fallback(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "high", "include_thoughts": True}),
            DEFAULT_REASONING_CAPS["openai_responses"],
            converter_type="openai_responses",
        )
        assert result["reasoning"]["summary"] == "auto"

    def test_summary_none_to_google(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "medium", "summary": "none"}),
            DEFAULT_REASONING_CAPS["google"],
            converter_type="google",
        )
        assert result["thinking_config"]["include_thoughts"] is False


# ── Custom shim tests ─────────────────────────────────────────────────────


class TestCustomShim:
    def test_effort_range_clamps_above_ceiling(self):
        custom = ReasoningCapability(
            effort_field="reasoning_effort",
            effort_range=("minimal", "high"),
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            custom,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_effort_range_clamps_below_floor(self):
        custom = ReasoningCapability(
            effort_field="output_config.effort",
            effort_range=("low", "max"),
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "minimal"}),
            custom,
            converter_type="anthropic",
        )
        assert result["output_config"]["effort"] == "low"

    def test_effort_range_none_full_pass_through(self):
        custom = ReasoningCapability(effort_field="reasoning_effort")
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            custom,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "max"

    def test_thinking_modes_adaptive_maps_enabled_to_adaptive(self):
        """Shim only supports adaptive → enabled request maps to adaptive."""
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "adaptive",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            effort_range=("low", "max"),
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled", "budget_tokens": 4096}),
            custom,
            converter_type="anthropic",
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_thinking_modes_enabled_with_budget(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            effort_range=("low", "max"),
        )
        result = apply_reasoning_config(
            cast(
                ReasoningConfig,
                {"mode": "auto", "effort": "high", "budget_tokens": 4096},
            ),
            custom,
            converter_type="anthropic",
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_budget_ratio_derives_tokens(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            effort_range=("low", "max"),
            budget_ratio=0.8,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
            max_tokens=10000,
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8000

    def test_budget_ratio_clamps_to_max_minus_one(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            budget_ratio=1.0,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
            max_tokens=2000,
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 1999

    def test_budget_ratio_floor_1024(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            budget_ratio=0.3,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
            max_tokens=2000,
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 1024

    def test_budget_ratio_max_tokens_too_small_falls_back(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            budget_ratio=0.8,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
            max_tokens=1024,
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_budget_ratio_none_falls_back_to_adaptive(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
            max_tokens=10000,
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_budget_ratio_without_max_tokens_falls_back(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="output_config.effort",
            budget_ratio=0.8,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="anthropic",
        )
        assert result["thinking"]["type"] == "adaptive"

    def test_budget_ratio_openai_chat(self):
        custom = ReasoningCapability(
            thinking_modes={
                "auto": "adaptive",
                "enabled": "enabled",
                "disabled": "disabled",
            },
            effort_field="reasoning_effort",
            budget_ratio=0.8,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled"}),
            custom,
            converter_type="openai_chat",
            max_tokens=10000,
        )
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8000

    def test_haiku_effort_field_none_drops_effort_keeps_thinking(self):
        custom = ReasoningCapability(
            thinking_modes={"enabled": "enabled", "disabled": "disabled"},
            effort_field="none",
            budget_ratio=0.8,
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "enabled", "effort": "medium"}),
            custom,
            converter_type="anthropic",
            max_tokens=8192,
        )
        assert "output_config" not in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 6553
