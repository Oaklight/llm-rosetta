"""Tests for shim-driven reasoning helpers — covers #244 scenarios.

Test matrix:
- Input normalisation: none → disabled, effort values pass through
- OpenAI (Chat+Responses): disabled → omit, xhigh/max capped to high
- Anthropic: disabled → thinking_disabled, minimal → low, xhigh/max pass through
- Google: disabled → thinkingBudget=0, effort skipped (no thinkingLevel)
- DeepSeek/Volcengine-style: disabled → thinking_disabled
- Custom shim override
"""

from __future__ import annotations

from typing import Any, cast


from llm_rosetta.converters.reasoning_helpers import (
    DEFAULT_REASONING_CAPS,
    apply_reasoning_config,
    normalize_reasoning_input,
)
from llm_rosetta.shims.provider_shim import ReasoningCapability
from llm_rosetta.types.ir.configs import ReasoningConfig


# ── Input normalisation ────────────────────────────────────────────────────


class TestNormalizeReasoningInput:
    """Test the P→IR normalisation step."""

    def test_none_becomes_disabled(self):
        """effort='none' → mode='disabled', no effort key."""
        result = normalize_reasoning_input(cast(ReasoningConfig, {"effort": "none"}))
        assert result["mode"] == "disabled"
        assert "effort" not in result

    def test_effort_values_pass_through(self):
        """Standard IR effort values pass through unchanged."""
        for level in ("minimal", "low", "medium", "high", "xhigh", "max"):
            result = normalize_reasoning_input(cast(ReasoningConfig, {"effort": level}))
            assert result["effort"] == level

    def test_none_preserves_other_fields(self):
        """none → disabled preserves budget_tokens."""
        result = normalize_reasoning_input(
            cast(ReasoningConfig, {"effort": "none", "budget_tokens": 4096})
        )
        assert result["mode"] == "disabled"
        assert result["budget_tokens"] == 4096
        assert "effort" not in result

    def test_empty_passes_through(self):
        """Empty config → empty config."""
        result = normalize_reasoning_input(cast(ReasoningConfig, {}))
        assert result == {}

    def test_does_not_mutate_original(self):
        """Input dict is not mutated."""
        original: dict[str, Any] = {"effort": "xhigh"}
        normalize_reasoning_input(cast(ReasoningConfig, original))
        assert original["effort"] == "xhigh"


# ── OpenAI Chat shim ──────────────────────────────────────────────────────


class TestOpenAIChatShim:
    """OpenAI Chat: disabled → omit, xhigh/max → high."""

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

    def test_effort_xhigh_maps_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_effort_max_maps_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"


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

    def test_xhigh_maps_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result["reasoning"]["effort"] == "high"

    def test_max_maps_to_high(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            self.cap,
            converter_type="openai_responses",
        )
        assert result["reasoning"]["effort"] == "high"


# ── Anthropic shim ────────────────────────────────────────────────────────


class TestAnthropicShim:
    """Anthropic: disabled → thinking_disabled, xhigh/max pass through."""

    cap = DEFAULT_REASONING_CAPS["anthropic"]

    def test_disabled_emits_thinking_disabled(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="anthropic",
        )
        assert result["thinking"]["type"] == "disabled"

    def test_minimal_maps_to_low(self):
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
    """Google: disabled → thinkingBudget=0, effort skipped."""

    cap = DEFAULT_REASONING_CAPS["google"]

    def test_disabled_emits_budget_zero(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_budget"] == 0

    def test_effort_skipped(self):
        """Google doesn't support thinkingLevel, effort is dropped."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = apply_reasoning_config(
                cast(ReasoningConfig, {"effort": "high"}),
                self.cap,
                converter_type="google",
            )
        # effort_field is "none" and effort_map is empty → nothing emitted
        assert result == {}

    def test_budget_still_works(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"budget_tokens": 8192}),
            self.cap,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_budget"] == 8192


# ── DeepSeek/Volcengine-style shim ────────────────────────────────────────


class TestDeepSeekShim:
    """DeepSeek: disabled → thinking_disabled (shim from YAML)."""

    cap = ReasoningCapability(
        disabled="thinking_disabled",
        effort_field="none",
        effort_map={},
    )

    def test_disabled_emits_thinking_disabled(self):
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            self.cap,
            converter_type="openai_chat",
        )
        assert result["thinking"]["type"] == "disabled"


# ── Custom shim override ──────────────────────────────────────────────────


class TestCustomShim:
    """Verify that custom ReasoningCapability overrides work."""

    def test_custom_effort_map(self):
        custom = ReasoningCapability(
            disabled="omit",
            effort_field="reasoning_effort",
            effort_map={
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "medium",  # unusual but valid
                "max": "low",
            },
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "xhigh"}),
            custom,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "medium"

    def test_custom_max_effort_caps_before_mapping(self):
        custom = ReasoningCapability(
            disabled="omit",
            effort_field="reasoning_effort",
            max_effort="high",
            effort_map={
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "max",
            },
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"effort": "max"}),
            custom,
            converter_type="openai_chat",
        )
        assert result["reasoning_effort"] == "high"

    def test_custom_thinking_budget_zero_disabled(self):
        custom = ReasoningCapability(
            disabled="thinking_budget_zero",
            effort_field="none",
            effort_map={},
        )
        result = apply_reasoning_config(
            cast(ReasoningConfig, {"mode": "disabled"}),
            custom,
            converter_type="google",
        )
        assert result["thinking_config"]["thinking_budget"] == 0
