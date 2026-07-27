"""Tests for the shared thinking_type override helper in reasoning.py."""

from __future__ import annotations

import warnings

import pytest

from llm_rosetta.converters.base.helpers.reasoning import (
    _apply_thinking_type_override,
    apply_reasoning_config,
)
from llm_rosetta.shims.provider_shim import (
    EffortField,
    EffortMap,
    ReasoningCapability,
    ThinkingType,
)


def _cap(
    *,
    effort_field: EffortField = "none",
    effort_map: EffortMap | None = None,
    thinking_type: ThinkingType | None = None,
    budget_tokens_default_ratio: float | None = None,
) -> ReasoningCapability:
    return ReasoningCapability(
        effort_field=effort_field,
        effort_map=effort_map or {},
        thinking_type=thinking_type,
        budget_tokens_default_ratio=budget_tokens_default_ratio,
    )


class TestApplyThinkingTypeOverride:
    def test_noop_when_cap_missing(self):
        result = {"thinking": {"type": "adaptive"}}
        _apply_thinking_type_override(result, None, None)
        assert result == {"thinking": {"type": "adaptive"}}

    def test_noop_when_thinking_type_none(self):
        result = {"thinking": {"type": "adaptive"}}
        _apply_thinking_type_override(result, _cap(thinking_type=None), None)
        assert result == {"thinking": {"type": "adaptive"}}

    def test_noop_when_no_thinking_key(self):
        result: dict = {}
        _apply_thinking_type_override(result, _cap(thinking_type="adaptive"), None)
        assert result == {}

    def test_switches_type_and_strips_budget_when_moving_to_adaptive(self):
        result = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        _apply_thinking_type_override(result, _cap(thinking_type="adaptive"), None)
        assert result == {"thinking": {"type": "adaptive"}}

    def test_leaves_type_unchanged_when_already_target(self):
        result = {"thinking": {"type": "adaptive"}}
        _apply_thinking_type_override(result, _cap(thinking_type="adaptive"), None)
        assert result == {"thinking": {"type": "adaptive"}}

    def test_enabled_target_derives_budget_from_ratio(self):
        result = {"thinking": {"type": "adaptive"}}
        cap = _cap(thinking_type="enabled", budget_tokens_default_ratio=0.5)
        _apply_thinking_type_override(result, cap, max_tokens=8192)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 4096

    def test_enabled_target_falls_back_to_adaptive_without_ratio(self):
        result = {"thinking": {"type": "adaptive"}}
        cap = _cap(thinking_type="enabled")
        _apply_thinking_type_override(result, cap, max_tokens=8192)
        assert result["thinking"] == {"type": "adaptive"}

    def test_enabled_target_falls_back_when_max_tokens_too_small(self):
        result = {"thinking": {"type": "adaptive"}}
        cap = _cap(thinking_type="enabled", budget_tokens_default_ratio=0.5)
        # max_tokens <= 1024 → derivation returns None
        _apply_thinking_type_override(result, cap, max_tokens=512)
        assert result["thinking"] == {"type": "adaptive"}

    def test_enabled_target_keeps_existing_budget(self):
        result = {"thinking": {"type": "enabled", "budget_tokens": 999}}
        cap = _cap(thinking_type="enabled", budget_tokens_default_ratio=0.5)
        _apply_thinking_type_override(result, cap, max_tokens=8192)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 999}


class TestOpenaiChatExtrasEndToEnd:
    """Guard-rails: apply_reasoning_config for openai_chat still delegates."""

    def test_shim_forces_adaptive_from_enabled(self):
        cap = _cap(
            effort_field="reasoning_effort",
            effort_map={"low": "low"},
            thinking_type="adaptive",
        )
        out = apply_reasoning_config(
            {"mode": "enabled", "budget_tokens": 2048},
            cap,
            converter_type="openai_chat",
        )
        assert out["thinking"] == {"type": "adaptive"}

    def test_shim_forces_enabled_derives_budget(self):
        cap = _cap(
            thinking_type="enabled",
            budget_tokens_default_ratio=0.5,
        )
        out = apply_reasoning_config(
            {"mode": "auto"},
            cap,
            converter_type="openai_chat",
            max_tokens=8192,
        )
        assert out["thinking"]["type"] == "enabled"
        assert out["thinking"]["budget_tokens"] == 4096


class TestAnthropicExtrasEndToEnd:
    def test_enabled_without_budget_warns_and_falls_back(self):
        cap = _cap(effort_field="output_config.effort")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = apply_reasoning_config(
                {"mode": "enabled"},
                cap,
                converter_type="anthropic",
            )
        assert out["thinking"] == {"type": "adaptive"}
        assert any("budget_tokens" in str(w.message) for w in caught)

    def test_enabled_derives_budget_from_ratio(self):
        cap = _cap(
            effort_field="output_config.effort",
            budget_tokens_default_ratio=0.5,
        )
        out = apply_reasoning_config(
            {"mode": "enabled"},
            cap,
            converter_type="anthropic",
            max_tokens=8192,
        )
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_shim_thinking_type_override_after_effort(self):
        # When mode=enabled with budget, then shim forces adaptive: budget stripped.
        cap = _cap(
            effort_field="output_config.effort",
            effort_map={"low": "low"},
            thinking_type="adaptive",
        )
        out = apply_reasoning_config(
            {"mode": "enabled", "budget_tokens": 2048},
            cap,
            converter_type="anthropic",
        )
        assert out["thinking"] == {"type": "adaptive"}

    def test_shim_forces_enabled_needs_budget_or_falls_back(self):
        cap = _cap(
            effort_field="output_config.effort",
            effort_map={"low": "low"},
            thinking_type="enabled",
        )
        out = apply_reasoning_config(
            {"effort": "low"},
            cap,
            converter_type="anthropic",
        )
        # No ratio → falls back to adaptive
        assert out["thinking"]["type"] == "adaptive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
