"""Shim-driven reasoning configuration helpers.

This module provides the central logic for converting IR ``ReasoningConfig``
to provider-specific parameters.  Instead of each converter hardcoding
effort downgrades and disabled serialization, the helpers read a
:class:`~llm_rosetta.shims.provider_shim.ReasoningCapability` config from the
provider shim and apply the mappings generically.

Input normalisation
~~~~~~~~~~~~~~~~~~~
External input values are normalised to the IR effort ladder before they
reach the converters:

- ``none`` → ``mode: disabled`` (NOT ``effort: none``)
- ``xhigh``, ``max`` → ``effort: ultra``
- All other values pass through if they are part of the canonical set
  ``{minimal, low, medium, high, ultra}``.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

from ..shims.provider_shim import ReasoningCapability
from ..types.ir.configs import ReasoningConfig

# ── Default reasoning capability configs per base converter type ──────────
# Used as fallback when no shim-level config is present.

_DEFAULT_OPENAI_CHAT = ReasoningCapability(
    disabled="omit",
    effort_field="reasoning_effort",
    effort_map={
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "ultra": "high",
    },
)

_DEFAULT_OPENAI_RESPONSES = ReasoningCapability(
    disabled="omit",
    effort_field="reasoning.effort",
    effort_map={
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "ultra": "high",
    },
)

_DEFAULT_ANTHROPIC = ReasoningCapability(
    disabled="thinking_disabled",
    effort_field="output_config.effort",
    effort_map={
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "ultra": "xhigh",
    },
)

_DEFAULT_GOOGLE = ReasoningCapability(
    disabled="thinking_budget_zero",
    effort_field="none",
    effort_map={},
)

DEFAULT_REASONING_CAPS: dict[str, ReasoningCapability] = {
    "openai_chat": _DEFAULT_OPENAI_CHAT,
    "openai_responses": _DEFAULT_OPENAI_RESPONSES,
    "anthropic": _DEFAULT_ANTHROPIC,
    "google": _DEFAULT_GOOGLE,
}


# ── Input normalisation ────────────────────────────────────────────────────

# External values that map to ``ultra`` in the IR effort ladder.
_EFFORT_TO_ULTRA = frozenset({"xhigh", "max"})

# The canonical IR effort levels.
_IR_EFFORT_LEVELS = frozenset({"minimal", "low", "medium", "high", "ultra"})


def normalize_reasoning_input(
    ir_reasoning: ReasoningConfig,
) -> ReasoningConfig:
    """Normalise external effort values into the canonical IR ladder.

    - ``none`` → ``mode: disabled``, effort removed.
    - ``xhigh`` / ``max`` → ``effort: ultra``.

    Returns a **new** dict; the original is not mutated.
    """
    result: dict[str, Any] = {**ir_reasoning}
    effort = result.get("effort")

    if effort == "none":
        # ``none`` means disabled, not an effort level.
        result["mode"] = "disabled"
        del result["effort"]

    elif effort is not None and effort in _EFFORT_TO_ULTRA:
        result["effort"] = "ultra"

    return cast(ReasoningConfig, result)


# ── Main helper ────────────────────────────────────────────────────────────


def apply_reasoning_config(
    ir_reasoning: ReasoningConfig,
    cap: ReasoningCapability,
    *,
    converter_type: str | None = None,
) -> dict[str, Any]:
    """Convert IR ``ReasoningConfig`` → provider parameters using *cap*.

    This is the single function each converter's ``ir_reasoning_config_to_p``
    should delegate to.  It handles:

    1. **Input normalisation** (``none`` → disabled, ``xhigh``/``max`` → ``ultra``).
    2. **Disabled serialisation** according to ``cap.disabled``.
    3. **Effort mapping** via ``cap.effort_map`` and ``cap.effort_field``.
    4. **Pass-through** of ``mode`` and ``budget_tokens`` for converters
       that support thinking objects (Anthropic, Google, OpenAI Chat extensions).

    Args:
        ir_reasoning: Normalised IR reasoning config.
        cap: Provider's reasoning capability descriptor.
        converter_type: The base converter type string (for extra pass-through
            logic).

    Returns:
        Dict of provider-specific request fields to merge.
    """
    # 1. Normalise input.
    ir = normalize_reasoning_input(ir_reasoning)

    mode = ir.get("mode")
    effort = ir.get("effort")
    budget_tokens = ir.get("budget_tokens")

    result: dict[str, Any] = {}

    # 2. Disabled mode.
    if mode == "disabled":
        return _serialize_disabled(cap)

    # 3. Effort mapping.
    if effort is not None:
        provider_effort = cap.effort_map.get(effort)
        if provider_effort is None:
            warnings.warn(
                f"Effort level '{effort}' not in shim effort_map, skipping",
                stacklevel=2,
            )
        else:
            effort_fields = _serialize_effort(cap.effort_field, provider_effort)
            _deep_merge(result, effort_fields)

    # 4. Converter-specific structural pass-through.
    if converter_type == "openai_chat":
        _apply_openai_chat_extras(ir, result, mode, budget_tokens)
    elif converter_type == "openai_responses":
        _apply_openai_responses_extras(ir, result, mode, budget_tokens)
    elif converter_type == "anthropic":
        _apply_anthropic_extras(ir, result, mode, effort, budget_tokens)
    elif converter_type == "google":
        _apply_google_extras(ir, result, mode, budget_tokens)

    return result


# ── Disabled serialisation ─────────────────────────────────────────────────


def _serialize_disabled(cap: ReasoningCapability) -> dict[str, Any]:
    """Serialize disabled state according to the shim strategy."""
    if cap.disabled == "omit":
        return {}
    elif cap.disabled == "thinking_disabled":
        return {"thinking": {"type": "disabled"}}
    elif cap.disabled == "thinking_budget_zero":
        return {"thinking_config": {"thinking_budget": 0}}
    return {}


# ── Effort serialisation ──────────────────────────────────────────────────


def _serialize_effort(
    effort_field: str,
    provider_effort: str,
) -> dict[str, Any]:
    """Place *provider_effort* at the location indicated by *effort_field*.

    Supported field paths:
    - ``reasoning_effort`` → ``{"reasoning_effort": value}``
    - ``reasoning.effort`` → ``{"reasoning": {"effort": value}}``
    - ``output_config.effort`` → ``{"output_config": {"effort": value}}``
    - ``thinking_level`` → ``{"thinking_config": {"thinking_level": value}}``
    - ``none`` → ``{}``  (provider does not support effort)
    """
    if effort_field == "none":
        return {}
    if effort_field == "reasoning_effort":
        return {"reasoning_effort": provider_effort}
    if effort_field == "reasoning.effort":
        return {"reasoning": {"effort": provider_effort}}
    if effort_field == "output_config.effort":
        return {"output_config": {"effort": provider_effort}}
    if effort_field == "thinking_level":
        return {"thinking_config": {"thinking_level": provider_effort}}
    # Unknown field — fall back to flat key
    warnings.warn(
        f"Unknown effort_field '{effort_field}', using as flat key",
        stacklevel=3,
    )
    return {effort_field: provider_effort}


# ── Helpers ────────────────────────────────────────────────────────────────


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """One-level deep merge of *source* into *target* (mutates target)."""
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k].update(v)
        else:
            target[k] = v


# ── Converter-specific pass-through extras ─────────────────────────────────


def _apply_openai_chat_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
) -> None:
    """OpenAI Chat extras: thinking object for mode/budget_tokens (DeepSeek ext)."""
    thinking: dict[str, Any] = {}
    if mode:
        thinking["type"] = mode
    if budget_tokens is not None:
        thinking["budget_tokens"] = budget_tokens
    if thinking:
        result["thinking"] = thinking


def _apply_openai_responses_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
) -> None:
    """OpenAI Responses extras: reasoning.type when mode is set."""
    if mode:
        reasoning = result.setdefault("reasoning", {})
        # auto → enabled for Responses API
        reasoning["type"] = "enabled" if mode == "auto" else mode

    if budget_tokens is not None:
        warnings.warn(
            "OpenAI Responses API does not support reasoning budget_tokens, ignored",
            stacklevel=2,
        )


def _apply_anthropic_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    effort: str | None,
    budget_tokens: int | None,
) -> None:
    """Anthropic extras: thinking object with type/budget_tokens."""
    if mode == "enabled":
        if budget_tokens is not None:
            result["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        else:
            warnings.warn(
                "Anthropic 'enabled' thinking requires budget_tokens, "
                "falling back to 'adaptive'",
                stacklevel=2,
            )
            result["thinking"] = {"type": "adaptive"}
    elif mode == "auto" or effort is not None:
        thinking: dict[str, Any] = {"type": "adaptive"}
        if budget_tokens is not None:
            thinking["budget_tokens"] = budget_tokens
        result["thinking"] = thinking
    elif budget_tokens is not None:
        result["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}


def _apply_google_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
) -> None:
    """Google extras: thinking_config with thinking_budget."""
    thinking_config = result.get("thinking_config", {})

    if mode == "auto" and budget_tokens is None:
        thinking_config["thinking_budget"] = -1

    if budget_tokens is not None:
        thinking_config["thinking_budget"] = budget_tokens

    if thinking_config:
        result["thinking_config"] = thinking_config
