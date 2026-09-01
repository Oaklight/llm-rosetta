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
- Effort values pass through if they are part of the canonical set
  ``{minimal, low, medium, high, xhigh, max}``.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

from llm_rosetta.shims.provider_shim import ReasoningCapability
from llm_rosetta.types.ir.reasoning import ReasoningConfig

# ── Default reasoning capability configs per base converter type ──────────
# Used as fallback when no shim-level config is present.

_DEFAULT_OPENAI_CHAT = ReasoningCapability(
    effort_field="reasoning_effort",
    effort_range=("minimal", "high"),
)

_DEFAULT_OPENAI_RESPONSES = ReasoningCapability(
    effort_field="reasoning.effort",
    effort_range=("minimal", "high"),
)

_DEFAULT_ANTHROPIC = ReasoningCapability(
    thinking_modes={"auto": "adaptive", "enabled": "enabled", "disabled": "disabled"},
    effort_field="output_config.effort",
    effort_range=("low", "max"),
)

_DEFAULT_GOOGLE = ReasoningCapability(
    effort_field="thinking_level",
    effort_range=("minimal", "high"),
)

DEFAULT_REASONING_CAPS: dict[str, ReasoningCapability] = {
    "openai_chat": _DEFAULT_OPENAI_CHAT,
    "openai_responses": _DEFAULT_OPENAI_RESPONSES,
    "anthropic": _DEFAULT_ANTHROPIC,
    "google": _DEFAULT_GOOGLE,
}


# ── IR effort ladder ──────────────────────────────────────────────────────

_EFFORT_LADDER: list[str] = ["minimal", "low", "medium", "high", "xhigh", "max"]
_EFFORT_RANK: dict[str, int] = {v: i for i, v in enumerate(_EFFORT_LADDER)}


# ── Input normalisation ────────────────────────────────────────────────────


def normalize_reasoning_input(
    ir_reasoning: ReasoningConfig,
) -> ReasoningConfig:
    """Normalise external effort values into the canonical IR ladder.

    - ``none`` → ``mode: disabled``, effort removed.

    Returns a **new** dict; the original is not mutated.
    """
    result: dict[str, Any] = {**ir_reasoning}
    effort = result.get("effort")

    if effort == "none":
        result["mode"] = "disabled"
        del result["effort"]

    return cast(ReasoningConfig, result)


# ── Main helper ────────────────────────────────────────────────────────────


def apply_reasoning_config(
    ir_reasoning: ReasoningConfig,
    cap: ReasoningCapability,
    *,
    converter_type: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Convert IR ``ReasoningConfig`` → provider parameters using *cap*.

    Handles:
    1. Input normalisation (``none`` → disabled).
    2. Disabled serialisation via ``cap.thinking_modes``.
    3. Effort clamping via ``cap.effort_range`` and placement via
       ``cap.effort_field``.
    4. Converter-specific structural pass-through (thinking blocks,
       budget, visibility).
    """
    ir = normalize_reasoning_input(ir_reasoning)

    mode = ir.get("mode")
    effort = ir.get("effort")
    budget_tokens = ir.get("budget_tokens")

    result: dict[str, Any] = {}

    # Disabled mode — most providers use thinking_modes["disabled"] via
    # _serialize_disabled, but Google disables via thinking_budget=0 (not
    # a type field), so it must go through _apply_google_extras instead.
    # Do NOT add thinking_modes to Google's shim — _serialize_disabled
    # would produce the wrong structure (thinking.type vs thinking_config).
    if mode == "disabled":
        if converter_type == "google":
            _apply_google_extras(ir, result, mode, budget_tokens, cap)
            return result
        return _serialize_disabled(cap)

    # Effort clamping + placement.
    if effort is not None:
        effort = _clamp_effort(effort, cap)
        effort_fields = _serialize_effort(cap.effort_field, effort)
        _deep_merge(result, effort_fields)

    # Converter-specific structural pass-through.
    if converter_type == "openai_chat":
        _apply_openai_chat_extras(ir, result, mode, budget_tokens, cap, max_tokens)
    elif converter_type == "openai_responses":
        _apply_openai_responses_extras(ir, result, mode, budget_tokens, cap)
    elif converter_type == "anthropic":
        _apply_anthropic_extras(
            ir, result, mode, effort, budget_tokens, cap, max_tokens
        )
    elif converter_type == "google":
        _apply_google_extras(ir, result, mode, budget_tokens, cap)

    return result


# ── Effort clamping ──────────────────────────────────────────────────────────


def _clamp_effort(effort: str, cap: ReasoningCapability) -> str:
    """Clamp *effort* to ``cap.effort_range`` boundaries."""
    if cap.effort_range is None:
        return effort
    floor, ceiling = cap.effort_range
    rank = _EFFORT_RANK.get(effort)
    lo = _EFFORT_RANK.get(floor)
    hi = _EFFORT_RANK.get(ceiling)
    if rank is None or lo is None or hi is None:
        return effort
    clamped = max(lo, min(hi, rank))
    return _EFFORT_LADDER[clamped]


# ── Disabled serialisation ─────────────────────────────────────────────────


def _serialize_disabled(cap: ReasoningCapability) -> dict[str, Any]:
    """Serialize disabled state using ``cap.thinking_modes``."""
    if cap.thinking_modes and "disabled" in cap.thinking_modes:
        return {"thinking": {"type": cap.thinking_modes["disabled"]}}
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


# ── Budget tokens default derivation ──────────────────────────────────────

_MIN_BUDGET_TOKENS = 1024


def _derive_budget_tokens(
    cap: ReasoningCapability,
    max_tokens: int | None,
) -> int | None:
    """Derive ``budget_tokens`` from ``cap.budget_ratio`` and *max_tokens*.

    Returns ``None`` when derivation is impossible (no ratio configured,
    no ``max_tokens``, or ``max_tokens`` too small to satisfy the minimum).
    """
    if cap.budget_ratio is None or max_tokens is None:
        return None
    if max_tokens <= _MIN_BUDGET_TOKENS:
        return None
    budget = max(_MIN_BUDGET_TOKENS, int(max_tokens * cap.budget_ratio))
    return min(budget, max_tokens - 1)


# ── Thinking block helpers ────────────────────────────────────────────────


def _resolve_thinking_type(
    mode: str | None,
    cap: ReasoningCapability,
) -> str | None:
    """Look up the provider thinking type for an IR *mode* via ``cap.thinking_modes``.

    Returns ``None`` when the provider doesn't support a thinking block
    or the specific mode is not in the map.
    """
    if not cap.thinking_modes:
        return None
    effective_mode = mode or cap.thinking_default
    if effective_mode is None:
        return None
    return cap.thinking_modes.get(effective_mode)


def _apply_visibility(
    ir: ReasoningConfig,
    result: dict[str, Any],
    cap: ReasoningCapability | None,
    *,
    target_key: str = "reasoning",
    target_subkey: str = "summary",
) -> None:
    """Apply visibility mapping from ``cap.visibility_modes`` or converter defaults.

    When ``visibility_modes`` is configured, uses the mapping.
    Otherwise falls back to the converter's hardcoded default behavior
    (writing to ``{target_key}.{target_subkey}``).
    """
    summary = ir.get("summary")
    include_thoughts = ir.get("include_thoughts")

    if cap and cap.visibility_modes:
        if summary and summary in cap.visibility_modes:
            result.setdefault(target_key, {})[target_subkey] = cap.visibility_modes[
                summary
            ]
        elif include_thoughts is True and "auto" in cap.visibility_modes:
            result.setdefault(target_key, {})[target_subkey] = cap.visibility_modes[
                "auto"
            ]
        return

    # Fallback: converter default behavior (summary pass-through)
    if summary in ("auto", "concise", "detailed"):
        result.setdefault(target_key, {})[target_subkey] = summary
    elif include_thoughts is True:
        result.setdefault(target_key, {})[target_subkey] = "auto"


# ── Converter-specific pass-through extras ─────────────────────────────────


def _apply_openai_chat_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
    cap: ReasoningCapability | None = None,
    max_tokens: int | None = None,
) -> None:
    """OpenAI Chat extras: thinking object for mode/budget_tokens (DeepSeek ext), summary."""
    if cap:
        thinking_type = _resolve_thinking_type(mode, cap)
        if thinking_type is not None:
            thinking: dict[str, Any] = {"type": thinking_type}
            if budget_tokens is not None:
                # "enabled" requires budget_tokens; "adaptive" ignores it.
                if thinking_type == "enabled" or thinking_type not in (
                    "adaptive",
                    "disabled",
                ):
                    thinking["budget_tokens"] = budget_tokens
            elif thinking_type == "enabled":
                derived = _derive_budget_tokens(cap, max_tokens)
                if derived is not None:
                    thinking["budget_tokens"] = derived
                elif cap.thinking_modes and "auto" in cap.thinking_modes:
                    thinking["type"] = cap.thinking_modes["auto"]
            result["thinking"] = thinking

    _apply_visibility(ir, result, cap)


def _apply_openai_responses_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
    cap: ReasoningCapability | None = None,
) -> None:
    """OpenAI Responses extras.

    OpenAI Responses API does **not** accept ``reasoning.type``.
    Reasoning is controlled via ``reasoning.effort`` + ``reasoning.summary``.
    """
    if budget_tokens is not None:
        warnings.warn(
            "OpenAI Responses API does not support reasoning budget_tokens, ignored",
            stacklevel=2,
        )

    _apply_visibility(ir, result, cap)


def _apply_anthropic_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    effort: str | None,
    budget_tokens: int | None,
    cap: ReasoningCapability | None = None,
    max_tokens: int | None = None,
) -> None:
    """Anthropic extras: thinking object with type/budget_tokens."""
    thinking_type = _resolve_thinking_type(mode, cap) if cap else None

    if thinking_type is None and cap and cap.thinking_modes:
        if effort is not None and "auto" in cap.thinking_modes:
            thinking_type = cap.thinking_modes["auto"]
        elif budget_tokens is not None and "enabled" in cap.thinking_modes:
            thinking_type = cap.thinking_modes["enabled"]

    if thinking_type is not None:
        result["thinking"] = _build_anthropic_thinking(
            thinking_type, budget_tokens, cap, max_tokens
        )

    if "thinking" in result:
        _apply_anthropic_visibility(ir, result, cap)


def _build_anthropic_thinking(
    thinking_type: str,
    budget_tokens: int | None,
    cap: ReasoningCapability | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the Anthropic ``thinking`` dict for a given type."""
    if thinking_type != "enabled":
        obj: dict[str, Any] = {"type": thinking_type}
        if budget_tokens is not None:
            obj["budget_tokens"] = budget_tokens
        return obj

    # "enabled" requires budget_tokens
    if budget_tokens is not None:
        return {"type": thinking_type, "budget_tokens": budget_tokens}
    derived = _derive_budget_tokens(cap, max_tokens) if cap else None
    if derived is not None:
        return {"type": thinking_type, "budget_tokens": derived}
    fallback = (
        cap.thinking_modes.get("auto") if cap and cap.thinking_modes else "adaptive"
    )
    if fallback:
        warnings.warn(
            f"Anthropic 'enabled' thinking requires budget_tokens, "
            f"falling back to '{fallback}'",
            stacklevel=2,
        )
    return {"type": fallback or thinking_type}


def _apply_anthropic_visibility(
    ir: ReasoningConfig,
    result: dict[str, Any],
    cap: ReasoningCapability | None,
) -> None:
    """Map IR summary → Anthropic ``thinking.display``."""
    summary = ir.get("summary")
    include_thoughts = ir.get("include_thoughts")

    if cap and cap.visibility_modes:
        if summary and summary in cap.visibility_modes:
            result["thinking"]["display"] = cap.visibility_modes[summary]
        elif include_thoughts is False and "none" in cap.visibility_modes:
            result["thinking"]["display"] = cap.visibility_modes["none"]
        elif (
            include_thoughts is True or summary in ("auto", "concise", "detailed")
        ) and "auto" in cap.visibility_modes:
            result["thinking"]["display"] = cap.visibility_modes["auto"]
        return

    if summary == "none" or include_thoughts is False:
        result["thinking"]["display"] = "omitted"
    elif summary in ("auto", "concise", "detailed") or include_thoughts is True:
        result["thinking"]["display"] = "summarized"


def _apply_google_extras(
    ir: ReasoningConfig,
    result: dict[str, Any],
    mode: str | None,
    budget_tokens: int | None,
    cap: ReasoningCapability | None = None,
) -> None:
    """Google extras: thinking_config with thinking_budget and include_thoughts."""
    thinking_config = result.get("thinking_config", {})

    if mode == "disabled":
        thinking_config["thinking_budget"] = 0
    elif (
        mode == "auto"
        and budget_tokens is None
        and "thinking_level" not in thinking_config
    ):
        thinking_config["thinking_budget"] = -1

    if budget_tokens is not None:
        thinking_config["thinking_budget"] = budget_tokens

    # Google visibility: include_thoughts boolean
    summary = ir.get("summary")
    if summary in ("auto", "concise", "detailed") or ir.get("include_thoughts") is True:
        thinking_config["include_thoughts"] = True
    elif summary == "none" or ir.get("include_thoughts") is False:
        thinking_config["include_thoughts"] = False

    if thinking_config:
        result["thinking_config"] = thinking_config
