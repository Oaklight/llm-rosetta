"""IR reasoning vocabulary and configuration types.

This module defines the fixed IR-side value sets and configuration
structure for reasoning/thinking behavior.  Provider-side values are
free strings defined in each provider's shim config.

Reference: https://llm-rosetta.readthedocs.io/en/latest/api/ir-types/
"""

from typing import Literal, TypedDict

# ── IR reasoning vocabulary ────────────────────────────────────────────────

#: IR reasoning mode: auto (model decides), enabled (explicit), disabled (off).
IRMode = Literal["auto", "enabled", "disabled"]

#: IR effort ladder (ordered): minimal < low < medium < high < xhigh < max.
#: External "none" maps to mode: disabled, not an effort level.
IREffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]

#: IR thinking output visibility control.
IRVisibility = Literal["auto", "concise", "detailed", "none"]

# Backward-compat alias
ReasoningEffortLevel = IREffort


# ── IR reasoning config ───────────────────────────────────────────────────


class ReasoningConfig(TypedDict, total=False):
    """Reasoning/thinking configuration.

    Controls whether and how the model performs explicit reasoning.

    Provider mappings for ``mode``:
    - ``"auto"``: Model decides when/how much to think.
      Anthropic: ``thinking.type="adaptive"``,
      Google: ``thinking_budget=-1``
    - ``"enabled"``: Explicit thinking with budget control.
      Anthropic: ``thinking.type="enabled"`` + ``budget_tokens``,
      OpenAI Responses: ``reasoning.type="enabled"``
    - ``"disabled"``: No thinking.
      Anthropic: ``thinking.type="disabled"``,
      Google: ``thinking_budget=0``,
      OpenAI Responses: ``reasoning.type="disabled"``

    Provider mappings for ``effort``:
    - Anthropic: ``output_config.effort``
    - OpenAI Chat: ``reasoning_effort``
    - OpenAI Responses: ``reasoning.effort``
    - Google: ``thinking_config.thinking_level``

    Provider mappings for ``budget_tokens``:
    - Anthropic: ``thinking.budget_tokens``
    - Google: ``thinking_config.thinking_budget``
    """

    mode: IRMode
    effort: IREffort
    budget_tokens: int
    summary: str  # "auto", "concise", "detailed", "none"
    include_thoughts: bool  # Google/Gemini thought inclusion flag


__all__ = [
    "IRMode",
    "IREffort",
    "IRVisibility",
    "ReasoningEffortLevel",
    "ReasoningConfig",
]
