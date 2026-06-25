"""Provider shim definitions with a global registry.

A **ProviderShim** is a lightweight identity card that declares which API
standard (converter) a provider uses, along with connection defaults and
optional transforms to bridge schema differences.

The global registry (``_SHIM_REGISTRY``) is a plain dict populated at
import time by ``shims/__init__.py``.  Registration functions
(``register_shim``, ``load_providers_from_dir``) write to it; query
functions (``get_shim``, ``list_shims``) read from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from .transforms import IRTransform, Transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning capability config
# ---------------------------------------------------------------------------

#: How a provider handles "reasoning disabled".
DisabledStrategy = Literal["omit", "thinking_disabled", "thinking_budget_zero"]

#: Where the provider expects the effort value to be serialised.
EffortField = Literal[
    "reasoning_effort",  # OpenAI Chat top-level
    "reasoning.effort",  # OpenAI Responses nested
    "output_config.effort",  # Anthropic
    "thinking_level",  # Google thinking_config.thinking_level
    "none",  # Provider has no effort field
]

#: Normalised IR effort ladder level.
EffortLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]

#: How the provider expects ``thinking.type`` to be serialised.
ThinkingType = Literal["enabled", "adaptive"]

#: How outbound unsigned reasoning blocks are handled.
UnsignedReasoningBlocks = Literal["as_is", "preserve"]

#: Mapping from normalised IR effort levels to provider-specific values.
#: Any IR level absent from the map is unsupported and will be warned/skipped.
EffortMap = dict[str, str]  # e.g. {"minimal": "low", "max": "high"}


@dataclass(frozen=True)
class ReasoningCapability:
    """Declares how a provider handles reasoning effort and disabled state.

    Attributes:
        disabled: How to serialise ``mode: disabled``.
        effort_field: Where the provider expects the effort value.
        max_effort: Highest normalised effort this shim should emit.
        thinking_type: Force ``thinking.type`` to this value.
        unsigned_reasoning_blocks: Policy for outbound unsigned reasoning blocks.
        effort_map: Map from normalised IR effort to provider effort string.
        budget_tokens_default_ratio: When ``thinking_type`` is ``"enabled"``
            but no ``budget_tokens`` is provided, derive a default budget as
            ``max(1024, int(max_tokens * ratio))`` clamped to ``max_tokens - 1``.
            Anthropic requires ``budget_tokens >= 1024`` and ``< max_tokens``.
            When ``None``, the existing ``"adaptive"`` fallback is used instead.
    """

    disabled: DisabledStrategy = "omit"
    effort_field: EffortField = "reasoning_effort"
    max_effort: EffortLevel | None = None
    thinking_type: ThinkingType | None = None
    unsigned_reasoning_blocks: UnsignedReasoningBlocks = "as_is"
    effort_map: EffortMap = field(
        default_factory=lambda: {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
        }
    )
    budget_tokens_default_ratio: float | None = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderShim:
    """Provider identity card with optional transforms.

    Attributes:
        name: Canonical provider identifier (e.g. ``"deepseek"``).
        base: API standard this provider follows.  Must be one of the
            converter type strings (``"openai_chat"``, ``"anthropic"``,
            ``"google"``, ``"openai_responses"``).
        default_base_url: Default upstream base URL.  Used by the gateway
            when the provider config does not specify ``base_url``.
        default_api_key_env: Default environment variable name for the
            API key (e.g. ``"DEEPSEEK_API_KEY"``).
        logo: URL to the provider's logo image (SVG preferred).
        model_id_field: JSON field name to use as model identifier when
            fetching the upstream model list.  Defaults to ``"id"``
            when ``None``.  Useful for providers like Argo that place
            the actual model identifier in a non-standard field.
        from_transforms: Transforms applied when data comes FROM this
            provider (normalise dialect → standard).
        to_transforms: Transforms applied when data goes TO this
            provider (standard → dialect).
        reasoning: Reasoning capability config for this provider.
            When ``None``, the converter uses its built-in default.
        model_reasoning: Per-model reasoning overrides keyed by
            **upstream model ID** (post-alias).  Each entry inherits
            from the provider-level ``reasoning`` for unset fields.
    """

    name: str
    base: str
    default_base_url: str | None = None
    default_api_key_env: str | None = None
    logo: str | None = None
    model_id_field: str | None = None
    from_transforms: tuple[Transform, ...] = ()
    to_transforms: tuple[Transform, ...] = ()
    ir_transforms: tuple[IRTransform, ...] = ()
    reasoning: ReasoningCapability | None = None
    model_reasoning: dict[str, ReasoningCapability] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SHIM_REGISTRY: dict[str, ProviderShim] = {}

# Base converter types — used by resolve_base() for pass-through detection
_BASE_TYPES: frozenset[str] = frozenset(
    {"openai_chat", "openai_responses", "open_responses", "anthropic", "google"}
)


def register_shim(shim: ProviderShim) -> None:
    """Register (or replace) a :class:`ProviderShim` in the global registry.

    If a shim with the same name is already registered, it is silently
    replaced and an INFO-level log is emitted.  This allows plugin shims
    to override built-in defaults without raising errors.
    """
    if shim.name in _SHIM_REGISTRY:
        logger.info("Shim %r overridden (base: %s)", shim.name, shim.base)
    _SHIM_REGISTRY[shim.name] = shim


def unregister_shim(name: str) -> ProviderShim | None:
    """Remove and return a shim by name.  Returns ``None`` if not found."""
    return _SHIM_REGISTRY.pop(name, None)


def get_shim(name: str) -> ProviderShim | None:
    """Look up a registered :class:`ProviderShim` by *name*."""
    return _SHIM_REGISTRY.get(name)


def resolve_shim(shim: ProviderShim | str | None) -> ProviderShim | None:
    """Resolve a shim argument to a :class:`ProviderShim` instance.

    Accepts a :class:`ProviderShim` (returned as-is), a registered name
    (looked up via :func:`get_shim`), or ``None`` (returns ``None``).
    """
    if shim is None:
        return None
    if isinstance(shim, ProviderShim):
        return shim
    return get_shim(shim)


def list_shims() -> list[ProviderShim]:
    """Return all registered provider shims."""
    return list(_SHIM_REGISTRY.values())


def resolve_base(name: str) -> str:
    """Resolve a provider/shim *name* to its base converter type.

    If *name* is already a known base type (e.g. ``"openai_chat"``),
    it is returned unchanged.  Otherwise the shim registry is consulted.
    If the name is not found in either, it is returned as-is (caller
    decides how to handle unknown names).
    """
    if name in _BASE_TYPES:
        return name
    shim = _SHIM_REGISTRY.get(name)
    if shim is not None:
        return shim.base
    return name


def _reset_registry() -> None:
    """Clear the registry.  Intended for testing only."""
    _SHIM_REGISTRY.clear()
