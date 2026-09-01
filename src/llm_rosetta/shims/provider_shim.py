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
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from ..types.ir.configs import IREffort, IRMode  # re-exported
from .transforms import IRTransform, Transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning capability config
# ---------------------------------------------------------------------------

# Provider-side types — strategy enums for shim behavior.

#: How outbound unsigned reasoning blocks are handled.
UnsignedBlocks = Literal["as_is", "preserve"]

#: Tool search capability mode.
ToolSearchMode = Literal["disabled", "native", "bridge"]

# Backward-compat aliases (deprecated — will be removed in a future version)
EffortLevel = IREffort
DisabledStrategy = Literal["omit", "thinking_disabled", "thinking_budget_zero"]
ThinkingType = Literal["enabled", "adaptive"]
UnsignedReasoningBlocks = UnsignedBlocks
EffortMap = dict[str, str]
EffortField = str


@dataclass(frozen=True)
class ReasoningCapability:
    """How a provider handles reasoning / thinking configuration.

    Each field group controls one dimension of reasoning behavior.
    IR-side values use the fixed vocabulary from ``types.ir.configs``;
    provider-side values are free strings specific to each upstream API.

    Reference: https://llm-rosetta.readthedocs.io/en/latest/api/reasoning/

    Naming convention:
    - ``_modes``: IR value → provider value mapping (dict)
    - ``_field``: provider-side field path (str)
    - ``_range``: IR-side constraint interval (tuple)
    """

    # ── Thinking toggle ──────────────────────────────────────────────
    # Maps IR mode → provider thinking type value.
    # None = provider does not support a thinking block.
    # Example: {"auto": "adaptive", "enabled": "enabled", "disabled": "disabled"}
    # IR modes not present in the map are silently dropped.
    thinking_modes: dict[str, str] | None = None

    # Default IR mode when the request has no explicit mode.
    # Must be a key in thinking_modes.
    thinking_default: IRMode | None = None

    # ── Effort ───────────────────────────────────────────────────────
    # Provider-side field path for the effort value.
    # "reasoning_effort"       → {reasoning_effort: v}
    # "reasoning.effort"       → {reasoning: {effort: v}}
    # "output_config.effort"   → {output_config: {effort: v}}
    # "thinking_level"         → {thinking_config: {thinking_level: v}}
    # "none"                   → provider does not accept effort
    effort_field: str = "reasoning_effort"

    # Supported IR effort range [floor, ceiling].
    # Values outside are clamped to the nearest boundary.
    # None = full IR ladder (minimal–max).
    effort_range: tuple[IREffort, IREffort] | None = None

    # ── Budget ───────────────────────────────────────────────────────
    # Derive budget_tokens as max(1024, int(max_tokens × ratio)),
    # clamped to max_tokens − 1.  None = no automatic derivation.
    budget_ratio: float | None = None

    # ── Visibility ───────────────────────────────────────────────────
    # Maps IR summary value → provider visibility value.
    # None = use converter default (hardcoded per API standard).
    # IR values not in the map → field is omitted from the request.
    # Example (Anthropic):  {"auto": "summarized", "none": "omitted"}
    # Example (OpenAI):     {"auto": "auto", "concise": "concise", "detailed": "detailed"}
    visibility_modes: dict[str, str] | None = None

    # ── Response handling ────────────────────────────────────────────
    # How to handle unsigned (non-redacted) reasoning blocks in responses.
    unsigned_blocks: UnsignedBlocks = "as_is"


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
        pre_ir_transforms: Body-level transforms applied BEFORE IR
            conversion (normalise provider dialect → standard).
            Aliased as ``from_transforms`` for backward compatibility.
        post_ir_transforms: Body-level transforms applied AFTER IR
            conversion (standard → provider dialect).
            Aliased as ``to_transforms`` for backward compatibility.
        reasoning: Reasoning capability config for this provider.
            When ``None``, the converter uses its built-in default.
        model_reasoning: Per-model reasoning overrides keyed by
            **upstream model ID** (post-alias).  Each entry inherits
            from the provider-level ``reasoning`` for unset fields.
        response_id_prefix: Provider-specific prefix for response IDs
            (e.g. ``"resp_"`` for OpenAI Responses, ``"chatcmpl-"`` for
            OpenAI Chat).  Default ``""`` means passthrough (no
            prefix stripping or adding).
        supports_custom_tools: Whether the provider's API natively
            accepts custom tool definitions (``{type: "custom"}``).
            Default ``False`` — only OpenAI's official API supports
            this.  When ``False``, custom tools are downgraded to
            function wrappers.
        multimodal_tool_result: Whether the provider supports multimodal
            content (images, files) in tool results natively.  ``None``
            (default) defers to the converter's class-level flag.
            ``True`` forces native multimodal pass-through; ``False``
            forces dual-encoding (text fallback + synthetic user message).
        tool_search_mode: How tool_search is handled.  ``"disabled"`` (default)
            drops tool_search items.  ``"native"`` passes through the provider's
            native protocol.  ``"bridge"`` emulates via BM25 search over request
            tool schemas.
    """

    name: str
    base: str
    default_base_url: str | None = None
    default_api_key_env: str | None = None
    logo: str | None = None
    model_id_field: str | None = None
    pre_ir_transforms: tuple[Transform, ...] = ()
    post_ir_transforms: tuple[Transform, ...] = ()
    ir_transforms: tuple[IRTransform, ...] = ()
    reasoning: ReasoningCapability | None = None
    model_reasoning: dict[str, ReasoningCapability] | None = None
    response_id_prefix: str = ""
    supports_custom_tools: bool = False
    hoist_system_messages: bool = True
    multimodal_tool_result: bool | None = None
    tool_search_mode: ToolSearchMode = "disabled"

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Accept both new and legacy kwarg names.

        Legacy ``from_transforms`` maps to ``pre_ir_transforms``;
        ``to_transforms`` maps to ``post_ir_transforms``.
        New names take precedence if both are provided.
        """
        # Map legacy names → new names (new names take precedence)
        if "from_transforms" in kwargs:
            warnings.warn(
                "ProviderShim(from_transforms=...) is deprecated, "
                "use pre_ir_transforms instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if "pre_ir_transforms" not in kwargs:
                kwargs["pre_ir_transforms"] = kwargs.pop("from_transforms")
            else:
                kwargs.pop("from_transforms")
        if "to_transforms" in kwargs:
            warnings.warn(
                "ProviderShim(to_transforms=...) is deprecated, "
                "use post_ir_transforms instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if "post_ir_transforms" not in kwargs:
                kwargs["post_ir_transforms"] = kwargs.pop("to_transforms")
            else:
                kwargs.pop("to_transforms")

        # Apply defaults for fields not in kwargs.
        # Keep in sync with dataclass field defaults above.
        _FIELD_DEFAULTS = {
            "default_base_url": None,
            "default_api_key_env": None,
            "logo": None,
            "model_id_field": None,
            "pre_ir_transforms": (),
            "post_ir_transforms": (),
            "ir_transforms": (),
            "reasoning": None,
            "model_reasoning": None,
            "response_id_prefix": "",
            "supports_custom_tools": False,
            "hoist_system_messages": True,
            "multimodal_tool_result": None,
            "tool_search_mode": "disabled",
        }
        _VALID_FIELDS = {"name", "base"} | _FIELD_DEFAULTS.keys()
        for k, v in _FIELD_DEFAULTS.items():
            kwargs.setdefault(k, v)

        # Reject unknown kwargs (match frozen dataclass behavior)
        unknown = set(kwargs) - _VALID_FIELDS
        if unknown:
            raise TypeError(
                f"ProviderShim.__init__() got unexpected keyword argument(s): "
                f"{', '.join(sorted(unknown))}"
            )

        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    # Backward-compatible aliases (read-only)
    @property
    def from_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``pre_ir_transforms`` (deprecated)."""
        return self.pre_ir_transforms

    @property
    def to_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``post_ir_transforms`` (deprecated)."""
        return self.post_ir_transforms


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
    # Also clear the converter cache since shim resolution may have
    # cached converters for names that are now unregistered.
    from llm_rosetta.auto_detect import _converter_cache

    _converter_cache.clear()

    # Clear convention-based model list transforms.
    from llm_rosetta.shims.providers import _model_list_transforms

    _model_list_transforms.clear()
