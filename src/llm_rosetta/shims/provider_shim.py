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
# Grouped config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionConfig:
    """How to reach the upstream provider."""

    base_url: str | None = None
    api_key_env: str | None = None
    models_path: str | None = None
    model_id_field: str | None = None


@dataclass(frozen=True)
class ToolsConfig:
    """Tool handling behavior for this provider."""

    custom_tools: bool = False
    max_description_length: int | None = None
    search_mode: ToolSearchMode = "disabled"
    multimodal_result: bool | None = None


# Legacy flat kwarg names → (grouped field, sub-field)
_LEGACY_CONNECTION_FIELDS: dict[str, str] = {
    "default_base_url": "base_url",
    "default_api_key_env": "api_key_env",
    "models_path": "models_path",
    "model_id_field": "model_id_field",
}
_LEGACY_TOOLS_FIELDS: dict[str, str] = {
    "supports_custom_tools": "custom_tools",
    "max_tool_description_length": "max_description_length",
    "tool_search_mode": "search_mode",
    "multimodal_tool_result": "multimodal_result",
}


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
        logo: URL to the provider's logo image (SVG preferred).
        connection: Upstream connection config (URL, API key env,
            models endpoint, model ID field).
        tools: Tool handling config (custom tools, description length,
            search mode, multimodal tool results).
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
        hoist_system_messages: Whether to hoist late system messages
            for prompt cache prefix stability.  Default ``True``.
    """

    name: str
    base: str
    logo: str | None = None
    connection: ConnectionConfig = ConnectionConfig()
    tools: ToolsConfig = ToolsConfig()
    pre_ir_transforms: tuple[Transform, ...] = ()
    post_ir_transforms: tuple[Transform, ...] = ()
    ir_transforms: tuple[IRTransform, ...] = ()
    response_body_transforms: tuple[Transform, ...] = ()
    reasoning: ReasoningCapability | None = None
    model_reasoning: dict[str, ReasoningCapability] | None = None
    response_id_prefix: str = ""
    hoist_system_messages: bool = True

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Accept both new and legacy kwarg names.

        Legacy ``from_transforms`` maps to ``pre_ir_transforms``;
        ``to_transforms`` maps to ``post_ir_transforms``.
        Legacy flat connection/tool kwargs are merged into their
        respective grouped config objects.
        New names take precedence if both are provided.
        """
        # ── Transform renames ────────────────────────────────────────
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

        # ── Legacy flat connection kwargs → ConnectionConfig ─────────
        conn_legacy = {
            new: kwargs.pop(old)
            for old, new in _LEGACY_CONNECTION_FIELDS.items()
            if old in kwargs
        }
        if conn_legacy:
            if "connection" not in kwargs:
                kwargs["connection"] = ConnectionConfig(**conn_legacy)
            else:
                warnings.warn(
                    "ProviderShim: flat connection kwargs "
                    f"({', '.join(conn_legacy)}) ignored because "
                    "'connection' was also provided",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # ── Legacy flat tool kwargs → ToolsConfig ────────────────────
        tools_legacy = {
            new: kwargs.pop(old)
            for old, new in _LEGACY_TOOLS_FIELDS.items()
            if old in kwargs
        }
        if tools_legacy:
            if "tools" not in kwargs:
                kwargs["tools"] = ToolsConfig(**tools_legacy)
            else:
                warnings.warn(
                    "ProviderShim: flat tool kwargs "
                    f"({', '.join(tools_legacy)}) ignored because "
                    "'tools' was also provided",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # ── Apply defaults ───────────────────────────────────────────
        _FIELD_DEFAULTS: dict[str, Any] = {
            "logo": None,
            "connection": ConnectionConfig(),
            "tools": ToolsConfig(),
            "pre_ir_transforms": (),
            "post_ir_transforms": (),
            "ir_transforms": (),
            "response_body_transforms": (),
            "reasoning": None,
            "model_reasoning": None,
            "response_id_prefix": "",
            "hoist_system_messages": True,
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

    # ── Backward-compatible aliases (read-only) ─────────────────────

    @property
    def from_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``pre_ir_transforms`` (deprecated)."""
        return self.pre_ir_transforms

    @property
    def to_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``post_ir_transforms`` (deprecated)."""
        return self.post_ir_transforms

    # Legacy flat connection accessors
    @property
    def default_base_url(self) -> str | None:
        """Alias for ``connection.base_url`` (deprecated)."""
        return self.connection.base_url

    @property
    def default_api_key_env(self) -> str | None:
        """Alias for ``connection.api_key_env`` (deprecated)."""
        return self.connection.api_key_env

    @property
    def models_path(self) -> str | None:
        """Alias for ``connection.models_path`` (deprecated)."""
        return self.connection.models_path

    @property
    def model_id_field(self) -> str | None:
        """Alias for ``connection.model_id_field`` (deprecated)."""
        return self.connection.model_id_field

    # Legacy flat tool accessors
    @property
    def supports_custom_tools(self) -> bool:
        """Alias for ``tools.custom_tools`` (deprecated)."""
        return self.tools.custom_tools

    @property
    def max_tool_description_length(self) -> int | None:
        """Alias for ``tools.max_description_length`` (deprecated)."""
        return self.tools.max_description_length

    @property
    def tool_search_mode(self) -> ToolSearchMode:
        """Alias for ``tools.search_mode`` (deprecated)."""
        return self.tools.search_mode

    @property
    def multimodal_tool_result(self) -> bool | None:
        """Alias for ``tools.multimodal_result`` (deprecated)."""
        return self.tools.multimodal_result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SHIM_REGISTRY: dict[str, ProviderShim] = {}

# Base converter types — used by resolve_base() for pass-through detection
_BASE_TYPES: frozenset[str] = frozenset(
    {
        "openai_chat",
        "openai_responses",
        "open_responses",
        "anthropic",
        "google",
        "google_generate",
        "google_interactions",
        "decision",
    }
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
