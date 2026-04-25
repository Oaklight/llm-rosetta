"""Provider and model shim definitions with a global registry.

A **ProviderShim** is a lightweight identity card that declares which API
standard (converter) a provider uses, along with connection defaults.

A **ModelShim** describes model-level capabilities and constraints.  It is
always nested within a ``ProviderShim`` — a model inherently belongs to a
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelShim:
    """Model-level capabilities and constraints.

    Nested within a :class:`ProviderShim` to express that every model
    belongs to exactly one provider.

    Attributes:
        pattern: Glob pattern to match model names (e.g. ``"o3-*"``).
        capabilities: Set of capability tags this model supports
            (e.g. ``{"reasoning", "tools", "vision"}``).
    """

    pattern: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    # hooks: reserved for #173 (declarative transform mechanism)


@dataclass(frozen=True)
class ProviderShim:
    """Provider identity card with optional nested model overrides.

    Attributes:
        name: Canonical provider identifier (e.g. ``"deepseek"``).
        base: API standard this provider follows.  Must be one of the
            converter type strings (``"openai_chat"``, ``"anthropic"``,
            ``"google"``, ``"openai_responses"``).
        default_base_url: Default upstream base URL.  Used by the gateway
            when the provider config does not specify ``base_url``.
        default_api_key_env: Default environment variable name for the
            API key (e.g. ``"DEEPSEEK_API_KEY"``).
        models: Tuple of :class:`ModelShim` entries that provide
            model-specific capability metadata.
    """

    name: str
    base: str
    default_base_url: str | None = None
    default_api_key_env: str | None = None
    models: tuple[ModelShim, ...] = ()
    # hooks: reserved for #173

    def get_model_shim(self, model: str) -> ModelShim | None:
        """Return the first :class:`ModelShim` whose pattern matches *model*.

        Uses :func:`fnmatch.fnmatch` for glob-style matching.  Returns
        ``None`` if no nested model shim matches.
        """
        for m in self.models:
            if fnmatch(model, m.pattern):
                return m
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SHIM_REGISTRY: dict[str, ProviderShim] = {}

# Base converter types — used by resolve_base() for pass-through detection
_BASE_TYPES: frozenset[str] = frozenset(
    {"openai_chat", "openai_responses", "open_responses", "anthropic", "google"}
)


def register_shim(shim: ProviderShim) -> None:
    """Register (or replace) a :class:`ProviderShim` in the global registry."""
    _SHIM_REGISTRY[shim.name] = shim


def unregister_shim(name: str) -> ProviderShim | None:
    """Remove and return a shim by name.  Returns ``None`` if not found."""
    return _SHIM_REGISTRY.pop(name, None)


def get_shim(name: str) -> ProviderShim | None:
    """Look up a registered :class:`ProviderShim` by *name*."""
    return _SHIM_REGISTRY.get(name)


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
