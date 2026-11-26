"""Turning a shim name into a shim: the one seam between conversion and shims.

Conversion code (``ir``, ``converters``, ``pipeline``) accepts a provider's
:class:`~llm_rosetta.contracts.shim.ProviderShim` as data.  A caller may also
pass a shim *name*; reading that name against a registry is not conversion's
business, so it goes through a :class:`ShimResolver` that the shim layer
registers (:func:`set_shim_resolver`) and ``llm_rosetta/__init__`` wires.

With no resolver registered the library is a plain conversion library: a
:class:`ProviderShim` object always works, a format name resolves to itself,
no name is a shim, and asserting that a name is one
(:func:`lookup_shim`, or a shim name passed to a pipeline) fails with
:class:`ShimResolverUnavailable`, which says what to do.  This module imports nothing from ``llm_rosetta``, so every layer may use it.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llm_rosetta.contracts.shim import ProviderShim, ReasoningCapability


@runtime_checkable
class ShimResolver(Protocol):
    """What the shim layer offers: a shim by its exact name."""

    def get_shim(self, name: str) -> ProviderShim | None:
        """Return the shim registered under *name*, or ``None``."""
        ...


class ShimResolverUnavailable(LookupError):
    """A shim name was given but no shim layer is loaded to read it."""


_resolver: ShimResolver | None = None


def set_shim_resolver(resolver: ShimResolver | None) -> None:
    """Register the resolver that reads shim names; ``None`` removes it."""
    global _resolver
    _resolver = resolver


def get_shim_resolver() -> ShimResolver | None:
    """Return the registered resolver, or ``None`` when there is none."""
    return _resolver


_dir_loader: Callable[..., list[ProviderShim]] | None = None


def set_providers_dir_loader(
    loader: Callable[..., list[ProviderShim]] | None,
) -> None:
    """Register the function that loads a shim tree; ``None`` removes it."""
    global _dir_loader
    _dir_loader = loader


def get_providers_dir_loader() -> Callable[..., list[ProviderShim]]:
    """Return the registered shim-tree loader.

    Raises:
        ShimResolverUnavailable: No shim layer is loaded to read a tree.
    """
    if _dir_loader is None:
        raise ShimResolverUnavailable(
            "cannot load a shim tree: no shim layer is loaded. "
            "Import llm_rosetta so that its shim layer registers itself."
        )
    return _dir_loader


def lookup_shim(name: str) -> ProviderShim | None:
    """Return the shim called *name*.

    Raises:
        ShimResolverUnavailable: No resolver is registered.
    """
    if _resolver is None:
        raise ShimResolverUnavailable(
            f"cannot resolve the shim name {name!r}: no shim layer is loaded. "
            "Pass a ProviderShim object, or import llm_rosetta so that its "
            "shims register themselves."
        )
    return _resolver.get_shim(name)


def resolve_shim(shim: ProviderShim | str | None) -> ProviderShim | None:
    """Resolve a shim argument to a :class:`ProviderShim`.

    A shim object is returned as it is and ``None`` stays ``None``.  A name is
    read through the registered resolver; with none registered no name is a
    shim, so it resolves to ``None`` (a provider name such as
    ``chat_completions`` is looked up this way and is not an error).  Use
    :func:`lookup_shim` where the caller asserts that the name is a shim.
    """
    if shim is None or not isinstance(shim, str):
        return shim
    if _resolver is None:
        return None
    return _resolver.get_shim(shim)


def resolve_format(name: str) -> str:
    """Resolve a provider or shim *name* to its wire format.

    A registered shim name gives the shim's format.  Any other name, with or
    without a resolver, comes back unchanged: the caller decides what an
    unknown name means.
    """
    if _resolver is not None:
        shim = _resolver.get_shim(name)
        if shim is not None:
            return shim.format
    return name


def match_model_reasoning(
    shim: ProviderShim | str | None, model: str | None
) -> ReasoningCapability | None:
    """Return the reasoning override a shim declares for *model*, if any.

    ``model_reasoning`` keys are matched against the **upstream model ID**.  An
    exact key wins; otherwise the keys are tried in declaration order as
    shell-style patterns (:func:`fnmatch.fnmatchcase`), so a vendor can cover a
    model family with one entry (``deepseek*``) while an exact ID still
    overrides it.  ``*`` matches ``/`` too, so one pattern covers
    vendor-prefixed IDs such as ``deepseek/deepseek-v3``.

    Args:
        shim: A shim object or registered name.
        model: The upstream model ID.  ``None`` or empty matches nothing.

    Returns:
        The matching per-model capability, or ``None`` (the caller then falls
        back to the provider-level one).
    """
    resolved: Any = resolve_shim(shim)
    if resolved is None or not resolved.model_reasoning or not model:
        return None
    overrides = resolved.model_reasoning
    exact = overrides.get(model)
    if exact is not None:
        return exact
    for pattern, capability in overrides.items():
        if fnmatch.fnmatchcase(model, pattern):
            return capability
    return None


__all__ = [
    "ShimResolver",
    "ShimResolverUnavailable",
    "get_providers_dir_loader",
    "get_shim_resolver",
    "lookup_shim",
    "match_model_reasoning",
    "resolve_format",
    "resolve_shim",
    "set_providers_dir_loader",
    "set_shim_resolver",
]
