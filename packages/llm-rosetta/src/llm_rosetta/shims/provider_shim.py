"""The shim registry, plus the 0.13.0 home of the shim data classes.

The global registry (``_SHIM_REGISTRY``) is a plain dict populated at
import time by ``shims/__init__.py``.  Registration functions
(``register_shim``, ``load_providers_from_dir``) write to it; query
functions (``get_shim``, ``list_shims``) read from it.  A host reaches these
functions through :mod:`llm_rosetta.api`.

The shim data classes (:class:`ProviderShim`, :class:`ConnectionConfig`,
:class:`ToolsConfig`, :class:`ReasoningCapability`, :class:`VendorManifest`,
:class:`QuotaDeclaration`, :class:`BatchDeclaration`,
:class:`SoftErrorPattern` and the ``TYPE_PARAM_KEYS`` / literal types) live in
:mod:`llm_rosetta.contracts.shim`.  This module re-exports them because
0.13.0 shipped them here; importing them from here is deprecated and the
re-exports will be removed in 0.15.0.
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta import retired_names
from llm_rosetta.contracts.shim import (  # deprecated re-exports, removed in 0.15.0
    TYPE_PARAM_KEYS,
    BatchDeclaration,
    ConnectionConfig,
    ProviderShim,
    QuotaDeclaration,
    ReasoningCapability,
    ReplayMissing,
    SoftErrorPattern,
    ToolSearchMode,
    ToolsConfig,
    UnsignedBlocks,
    VendorManifest,
)
from llm_rosetta.ir.primitives.body import Transform  # re-exported (0.13.0)
from llm_rosetta.ir.primitives.ir import IRTransform  # re-exported (0.13.0)
from llm_rosetta.model_vocabulary import (
    FORMATS_BY_TYPE,
    MODEL_TYPES,
)

from ..types.ir.configs import IREffort, IRMode  # re-exported (0.13.0)

logger = logging.getLogger(__name__)

__all__ = [
    # Registry
    "RegistrySnapshot",
    "get_shim",
    "list_shims",
    "match_model_reasoning",
    "register_shim",
    "resolve_format",
    "resolve_model_type",
    "resolve_shim",
    "restore_registry",
    "snapshot_registry",
    "staging_registry",
    "unregister_shim",
    # Deprecated re-exports of llm_rosetta.contracts.shim (removed in 0.15.0)
    "TYPE_PARAM_KEYS",
    "BatchDeclaration",
    "ConnectionConfig",
    "IREffort",
    "IRMode",
    "IRTransform",
    "ProviderShim",
    "QuotaDeclaration",
    "ReasoningCapability",
    "ReplayMissing",
    "SoftErrorPattern",
    "ToolSearchMode",
    "ToolsConfig",
    "Transform",
    "UnsignedBlocks",
    "VendorManifest",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SHIM_REGISTRY: dict[str, ProviderShim] = {}

# Per-thread view override for :func:`staging_registry`.  Set only on the
# thread that is building a replacement registry, so every other thread keeps
# reading ``_SHIM_REGISTRY`` and never observes a half-built one.
_registry_staging = threading.local()


def _current_registry() -> dict[str, ProviderShim]:
    """Return the registry *this thread* reads and writes.

    Normally the process-wide ``_SHIM_REGISTRY``; while the thread is inside
    :func:`staging_registry` it is that thread's private staging dict instead.
    """
    staged = getattr(_registry_staging, "registry", None)
    return staged if staged is not None else _SHIM_REGISTRY


@contextlib.contextmanager
def staging_registry() -> Iterator[dict[str, ProviderShim]]:
    """Build a replacement registry off to the side, then swap it in at once.

    Registration made inside the block lands in a private staging dict; other
    threads keep reading the live registry untouched, so they never see a
    partially-populated one.  On a clean exit the staging dict replaces the
    global in a **single reference assignment**; if the block raises, the live
    registry is left exactly as it was and the staging dict is discarded.

    Everything derived from the registry (converter instances, convention
    model-list transforms, plugin transform namespaces) is dropped on entry,
    so the staged registry starts from a clean slate and a caller does not need
    to know which caches exist.  If the block raises, the convention model-list
    transforms are put back, because they are registered while shims load and
    the live registry still needs them.
    """
    from .providers import _model_list_transforms

    kept_transforms = dict(_model_list_transforms)
    _clear_registry_caches()
    previous = getattr(_registry_staging, "registry", None)
    staging: dict[str, ProviderShim] = {}
    _registry_staging.registry = staging
    try:
        yield staging
    except BaseException:
        _registry_staging.registry = previous
        _model_list_transforms.clear()
        _model_list_transforms.update(kept_transforms)
        raise
    _registry_staging.registry = previous
    global _SHIM_REGISTRY
    _SHIM_REGISTRY = staging


# Chat formats and model-type names are not shims.  Non-chat format names are
# deliberately absent: ``openai``, ``cohere``, ``jina`` and ``voyage`` are also
# vendor names, so a bare name is only resolved within a model-type context.
_CHAT_FORMATS: frozenset[str] = frozenset(FORMATS_BY_TYPE["llm"])


def register_shim(shim: ProviderShim) -> None:
    """Register (or replace) a :class:`ProviderShim` in the global registry.

    The shim is keyed by its **exact** declared name.

    If a shim with the same name is already registered, it is silently
    replaced and an INFO-level log is emitted.  This allows plugin shims
    to override built-in defaults without raising errors.

    Inside :func:`staging_registry` the write goes to that thread's staging
    dict instead of the live registry.
    """
    registry = _current_registry()
    key = shim.name
    if key in registry:
        logger.info("Shim %r overridden (format: %s)", key, shim.format)
    registry[key] = shim


def unregister_shim(name: str) -> ProviderShim | None:
    """Remove and return a shim by name.  Returns ``None`` if not found."""
    return _current_registry().pop(name, None)


def get_shim(name: str) -> ProviderShim | None:
    """Look up a registered :class:`ProviderShim` by *name*.

    Only the exact registered name resolves.  A **bare chat format**
    (``chat_completions``, ``open_responses``, ``messages``, ``google_generate``,
    ``google_interactions``), a **model-type name** and a retired spelling
    (``"google"``, ``"openai_responses"``) are not shim names: each resolves to
    ``None``, so a format name can never silently inherit a vendor shim's
    connection defaults (#917).  The boundaries that accept names from outside
    reject a retired spelling with the retired-names message instead.
    """
    return _current_registry().get(name)


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


def match_model_reasoning(
    shim: ProviderShim | str | None,
    model: str | None,
) -> ReasoningCapability | None:
    """Return the reasoning override declared for *model*, if any.

    ``model_overrides`` keys are matched against the **upstream model ID**.
    An exact key wins; otherwise the keys are tried in declaration order as
    shell-style patterns (:func:`fnmatch.fnmatchcase`), so a vendor can cover
    a model family with one entry (``deepseek*``) while an exact ID still
    overrides it.  ``fnmatch``'s ``*`` matches ``/`` too, so the same
    pattern covers vendor-prefixed IDs such as ``deepseek/deepseek-v3``.

    Args:
        shim: A shim object or registered name.
        model: The upstream model ID.  ``None`` or empty matches nothing.

    Returns:
        The matching per-model capability, or ``None`` when the model has no
        override (the caller then falls back to the provider-level one).
    """
    resolved = resolve_shim(shim)
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


def list_shims() -> list[ProviderShim]:
    """Return all registered provider shims, in registration order.

    The registry holds canonical names only (see :func:`register_shim`), so
    each shim appears exactly once.
    """
    seen: set[int] = set()
    out: list[ProviderShim] = []
    for shim in _current_registry().values():
        if id(shim) in seen:
            continue
        seen.add(id(shim))
        out.append(shim)
    return out


def resolve_format(name: str) -> str:
    """Resolve a provider/shim *name* to its wire format.

    An exact registered shim name is tried first.  If *name* is a known chat
    format or a model-type name, it is returned unchanged.  If the name is not
    found in either, it is returned as-is (caller decides how to handle unknown
    names, including a retired spelling).
    """
    shim = _current_registry().get(name)
    if shim is not None:
        return shim.format
    return name


def resolve_model_type(name: str) -> str | None:
    """Resolve a provider/shim *name* to its capability category.

    A registered shim declares the category.  A bare chat format or a
    model-type name resolves without the registry.  A non-chat format name is
    ambiguous at this boundary (``openai`` is both a vendor and an embedding
    format), so it must come from a shim or from a caller that already knows the
    category; this function returns ``None`` for it.

    Returns:
        ``"llm"``, ``"embeddings"``, ``"rerank"`` or ``"decisions"``, or
        ``None`` when the name does not identify a category.

    Raises:
        ValueError: If *name* is a retired model-type spelling.
    """
    shim = _current_registry().get(name)
    if shim is not None:
        return shim.model_type
    retired_names.check_not_retired(name, retired_names.MODEL_TYPE)
    if name in MODEL_TYPES:
        return name
    if name in _CHAT_FORMATS:
        return "llm"
    return None


def _clear_registry_caches() -> None:
    """Drop everything derived from the registry.

    Converter instances, convention model-list transforms and plugin transform
    namespaces all cache per-shim state, so they must not outlive a rebuild.
    """
    # Also clear the converter cache since shim resolution may have
    # cached converters for names that are now unregistered.
    from llm_rosetta.auto_detect import _converter_cache

    _converter_cache.clear()

    # Clear convention-based model list transforms.
    from llm_rosetta.shims.providers import _model_list_transforms

    _model_list_transforms.clear()

    # Drop the synthetic packages registered for plugin shim transforms.
    from llm_rosetta.shims.providers import _clear_plugin_namespaces

    _clear_plugin_namespaces()


def _reset_registry() -> None:
    """Clear the registry and everything derived from it.  Intended for testing."""
    _SHIM_REGISTRY.clear()
    _clear_registry_caches()


@dataclass(frozen=True)
class RegistrySnapshot:
    """A restorable view of the process-wide shim registry.

    The registry is not the only state a test can disturb: the convention
    model-list transforms are filled while the shims load and cleared with
    them, so a snapshot carries both.  Converter instances are not carried —
    they are derived from a shim and are rebuilt on the next lookup.

    Attributes:
        registry: The shim mapping, as :func:`snapshot_registry` copied it.
        model_list_transforms: The convention transforms registered by the
            same load, keyed by shim name.
    """

    registry: dict[str, ProviderShim] = field(default_factory=dict)
    model_list_transforms: dict[str, Any] = field(default_factory=dict)


def snapshot_registry() -> RegistrySnapshot:
    """Return a restorable copy of the process-wide registry.

    The registry is process-wide and a test may replace it — a unit suite that
    calls ``rebuild_shims`` with its own fixtures, say.  Take a snapshot before
    such a test and hand it back to :func:`restore_registry` afterwards, so the
    next test sees the registry it expects and file order stops mattering
    (#1332).

    This reads the live registry rather than a thread's staging view: the pair
    exists to restore the state other threads see, and the restore always
    rebinds it.

    Returns:
        A snapshot whose mappings are copies; mutating them does not touch the
        live state.
    """
    from .providers import _model_list_transforms

    return RegistrySnapshot(
        registry=dict(_SHIM_REGISTRY),
        model_list_transforms=dict(_model_list_transforms),
    )


def restore_registry(snapshot: RegistrySnapshot) -> None:
    """Put the registry and its derived state back.

    Rebinds the global to a copy of the snapshot, matching
    :func:`staging_registry`, which also rebinds rather than mutating in place.
    The convention transforms are restored from the snapshot rather than
    cleared: they are registered while shims load, not rebuilt on demand, so
    clearing them would leave the built-ins' transforms missing for the rest
    of the process.

    Args:
        snapshot: A snapshot from :func:`snapshot_registry`.
    """
    global _SHIM_REGISTRY
    from llm_rosetta.auto_detect import _converter_cache

    from .providers import _model_list_transforms

    _SHIM_REGISTRY = dict(snapshot.registry)
    _model_list_transforms.clear()
    _model_list_transforms.update(snapshot.model_list_transforms)
    # Cached converters are derived from shims that may no longer exist.
    _converter_cache.clear()
