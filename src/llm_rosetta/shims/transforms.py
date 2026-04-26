"""Transform primitives for the provider/model shim layer.

A **Transform** is a pure data transformation: ``dict → dict``.  Transforms
bridge the gap between a real provider's API dialect and the "ideal" standard
that the corresponding base converter expects.

Transforms are NOT a replacement for converter functionality — they handle
provider/model-specific field-level quirks (rename, strip, inject defaults),
while converters handle semantic API-standard translation.

Design principles:

* **Idempotent**: applying the same transform twice should be harmless.
* **Non-overlapping**: provider-level and model-level transforms should
  operate on different fields by convention.
* **Merge by concatenation**: ``resolve_transforms`` merges provider + model
  transforms.  Provider transforms run first (broad dialect normalisation),
  model transforms run after (model-specific adjustments).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Core type
# ---------------------------------------------------------------------------

Transform = Callable[[dict[str, Any]], dict[str, Any]]
"""A pure data transformation: receives a provider body dict and returns
a (possibly mutated) body dict."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transformable(Protocol):
    """Shared interface satisfied by both ``ProviderShim`` and ``ModelShim``."""

    from_transforms: tuple[Transform, ...]
    to_transforms: tuple[Transform, ...]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class _NamedTransform:
    """Thin wrapper that gives a factory-produced callable a readable repr."""

    __slots__ = ("_fn", "_repr")

    def __init__(self, fn: Transform, repr_str: str) -> None:
        self._fn = fn
        self._repr = repr_str

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._fn(body)

    def __repr__(self) -> str:
        return self._repr


def strip_fields(*keys: str) -> Transform:
    """Return a transform that removes *keys* from the body.

    No-op for keys that do not exist (idempotent).
    """

    def _strip(body: dict[str, Any]) -> dict[str, Any]:
        for k in keys:
            body.pop(k, None)
        return body

    return _NamedTransform(_strip, f"strip_fields({', '.join(repr(k) for k in keys)})")


def rename_field(old: str, new: str) -> Transform:
    """Return a transform that renames a top-level field.

    No-op if *old* does not exist (idempotent).
    """

    def _rename(body: dict[str, Any]) -> dict[str, Any]:
        if old in body:
            body[new] = body.pop(old)
        return body

    return _NamedTransform(_rename, f"rename_field({old!r}, {new!r})")


def set_defaults(**defaults: Any) -> Transform:
    """Return a transform that sets fields only when they are absent.

    Idempotent: existing values are never overwritten.
    """

    def _defaults(body: dict[str, Any]) -> dict[str, Any]:
        for k, v in defaults.items():
            body.setdefault(k, v)
        return body

    return _NamedTransform(
        _defaults,
        f"set_defaults({', '.join(f'{k}={v!r}' for k, v in defaults.items())})",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_transforms(
    transforms: tuple[Transform, ...], body: dict[str, Any]
) -> dict[str, Any]:
    """Apply *transforms* sequentially to *body*, returning the result."""
    for t in transforms:
        body = t(body)
    return body


def resolve_transforms(
    provider: Transformable,
    model: Transformable | None,
) -> tuple[tuple[Transform, ...], tuple[Transform, ...]]:
    """Merge provider and model transforms by concatenation.

    Provider transforms run first (broad dialect normalisation), model
    transforms run after (model-specific adjustments).

    Returns:
        ``(from_transforms, to_transforms)`` ready for ``apply_transforms``.
    """
    if model is None:
        return provider.from_transforms, provider.to_transforms
    return (
        provider.from_transforms + model.from_transforms,
        provider.to_transforms + model.to_transforms,
    )
