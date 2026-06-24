"""Transform primitives for the provider shim layer.

A **Transform** is a pure data transformation: ``dict → dict``.  Transforms
bridge the gap between a real provider's API dialect and the "ideal" standard
that the corresponding base converter expects.

Transforms are NOT a replacement for converter functionality — they handle
provider-specific field-level quirks (rename, strip, inject defaults),
while converters handle semantic API-standard translation.

Design principles:

* **Idempotent**: applying the same transform twice should be harmless.
* **Non-overlapping**: transforms should operate on different fields by
  convention.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Core type
# ---------------------------------------------------------------------------

Transform = Callable[[dict[str, Any]], dict[str, Any]]
"""A pure data transformation: receives a provider body dict and returns
a (possibly mutated) body dict."""


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
# Message-level factory functions
# ---------------------------------------------------------------------------


def replace_message_field(field: str, old_value: Any, new_value: Any) -> Transform:
    """Return a transform that replaces *old_value* with *new_value* in
    ``messages[].field``.

    Iterates all entries in ``body["messages"]`` and replaces the field
    value where it matches *old_value*.  No-op if ``messages`` is absent
    or no entry has the matching value (idempotent).

    Example::

        replace_message_field("role", "developer", "system")
    """

    def _replace(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body
        for msg in messages:
            if isinstance(msg, dict) and msg.get(field) == old_value:
                msg[field] = new_value
        return body

    return _NamedTransform(
        _replace,
        f"replace_message_field({field!r}, {old_value!r}, {new_value!r})",
    )


def default_message_field(field: str, default: Any) -> Transform:
    """Return a transform that sets a default for ``messages[].field``
    when the field is absent or its value is ``None``.

    No-op if ``messages`` is absent or no entry needs defaulting
    (idempotent).

    Example::

        default_message_field("content", "")
    """

    def _default(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body
        for msg in messages:
            if isinstance(msg, dict) and msg.get(field) is None:
                msg[field] = default
        return body

    return _NamedTransform(
        _default,
        f"default_message_field({field!r}, {default!r})",
    )


def strip_fields_for_model(pattern: str, *keys: str) -> Transform:
    """Return a transform that removes *keys* from the body only when
    ``body["model"]`` matches *pattern* (regex search).

    No-op if ``model`` is absent or doesn't match (idempotent).

    Example::

        strip_fields_for_model(r"^claudeopus47", "temperature")
    """
    compiled = re.compile(pattern)

    def _strip(body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model")
        if not model or not isinstance(model, str):
            return body
        # Normalise for matching: strip non-alphanumeric, lowercase
        normalised = re.sub(r"[^a-z0-9]", "", model.lower())
        if not compiled.search(normalised):
            return body
        for k in keys:
            body.pop(k, None)
        return body

    return _NamedTransform(
        _strip,
        f"strip_fields_for_model({pattern!r}, {', '.join(repr(k) for k in keys)})",
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
