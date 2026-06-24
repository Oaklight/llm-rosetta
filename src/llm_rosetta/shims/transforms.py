"""Transform primitives for the provider shim layer.

Two transform types:

* **Transform** — body-level: ``dict → dict``.  Handles provider-specific
  field-level quirks (rename, strip, inject defaults) on the raw provider
  request/response body.
* **IRTransform** — IR-level: ``(dict, TransformContext) → dict``.  Operates
  on the IR request with access to route-level context (model capabilities,
  upstream model name, etc.).

Design principles:

* **Idempotent**: applying the same transform twice should be harmless.
* **Non-overlapping**: transforms should operate on different fields by
  convention.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# IR-level transforms (context-aware)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransformContext:
    """Context available to IR-level transforms.

    Carries route-level information that body-level transforms don't
    need but IR transforms do (e.g. model capabilities for vision
    stripping).

    Attributes:
        model: Upstream model identifier (post-alias).
        model_capabilities: Declared capabilities of the model
            (e.g. ``["text", "vision"]``).  ``None`` means unknown.
        request_id: Request identifier for logging.
    """

    model: str = ""
    model_capabilities: list[str] | None = None
    request_id: str = "-"


IRTransform = Callable[[dict[str, Any], TransformContext], dict[str, Any]]
"""An IR-level data transformation: receives an IR request dict and a
:class:`TransformContext`, returns the (possibly mutated) IR dict."""


class _NamedIRTransform:
    """Thin wrapper that gives a factory-produced IR transform a readable repr."""

    __slots__ = ("_fn", "_repr")

    def __init__(self, fn: IRTransform, repr_str: str) -> None:
        self._fn = fn
        self._repr = repr_str

    def __call__(
        self, body: dict[str, Any], context: TransformContext
    ) -> dict[str, Any]:
        return self._fn(body, context)

    def __repr__(self) -> str:
        return self._repr


def apply_ir_transforms(
    transforms: tuple[IRTransform, ...],
    body: dict[str, Any],
    context: TransformContext,
) -> dict[str, Any]:
    """Apply IR-level *transforms* sequentially with *context*."""
    for t in transforms:
        body = t(body, context)
    return body


# ---------------------------------------------------------------------------
# IR-level factory functions
# ---------------------------------------------------------------------------


def strip_non_vision_images() -> IRTransform:
    """Return an IR transform that replaces all images with text placeholders
    when the model lacks ``"vision"`` capability.

    No-op if ``model_capabilities`` is ``None`` (unknown) or includes
    ``"vision"`` (idempotent).
    """

    def _strip(body: dict[str, Any], context: TransformContext) -> dict[str, Any]:
        if context.model_capabilities is None or "vision" in context.model_capabilities:
            return body
        from llm_rosetta.converters.base.helpers.image_limit import (
            strip_images_for_non_vision,
        )

        return strip_images_for_non_vision(
            body, model=context.model, request_id=context.request_id
        )

    return _NamedIRTransform(_strip, "strip_non_vision_images()")


def truncate_images(max_images: int, pattern: str | None = None) -> IRTransform:
    """Return an IR transform that truncates images exceeding *max_images*.

    When *pattern* is set, truncation only fires if the upstream model
    matches the regex (search on raw model string).

    Example::

        truncate_images(50, pattern=r"^(gpt|o\\d)")
    """
    compiled = re.compile(pattern) if pattern else None

    def _truncate(body: dict[str, Any], context: TransformContext) -> dict[str, Any]:
        if compiled is not None:
            if not context.model or not compiled.search(context.model):
                return body
        from llm_rosetta.converters.base.helpers.image_limit import (
            truncate_images as _truncate_impl,
        )

        return _truncate_impl(body, max_images, request_id=context.request_id)

    label = f"truncate_images({max_images}"
    if pattern:
        label += f", pattern={pattern!r}"
    label += ")"
    return _NamedIRTransform(_truncate, label)


def unwind_parallel_tool_calls(pattern: str | None = None) -> IRTransform:
    """Return an IR transform that splits parallel tool calls into
    sequential call-result pairs.

    When *pattern* is set, unwinding only fires if the upstream model
    matches the regex (search on raw model string).

    Example::

        unwind_parallel_tool_calls(pattern=r"^gemini")
    """
    compiled = re.compile(pattern) if pattern else None

    def _unwind(body: dict[str, Any], context: TransformContext) -> dict[str, Any]:
        if compiled is not None:
            if not context.model or not compiled.search(context.model):
                return body
        from llm_rosetta.converters.base.helpers.tool_call_unwind import (
            unwind_parallel_tool_calls_ir,
        )

        return unwind_parallel_tool_calls_ir(body)

    label = "unwind_parallel_tool_calls("
    if pattern:
        label += f"pattern={pattern!r}"
    label += ")"
    return _NamedIRTransform(_unwind, label)
