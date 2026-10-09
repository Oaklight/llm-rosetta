"""IR-level transform primitives.

An **IRTransform** operates on the intermediate representation (IR) request
dict with access to route-level context (model capabilities, upstream model
name, etc.): ``(dict, TransformContext) → dict``.

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
        hoist_system_messages: Whether to hoist late system messages
            into user-role envelopes for cache prefix stability.
    """

    model: str = ""
    model_capabilities: list[str] | None = None
    request_id: str = "-"
    hoist_system_messages: bool = True


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
        from llm_rosetta.converters.base.tools.call_unwind import (
            unwind_parallel_tool_calls_ir,
        )

        return unwind_parallel_tool_calls_ir(body)

    label = "unwind_parallel_tool_calls("
    if pattern:
        label += f"pattern={pattern!r}"
    label += ")"
    return _NamedIRTransform(_unwind, label)


def auto_cache_breakpoints(mode: str = "none_only") -> IRTransform:
    """Return an IR transform that injects ``cache_hint`` breakpoints.

    Targets cross-format requests (OpenAI/Gemini → Anthropic) where the
    source format has no explicit cache semantics.  Injects up to 4
    ``cache_hint`` markers matching Anthropic's breakpoint limit.

    Args:
        mode: ``"none_only"`` (default) skips injection if any
            ``cache_hint`` already exists.  ``"fill_gaps"`` fills each
            segment (tools, system, messages) independently.

    Example::

        auto_cache_breakpoints()                  # conservative
        auto_cache_breakpoints(mode="fill_gaps")  # per-segment
    """

    def _inject(body: dict[str, Any], context: TransformContext) -> dict[str, Any]:
        from llm_rosetta.converters.base.helpers.cache_breakpoints import (
            inject_cache_breakpoints,
        )

        return inject_cache_breakpoints(body, mode=mode, request_id=context.request_id)

    label = "auto_cache_breakpoints("
    if mode != "none_only":
        label += f"mode={mode!r}"
    label += ")"
    return _NamedIRTransform(_inject, label)


def hoist_late_system_messages() -> IRTransform:
    """Return an IR transform that hoists late system messages.

    Leading system messages (before any non-system message) are moved to
    ``system_instruction``.  Mid-conversation system messages are rewritten
    as ``UserMessage`` with a ``[System: ...]`` envelope so they don't
    break the prompt cache prefix or get silently dropped by target
    converters.

    Idempotent: once rewritten to ``role: "user"``, a second pass is a
    no-op.  Should run **before** ``auto_cache_breakpoints()`` so cache
    hints target the post-hoist message structure.
    """

    def _hoist(body: dict[str, Any], context: TransformContext) -> dict[str, Any]:
        if not context.hoist_system_messages:
            return body
        from llm_rosetta.converters.base.helpers.system_message_hoist import (
            hoist_late_system_messages_ir,
        )

        return hoist_late_system_messages_ir(body, request_id=context.request_id)

    return _NamedIRTransform(_hoist, "hoist_late_system_messages()")
