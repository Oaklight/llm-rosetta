"""Apply a dialect's declared request constraints to the IR request.

The declaration is a :class:`~llm_rosetta.types.ir.constraints.RequestConstraints`
on the target converter (its own dialect's requirement) overlaid by the target
shim (its vendor's requirement, per path).  Three things happen here, in this
order:

1. **Explicit nulls are normalized to absence.**  ``{"max_tokens": null}`` and
   a missing key mean the same thing in IR — the client named no value — and
   a ``null`` that reached a writer became a value the upstream rejects
   (#1459).  This runs for every field of
   :class:`~llm_rosetta.types.ir.configs.GenerationConfig`, declared or not,
   because it is IR semantics rather than a dialect quirk.
2. **Declared defaults fill an absent path.**  The value is recorded in the
   conversion metadata (``injected``), the same carrier the writers use for
   the defaults they inject, so the request row names every value the client
   did not send.
3. **Declared ranges adjust a value.**  ``clamp`` writes the nearest bound,
   ``drop`` removes the field, and ``reject`` raises
   :class:`RequestConstraintViolation`; each is reported under the IR path it
   changed, except ``reject``, whose exception carries it.

A constraint object that declares nothing is still applied for step 1: every
conversion gets the null rule, and no other step runs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

from llm_rosetta.conversion_warnings import DOWNGRADED, DROPPED, WarningSink
from llm_rosetta.types.ir.configs import GenerationConfig
from llm_rosetta.types.ir.constraints import Range, RequestConstraints

logger = logging.getLogger(__name__)


class RequestConstraintViolation(ValueError):
    """A declared range rejected the value the client sent.

    Carries the IR path and the range so a caller can name both without
    parsing the message.

    Attributes:
        path: The IR path whose value was rejected.
        value: The value the request carried.
        range: The declared :class:`Range`.
    """

    def __init__(self, path: str, value: Any, declared: Range) -> None:
        self.path = path
        self.value = value
        self.range = declared
        super().__init__(
            f"{path} = {value!r} is outside the range this target accepts "
            f"({declared.describe()}); the target rejects the request"
        )


#: IR paths an explicit null is normalized on.  Derived from the IR type, so a
#: field added there is covered without a second list to keep in step.
_GENERATION_PATHS = tuple(
    f"generation.{key}" for key in GenerationConfig.__annotations__
)


def _container(ir: Mapping[str, Any], path: str) -> tuple[MutableMapping | None, str]:
    """Split *path* into the mapping that holds it and its final key.

    Args:
        ir: The IR request (or any nested mapping).
        path: A dotted IR path, e.g. ``generation.max_tokens``.

    Returns:
        ``(container, key)``; ``container`` is ``None`` when an intermediate
        segment is absent or not a mapping.
    """
    parts = path.split(".")
    holder: Any = ir
    for part in parts[:-1]:
        holder = holder.get(part) if isinstance(holder, Mapping) else None
        if not isinstance(holder, Mapping):
            return None, ""
    if not isinstance(holder, MutableMapping):
        return None, ""
    return holder, parts[-1]


def _is_number(value: Any) -> bool:
    """Whether *value* is a real number (booleans excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_nulls(ir: Mapping[str, Any], warnings: WarningSink | None) -> None:
    """Remove explicit nulls from the generation block, reporting each one."""
    for path in _GENERATION_PATHS:
        holder, key = _container(ir, path)
        if holder is None or key not in holder or holder[key] is not None:
            continue
        del holder[key]
        if warnings is not None:
            warnings.warn(
                "an explicit null means the client named no value; the key is "
                "left out so the target can apply its own default",
                code=DROPPED,
                subject=path,
            )


def _ensure_container(
    ir: MutableMapping[str, Any], path: str
) -> tuple[MutableMapping[str, Any], str]:
    """Return the mapping *path* names, creating absent intermediates."""
    parts = path.split(".")
    holder = ir
    for part in parts[:-1]:
        nxt = holder.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            holder[part] = nxt
        holder = nxt
    return holder, parts[-1]


def _apply_defaults(
    ir: MutableMapping[str, Any],
    defaults: Mapping[str, Any],
    metadata: MutableMapping[str, Any] | None,
) -> None:
    """Write each declared default into an absent path.

    A key the request carries is left alone — the client always wins — and a
    ``null`` no longer counts as carrying one, because nulls were normalized
    first.
    """
    for path, value in defaults.items():
        holder, key = _ensure_container(ir, path)
        if key in holder and holder[key] is not None:
            continue
        holder[key] = value
        if metadata is not None:
            metadata.setdefault("injected", {})[key] = value


def _apply_ranges(
    ir: Mapping[str, Any],
    ranges: Mapping[str, Range],
    warnings: WarningSink | None,
) -> None:
    """Adjust or reject a value outside its declared range."""
    for path, declared in ranges.items():
        holder, key = _container(ir, path)
        if holder is None or key not in holder:
            continue
        value = holder[key]
        if not _is_number(value) or declared.contains(value):
            continue
        if declared.on_out_of_range == "reject":
            raise RequestConstraintViolation(path, value, declared)
        if declared.on_out_of_range == "drop":
            del holder[key]
            if warnings is not None:
                warnings.warn(
                    f"{value!r} is outside the range this target accepts "
                    f"({declared.describe()}); the field is dropped so the "
                    f"target applies its own default",
                    code=DROPPED,
                    subject=path,
                )
            continue
        clamped = declared.clamp(value)
        holder[key] = clamped
        if warnings is not None:
            warnings.warn(
                f"{value!r} is outside the range this target accepts "
                f"({declared.describe()}); written as {clamped!r}",
                code=DOWNGRADED,
                subject=path,
            )


def apply_request_constraints(
    ir_request: dict[str, Any],
    *,
    constraints: RequestConstraints | None = None,
    warnings: WarningSink | None = None,
    metadata: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the target's declared request constraints, in place.

    Args:
        ir_request: The IR request about to be written.
        constraints: The merged declaration (dialect defaults with the shim's
            overlay already applied), or ``None``.  An empty declaration still
            gets the null normalization.
        warnings: The conversion's sink, when the caller has one.  A null, a
            clamp and a drop are reported there under their IR path; a default
            is recorded in *metadata* instead, because a value the target
            required is not something the request lost.
        metadata: The conversion metadata (``ctx.metadata``), which carries
            ``injected`` to the request row.

    Returns:
        The IR request passed in, for chaining.

    Raises:
        RequestConstraintViolation: If a range declares ``on_out_of_range:
            reject`` and the value is outside it.
    """
    _normalize_nulls(ir_request, warnings)
    if constraints:
        _apply_defaults(ir_request, constraints.defaults, metadata)
        _apply_ranges(ir_request, constraints.ranges, warnings)
    return ir_request


__all__ = ["RequestConstraintViolation", "apply_request_constraints"]
