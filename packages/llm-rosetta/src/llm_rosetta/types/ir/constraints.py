"""Request constraints a dialect declares on IR request fields.

A vendor or a wire format constrains a request field in three ways, and only
two of them had a home before this module: a *refused* field is removed by a
shim transform, and everything else lived in converter code.  The remaining two
are declared here:

- a field the dialect **requires**, and the value to write when the client
  named none (``defaults``);
- a **value range** the upstream accepts, and what to do outside it
  (``ranges``).

Both are keyed by the IR path they constrain (``generation.max_tokens``,
``generation.temperature``), because the constraint is applied to the IR
request before a dialect writes it: the same path names one field in every
chat format, and the conversion warning is filed under that path.

The type is shared by three layers: a converter declares its own dialect
defaults as a class attribute, a shim may declare or override per vendor, and
the pass applies the merged result.  A constraint object is therefore data
only — applying it is ``ir.passes.request_constraints``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

#: What to do with a value outside a declared range.
#:
#: ``clamp`` writes the nearest bound and reports; ``drop`` removes the field
#: so the upstream default applies and reports; ``reject`` raises.
OnOutOfRange = Literal["clamp", "drop", "reject"]


@dataclass(frozen=True)
class Range:
    """An accepted value range for one numeric IR field.

    Attributes:
        min: Lowest accepted value, or ``None`` for no lower bound.
        max: Highest accepted value, or ``None`` for no upper bound.
        on_out_of_range: What to do with a value outside the bounds.
    """

    min: float | None = None
    max: float | None = None
    on_out_of_range: OnOutOfRange = "clamp"

    def __post_init__(self) -> None:
        if self.on_out_of_range not in ("clamp", "drop", "reject"):
            raise ValueError(
                f"Range: on_out_of_range must be clamp, drop or reject, "
                f"got {self.on_out_of_range!r}"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"Range: min {self.min} is greater than max {self.max}")

    def clamp(self, value: float) -> float:
        """Return *value* pulled into the bounds (never widens)."""
        if self.min is not None and value < self.min:
            return self.min
        if self.max is not None and value > self.max:
            return self.max
        return value

    def contains(self, value: float) -> bool:
        """Whether *value* is inside the bounds."""
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


@dataclass(frozen=True)
class RequestConstraints:
    """What a dialect requires of its request fields, by IR path.

    Attributes:
        defaults: IR path → the value to write when the client named none.
            An explicit ``null`` counts as naming none (see
            :func:`~llm_rosetta.ir.passes.request_constraints.apply_request_constraints`).
        ranges: IR path → accepted range and out-of-range policy.
    """

    defaults: Mapping[str, Any] = field(default_factory=dict)
    ranges: Mapping[str, Range] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether anything is declared at all."""
        return bool(self.defaults or self.ranges)

    def merged_with(self, overlay: RequestConstraints | None) -> RequestConstraints:
        """Return these constraints with *overlay* applied per path.

        A shim speaks for its own upstream, so its entry wins over the
        dialect's for the same path; paths only one of them names are kept.

        Args:
            overlay: Constraints declared by a shim, or ``None``.

        Returns:
            The merged constraints; ``self`` when the overlay is empty.
        """
        if not overlay:
            return self
        return RequestConstraints(
            defaults={**self.defaults, **overlay.defaults},
            ranges={**self.ranges, **overlay.ranges},
        )


__all__ = ["OnOutOfRange", "Range", "RequestConstraints"]
