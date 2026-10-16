"""The stages selection is made of, and the chain that runs them.

A stage is one decision.  Which stages exist, and in what order, is stated
once, where the chain is built — the two ends of that chain run at different
points in a request (see :mod:`~llm_rosetta.gateway.routing.context`), and the
callers that run them are what picks the split.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .context import SelectionContext


class RoutingStage(ABC):
    """One step of selection.

    A stage reads what earlier stages left on the context and writes what it
    decides, through the context's own setters.

    ``requires`` and ``provides`` name those context fields.  The chain checks
    them when it is built, so a stage slotted in ahead of something it needs
    fails at construction rather than on the first request — which matters most
    for a stage added later, by which time nobody is holding the whole order in
    their head.
    """

    __slots__ = ()

    name: ClassVar[str]
    """Registry-friendly name, used in error messages."""

    requires: ClassVar[frozenset[str]] = frozenset()
    """Context fields this stage reads."""

    provides: ClassVar[frozenset[str]] = frozenset()
    """Context fields this stage writes."""

    @abstractmethod
    def run(self, ctx: SelectionContext) -> None:
        """Do this stage's part of selecting for *ctx*.

        Args:
            ctx: The request's selection state, updated in place.
        """


class StageOrderError(ValueError):
    """A stage asked for something no stage before it provides."""


class RoutingChain:
    """An ordered list of stages, checked for order once at build time."""

    __slots__ = ("_stages",)

    #: Fields the context carries before any stage runs.
    SEEDED: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "source_format",
            "identity",
            "model_route",
            "providers",
            "provider_types",
            "provider_shim_names",
            "provider_account_tokens",
        }
    )

    def __init__(self, stages: Sequence[RoutingStage]) -> None:
        """Build a chain, refusing one whose stages are out of order.

        Args:
            stages: The stages, in the order they will run.

        Raises:
            StageOrderError: If a stage reads a field that nothing before it —
                neither the seeded context nor an earlier stage — provides.
        """
        available = set(self.SEEDED)
        for stage in stages:
            missing = stage.requires - available
            if missing:
                names = ", ".join(sorted(missing))
                raise StageOrderError(
                    f"stage {stage.name!r} needs {names}, which nothing before "
                    f"it provides"
                )
            available |= stage.provides
        self._stages = tuple(stages)

    @property
    def stages(self) -> tuple[RoutingStage, ...]:
        """The stages, in order."""
        return self._stages

    def run(self, ctx: SelectionContext) -> None:
        """Run every stage against *ctx*, in order."""
        for stage in self._stages:
            stage.run(ctx)
