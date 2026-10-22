"""Provider selection: which of a model's providers serves this request.

The provider is chosen before the request body has been converted, so nothing
here reads the IR.  What it reads is the model's route and the readiness of the
providers in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping

from .stage import RoutingStage

if TYPE_CHECKING:
    from collections.abc import Callable

    from .context import SelectionContext
    from .strategy import ModelRoute, ProviderEntry


#: How much of a provider's configured weight survives a spent window.  The
#: floor keeps a live provider in rotation — a headroom of 0.0 is "nearly
#: spent", and the quota state, not this factor, is what removes it.
HEADROOM_FLOOR = 0.25


def quota_weight(headroom: float | None) -> float:
    """Scale a weight by the provider's remaining allowance.

    Args:
        headroom: Smallest remaining share, 0..1, or ``None`` when nothing has
            measured this provider.

    Returns:
        The multiplier: ``1.0`` for an unmeasured or untouched provider, the
        floor for a spent one.
    """
    if headroom is None:
        return 1.0
    return HEADROOM_FLOOR + (1.0 - HEADROOM_FLOOR) * max(0.0, min(1.0, headroom))


def select_ready_entry(
    model_route: ModelRoute,
    *,
    is_ready: Callable[[str], bool],
    identity: str | None = None,
    headroom: Mapping[str, float] | None = None,
) -> ProviderEntry | None:
    """Pick a ready provider from *model_route*, or ``None`` when none is.

    Alternatives are scanned over the provider list directly rather than via
    the strategy, leaving the round-robin counter alone and checking every
    candidate whatever the strategy.

    Args:
        model_route: The model's candidates and how they are chosen between.
        is_ready: Whether a named provider can take traffic right now.
        identity: Optional client identity for affinity routing.

        headroom: Per-provider remaining share (see :func:`quota_weight`).

    Returns:
        The chosen entry, or ``None`` when no candidate is ready.
    """
    weights = (
        {name: quota_weight(share) for name, share in headroom.items()}
        if headroom
        else None
    )
    entry = model_route.select_entry(identity=identity, weights=weights)
    if is_ready(entry.name):
        return entry
    for alt in model_route.providers:
        if alt.name != entry.name and is_ready(alt.name):
            return alt
    return None


class AnchorStage(RoutingStage):
    """Pick the provider that routing would choose with no preference applied.

    This is the terminal choice: whatever the stages after it decide, they
    decide *relative to this*, which is what lets the preference be a
    preference — the full route's counter advances exactly as it would with the
    preference off.

    Leaving :attr:`SelectionContext.chosen` unset means no candidate was ready,
    which the caller turns into a failure; a later stage that cannot improve on
    the anchor leaves it alone instead.
    """

    __slots__ = ()

    name = "anchor"
    # ``quota_headroom`` is read (it weights the candidates) and not seeded, so
    # declaring it is what makes a chain that puts this stage first refuse to
    # build — otherwise a reversed order would silently weight nothing.
    requires = frozenset({"model_route", "providers", "identity", "quota_headroom"})
    provides = frozenset({"chosen"})

    def run(self, ctx: SelectionContext) -> None:
        """Choose the anchor, if any candidate is ready."""
        assert ctx.model_route is not None  # seeded by install_route()
        entry = select_ready_entry(
            ctx.model_route,
            is_ready=ctx.is_ready,
            identity=ctx.identity.client,
            headroom=ctx.quota_headroom or None,
        )
        if entry is not None:
            ctx.choose(entry)
