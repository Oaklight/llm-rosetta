"""Provider selection: which of a model's providers serves this request.

The provider is chosen before the request body has been converted, so nothing
here reads the IR.  What it reads is the model's route and the eligibility of
the providers in it — the one predicate that answers "may this provider serve",
computed for every candidate once, by the stage that runs before this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping

from .eligibility import HEADROOM_FLOOR, Eligibility, quota_weight
from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext
    from .strategy import EntryKey, ModelRoute, ProviderEntry

__all__ = [
    "HEADROOM_FLOOR",
    "AnchorStage",
    "quota_weight",
    "select_ready_entry",
]


def select_ready_entry(
    model_route: ModelRoute,
    *,
    eligibility: Mapping[EntryKey, Eligibility],
    identity: str | None = None,
    excluded: frozenset[EntryKey] = frozenset(),
) -> ProviderEntry | None:
    """Pick an eligible provider from *model_route*, or ``None`` when none is.

    The strategy only ever sees providers that may serve, so an ineligible one
    cannot consume a turn of the round-robin — that was the reason the old
    code checked readiness *after* the pick and then scanned for an
    alternative.  Providers with a request-time weight (an allowance headroom)
    are handed over scaled; providers a recent refusal sank are handed over
    last, which every strategy does by only being shown them when nothing else
    is left.

    Args:
        model_route: The model's candidates and how they are chosen between.
        eligibility: The one predicate's answer for every candidate, keyed by
            the candidate's :attr:`~.strategy.ProviderEntry.key`.  A candidate
            without an answer is not a candidate.
        identity: Optional client identity for affinity routing.
        excluded: Providers this request must not be served by, on top of the
            ineligible ones — the ones it already failed over from, and every
            provider a pin rules out.

    Returns:
        The chosen entry, or ``None`` when nothing may serve.
    """
    weights: dict[EntryKey, float] = {}
    sunk: set[EntryKey] = set()
    blocked = set(excluded)
    for entry in model_route.providers:
        answer = eligibility.get(entry.key)
        if answer is None or not answer.eligible:
            blocked.add(entry.key)
            continue
        if answer.sunk:
            sunk.add(entry.key)
        factor = answer.weight_factor
        if factor != 1.0:
            weights[entry.key] = factor

    if all(p.key in blocked for p in model_route.providers):
        return None
    return model_route.select(
        identity=identity,
        weights=weights or None,
        excluded=frozenset(blocked),
        sunk=frozenset(sunk),
    )


class AnchorStage(RoutingStage):
    """Pick the provider that routing would choose with no preference applied.

    This is the terminal choice: whatever the stages after it decide, they
    decide *relative to this*, which is what lets the preference be a
    preference — the full route's counter advances exactly as it would with the
    preference off.

    Leaving :attr:`SelectionContext.chosen` unset means no candidate was
    eligible, which the caller turns into a failure; a later stage that cannot
    improve on the anchor leaves it alone instead.
    """

    __slots__ = ()

    name = "anchor"
    # ``eligibility`` is read and not seeded, so declaring it is what makes a
    # chain that puts this stage first refuse to build — otherwise a reversed
    # order would silently choose without the predicate.
    requires = frozenset(
        {
            "model_route",
            "identity",
            "eligibility",
            "failover_excluded",
        }
    )
    provides = frozenset({"chosen"})

    def run(self, ctx: SelectionContext) -> None:
        """Choose the anchor, if any candidate is eligible."""
        assert ctx.model_route is not None  # seeded by install_route()
        entry = select_ready_entry(
            ctx.model_route,
            eligibility=ctx.eligibility,
            identity=ctx.identity.client,
            excluded=ctx.excluded_providers(),
        )
        if entry is not None:
            ctx.choose(entry)
