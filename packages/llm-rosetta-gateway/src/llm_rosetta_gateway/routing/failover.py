"""Trying another provider after the serving one refuses.

The decision has two halves.  *Whether* a model may fail over at all is a
product choice the config owns (``failover_on_quota``): another provider is a
different prompt cache, a different rate-limit bucket and possibly a different
model build.  *Where* it goes is routing's, and that is this module: the
provider that refused is excluded for the rest of the request, and the same
selection chain runs again over what is left.

The proxy keeps the consequences — a rewritten body, a log line, a profile
entry — because none of them is a routing fact.

Resolving is passed in rather than imported: routing must not depend on the
config package, and the one thing this function does with a resolver is call
it and treat "nothing left to choose" as an answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_rosetta.routing import ResolvedRoute

    from ..transport import ProviderInfo
    from .context import SelectionContext


def reselect_after_refusal(
    selection: SelectionContext,
    *,
    exclude: str,
    resolve: Callable[[SelectionContext], tuple[ResolvedRoute, ProviderInfo]],
) -> tuple[ResolvedRoute, ProviderInfo] | None:
    """Run selection again with *exclude* out of the running.

    The provider that refused is excluded for the rest of the request, and the
    previous choice is voided with it, so a stage that finds nothing cannot
    leave the refused provider standing.

    Args:
        selection: This request's selection state, already installed with the
            config's tables.
        exclude: The provider that just refused.
        resolve: The caller's resolver for a context — ``GatewayConfig.resolve``
            in the gateway.  Passed in so this module imports no config.

    Returns:
        The new route and transport config, or ``None`` when nothing else can
        serve — in which case the caller reports the upstream's own refusal
        rather than inventing a different story.
    """
    from ..maintenance.startup import ProviderNotReady

    selection.with_failover_excluded(selection.failover_excluded | {exclude})
    try:
        return resolve(selection)
    except (ProviderNotReady, KeyError):
        return None
