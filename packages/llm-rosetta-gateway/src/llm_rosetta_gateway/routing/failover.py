"""Trying another lane after the one serving refused.

The decision has two halves.  *Whether* a model may fail over at all is a
product choice the config owns (``failover_on_error``): another lane is another
prompt cache and possibly another model build.  *Where* it goes is routing's,
and that is this module.

Two kinds of refusal reach here, and they exclude different things:

- A **credential-level** refusal — a ``429``, a ``402``, a spent allowance — is
  one key's state.  Every lane of that provider shares the key, so the
  provider is excluded by **name** for the rest of the request
  (:func:`reselect_after_credential_refusal`).
- A **content-level** refusal — a transient upstream status, a first token
  that never came — is about the upstream model that answered, not the key.
  Only that **lane** leaves the route, and the provider's siblings are tried
  before anyone else: same vendor, same account, one base URL
  (:func:`reselect_after_lane_refusal`).

The proxy keeps the consequences — a rewritten body, a log line, a profile
entry — because none of them is a routing fact.

Resolving is passed in rather than imported: routing must not depend on the
config package, and the one thing these functions do with a resolver is call
it and treat "nothing left to choose" as an answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_rosetta.routing import ResolvedRoute

    from ..transport import ProviderInfo
    from .context import SelectionContext
    from .strategy import EntryKey

    _Resolve = Callable[[SelectionContext], tuple[ResolvedRoute, ProviderInfo]]


def _resolve_or_none(
    selection: SelectionContext, resolve: _Resolve
) -> tuple[ResolvedRoute, ProviderInfo] | None:
    """Ask the resolver, and read "nothing left" as an answer rather than an error."""
    from ..maintenance.startup import ProviderNotReady

    try:
        return resolve(selection)
    except (ProviderNotReady, KeyError):
        return None


def reselect_after_credential_refusal(
    selection: SelectionContext,
    *,
    exclude: str,
    resolve: _Resolve,
) -> tuple[ResolvedRoute, ProviderInfo] | None:
    """Run selection again with every lane of *exclude* out of the running.

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
    selection.with_failover_excluded(selection.failover_excluded | {exclude})
    return _resolve_or_none(selection, resolve)


def reselect_after_lane_refusal(
    selection: SelectionContext,
    *,
    lane: EntryKey,
    resolve: _Resolve,
    try_siblings: bool = True,
) -> tuple[ResolvedRoute, ProviderInfo] | None:
    """Run selection again with *lane* out, its siblings before anyone else.

    Only the lane that refused leaves the route: a content-level refusal says
    nothing about the credential, so the provider's other lanes are still
    candidates.

    They are tried **first**, and that is what the temporary pin does: pinning
    a provider is the same statement — "only these lanes are candidates" — the
    operator's own pin makes, so the mechanism is shared rather than
    reimplemented.  With no sibling left, the pin is lifted and the search
    widens to the rest of the route; a caller's own pin is never widened,
    because the pin is an instruction and a failure is not a licence to leave
    it.

    Args:
        selection: This request's selection state, already installed with the
            config's tables.
        lane: The candidate key — ``(provider, upstream model)`` — that
            refused, as the selection used it (``selection.chosen.key``).
        resolve: The caller's resolver, as above.
        try_siblings: Whether the provider's other lanes may be tried.  The
            budget passes ``False`` once a sibling has refused in turn: from
            there only another provider is left.

    Returns:
        The new route and transport config, or ``None`` when no lane may serve
        — the caller then reports the refusal it already has.
    """
    selection.with_lane_excluded(selection.lane_excluded | {lane})
    provider = lane[0]
    pinned = selection.pinned
    if pinned is not None and pinned != provider:
        return None

    if try_siblings:
        selection.with_pinned_provider(provider)
        try:
            sibling = _resolve_or_none(selection, resolve)
        finally:
            selection.with_pinned_provider(pinned)
        if sibling is not None:
            return sibling
    if pinned is not None:
        return None
    return _resolve_or_none(selection, resolve)
