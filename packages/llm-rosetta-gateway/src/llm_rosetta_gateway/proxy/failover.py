"""Failover when the lane serving a request cannot answer.

Two kinds of refusal qualify, and they leave different parts of the route
standing:

- A **credential-level** one — an allowance that ran out (a ``429`` the quota
  probe can explain, or a ``402``) — is one key's state.  Every lane of that
  provider shares the key, so the provider leaves the route for the rest of the
  request and the model's other providers may serve.
- A **content-level** one — a transient upstream status (``5xx``, ``408``,
  ``529``), or a first token that never came — is about the upstream model that
  answered.  Only that lane leaves, the provider's siblings are tried first,
  and another provider is reached only once none of them is left.

Trying another lane or provider at all is a product decision rather than a
routing one — the second one is a different prompt cache and possibly a
different model build — so it is opt-in per model (``failover_on_error``, or
the older ``failover_on_quota``).  What the budget allows is
:class:`FailoverBudget`: one sibling retry and one cross-provider retry for a
content-level refusal, one cross-provider retry for a credential-level one.

Both handlers use this module the same way: ask :func:`prepare_failover`, and
when it returns a target, run the whole request again against it and record the
outcome with :func:`record_failover`.  The streaming handler reaches this
before it has written anything to the client, which is the only point at which
a different answer can still be given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..routing.failover import (
    reselect_after_credential_refusal,
    reselect_after_lane_refusal,
)

if TYPE_CHECKING:
    from llm_rosetta.routing import ResolvedRoute

    from ..config import GatewayConfig
    from ..routing import SelectionContext
    from ..routing.strategy import EntryKey
    from ..transport import ProviderInfo
    from .errors import RefusalDecision


@dataclass(frozen=True)
class FailoverBudget:
    """What this request has already spent on retries.

    A credential-level refusal earns the same single cross-provider retry it
    always did.  A content-level one earns two: the sibling lane first, then —
    only if that lane has none left, or its sibling refused in turn — another
    provider.  That bounds a request at three upstream attempts.
    """

    #: How many retries this request has made.  The bound is on the count, not
    #: on either kind below: a provider with no sibling spends a step that
    #: lands on another provider, and a rule that only looked at
    #: ``lane_retried`` would let that shape walk the whole route, one
    #: provider per refusal.
    retries: int = 0
    #: A same-provider sibling has already been tried.
    lane_retried: bool = False
    #: Another provider has already been tried.
    cross_retried: bool = False

    @property
    def spent(self) -> int:
        """How many retries this request has made."""
        return self.retries

    def after(self, route: ResolvedRoute, new_route: ResolvedRoute) -> FailoverBudget:
        """The budget once the retry on *new_route* has been made."""
        same_provider = new_route.provider_name == route.provider_name
        return FailoverBudget(
            retries=self.retries + 1,
            lane_retried=self.lane_retried or same_provider,
            cross_retried=self.cross_retried or not same_provider,
        )


#: How many retries a content-level refusal may earn: the sibling lane, then
#: another provider.  Two steps, so a request makes at most three upstream
#: attempts however many providers its route lists.
_CONTENT_RETRIES = 2


def lane_label(provider: str, upstream_model: str | None) -> str:
    """How a lane is named in a log line: ``provider`` or ``provider/upstream``."""
    return f"{provider}/{upstream_model}" if upstream_model else provider


def prepare_failover(
    config: GatewayConfig | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    decision: RefusalDecision,
    budget: FailoverBudget,
) -> tuple[ResolvedRoute, ProviderInfo, dict[str, Any], FailoverBudget] | None:
    """Decide whether this request may be tried on another lane or provider.

    The model must have opted in — a different lane is a different prompt
    cache and possibly a different model build — and the budget must still
    allow the attempt the refusal earns.

    Returns:
        ``(route, provider_info, body, budget)`` for the next attempt, with the
        body carrying the new lane's upstream model name and the budget
        recording what the attempt spends, or ``None`` when there will not be
        one.
    """
    from ..logging import get_logger
    from .errors import RefusalScope

    if config is None:
        return None
    if selection.model not in getattr(config, "model_failover", {}):
        return None
    if decision.scope is RefusalScope.LANE:
        if budget.spent >= _CONTENT_RETRIES:
            return None
        if not budget.lane_retried:
            target = reselect_after_lane_refusal(
                selection, lane=_refused_lane(selection, route), resolve=config.resolve
            )
        elif not budget.cross_retried:
            # The sibling has been tried: only another provider is left.
            target = reselect_after_lane_refusal(
                selection,
                lane=_refused_lane(selection, route),
                resolve=config.resolve,
                try_siblings=False,
            )
        else:
            return None
    else:
        if budget.cross_retried:
            return None
        target = reselect_after_credential_refusal(
            selection, exclude=route.provider_name, resolve=config.resolve
        )
    if target is None:
        return None
    new_route, new_pinfo = target
    reason = getattr(decision.signal, "reason", None) or "upstream refusal"
    get_logger().warning(
        "Upstream failover for model %s: %s -> %s (%s)",
        selection.model,
        lane_label(route.provider_name, route.upstream_model),
        lane_label(new_route.provider_name, new_route.upstream_model),
        reason,
    )
    retry_body = dict(body)
    # The client's model name may have been replaced by the first lane's
    # upstream alias; the next lane gets its own, or the original name.
    retry_body["model"] = new_route.upstream_model or selection.model
    return new_route, new_pinfo, retry_body, budget.after(route, new_route)


def _refused_lane(selection: SelectionContext, route: ResolvedRoute) -> EntryKey:
    """The candidate key the attempt was made on.

    The selection's own entry is authoritative: a model-level upstream alias
    can make ``route.upstream_model`` name something the entry key does not,
    and excluding the wrong key would leave the refused lane in the route.
    """
    if selection.chosen is not None:
        return selection.chosen.key
    return (route.provider_name, route.upstream_model)


def record_failover(
    profile: dict[str, Any],
    *,
    route: ResolvedRoute,
    new_route: ResolvedRoute,
    decision: RefusalDecision,
    sub_profile: dict[str, Any],
) -> None:
    """Write the failover into the profile, and point the log at the server.

    The profile entry is what the request log carries, so a reader can see
    that the answer came from the second lane and what kind of refusal moved
    it; the lifecycle row must name the lane whose response the client
    actually received.
    """
    from ..lifecycle import current_lifecycle
    from ..logging import get_logger

    try:
        scope = getattr(decision.scope, "value", None)
        profile["failover"] = {
            "from": route.provider_name,
            "to": new_route.provider_name,
            "from_lane": route.upstream_model,
            "to_lane": new_route.upstream_model,
            "scope": scope,
            "reason": getattr(decision.signal, "reason", None) or "upstream refusal",
            "profile": sub_profile,
        }
        ctx = current_lifecycle()
        # The outermost call runs last, so an attempt that failed over again
        # has already named the lane that served: a second hop's row must not
        # be overwritten with the intermediate one.
        if ctx is not None and "failover" not in sub_profile:
            ctx.note_route(
                target_format=new_route.target_provider,
                provider_name=new_route.provider_name,
                upstream_model=new_route.upstream_model,
            )
    except Exception:  # noqa: BLE001 - bookkeeping must not mask the outcome
        # This runs from a ``finally`` on the failure path, where an exception
        # here would replace the client's error with a bookkeeping one.  The
        # answer always wins; the record is best-effort.
        get_logger().warning("Could not record the failover", exc_info=True)
