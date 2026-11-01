"""Why nothing was chosen, in the words the client needs.

``resolve`` asks this module once, after selection left ``ctx.chosen`` unset,
instead of re-deriving the four states inline.  The order of the checks is the
rule:

1. a **quota block** covering every provider the request can still use has a
   known end and must not be reported as startup — a client that just
   received a plan refusal would go looking for a boot problem that does not
   exist;
2. a route every remaining candidate of which a **reading** removed is
   ``exhausted``, with the reset when the vendor named one and without one
   when it did not;
3. otherwise the route is down for a startup reason: ``failed`` when every
   unready provider failed to initialise, else ``initializing``.

The module answers a question; it does not raise.  ``ProviderNotReady`` stays
in ``maintenance`` and is built from :meth:`Unavailable.as_kwargs`, so the
routing package keeps its promise not to own the server's error vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..config.core import GatewayConfig
    from .context import SelectionContext

#: ``blocked`` and ``exhausted`` are quota states; ``failed`` and
#: ``initializing`` are startup states.
Reason = Literal["blocked", "exhausted", "failed", "initializing"]


@dataclass(frozen=True)
class Unavailable:
    """The state of a route no candidate could serve.

    Attributes:
        reason: Which of the four stories this is.
        until: When the route comes back, for the two quota reasons.  ``None``
            on ``exhausted`` means the vendor named no reset, and the gateway
            will not invent one.
    """

    reason: Reason
    until: datetime | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """The keyword arguments ``ProviderNotReady`` takes for this state."""
        if self.reason == "blocked":
            return {"blocked_until": self.until}
        if self.reason == "exhausted":
            return {"exhausted_until": self.until, "exhausted": self.until is None}
        return {"failed": self.reason == "failed"}


def blocked_until(unready: list[Any]) -> datetime | None:
    """The end of a quota block covering every unready provider, if any.

    Returns the **latest** end, because the route is unavailable until the last
    of its providers is back.  A provider that is unready for any other reason,
    or a block the state cannot date, makes the answer unknown — ``None``.
    """
    from ..quota.state import QUOTA_STATE

    ends: list[datetime] = []
    for pinfo in unready:
        blocked = getattr(pinfo, "quota_provider_blocked", None)
        if blocked is None or not blocked():
            return None
        identity = getattr(pinfo, "quota_identity", None) or f"@{pinfo.name}"
        block = QUOTA_STATE.provider_block(identity)
        if block is None:
            return None
        ends.append(block.until)
    return max(ends) if ends else None


def exhausted_until(
    config: GatewayConfig,
    ctx: SelectionContext,
    names: set[str] | None = None,
) -> datetime | None:
    """The end of a spent allowance covering every candidate, if one is known.

    The route is unavailable until the **last** provider comes back, so this is
    the latest of their earliest serve times; any provider whose vendor named
    no reset makes the end unknown.

    Args:
        config: The live config, for the provider table.
        ctx: The request's selection state, for the route.
        names: The candidates the question is about.  ``None`` asks about the
            whole route; the caller passes what a failed-over request can
            still use.
    """
    from ..quota.service import provider_exhausted_until

    ends: list[datetime] = []
    for entry in ctx.candidates:
        if names is not None and entry.name not in names:
            continue
        pinfo = config.providers.get(entry.name)
        if pinfo is None:
            return None
        moment = provider_exhausted_until(pinfo)
        if moment is None:
            return None
        ends.append(moment)
    return max(ends) if ends else None


def classify(config: GatewayConfig, ctx: SelectionContext) -> Unavailable:
    """Say why no candidate was chosen.

    Only meaningful after the selection chain left ``ctx.chosen`` unset.

    Args:
        config: The live config, whose provider table the readiness check reads.
        ctx: The request's selection state, already run through the chain.

    Returns:
        The state to report, in the order documented on the module.
    """
    from ..maintenance.startup import ProviderInitState

    # The route as this request can still use it: a provider it already failed
    # over from is not "available", so answering "every provider" over the full
    # list would describe a route that includes the one that just refused.
    # Everything else is whole-route -- ``len(unready) == len(remaining)`` and
    # ``names <= quota_excluded`` are "every provider is blocked" / "... spent"
    # -- and both were asked that way by ``resolve`` before this module
    # existed.  A list filtered by anything else here would make them answer
    # about that filtered set instead.
    remaining = [p for p in ctx.candidates if p.name not in ctx.failover_excluded]
    unready = [
        config.providers[p.name]
        for p in remaining
        if not config.providers[p.name].ready
    ]

    end = blocked_until(unready)
    if end is not None and len(unready) == len(remaining):
        return Unavailable("blocked", end)

    names = {p.name for p in remaining}
    if names and names <= set(ctx.quota_excluded):
        return Unavailable("exhausted", exhausted_until(config, ctx, names))

    deferred = getattr(config, "_deferred_startup", None)
    failed = deferred is not None and all(
        deferred.provider_state(p.name) == ProviderInitState.FAILED for p in unready
    )
    return Unavailable("failed" if failed else "initializing")
