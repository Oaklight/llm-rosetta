"""Why nothing was chosen, in the words the client needs.

``resolve`` asks this module once, after selection left ``ctx.chosen`` unset,
instead of re-deriving the four states inline.  The order of the checks is the
rule:

1. a **rate-limit disable** covering every candidate the request can still use
   — the credential ladder's 429s or a lane's capacity ladder — is temporary
   and dated, and it is asked first because ``ready`` cannot tell it apart
   from a quota block: the client is told when the lane comes back;
2. a **quota block** covering every provider the request can still use has a
   known end and must not be reported as startup — a client that just
   received a plan refusal would go looking for a boot problem that does not
   exist;
3. a route every remaining candidate of which a **reading** removed is
   ``exhausted``, with the reset when the vendor named one and without one
   when it did not;
4. a route every remaining candidate of which **lacks the capability** this
   request needs is ``capability``;
5. otherwise the route is down for a startup reason: ``failed`` when every
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
    from collections.abc import Mapping

    from ..config.core import GatewayConfig
    from .context import SelectionContext

#: ``rate_limited`` and ``blocked`` are temporary and dated; ``exhausted`` is
#: a spent allowance; ``failed`` and ``initializing`` are startup states;
#: ``capability`` means every candidate serves the model but not this request.
Reason = Literal[
    "rate_limited", "blocked", "exhausted", "failed", "initializing", "capability"
]


@dataclass(frozen=True)
class Unavailable:
    """The state of a route no candidate could serve.

    Attributes:
        reason: Which of the stories this is.
        until: When the route comes back, for ``rate_limited`` and the other
            dated reasons.  ``None`` on ``exhausted`` means the vendor named no
            reset, and the gateway will not invent one.
    """

    reason: Reason
    until: datetime | None = None
    #: For ``capability``: the capabilities no candidate declared.
    unsupported: tuple[str, ...] = ()

    def as_kwargs(self) -> dict[str, Any]:
        """The keyword arguments ``ProviderNotReady`` takes for this state."""
        if self.reason == "capability":
            return {"unsupported": self.unsupported}
        if self.reason == "rate_limited":
            return {
                "rate_limited_until": self.until,
                "rate_limited": self.until is None,
            }
        if self.reason == "blocked":
            return {"blocked_until": self.until}
        if self.reason == "exhausted":
            return {"exhausted_until": self.until, "exhausted": self.until is None}
        return {"failed": self.reason == "failed"}


def blocked_until(unready: list[Any]) -> datetime | None:
    """The end of a block covering every unready provider, if any.

    Returns the **latest** end, because the route is unavailable until the last
    of its providers is back.  A provider that is unready for any other reason,
    or a block the state cannot date, makes the answer unknown — ``None``.

    A block is either kind the state records: a quota block from an in-band
    refusal, or a rate-limit disable from a run of 429s.  Both name an end, so
    both let the client be told "blocked until" rather than "initializing".
    """
    from ..quota.service import provider_offline_until

    ends: list[datetime] = []
    for pinfo in unready:
        moment = provider_offline_until(pinfo)
        if moment is None:
            return None
        ends.append(moment)
    return max(ends) if ends else None


def exhausted_until(
    config: GatewayConfig,
    ctx: SelectionContext,
    names: set[str] | None = None,
    *,
    providers: Mapping[str, Any] | None = None,
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
        providers: The provider table to read, when it is not the chat pool —
            a typed route keeps its own.
    """
    from ..quota.service import provider_exhausted_until

    table = config.providers if providers is None else providers
    ends: list[datetime] = []
    for entry in ctx.candidates:
        if names is not None and entry.name not in names:
            continue
        pinfo = table.get(entry.name)
        if pinfo is None:
            return None
        moment = provider_exhausted_until(pinfo)
        if moment is None:
            return None
        ends.append(moment)
    return max(ends) if ends else None


def _latest_until(answers: list[Any]) -> datetime | None:
    """The latest end among the answers, or ``None`` when one has none.

    The route is unavailable until the **last** candidate comes back, so the
    latest end is the answer — unless a candidate cannot date its own, in
    which case the end is unknown and this module does not invent one.
    """
    ends = [answer.until for answer in answers if answer is not None]
    if not ends or any(end is None for end in ends):
        return None
    return max(ends)


def classify(
    config: GatewayConfig,
    ctx: SelectionContext,
    *,
    providers: Mapping[str, Any] | None = None,
) -> Unavailable:
    """Say why no candidate was chosen.

    Only meaningful after the selection chain left ``ctx.chosen`` unset.

    Args:
        config: The live config, whose provider table the readiness check reads.
        ctx: The request's selection state, already run through the chain.
        providers: The provider table to read, when it is not the chat pool —
            the typed pools keep their own, and a typed model is answered
            about its own candidates.

    Returns:
        The state to report, in the order documented on the module.
    """
    from ..maintenance.startup import ProviderInitState

    table = config.providers if providers is None else providers

    # The route as this request can still use it: a provider it already failed
    # over from is not "available", and neither is a lane it was refused by, so
    # answering "every provider" over the full list would describe a route that
    # includes what just refused.  Everything else is whole-route --
    # ``len(unready) == len(remaining)`` and ``names <= quota_excluded`` are
    # "every provider is blocked" / "... spent" -- and both were asked that way
    # by ``resolve`` before this module existed.  A list filtered by anything
    # else here would make them answer about that filtered set instead.
    remaining = [
        p
        for p in ctx.candidates
        if p.name not in ctx.failover_excluded and p.key not in ctx.lane_excluded
    ]

    # A rate-limit disable — the credential ladder's 429s, or the capacity
    # ladder of one upstream lane — is temporary and dated.  Asked before the
    # block check because ``ready`` cannot tell the two apart: a disabled
    # provider is unready, so the block check would call it a quota block
    # ("plan refusal") when a ladder is what actually holds it out.
    from .eligibility import RATE_LIMITED

    answers = [ctx.eligibility.get(p.key) for p in remaining]
    if remaining and all(
        answer is not None and answer.reason == RATE_LIMITED for answer in answers
    ):
        return Unavailable("rate_limited", _latest_until(answers))

    unready = [
        table[p.name] for p in remaining if p.name in table and not table[p.name].ready
    ]

    end = blocked_until(unready)
    if end is not None and len(unready) == len(remaining):
        return Unavailable("blocked", end)

    names = {p.name for p in remaining}
    if names and names <= set(ctx.quota_excluded):
        return Unavailable(
            "exhausted", exhausted_until(config, ctx, names, providers=table)
        )

    from .eligibility import CAPABILITY

    if remaining and all(
        answer is not None and answer.reason == CAPABILITY
        for answer in (ctx.eligibility.get(p.key) for p in remaining)
    ):
        return Unavailable(
            "capability", unsupported=tuple(sorted(ctx.required_capabilities))
        )

    deferred = getattr(config, "_deferred_startup", None)
    failed = deferred is not None and all(
        deferred.provider_state(p.name) == ProviderInitState.FAILED for p in unready
    )
    return Unavailable("failed" if failed else "initializing")
