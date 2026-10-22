"""How much allowance each candidate has left, for the stages after it.

Quota is not a selection strategy: it is a *fact* about the candidates that the
strategy weighs.  Keeping it a stage means the strategies stay what they are —
weighted round-robin still round-robins, client affinity still hashes — and the
only thing that changes is the weight they are given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext


class QuotaStage(RoutingStage):
    """Record the smallest remaining share per candidate provider.

    Reads the quota state the probe layer already keeps; costs one dictionary
    lookup per credential.  A provider nothing has measured is recorded as
    ``1.0`` — it is not punished for being unmeasured, and a lane with no probe
    at all behaves exactly as it did before this stage existed.
    """

    __slots__ = ()

    name = "quota"
    requires = frozenset({"model_route", "providers"})
    provides = frozenset({"quota_headroom"})

    def run(self, ctx: SelectionContext) -> None:
        """Fill ``ctx.quota_headroom`` for every candidate."""
        from ..quota.service import provider_headroom

        headroom: dict[str, float] = {}
        for entry in ctx.candidates:
            pinfo = ctx.providers.get(entry.name)
            share = provider_headroom(pinfo) if pinfo is not None else None
            headroom[entry.name] = 1.0 if share is None else share
        ctx.with_quota_headroom(headroom)
