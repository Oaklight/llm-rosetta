"""The bookkeeping of one upstream attempt (#1830 A4).

Every path that sends a request upstream counts the attempt the same way: it
files it under a lane, clears that lane's record when the upstream answered,
and sinks the lane when the host did not.  The JSON and streaming chat paths,
the typed pipelines and the relayed ``compact`` / ``count_tokens`` routes used
to spell that out each for itself.  :func:`attempt_upstream` is the one
place; a site holds the :class:`UpstreamAttempt` it returns and reads the lane
and the rate-limit rule from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..observability.logging import get_logger
from .errors import note_transport_refusal
from .failover import RoutedModel, attempted_lane

if TYPE_CHECKING:
    from ..routing import SelectionContext
    from ..transport import ProviderInfo

logger = get_logger()


@dataclass(frozen=True, slots=True)
class UpstreamAttempt:
    """One attempt on one lane, and the two ways it is recorded.

    Attributes:
        provider_info: The provider the request was sent to.
        lane: The lane's key, as :func:`proxy.failover.attempted_lane` builds
            it.
        rate_limit: The rate-limit rule in force, or ``None`` for the default.
    """

    provider_info: ProviderInfo
    lane: str
    rate_limit: Any | None

    def served(self) -> None:
        """Clear the ladder entries a served request is evidence about.

        Bookkeeping must not fail a served request, so a recording error is
        logged and not raised.
        """
        from ..quota.service import note_upstream_success

        try:
            note_upstream_success(self.provider_info, lane=self.lane)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail a served request
            logger.warning("Could not clear the transient refusals", exc_info=True)

    def unreachable(self) -> None:
        """Count a host that did not answer against this lane.

        For the paths that answer the failure themselves; the chat paths hand
        the same count to failover through ``connection_refusal_decision``.
        """
        note_transport_refusal(
            self.provider_info, rate_limit=self.rate_limit, lane=self.lane
        )


def attempt_upstream(
    config: Any,
    provider_info: ProviderInfo,
    *,
    route: RoutedModel,
    model: str,
    selection: SelectionContext | None = None,
) -> UpstreamAttempt:
    """Open the record of one upstream attempt.

    Args:
        config: The gateway config; its ``routing_rate_limit`` is the rule.
        provider_info: The provider the request is sent to.
        route: The route the attempt uses.  A typed route has no selection.
        model: The model the client asked for, the lane key when the
            candidate names no upstream model.
        selection: This request's selection state, or ``None`` when the
            caller has none.

    Returns:
        The attempt, carrying its lane key and the rule.
    """
    return UpstreamAttempt(
        provider_info=provider_info,
        lane=attempted_lane(selection, route, model),
        rate_limit=getattr(config, "routing_rate_limit", None),
    )
