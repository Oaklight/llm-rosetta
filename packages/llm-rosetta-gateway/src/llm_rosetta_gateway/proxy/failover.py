"""Cross-provider failover when the serving provider's allowance runs out.

An in-band 429 demotes the credential that earned it.  When that leaves the
provider with nothing to retry on, the request usually still has somewhere to
go: the model's other providers.  Trying one of them is a product decision
rather than a routing one — the second provider is a different prompt cache, a
different rate-limit bucket and possibly a different model build — so it is
opt-in per model (``failover_on_quota`` on the model entry) and spent at most
once per request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_rosetta.routing import ResolvedRoute

    from ..config import GatewayConfig
    from ..routing import SelectionContext
    from ..transport import ProviderInfo


def reselect_after_quota(
    config: GatewayConfig,
    selection: SelectionContext,
    *,
    exclude: str,
) -> tuple[ResolvedRoute, ProviderInfo] | None:
    """Run selection again with *exclude* out of the running.

    The provider that refused is excluded for the rest of the request, and the
    previous choice is voided with it, so a stage that finds nothing cannot
    leave the refused provider standing.

    Args:
        config: The live config, whose tables ``resolve`` needs.
        selection: This request's selection state, already installed with the
            config's tables.
        exclude: The provider that just refused.

    Returns:
        The new route and transport config, or ``None`` when nothing else can
        serve — in which case the caller reports the upstream's own refusal
        rather than inventing a different story.
    """
    from ..maintenance.startup import ProviderNotReady

    selection.with_failover_excluded(selection.failover_excluded | {exclude})
    try:
        return config.resolve(selection)
    except (ProviderNotReady, KeyError):
        return None
