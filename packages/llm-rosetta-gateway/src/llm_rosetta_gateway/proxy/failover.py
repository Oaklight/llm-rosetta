"""Cross-provider failover when the serving provider's allowance runs out.

An in-band 429 demotes the credential that earned it.  When that leaves the
provider with nothing to retry on, the request usually still has somewhere to
go: the model's other providers.  Trying one of them is a product decision
rather than a routing one — the second provider is a different prompt cache, a
different rate-limit bucket and possibly a different model build — so it is
opt-in per model (``failover_on_quota`` on the model entry) and spent at most
once per request.

Both handlers use this module the same way: ask :func:`prepare_failover`, and
when it returns a target, run the whole request again against it and record the
outcome with :func:`record_failover`.  The streaming handler reaches this
before it has written anything to the client, which is the only point at which
a different answer can still be given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..routing.failover import reselect_after_refusal

if TYPE_CHECKING:
    from llm_rosetta.routing import ResolvedRoute

    from ..config import GatewayConfig
    from ..routing import SelectionContext
    from ..transport import ProviderInfo


def prepare_failover(
    config: GatewayConfig | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    signal: Any,
    attempted: str | None,
) -> tuple[ResolvedRoute, ProviderInfo, dict[str, Any]] | None:
    """Decide whether this request may be tried on another provider.

    Spent at most once per request: *attempted* names the provider the first
    attempt was made on, so the second attempt cannot fail over again.  The
    model must have opted in — a different provider is a different prompt
    cache and possibly a different model build.

    Returns:
        ``(route, provider_info, body)`` for the second attempt, with the body
        carrying the new provider's upstream model name, or ``None`` when
        there will not be a second attempt.
    """
    from ..logging import get_logger

    if config is None or attempted is not None:
        return None
    if selection.model not in getattr(config, "model_failover_on_quota", {}):
        return None
    target = reselect_after_refusal(
        selection, exclude=route.provider_name, resolve=config.resolve
    )
    if target is None:
        return None
    new_route, new_pinfo = target
    reason = getattr(signal, "reason", None) or "quota refusal"
    get_logger().warning(
        "Quota failover for model %s: %s -> %s (%s)",
        selection.model,
        route.provider_name,
        new_route.provider_name,
        reason,
    )
    retry_body = dict(body)
    # The client's model name may have been replaced by the first provider's
    # upstream alias; the second provider gets its own, or the original name.
    retry_body["model"] = new_route.upstream_model or selection.model
    return new_route, new_pinfo, retry_body


def record_failover(
    profile: dict[str, Any],
    *,
    route: ResolvedRoute,
    new_route: ResolvedRoute,
    signal: Any,
    sub_profile: dict[str, Any],
) -> None:
    """Write the failover into the profile, and point the log at the server.

    The profile entry is what the request log carries, so a reader can see
    that the answer came from the second provider; the lifecycle row must name
    the provider whose response the client actually received.
    """
    from ..lifecycle import current_lifecycle
    from ..logging import get_logger

    try:
        profile["failover"] = {
            "from": route.provider_name,
            "to": new_route.provider_name,
            "reason": getattr(signal, "reason", None) or "quota refusal",
            "profile": sub_profile,
        }
        ctx = current_lifecycle()
        if ctx is not None:
            ctx.note_route(
                target_format=new_route.target_provider,
                provider_name=new_route.provider_name,
            )
    except Exception:  # noqa: BLE001 - bookkeeping must not mask the outcome
        # This runs from a ``finally`` on the failure path, where an exception
        # here would replace the client's error with a bookkeeping one.  The
        # answer always wins; the record is best-effort.
        get_logger().warning("Could not record the quota failover", exc_info=True)
