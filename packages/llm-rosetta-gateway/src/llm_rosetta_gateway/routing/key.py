"""Upstream key selection: keep one conversation on one key.

Rendezvous hashing over the provider's key ring, keyed on a value derived from
the client and the request's leading messages, so the same conversation from
the same client keeps reaching the same upstream key — and with it the upstream
prompt cache — while a key-ring change moves only about one mapping in *n*.

Two messages rather than one here: key selection wants the entropy, and unlike
the session identity below nothing depends on this value surviving a turn, so a
longer prefix that changes as the conversation grows is the better trade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prefix import compute_affinity_identity, extract_prefix_from_ir
from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext


class ProviderKeyAffinity(RoutingStage):
    """Pin a conversation to one of the provider's keys.

    Declines when the provider has a single key (nothing to choose), when no
    client identity is available (no stable value to pin on), or when the
    provider turned the feature off — in which case ``auth_headers()`` falls
    back to round-robin as it would have anyway.
    """

    __slots__ = ()

    name = "provider_key_affinity"
    requires = frozenset({"identity", "ir_request", "provider_info"})
    provides = frozenset({"provider_info"})

    def run(self, ctx: SelectionContext) -> None:
        """Set the affinity identity on the provider's transport config."""
        provider_info = ctx.provider_info
        if provider_info is None:
            return
        if not (
            ctx.key_affinity and ctx.identity.client and len(provider_info.key_ring) > 1
        ):
            return
        identity = compute_affinity_identity(
            ctx.identity.client, extract_prefix_from_ir(ctx.ir_request)
        )
        if identity is not None:
            ctx.with_provider_info(provider_info.with_affinity(identity))
