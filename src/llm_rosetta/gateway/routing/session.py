"""The session identity an upstream is told, derived from the request.

A shim may declare a session header (see :attr:`ConnectionConfig.session_header`)
that its upstream requires on every inference request and uses to route the
request and to keep a prompt-cache-affine backend for the conversation.  The
value has to be *the same on every turn of one conversation*, which is why it
is derived from the first message alone: history is append-only, so the first
message is identical on every turn, where a two-message prefix would change
between turn one and turn two and hand the opening request a different session
than the rest of the conversation.

Nothing here is stored.  The value is recomputed from the request every time,
so any worker, and any restart, derives the same one.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from .prefix import compute_affinity_identity, extract_prefix_from_ir
from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext

# Stable per-process fallback: used only when a provider declares a session
# header but no per-conversation identity could be derived, so the header is
# never absent (some upstreams reject a missing session with 400).  Every such
# request shares this one id — acceptable for the fallback, since a request
# with neither a client key nor a message prefix carries nothing to key a
# conversation on.
_SESSION_FALLBACK = uuid.uuid4().hex


class SessionHeaderAffinity(RoutingStage):
    """Tell the upstream which conversation the request belongs to."""

    __slots__ = ()

    name = "session_header"
    requires = frozenset({"identity", "ir_request", "provider_info"})
    provides = frozenset({"provider_info", "upstream_session_id"})

    def run(self, ctx: SelectionContext) -> None:
        """Derive the session id and set it on the provider's transport config."""
        provider_info = ctx.provider_info
        if provider_info is None or not provider_info.session_header:
            return
        first_prefix = extract_prefix_from_ir(ctx.ir_request, max_messages=1)
        seed = (
            compute_affinity_identity(ctx.identity.client or "", first_prefix)
            or first_prefix
            or _SESSION_FALLBACK
        )
        session_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
        ctx.with_upstream_session_id(session_id)
        ctx.with_provider_info(provider_info.with_session(session_id))
