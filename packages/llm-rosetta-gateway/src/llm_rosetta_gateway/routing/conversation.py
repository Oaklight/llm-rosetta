"""The conversation a request belongs to, for the routing page and the lanes.

The identity itself lives in
:mod:`~llm_rosetta_gateway.routing.session_identity`, which reads the carrier a
client sets before falling back to a guess.  This module is the stage that puts
it on the selection context, and the ``ContextVar`` the request log reads after
the handler returns.

The stage runs for every chat request, not only for providers that declare a
session header: the request log records the id whatever lane served the
request, so the routing page can group by conversation across lanes.  Only the
digest is published.  The carrier is not stored and the digest is not
reversible: it names a conversation, it does not describe one.
"""

from __future__ import annotations

import contextvars
import hashlib
from typing import TYPE_CHECKING, Any

from .prefix import compute_affinity_identity, extract_prefix_from_ir
from .session_identity import SessionIdentity, identify
from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext

#: The conversation id of the request being served, once one has been
#: derived.  ``None`` on a request that carries nothing to key a conversation
#: on, and on every typed (non-chat) request, which derives none.
conversation_context_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_rosetta_gateway_conversation", default=None
)


def conversation_id(
    client_key_hash: str | None,
    ir_request: dict[str, Any] | None,
) -> str | None:
    """Derive the conversation id for one request from its converted body.

    A fallback for callers that hold no raw request body — the typed routes,
    and the stages' own tests.  A real chat request is identified from the
    carrier the client set, which this function cannot see.

    Args:
        client_key_hash: The authenticated client's key hash, or ``None``
            when the request carried no key.
        ir_request: The request converted to the IR, or ``None`` before one
            exists.

    Returns:
        A 32-character hex digest, or ``None`` when the request has no client
        or no first message to key on.
    """
    seed = compute_affinity_identity(
        client_key_hash or "",
        extract_prefix_from_ir(ir_request, max_messages=1),
    )
    if not seed:
        return None
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


class ConversationIdentity(RoutingStage):
    """Record which conversation the request belongs to.

    A carrier the client set wins; failing one, the request's own opening is
    hashed, and the source says so.  A consumer that stores state must read
    :attr:`SessionIdentity.declared` — two conversations in one project can
    share an opening, so a derived id may name both.
    """

    __slots__ = ()

    name = "conversation_identity"
    requires = frozenset({"identity", "ir_request"})
    provides = frozenset({"identity"})

    def run(self, ctx: SelectionContext) -> None:
        """Set the session identity on the selection context."""
        fallback = conversation_id(ctx.identity.client, ctx.ir_request)
        ctx.with_session(
            identify(
                ctx.headers,
                ctx.source_body,
                body_fallback=SessionIdentity(fallback, "derived:opening")
                if fallback is not None
                else None,
                client=ctx.identity.client,
            )
        )


def note_conversation(value: str | None) -> None:
    """Publish the request's conversation id for the log row to read."""
    conversation_context_var.set(value)


def current_conversation() -> str | None:
    """Return the conversation id of the request being served, if any."""
    return conversation_context_var.get()
