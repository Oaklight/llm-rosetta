"""The conversation a request belongs to, derived from the request itself.

The admin routing page shows one line per conversation, so the request log has
to carry an identity that is the same on every turn of one conversation and
different between conversations.  This module derives it from the client that
sent the request and the first message of its history — the same seed the
upstream session identity uses, and for the same reason: history is
append-only, so the first message is identical on every turn where a longer
prefix would change as the conversation grows.

Only the digest is written to the log.  The seed is not stored and the digest
is not reversible: it names a conversation, it does not describe one.

A request whose client or first message cannot be read derives no id at all.
Readers then fall back to the grouping they used before this value existed
rather than inventing an identity two different conversations could share.
"""

from __future__ import annotations

import contextvars
import hashlib
from typing import TYPE_CHECKING, Any

from .prefix import compute_affinity_identity, extract_prefix_from_ir
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
    """Derive the conversation id for one request.

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

    Runs for every chat request, not only for the providers that declare a
    session header: the request log records the id whatever lane served the
    request, so the routing page can group by conversation across lanes.
    """

    __slots__ = ()

    name = "conversation_identity"
    requires = frozenset({"identity", "ir_request"})
    provides = frozenset({"identity"})

    def run(self, ctx: SelectionContext) -> None:
        """Set the conversation id on the selection context."""
        ctx.with_conversation(
            conversation_id(ctx.identity.client, ctx.ir_request),
        )


def note_conversation(value: str | None) -> None:
    """Publish the request's conversation id for the log row to read."""
    conversation_context_var.set(value)


def current_conversation() -> str | None:
    """Return the conversation id of the request being served, if any."""
    return conversation_context_var.get()
