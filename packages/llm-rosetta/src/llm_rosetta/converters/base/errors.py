"""Errors a writer raises when its wire cannot express the IR it was given.

A conversion is not an isomorphism: a dialect may have no form for a field,
a part or a terminal state.  Most of those losses are reported as warnings —
the item is dropped and the rest of the turn still goes out.  A *terminal
state* is different: a response that ended because the upstream failed
cannot be written as a normal turn without telling the client the request
succeeded.  Those raise, and the error carries the payload a caller needs to
log, dump and map it instead of parsing a message string.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UnrepresentableFinishReason(ValueError):
    """A terminal turn this dialect has no way to express.

    Chat Completions, for instance, has no finish reason for a generation
    that ended in an error or was cancelled.  The alternatives to raising are
    equally wrong: ``stop`` hides a failed request, and inventing a reason
    would put a word on the wire the vendor never sends.

    The payload rides as attributes so a gateway can dump it and choose a
    status code without ``json.loads`` on a string.

    Attributes:
        reason: The IR finish reason, e.g. ``"error"`` or ``"cancelled"``.
        response_id: The response the reason belongs to.
        provider_metadata: Upstream metadata recorded with the reason.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        response_id: str = "",
        provider_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the error from its message and the reason it is about."""
        self.reason = reason
        self.response_id = response_id
        self.provider_metadata = dict(provider_metadata or {})
        super().__init__(message)
