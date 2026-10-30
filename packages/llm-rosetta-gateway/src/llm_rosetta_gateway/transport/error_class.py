"""How a terminal status is worded in every client dialect.

Two envelopes answer the same question — the JSON error body and the in-band
terminal event of a stream — and a client that asks with ``stream: true`` must
not get a different verdict from the one that asks without.  The verdict has
three cells, and each dialect spells them in its own words:

===========  ========================  ===================  ===============
status       meaning                   OpenAI               Anthropic
===========  ========================  ===================  ===============
< 500        the client's request      ``invalid_request_  ``invalid_request_
             was wrong                 error``              error``
500–501      the gateway broke         ``server_error``     ``api_error``
>= 502       the upstream produced     ``api_error``        ``api_error``
             what the gateway could
             not carry
===========  ========================  ===================  ===============

Anthropic has no ``server_error`` at all — every 5xx is ``api_error``, "an
unexpected error has occurred internal to Anthropic's systems" — which is why
the gateway's own fault and the upstream's read the same there.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalClass:
    """One status cell, worded for every dialect that has a word for it.

    Attributes:
        kind: ``"client"``, ``"gateway"`` or ``"upstream"``.
        openai_type: OpenAI's ``error.type`` (Responses uses the same word for
            its in-band ``error.code``).
        anthropic_type: Anthropic's ``error.type``.
        google_status: Google's ``error.status``.
    """

    kind: str
    openai_type: str
    anthropic_type: str
    google_status: str


#: The request was fine; the client has something to fix.
CLIENT = TerminalClass(
    "client", "invalid_request_error", "invalid_request_error", "INVALID_ARGUMENT"
)
#: The gateway itself failed.
GATEWAY = TerminalClass("gateway", "server_error", "api_error", "INTERNAL")
#: The upstream produced something the gateway could not turn into a reply.
UPSTREAM = TerminalClass("upstream", "api_error", "api_error", "UNAVAILABLE")


def classify_status(status_code: int) -> TerminalClass:
    """Return the cell *status_code* belongs to.

    Args:
        status_code: The HTTP status the failure would have carried.

    Returns:
        The :class:`TerminalClass` both envelopes must agree on.
    """
    if status_code < 500:
        return CLIENT
    if status_code < 502:
        return GATEWAY
    return UPSTREAM
