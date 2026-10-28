"""Helpers shared by the stream readers.

A finish reason can arrive in a chunk that carries no content at all — Google
answer in an empty final chunk, and Chat's finish reason rides a chunk with
deltas but no call.  The turn's meaning therefore has to come from what the
stream already opened, which is what the stream context records.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def stream_opened_client_call(
    context: Any,
    events: Iterable[Any],
    since: int = 0,
) -> bool:
    """Whether the stream opened a tool call the client owes an answer to.

    An intrinsic (provider-hosted) call is executed on the provider's side, so
    it does not make the turn a tool-call turn.  The stream context knows every
    call the response opened; without one, only the events produced since
    *since* are visible.

    Args:
        context: The stream context, or ``None`` when the caller keeps none.
        events: Events emitted while reading the current chunk.
        since: Index in *events* where the chunk's own parts start.

    Returns:
        ``True`` when at least one non-intrinsic call was opened.
    """
    if context is not None:
        call_ids: Sequence[str] = context.tool_call_ids
        if call_ids:
            return any(
                context.get_tool_type(call_id) != "intrinsic" for call_id in call_ids
            )
    return any(
        event.get("type") == "tool_call_start" and event.get("tool_type") != "intrinsic"
        for event in list(events)[since:]
    )
