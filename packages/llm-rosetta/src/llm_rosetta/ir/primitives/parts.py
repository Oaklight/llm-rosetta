"""Walk an IR response body for the call parts a pass rewrites.

A response body can carry calls in two places — each choice's ``message`` and
the ``messages`` list an upstream hands back — and more than one pass walks
both (the tool-name restore and the custom-tool mark).  The walk lives here so
that a change to where calls live is made once.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_response_messages(ir_response: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every message dict of a response, from choices and messages.

    Args:
        ir_response: An IR response body.

    Yields:
        Each message, choices first, in the order the response carries them.
    """
    for choice in ir_response.get("choices") or []:
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                yield message
    for message in ir_response.get("messages") or []:
        if isinstance(message, dict):
            yield message


def iter_tool_call_parts(
    ir_response: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield every ``tool_call`` part in every message of *ir_response*.

    Args:
        ir_response: An IR response body.

    Yields:
        Each tool-call part, in message order.
    """
    for message in iter_response_messages(ir_response):
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_call":
                yield part
