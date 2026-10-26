"""The IR stop reason for a turn that carries client tool calls.

Not every provider has a stop reason of its own for a tool call.  Anthropic
reports ``stop_reason: "tool_use"`` and the Interactions API ``status:
"requires_action"``, so both map to the IR's ``tool_calls`` directly.  OpenAI's
Responses API answers ``status: "completed"`` with a ``function_call`` output
item, and Google generateContent answers ``finishReason: "STOP"`` with a
``functionCall`` part: read literally, both say the model finished talking.

The IR is the hub, so the rule belongs here: a turn that finished normally but
carries tool calls the client owes an answer to is a ``tool_calls`` turn.  Every
source reader that lacks an explicit reason of its own applies it, and every
target writer then reads the same reason — Chat's ``finish_reason:
"tool_calls"``, Interactions' ``requires_action`` — without a per-dialect
compensation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypeVar

from llm_rosetta.types.ir import is_tool_call_part
from ..tools.intrinsic import is_intrinsic_part

ReasonT = TypeVar("ReasonT", bound=str)


def client_tool_calls_in(parts: Iterable[Any]) -> bool:
    """Return whether *parts* hold a tool call the client has to answer.

    An intrinsic (provider-hosted) call is executed on the provider's side, so
    the turn is finished rather than waiting on the client.

    Args:
        parts: IR content parts of an assistant message, or stream events.

    Returns:
        ``True`` when at least one non-intrinsic tool call is present.
    """
    return any(
        is_tool_call_part(part) and not is_intrinsic_part(part) for part in parts
    )


def reason_for_completed_turn(
    reason: ReasonT, has_client_tool_calls: bool
) -> ReasonT | Literal["tool_calls"]:
    """Add the tool-call meaning a normal stop does not carry.

    Args:
        reason: The reason the source dialect reported, already in IR terms.
        has_client_tool_calls: Whether the turn carries a tool call the client
            must answer.

    Returns:
        ``"tool_calls"`` when the turn finished normally while owing the client
        an answer; *reason* unchanged otherwise.  A length stop, a content
        filter or a failure keeps its own meaning — the client is not being
        asked to run anything.
    """
    if reason == "stop" and has_client_tool_calls:
        return "tool_calls"
    return reason
