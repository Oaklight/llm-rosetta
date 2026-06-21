"""Orphaned tool call/result fix utilities (IR level).

Shared helpers that fix mismatches between tool calls and tool results
in IR messages.  Used by all 4 provider converters to ensure
bidirectional pairing before downstream conversion.

Also provides ``strip_orphaned_tool_config`` which removes ``tool_choice``
/ ``tool_config`` when no tool definitions are present (Codex CLI context
compaction workaround).
"""

import logging
from collections.abc import Sequence
from typing import Any, cast

from ....types.ir import Message
from ....types.ir.request import IRRequest

logger = logging.getLogger(__name__)


# ==================== Shared Helpers ====================


def extract_part_ids(parts: Sequence[Any], part_type: str, id_key: str) -> set[str]:
    """Extract IDs from content parts/items matching a given type.

    Generic helper reused across IR, Anthropic, and OpenAI Responses
    orphaned-tool-call fixers.

    Args:
        parts: List of dicts (content parts, blocks, or flat items).
        part_type: Value of the ``type`` key to match.
        id_key: Key whose value is collected as an ID.

    Returns:
        Set of extracted ID strings.
    """
    return {
        part[id_key]
        for part in parts
        if isinstance(part, dict) and part.get("type") == part_type and part.get(id_key)
    }


def log_orphan_warnings(
    log: logging.Logger,
    call_ids: Sequence[str],
    result_ids: Sequence[str],
    call_label: str = "tool_call",
    result_label: str = "tool_result",
) -> None:
    """Log warnings for orphaned tool calls and results.

    Shared by all per-format ``fix_orphaned_tool_calls`` functions.

    Args:
        log: Logger instance to emit warnings on.
        call_ids: IDs of orphaned calls that were patched.
        result_ids: IDs of orphaned results that were removed.
        call_label: Human-readable label for call items (e.g. "tool_use").
        result_label: Human-readable label for result items.
    """
    if call_ids:
        log.warning(
            "Fixed %d orphaned %s(s) by injecting synthetic results: %s",
            len(call_ids),
            call_label,
            ", ".join(call_ids),
        )
    if result_ids:
        log.warning(
            "Removed %d orphaned %s(s) with no matching %s: %s",
            len(result_ids),
            result_label,
            call_label,
            ", ".join(result_ids),
        )


# ==================== IR-Level Orphan Fix ====================


def _collect_ir_tool_ids(
    messages: Sequence[Message],
) -> tuple[set[str], set[str]]:
    """Collect all tool_call IDs and answered (tool_result) IDs from IR messages."""
    known_call_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg in messages:
        content = msg.get("content", [])
        role = msg.get("role")
        if role == "assistant":
            known_call_ids |= extract_part_ids(content, "tool_call", "tool_call_id")
        elif role == "tool":
            answered_ids |= extract_part_ids(content, "tool_result", "tool_call_id")
    return known_call_ids, answered_ids


def fix_orphaned_tool_calls_ir(
    messages: Sequence[Message],
    *,
    placeholder: str = "[No output available yet]",
) -> list[Message]:
    """Fix mismatched tool_calls and tool results at IR level.

    Both the OpenAI Chat Completions API and the Responses API **strictly
    require** bidirectional pairing between tool calls and tool results:

    1. Every ``tool_call_id`` in an assistant message must have a matching
       ``role: "tool"`` result message (**orphaned tool_call**).
    2. Every ``role: "tool"`` result message must have a preceding assistant
       message containing the matching ``tool_call_id``
       (**orphaned tool_result**).

    Anthropic enforces the same strict pairing; only Google Gemini is
    lenient.  This function patches IR messages so that downstream
    converters produce valid output for any target provider.

    This function handles both directions:

    - **Orphaned tool_calls**: injects a synthetic ``role: "tool"`` IR
      message with *placeholder* content immediately after the assistant
      message.
    - **Orphaned tool_results**: removes ``role: "tool"`` messages whose
      ``tool_call_id`` does not appear in any preceding assistant
      ``tool_call`` content part.

    The original iterable is **not** modified; a new list is returned.

    Args:
        messages: IR messages (any iterable of Message dicts).
        placeholder: Content string for injected synthetic tool results.

    Returns:
        A new messages list with orphaned tool_calls/results fixed.
    """
    msg_list = list(messages)

    known_call_ids, answered_ids = _collect_ir_tool_ids(msg_list)

    # Fast path: nothing to fix
    if not known_call_ids and not answered_ids:
        return msg_list

    patched: list[Message] = []
    orphaned_call_ids: list[str] = []
    orphaned_result_ids: list[str] = []

    for msg in msg_list:
        role = msg.get("role")
        content = msg.get("content", [])

        # Remove orphaned tool results (result without preceding tool_call)
        if role == "tool":
            result_ids = extract_part_ids(content, "tool_result", "tool_call_id")
            if result_ids and result_ids.isdisjoint(known_call_ids):
                orphaned_result_ids.extend(result_ids)
                continue

        patched.append(msg)

        # Inject synthetic results for orphaned tool_calls
        if role == "assistant":
            unanswered = (
                extract_part_ids(content, "tool_call", "tool_call_id") - answered_ids
            )
            for tc_id in unanswered:
                orphaned_call_ids.append(tc_id)
                patched.append(
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": tc_id,
                                "result": placeholder,
                            }
                        ],
                    }
                )

    log_orphan_warnings(logger, orphaned_call_ids, orphaned_result_ids)
    return patched


# ==================== Orphaned Tool Config Fix (IR level) ====================


def strip_orphaned_tool_config(ir_request: IRRequest) -> list[str]:
    """Strip ``tool_choice`` and ``tool_config`` when no tools are defined.

    Codex CLI context compaction can remove all tool definitions from a
    request while keeping ``tool_choice`` (e.g. ``"auto"``).  This produces
    an invalid request that upstream APIs reject with *"tool_choice is set
    but no tools are provided"*.

    This is part of the same problem family as
    :func:`fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing)
    and ``_reorder_tool_messages`` (tool message ordering) — all stem from
    Codex context compaction breaking request structural integrity.

    The request dict is modified **in-place**.

    Args:
        ir_request: IR request dict (mutated in-place).

    Returns:
        List of warning strings for each stripped field.
    """
    tools = ir_request.get("tools")
    has_tools = bool(tools)

    if has_tools:
        return []

    # Cast to plain dict for mutation — IRRequest is a TypedDict at
    # type-check time but a regular dict at runtime.
    request_dict = cast(dict[str, Any], ir_request)

    warnings: list[str] = []
    for field in ("tool_choice", "tool_config"):
        if field in request_dict:
            value = request_dict.pop(field)
            warnings.append(
                f"Stripped orphaned '{field}' (value: {value!r}) — "
                "no tool definitions present in request"
            )
            logger.warning(
                "Stripped orphaned '%s' from IR request — "
                "no tool definitions present (Codex context compaction workaround)",
                field,
            )

    return warnings
