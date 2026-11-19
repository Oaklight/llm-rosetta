"""Unwind parallel tool calls into sequential call-result pairs (IR level).

Some upstream gateways reject requests where a single model turn contains
multiple tool calls paired with separate tool result messages.  This module
converts parallel tool calls into sequential assistant+tool message pairs
so that every assistant message contains exactly one tool call followed by
its corresponding tool result.

The algorithm operates on **IR messages** (``role: "assistant"`` with
``type: "tool_call"`` content parts, ``role: "tool"`` with
``type: "tool_result"`` content parts).

Enabled per-provider via ``ProviderShim.unwind_parallel_tool_calls`` and
``unwind_parallel_tool_calls_pattern`` (model-scoped regex).
"""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss
from llm_rosetta.types.ir import (
    is_message,
    is_tool_call_part,
    is_tool_result_part,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_parallel_tool_call_message(message: dict[str, Any]) -> bool:
    """Check if an IR message has multiple tool_call content parts."""
    if message.get("role") != "assistant":
        return False
    content = message.get("content", [])
    tool_calls = [p for p in content if is_tool_call_part(p)]
    return len(tool_calls) > 1


def _parts_after_first_call(content: list[Any]) -> list[str]:
    """Return the types of the non-call parts that sit after the first call.

    The unwind puts every non-call part ahead of the first tool call, so
    these are the parts whose position it changes.  Their types come back in
    first-appearance order — the report names them, and the count is how
    many parts moved.

    Args:
        content: The assistant turn's content parts, in their original order.

    Returns:
        One entry per moved part; empty when every non-call part already
        precedes the calls.
    """
    seen_call = False
    moved: list[str] = []
    for part in content:
        if is_tool_call_part(part):
            seen_call = True
        elif seen_call:
            kind = part.get("type") if isinstance(part, dict) else None
            moved.append(str(kind) if kind else "unknown")
    return moved


def _collect_tool_result_messages(
    messages: list[dict[str, Any]],
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Collect consecutive ``role: "tool"`` messages from *start_index*.

    Returns:
        ``(tool_messages, next_index)`` where *next_index* points to the
        first message after the collected tool results.
    """
    tool_messages: list[dict[str, Any]] = []
    j = start_index
    while (
        j < len(messages)
        and is_message(messages[j])
        and messages[j].get("role") == "tool"
    ):
        tool_messages.append(messages[j])
        j += 1
    return tool_messages, j


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _build_result_id_map(
    tool_messages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map ``tool_call_id`` → tool result message for O(1) lookup."""
    mapping: dict[str, dict[str, Any]] = {}
    for msg in tool_messages:
        for part in msg.get("content", []):
            if is_tool_result_part(part):
                tcid = part.get("tool_call_id", "")
                if tcid:
                    mapping[tcid] = msg
    return mapping


def _find_matching_result(
    tool_call_id: str,
    result_id_map: dict[str, dict[str, Any]],
    tool_messages: list[dict[str, Any]],
    index: int,
    warnings: Any = None,
) -> dict[str, Any] | None:
    """Find the matching tool result message for a given tool_call_id.

    Tries ID-based matching first; falls back to positional matching.

    Args:
        tool_call_id: The tool_call_id to match.
        result_id_map: Mapping of tool_call_id → tool result message.
        tool_messages: Ordered list of tool messages (positional fallback).
        index: Position index for fallback matching.
        warnings: The conversion's sink, when the caller has one.

    Returns:
        The matching tool result message, or None.
    """
    # ID-based matching
    if tool_call_id and tool_call_id in result_id_map:
        logger.debug(
            "Unwind: ID match for tool_call_id=%s",
            tool_call_id,
        )
        return result_id_map[tool_call_id]

    # Positional fallback
    if index < len(tool_messages):
        logger.debug(
            "Unwind: positional match for tool call %d",
            index + 1,
        )
        return tool_messages[index]

    report_loss(
        f"Unwind: no matching result for tool_call_id={tool_call_id} (index {index})",
        subject=tool_call_id,
        warnings=warnings,
    )
    return None


# ---------------------------------------------------------------------------
# Core unwind
# ---------------------------------------------------------------------------


def _create_sequential_pairs(
    tool_call_parts: list[Any],
    tool_messages: list[dict[str, Any]],
    non_tool_call_parts: list[Any],
    warnings: Any = None,
) -> list[dict[str, Any]]:
    """Convert parallel tool calls into sequential call-result pairs.

    Args:
        tool_call_parts: List of tool_call content parts from the assistant.
        tool_messages: List of corresponding tool result messages.
        non_tool_call_parts: Non-tool-call content parts (text, reasoning)
            from the original assistant message.
        warnings: The conversion's sink, when the caller has one.

    Returns:
        List of alternating assistant and tool messages.
    """
    sequential: list[dict[str, Any]] = []
    result_map = _build_result_id_map(tool_messages)

    for idx, tc_part in enumerate(tool_call_parts):
        tc_id = tc_part.get("tool_call_id", "")
        result_msg = _find_matching_result(
            tc_id, result_map, tool_messages, idx, warnings
        )
        if result_msg is None:
            continue

        # Build assistant message with a single tool call.
        # First pair gets the non-tool-call parts (text, reasoning);
        # subsequent pairs get only the tool call.
        if idx == 0 and non_tool_call_parts:
            content = non_tool_call_parts + [tc_part]
        else:
            content = [tc_part]

        sequential.append({"role": "assistant", "content": content})
        sequential.append(result_msg)

    return sequential


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def unwind_parallel_tool_calls_ir(
    ir_request: dict[str, Any],
    *,
    warnings: Any = None,
) -> dict[str, Any]:
    """Unwind parallel tool calls into sequential call-result pairs.

    Transforms IR messages like::

        [user, assistant(tc_1, tc_2), tool(tr_1), tool(tr_2)]

    into::

        [user, assistant(tc_1), tool(tr_1), assistant(tc_2), tool(tr_2)]

    Messages without parallel tool calls are passed through unchanged.
    When tool call count does not match tool result count, the message
    group is left unchanged (safety).

    A turn whose non-call parts (text, reasoning) sit after its first call
    has those parts moved ahead of the first pair — the split cannot keep
    their position.  That move is reported as a downgrade: the parts are
    kept, their place in the turn is not.  A turn whose non-call parts
    already precede every call keeps its order and is not reported.

    The original request is NOT modified; a new dict is returned only
    when changes are needed.

    Args:
        ir_request: IR request dict with ``messages`` key.
        warnings: The conversion's sink, when the caller has one.  ``None``
            falls back to the conversion in progress, then to the log.

    Returns:
        The original or a new IR request with unwound messages.
    """
    messages = ir_request.get("messages", [])
    if not messages:
        return ir_request

    transformed: list[dict[str, Any]] = []
    changed = False
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_parallel_tool_call_message(msg):
            content = msg.get("content", [])
            tool_call_parts = [p for p in content if is_tool_call_part(p)]
            non_tool_call_parts = [p for p in content if not is_tool_call_part(p)]
            n_calls = len(tool_call_parts)

            # Collect consecutive tool result messages
            tool_messages, next_index = _collect_tool_result_messages(messages, i + 1)

            if len(tool_messages) != n_calls:
                report_loss(
                    f"Unwind: mismatch — {n_calls} tool calls but "
                    f"{len(tool_messages)} tool results; skipping reorder",
                    code=DOWNGRADED,
                    warnings=warnings,
                )
                transformed.append(msg)
                i += 1
                continue

            # Name the parts the split moves.  Only a part that sat after the
            # first call changes position; the report is what keeps that
            # reorder from being silent.
            moved = _parts_after_first_call(content)
            if moved:
                kinds = ", ".join(dict.fromkeys(moved))
                report_loss(
                    f"Unwind: moved {len(moved)} non-tool-call part(s) "
                    f"({kinds}) of an assistant turn with {n_calls} tool "
                    "calls ahead of the first call; their position relative "
                    "to the calls is not preserved",
                    code=DOWNGRADED,
                    subject="content_order",
                    warnings=warnings,
                )

            pairs = _create_sequential_pairs(
                tool_call_parts, tool_messages, non_tool_call_parts, warnings
            )
            transformed.extend(pairs)
            changed = True

            logger.debug(
                "Unwind: converted %d parallel tool calls to %d sequential pairs",
                n_calls,
                len(pairs) // 2,
            )

            i = next_index
        else:
            transformed.append(msg)
            i += 1

    if not changed:
        return ir_request

    return {**ir_request, "messages": transformed}
