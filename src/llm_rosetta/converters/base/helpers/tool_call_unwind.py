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
) -> dict[str, Any] | None:
    """Find the matching tool result message for a given tool_call_id.

    Tries ID-based matching first; falls back to positional matching.

    Args:
        tool_call_id: The tool_call_id to match.
        result_id_map: Mapping of tool_call_id → tool result message.
        tool_messages: Ordered list of tool messages (positional fallback).
        index: Position index for fallback matching.

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

    logger.warning(
        "Unwind: no matching result for tool_call_id=%s (index %d)",
        tool_call_id,
        index,
    )
    return None


# ---------------------------------------------------------------------------
# Core unwind
# ---------------------------------------------------------------------------


def _create_sequential_pairs(
    tool_call_parts: list[Any],
    tool_messages: list[dict[str, Any]],
    non_tool_call_parts: list[Any],
) -> list[dict[str, Any]]:
    """Convert parallel tool calls into sequential call-result pairs.

    Args:
        tool_call_parts: List of tool_call content parts from the assistant.
        tool_messages: List of corresponding tool result messages.
        non_tool_call_parts: Non-tool-call content parts (text, reasoning)
            from the original assistant message.

    Returns:
        List of alternating assistant and tool messages.
    """
    sequential: list[dict[str, Any]] = []
    result_map = _build_result_id_map(tool_messages)

    for idx, tc_part in enumerate(tool_call_parts):
        tc_id = tc_part.get("tool_call_id", "")
        result_msg = _find_matching_result(tc_id, result_map, tool_messages, idx)
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
) -> dict[str, Any]:
    """Unwind parallel tool calls into sequential call-result pairs.

    Transforms IR messages like::

        [user, assistant(tc_1, tc_2), tool(tr_1), tool(tr_2)]

    into::

        [user, assistant(tc_1), tool(tr_1), assistant(tc_2), tool(tr_2)]

    Messages without parallel tool calls are passed through unchanged.
    When tool call count does not match tool result count, the message
    group is left unchanged (safety).

    The original request is NOT modified; a new dict is returned only
    when changes are needed.

    Args:
        ir_request: IR request dict with ``messages`` key.

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
                logger.warning(
                    "Unwind: mismatch — %d tool calls but %d tool results, "
                    "skipping reorder",
                    n_calls,
                    len(tool_messages),
                )
                transformed.append(msg)
                i += 1
                continue

            pairs = _create_sequential_pairs(
                tool_call_parts, tool_messages, non_tool_call_parts
            )
            transformed.extend(pairs)
            changed = True

            logger.info(
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
