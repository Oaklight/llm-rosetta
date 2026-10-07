"""Tool result batch tracking — assign and merge by batch_id.

When multiple tool results belong to the same assistant tool-call turn,
they share a ``batch_id``.  Outbound converters for grouped formats
(Anthropic, Google generateContent) use this to merge tool results into
a single container message.  When ``batch_id`` is absent, the merge
falls back to adjacency for backward compatibility.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any


def assign_tool_batch_ids(ir_messages: list[Any]) -> list[Any]:
    """Assign ``batch_id`` to tool messages from preceding assistant tool_calls.

    Walks the message list.  Each assistant message that contains at least
    one ``tool_call`` part starts a new batch (sequential counter).  All
    immediately following ``tool``-role messages receive that batch_id.
    A non-tool message resets the batch.  Items without a ``role`` key
    (passthrough / extension items) are skipped transparently — they do
    not reset the current batch.

    Messages that already carry a ``batch_id`` are left unchanged.
    Mutates in place and returns the same list for chaining.
    """
    counter = 0
    current_batch: str | None = None

    for item in ir_messages:
        role = item.get("role") if isinstance(item, dict) else None

        if role is None:
            continue

        if role == "assistant":
            has_tool_calls = any(
                isinstance(p, dict) and p.get("type") == "tool_call"
                for p in item.get("content", [])
            )
            if has_tool_calls:
                current_batch = str(counter)
                counter += 1
            else:
                current_batch = None
        elif role == "tool":
            if current_batch is not None and "batch_id" not in item:
                item["batch_id"] = current_batch
        else:
            current_batch = None

    return ir_messages


def merge_tool_messages(ir_messages: Sequence[Any]) -> list[Any]:
    """Merge consecutive tool messages that belong to the same batch.

    When ``batch_id`` is present, tool messages with the same value are
    merged into a single message (content lists concatenated).  When
    ``batch_id`` is absent on consecutive tool messages, they are merged
    by adjacency as a backward-compatibility fallback.

    Non-tool messages and items without a ``role`` key pass through
    unchanged.  The returned list is always a new list; input messages
    that are merged are shallow-copied.
    """
    if not ir_messages:
        return list(ir_messages)

    merged: list[Any] = []
    last_batch_id: str | None = None
    last_was_tool = False

    for item in ir_messages:
        role = item.get("role") if isinstance(item, dict) else None

        if role == "tool":
            batch_id = item.get("batch_id") if isinstance(item, dict) else None

            should_merge = False
            if merged and last_was_tool:
                if batch_id is not None and batch_id == last_batch_id:
                    should_merge = True
                elif batch_id is None and last_batch_id is None:
                    should_merge = True

            if should_merge:
                prev = merged[-1]
                merged[-1] = {
                    **prev,
                    "content": prev["content"] + item["content"],
                }
            else:
                merged.append(copy.copy(item))

            last_batch_id = batch_id
            last_was_tool = True
        else:
            last_batch_id = None
            last_was_tool = False
            merged.append(item)

    return merged
