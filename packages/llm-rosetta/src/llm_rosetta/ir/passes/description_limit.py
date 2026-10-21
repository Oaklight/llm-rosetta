"""Oversized tool descriptions — relocation out of the tool list.

A description too long for the target is moved into a late system message rather
than dropped, so the model still reads it and the tool list stays within the
provider's limit.
"""

from __future__ import annotations

import copy

import logging
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oversized tool description relocation
# ---------------------------------------------------------------------------

_POINTER_TEMPLATE = "[Full documentation for '{}' provided separately.]"


def relocate_oversized_tool_descriptions(
    ir_request: dict[str, Any],
    *,
    max_description_length: int | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Move oversized tool descriptions into a late system message.

    When a tool's description exceeds *max_description_length*, the full
    text is moved into an appended system message and the tool's
    description is replaced with a short pointer.  The downstream
    ``hoist_late_system_messages`` transform adapts the system message
    for each provider automatically.

    No-op when *max_description_length* is ``None`` (provider has no
    limit) or no tools exceed the threshold.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        max_description_length: Maximum allowed description length, or
            ``None`` to skip relocation entirely.
        request_id: For logging.

    Returns:
        The IR request with oversized descriptions relocated, or the
        original request unchanged.
    """
    if max_description_length is None:
        return ir_request

    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    oversized: list[tuple[int, str, str]] = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        desc = tool.get("description", "")
        if len(desc) > max_description_length:
            oversized.append((i, tool.get("name", f"tool_{i}"), desc))

    if not oversized:
        return ir_request

    logger = logging.getLogger(__name__)

    tools = copy.deepcopy(tools)
    ir_request = {**ir_request, "tools": tools}

    sections: list[str] = []
    for idx, name, full_desc in oversized:
        tools[idx]["description"] = _POINTER_TEMPLATE.format(name)
        meta = tools[idx].get("metadata") or {}
        meta["_description_relocated"] = True
        tools[idx]["metadata"] = meta
        sections.append(f"## Tool: {name}\n\n{full_desc}")

    relocated_text = "\n\n---\n\n".join(sections)
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": [{"type": "text", "text": relocated_text}],
    }

    messages = list(ir_request.get("messages", []))
    messages.append(system_msg)
    ir_request["messages"] = messages

    logger.debug(
        "[%s] relocated %d oversized tool description(s) to late system message",
        request_id,
        len(oversized),
    )
    return ir_request
