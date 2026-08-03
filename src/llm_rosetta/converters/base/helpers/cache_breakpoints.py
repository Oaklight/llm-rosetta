"""Auto-inject cache breakpoints into IR requests.

When cross-format requests (OpenAI/Gemini → Anthropic) pass through
conversion, the source format has no cache semantics.  This helper
injects ``cache_hint`` markers on IR parts so the Anthropic converter
emits ``cache_control`` breakpoints, enabling prompt caching.

Strategy (up to 4 breakpoints, matching Anthropic's limit):

1. Last tool definition       → caches all tool schemas
2. System instruction tail    → caches tools + system prefix
3. Last user message          → caches conversation so far
4. Second-to-last user msg    → keeps a rolling prefix cache warm
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


def _has_cache_hint_in_parts(parts: list[dict[str, Any]]) -> bool:
    """Check if any part in a list carries ``cache_hint``."""
    return any(p.get("cache_hint") is not None for p in parts)


def _has_cache_hint_in_messages(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and _has_cache_hint_in_parts(content):
            return True
    return False


def _has_any_cache_hint(ir_request: dict[str, Any]) -> bool:
    """Return True if the IR request contains any ``cache_hint`` marker."""
    system = ir_request.get("system_instruction")
    if isinstance(system, list) and _has_cache_hint_in_parts(system):
        return True

    tools = ir_request.get("tools")
    if isinstance(tools, list) and _has_cache_hint_in_parts(tools):
        return True

    messages = ir_request.get("messages")
    if isinstance(messages, list) and _has_cache_hint_in_messages(messages):
        return True

    return False


def _mark_last_part(parts: list[dict[str, Any]]) -> bool:
    """Set ``cache_hint`` on the last part that can carry it.

    Shallow-copies the target entry before mutation to avoid corrupting
    the conversion pipeline's LRU cache (cached IR entries are shared
    by reference).

    Returns True if a hint was placed.
    """
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if isinstance(part, dict) and part.get("cache_hint") is None:
            parts[i] = {**part, "cache_hint": dict(_EPHEMERAL)}
            return True
    return False


def _mark_last_user_messages(messages: list[dict[str, Any]], count: int) -> int:
    """Mark the last *count* user messages with cache breakpoints.

    Returns the number of breakpoints actually placed.
    """
    placed = 0
    user_indices = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    for idx in reversed(user_indices[-count:]):
        content = messages[idx].get("content")
        if isinstance(content, list) and content and _mark_last_part(content):
            placed += 1
    return placed


def inject_cache_breakpoints(
    ir_request: dict[str, Any],
    *,
    mode: str = "none_only",
    request_id: str = "-",
) -> dict[str, Any]:
    """Inject ``cache_hint`` breakpoints into an IR request.

    Args:
        ir_request: The IR request dict (mutated in place).
        mode: ``"none_only"`` skips injection if any hint exists;
            ``"fill_gaps"`` fills each segment independently.
        request_id: For logging context.

    Returns:
        The (potentially mutated) IR request.
    """
    if mode == "none_only" and _has_any_cache_hint(ir_request):
        logger.debug("[%s] cache hints already present, skipping injection", request_id)
        return ir_request

    remaining = 4
    placed_segments: list[str] = []

    # 1. Last tool definition
    tools = ir_request.get("tools")
    if isinstance(tools, list) and tools and remaining > 0:
        if mode == "fill_gaps" and _has_cache_hint_in_parts(tools):
            pass
        elif _mark_last_part(tools):
            remaining -= 1
            placed_segments.append("tools")

    # 2. System instruction tail
    system = ir_request.get("system_instruction")
    if isinstance(system, list) and system and remaining > 0:
        if mode == "fill_gaps" and _has_cache_hint_in_parts(system):
            pass
        elif _mark_last_part(system):
            remaining -= 1
            placed_segments.append("system")

    # 3 & 4. Last two user messages
    messages = ir_request.get("messages")
    if isinstance(messages, list) and messages and remaining > 0:
        if mode == "fill_gaps" and _has_cache_hint_in_messages(messages):
            pass
        else:
            n = _mark_last_user_messages(messages, min(2, remaining))
            remaining -= n
            if n:
                placed_segments.append(f"user_messages({n})")

    if placed_segments:
        logger.debug(
            "[%s] injected cache breakpoints: %s",
            request_id,
            ", ".join(placed_segments),
        )

    return ir_request
