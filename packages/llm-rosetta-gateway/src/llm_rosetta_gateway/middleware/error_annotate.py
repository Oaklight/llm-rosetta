"""Annotate upstream error bodies with gateway-side diagnostics.

Some requests the gateway forwards carry a defect no converter can repair —
a tool list that declares one name twice with different definitions, for
instance.  The upstream's own rejection says nothing about that, so the
failure reads like a gateway fault even though the client's request caused it.

This module appends a short attribution to the error body the client receives.
The rewrite is deliberately narrow: of the parsed document, only the
``error.message`` string changes, and only when the request actually carried a
collision.  The body is re-serialized, so whitespace and escaping may come back
normalized (``ensure_ascii=False`` keeps non-ASCII text readable) — harmless to
a JSON client, visible to one that diffs the raw bytes.  A body that is not
JSON, or whose error envelope carries no string message, is returned untouched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# Marks a message this module already annotated, so a chained gateway or a
# client retry does not stack a second copy of the hint.
_MARKER = "[llm-rosetta]"

# Names listed before the hint summarizes the rest.  Each collision needs two
# definitions in the client's own request, so the input bounds this — the cap
# only keeps a pathological tool list from producing a multi-kilobyte message.
_MAX_HINT_NAMES = 10


def collision_hint(collisions: Mapping[str, int]) -> str:
    """Build the attribution sentence for a set of conflicting tool names.

    Args:
        collisions: Conflicting tool name -> number of distinct definitions
            declared for it.

    Returns:
        One sentence naming the tools in sorted order, capped at
        :data:`_MAX_HINT_NAMES` names with a count of the remainder.
    """
    names = sorted(collisions.items())
    parts = "; ".join(
        f"declared tool name {name!r} with {count} distinct definitions"
        for name, count in names[:_MAX_HINT_NAMES]
    )
    if len(names) > _MAX_HINT_NAMES:
        parts += f"; and {len(names) - _MAX_HINT_NAMES} more"
    return (
        f"{_MARKER} This request {parts}. Some upstreams require unique tool "
        "names, so this rejection may come from the request's own tool list."
    )


def annotate_tool_collisions(
    raw: str | bytes, collisions: Mapping[str, int] | None
) -> str | bytes:
    """Append a tool-name-collision hint to a serialized error body.

    The body is treated as opaque: it may be in the upstream's format rather
    than the client's, and any envelope whose ``error`` object carries a
    string ``message`` (OpenAI, Anthropic and Google all do) is accepted
    without format-specific handling.  Only the parsed ``error.message``
    changes; the returned body is re-serialized, so whitespace and escaping
    may differ from the input.

    Args:
        raw: The serialized error body, from an upstream or a synthesized
            soft-error message.
        collisions: Conflicting tool names to report.  An empty or ``None``
            mapping makes this a no-op.

    Returns:
        The body with the hint appended to ``error.message``, matching the
        input type, or *raw* unchanged when there is nothing to annotate or
        the body cannot be rewritten.
    """
    if not collisions:
        return raw
    if isinstance(raw, bytes):
        try:
            text: str | None = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
    else:
        text = raw
    annotated = _annotate_text(text, collisions)
    if annotated is None:
        return raw
    return annotated.encode("utf-8") if isinstance(raw, bytes) else annotated


def _annotate_text(text: str, collisions: Mapping[str, int]) -> str | None:
    """Rewrite one JSON error body, or return ``None`` if it does not fit.

    Returns ``None`` for anything this module should leave alone: a body that
    does not parse, is not a JSON object, has no ``error`` object, has a
    non-string or already-annotated message.
    """
    try:
        data: Any = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    if not isinstance(message, str) or _MARKER in message:
        return None
    hint = collision_hint(collisions)
    error["message"] = f"{message}\n\n{hint}" if message else hint
    return json.dumps(data, ensure_ascii=False)
