"""The ACP transport's wire codec: a request body in, prompt blocks out; agent
updates in, chunks out.

:class:`~.acp.AcpTransport` runs the agent process and its sessions; this
module only translates.  It turns the converted request body into the prompt
blocks a session takes, and the agent's ``session/update`` notifications into
the chunk dicts the transport emits (``CHUNK_*``), including the stop reason,
the usage fields and the failure chunk.  Nothing here holds state.
"""

from __future__ import annotations

from typing import Any

#: Chunk shapes this transport emits.  The ``acp`` converter reads exactly
#: these; keeping them in one place is what lets both sides agree without a
#: shared schema module.
CHUNK_SESSION_START = "session_start"
CHUNK_TEXT = "text"
CHUNK_THOUGHT = "thought"
CHUNK_USAGE = "usage"
CHUNK_TOOL_CALL = "tool_call"
CHUNK_STOP = "stop"
CHUNK_ERROR = "error"


def _blocks_of(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The prompt blocks this turn sends, system instruction first.

    The converter keeps the system instruction in its own body field, because
    that is the shape an IR round-trip reads back; an agent needs it inside
    the prompt, where every other thing it is told lives.  Rendering it as the
    first block — labelled the way :func:`render_messages` labels a system
    message — is what stops a chat client's system instruction from vanishing:
    a lane that silently drops it is the one behaviour a chat lane must not
    have.

    Because the block is first and never moves, the ledger's digest of "what
    the agent has seen" covers it on every turn, and a session opened before
    it changed diverges rather than continuing.
    """
    blocks: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        blocks.append({"type": "text", "text": f"[System]\n{system}"})
    blocks.extend(
        block for block in body.get("prompt") or [] if isinstance(block, dict)
    )
    return blocks


def _failure(collect: list[dict[str, Any]] | None, message: str) -> dict[str, Any]:
    """The turn's error chunk, recorded for a caller that collects chunks."""
    chunk = {"type": CHUNK_ERROR, "message": message}
    if collect is not None:
        collect.append(chunk)
    return chunk


def _error_of(updates: list[dict[str, Any]]) -> str | None:
    """The message of the turn's error chunk, if it produced one."""
    for chunk in updates:
        if chunk.get("type") == CHUNK_ERROR:
            return str(chunk.get("message") or "") or "the ACP turn failed"
    return None


def _prompt_text(prompt: list[dict[str, Any]]) -> str:
    """Join the prompt's text blocks; images travel beside them."""
    return "\n\n".join(
        str(block.get("text") or "")
        for block in prompt
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _chunk_of(update: dict[str, Any]) -> dict[str, Any] | None:
    """Translate one ``session/update`` payload into this transport's chunk.

    ``ACPClient.prompt`` yields the notification's *params*, so the update
    itself sits under ``"update"``.  Only the update kinds the ``acp``
    converter understands are returned.  A tool frame is not one of them: a
    bridged client tool travels as ``CHUNK_TOOL_CALL`` from the MCP bridge,
    and a built-in tool frame stops the turn in
    :class:`~.acp._TurnRunner`.
    """
    body = update.get("update")
    if isinstance(body, dict):
        update = body
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    if kind == "agent_message_chunk":
        content = update.get("content") or {}
        if isinstance(content, dict) and content.get("type") == "text":
            return {"type": CHUNK_TEXT, "text": str(content.get("text") or "")}
        return None
    if kind == "agent_thought_chunk":
        content = update.get("content") or {}
        if isinstance(content, dict) and content.get("type") == "text":
            return {"type": CHUNK_THOUGHT, "text": str(content.get("text") or "")}
        return None
    if kind == "usage_update":
        used = update.get("used")
        meta = update.get("_meta") or update.get("meta")
        if not isinstance(used, int) and meta is None:
            return None
        chunk: dict[str, Any] = {
            "type": CHUNK_USAGE,
            "used": used if isinstance(used, int) else None,
            "size": update.get("size"),
        }
        # The adapter's own accounting rides beside the context size: Claude
        # reports the account's rate-limit windows under ``_meta``.  Passing
        # them through is what lets the quota card read a keyless lane.
        cost = update.get("cost")
        if isinstance(cost, dict):
            chunk["cost"] = cost
        if meta is not None:
            chunk["meta"] = meta
        return chunk
    return None


def _stop_reason_of(updates: list[dict[str, Any]]) -> str:
    for chunk in reversed(updates):
        if chunk.get("type") == CHUNK_STOP:
            return str(chunk.get("stop_reason") or "end_turn")
    return "end_turn"


def _field_of(updates: list[dict[str, Any]], kind: str, field_name: str) -> str:
    for chunk in updates:
        if chunk.get("type") == kind:
            return str(chunk.get(field_name) or "")
    return ""
