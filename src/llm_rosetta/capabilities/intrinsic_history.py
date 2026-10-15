"""Intrinsic *history* enforcement — degrade what the target cannot replay.

An intrinsic (provider-executed) tool call in message history is context, not
a capability declaration: the target's shim gates the *definitions*
(:func:`resolve_intrinsic_tools`), while history is whatever the conversation
already contains.  What the target's wire can express, though, is its own
business — and a server call it has no part for cannot be replayed at all.

This pass asks the **target** converter the one question that decides the
shape, :meth:`~llm_rosetta.converters.base.tools.ops.BaseToolOps.intrinsic_renderable`
with ``leg="history"``, and for every call it cannot render:

- rewrites the call into an ordinary ``function`` tool call;
- rewrites every ``tool_result`` carrying the same ``tool_call_id`` — wherever
  it lives, inline in the assistant turn or in a ``role: "tool"`` message —
  into an ordinary function result;
- keeps the id, the input and the result, and records what it was in
  ``provider_metadata["_downgraded_from"]``.

The pair is decided once, by ``tool_call_id``, so a call and its result can
never disagree about their shape.  A part the target *can* render natively is
left exactly as it was, whatever dialect the kind came from.

This is the history half of a division of labour: the pipeline owns the
native/degrade decision for history here, while the converters own the
response leg (never degrade a server call into a client call — drop it) and
their own native rendering.  A dialect used as a library, without a pipeline,
therefore still writes the parent client shape for an unrenderable history
part rather than raising.

The custom-tool analogue is :func:`enforce_custom_tools` in
:mod:`.enforcement`; this one is keyed on the *part* rather than the tool
definition, and on the target's renderability rather than a shim flag.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, cast

from ..converters.base.tools.intrinsic import intrinsic_kind_of, is_intrinsic_part

logger = logging.getLogger(__name__)

#: A degraded name must be accepted by every dialect this library speaks.
_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

#: Marker key written into ``provider_metadata`` of a degraded part.
DOWNGRADED_FROM_KEY = "_downgraded_from"


def _degraded_name(kind: str, *, taken: set[str]) -> str:
    """Return a name-safe function name standing in for *kind*.

    Deterministic: the same kind always yields the same name for a given
    ``taken`` set.  The kind itself when nothing else claims it, otherwise the
    kind with a deterministic qualifier appended — never a name already used by
    a client tool in the same request, or by another degraded kind.

    Args:
        kind: The canonical intrinsic kind the call carried.
        taken: Names already claimed; the returned name is not in it.

    Returns:
        A name matching ``^[a-zA-Z0-9_-]+$``.
    """
    base = _NAME_SAFE.sub("_", kind).strip("_") or "intrinsic_tool"
    if base not in taken:
        return base
    stem = f"{base}_intrinsic"
    candidate = stem
    suffix = 2
    while candidate in taken:
        candidate = f"{stem}{suffix}"
        suffix += 1
    return candidate


def _claimed_names(ir_request: dict[str, Any]) -> set[str]:
    """Names the request's own tool list already uses.

    A native (``type="intrinsic"``) definition is not a claim: it *is* the
    server tool the degraded call stands in for, so a degraded ``web_search``
    may keep the name the intrinsic definition used.

    Args:
        ir_request: The IR request.

    Returns:
        The set of declared client tool names.
    """
    names: set[str] = set()
    for tool in ir_request.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") == "intrinsic":
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


_CONTENT_BLOCK_TYPES = frozenset({"text", "image", "input_text", "output_text"})


def _plain_result(result: Any) -> Any:
    """Make a degraded result safe for an ordinary tool result.

    A server result is provider-shaped (search result blocks with no ``type``,
    a structured code-execution object).  An ordinary tool result accepts a
    string or typed content blocks, and live targets reject anything else
    (``tool_result.content.0.type: Field required``), so an untyped payload is
    carried as JSON text.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        if all(
            isinstance(b, dict) and b.get("type") in _CONTENT_BLOCK_TYPES
            for b in result
        ):
            return result
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False)


def _downgrade_part(
    part: dict[str, Any], kind: str, name: str | None
) -> dict[str, Any]:
    """Rewrite one intrinsic part into the ordinary function part it becomes.

    The result carries no name of its own, so *name* is ``None`` for it.

    Args:
        part: The IR intrinsic ``tool_call`` or ``tool_result`` part.
        kind: The intrinsic kind the part carried.
        name: The degraded function name (calls only).

    Returns:
        A new part dict with the intrinsic identity replaced by the marker.
    """
    degraded = copy.deepcopy(part)
    degraded["tool_type"] = "function"
    if part.get("type") == "tool_result":
        degraded["result"] = _plain_result(degraded.get("result"))
    degraded.pop("intrinsic_kind", None)
    if name is not None:
        degraded["tool_name"] = name

    metadata = dict(degraded.get("provider_metadata") or {})
    # The pre-#999 spelling lived here; leaving it would keep
    # ``intrinsic_kind_of`` answering a kind the part no longer is.
    metadata.pop("intrinsic_kind", None)
    metadata[DOWNGRADED_FROM_KEY] = {"type": "intrinsic", "kind": kind}
    degraded["provider_metadata"] = metadata
    return degraded


def _decide(
    messages: list[Any],
    target_tool_ops: Any,
    taken: set[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Decide which calls to degrade, and the name each takes.

    Args:
        messages: The IR message list.
        target_tool_ops: The target converter's tool operations.
        taken: Names already claimed; extended in place.

    Returns:
        ``(degraded, names, reasons)``: pairing key → kind for every call the
        target cannot render on the history leg, pairing key → the client
        function name its call is rewritten to, and pairing key → why it could
        not stay native.
    """
    by_kind: dict[str, str] = {}
    degraded: dict[str, str] = {}
    names: dict[str, str] = {}
    reasons: dict[str, str] = {}
    answered = {
        part.get("tool_call_id")
        for message in messages
        if isinstance(message, dict)
        for part in message.get("content") or []
        if isinstance(part, dict) and part.get("type") == "tool_result"
    }
    keys = _pair_keys(messages)
    for (message_index, part_index), key in keys.items():
        part = messages[message_index]["content"][part_index]
        if part.get("type") != "tool_call" or not is_intrinsic_part(part):
            continue
        kind = intrinsic_kind_of(part)
        fields = part.get("tool_input")
        reason = _degrade_reason(
            part,
            kind,
            fields,
            target_tool_ops,
            answered,
            is_anonymous=key != (part.get("tool_call_id") or ""),
        )
        if reason is None:
            continue
        name = by_kind.get(kind)
        if name is None:
            name = _degraded_name(kind, taken=taken)
            by_kind[kind] = name
            taken.add(name)
        degraded[key] = kind
        names[key] = name
        reasons[key] = reason
    return degraded, names, reasons


def _degrade_reason(
    part: Any,
    kind: str,
    fields: Any,
    target_tool_ops: Any,
    answered: set[Any],
    *,
    is_anonymous: bool,
) -> str | None:
    """Why this call cannot stay native on the history leg, or ``None``.

    The three reasons read very differently to whoever sees the warning, and
    only the first is about the target's vocabulary: a kind the target simply
    does not have, a kind whose native form needs a field the IR part does not
    carry (the same kind is written natively when it arrives in a dialect that
    fills that field), and a turn this target requires a paired result in.

    Args:
        part: The IR intrinsic tool call part.
        kind: Its canonical kind (``""`` when the part carries none).
        fields: Its ``tool_input``.
        target_tool_ops: The target converter's tool operations.
        answered: Every ``tool_call_id`` a result in this history answers.
        is_anonymous: Whether the part has no ``tool_call_id``.

    Returns:
        A short reason, or ``None`` when the call stays native.
    """
    if not kind:
        return "the part carries no intrinsic kind"
    if target_tool_ops.intrinsic_requires_result(kind) and (
        is_anonymous or part.get("tool_call_id") not in answered
    ):
        return f"the target needs a result for its {kind!r} call and this turn has none"
    if target_tool_ops.intrinsic_renderable(
        kind, fields if isinstance(fields, Mapping) else {}, leg="history"
    ):
        return None
    # Two different situations land here and a reader has to know which to
    # look at: a kind the target has no form for at all, and a kind it does
    # have whose form needs a field this part does not carry (the same kind is
    # written natively when it arrives in a dialect that fills that field).
    return (
        f"the target cannot write the {kind!r} call as it stands: either it has "
        f"no {kind!r} form, or the payload lacks a field that form needs"
    )


def _pair_keys(messages: list[Any]) -> dict[tuple[int, int], str]:
    """Map each intrinsic call/result position to the key that pairs them.

    The key is the ``tool_call_id`` when the part has one.  Ids are optional on
    :class:`~llm_rosetta.types.ir.parts.ToolCallPart`, and two id-less calls of
    different kinds share the empty string, so an anonymous call is keyed by its
    own position instead — each keeps its own kind and name.  An anonymous
    result pairs with the most recent anonymous call before it, which is the
    order a provider writes them in.

    Args:
        messages: The IR message list.

    Returns:
        ``(message index, part index)`` → pairing key.
    """
    keys: dict[tuple[int, int], str] = {}
    pending: str | None = None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for part_index, part in enumerate(message.get("content") or []):
            if not isinstance(part, dict) or not is_intrinsic_part(part):
                continue
            if part.get("type") not in ("tool_call", "tool_result"):
                continue
            call_id = part.get("tool_call_id") or ""
            if call_id:
                keys[(message_index, part_index)] = call_id
                continue
            if part.get("type") == "tool_call":
                pending = f"\x00anonymous:{message_index}:{part_index}"
                keys[(message_index, part_index)] = pending
            elif pending is not None:
                keys[(message_index, part_index)] = pending
    return keys


def _rewrite(
    messages: list[Any], degraded: dict[str, str], names: dict[str, str]
) -> list[Any]:
    """Rewrite every degraded call and result, keeping the other parts as they are."""
    new_messages: list[Any] = []
    pair_keys = _pair_keys(messages)
    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            new_messages.append(message)
            continue
        new_content: list[Any] = []
        changed = False
        for part_index, part in enumerate(content):
            if not isinstance(part, dict) or not is_intrinsic_part(part):
                new_content.append(part)
                continue
            key = pair_keys.get((message_index, part_index))
            if key is None or key not in degraded:
                new_content.append(part)
                continue
            # Only a call carries the function name; a result pairs by id.
            name = names.get(key) if part.get("type") == "tool_call" else None
            new_content.append(
                _downgrade_part(cast(dict[str, Any], part), degraded[key], name)
            )
            changed = True
        new_messages.append({**message, "content": new_content} if changed else message)
    return new_messages


def degrade_intrinsic_history(
    ir_request: dict[str, Any],
    target_tool_ops: Any,
    *,
    warnings: list[str] | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Degrade intrinsic history the target cannot replay into function pairs.

    Every intrinsic ``tool_call`` in message history is asked of
    *target_tool_ops*: can this dialect write you on the history leg
    (``intrinsic_renderable(kind, input, leg="history")``)?  A kind it can is
    left alone; one it cannot — a sibling dialect's kind, a kind no dialect
    knows, a kind whose native part needs a field the IR does not carry, or a
    part with no kind at all — becomes an ordinary function call, together
    with every result answering the same ``tool_call_id``.

    The pairing is by ``tool_call_id`` and is decided once, so the call and its
    result always take the same shape.  Each degraded pair is reported once in
    *warnings*.

    Runs unconditionally, same-format routes included: the judge is what the
    target can render, and a part the target renders natively is untouched
    either way.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        target_tool_ops: The target converter's
            :class:`~llm_rosetta.converters.base.tools.ops.BaseToolOps`
            (class or instance): the native/degrade vocabulary being asked.
        warnings: Optional list to append one entry per degraded pair to.
        request_id: Request id for the log line.

    Returns:
        The IR request with unrenderable intrinsic history degraded, or the
        original request unchanged when every part is renderable.
    """
    messages = ir_request.get("messages")
    if not isinstance(messages, list) or not messages:
        return ir_request

    degraded, names, reasons = _decide(
        messages, target_tool_ops, _claimed_names(ir_request)
    )
    if not degraded:
        return ir_request

    if warnings is not None:
        for key, kind in degraded.items():
            where = (
                "(no tool_call_id)" if key.startswith("\x00anonymous") else repr(key)
            )
            warnings.append(
                f"Intrinsic {kind!r} tool call {where}: {reasons[key]}; degraded "
                f"to the client function {names[key]!r}"
            )

    logger.info(
        "[%s] degraded %d intrinsic history pair(s) the target cannot replay: %s",
        request_id,
        len(degraded),
        ", ".join(
            f"{kind}({call_id or 'no id'})->{names[call_id]}"
            for call_id, kind in degraded.items()
        ),
    )

    return {**ir_request, "messages": _rewrite(messages, degraded, names)}
