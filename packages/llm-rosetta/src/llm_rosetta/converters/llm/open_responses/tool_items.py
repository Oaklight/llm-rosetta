"""Responses tool item builders and readers.

Module-level helpers behind :class:`OpenResponsesToolOps`: namespace tools,
function/custom/server call items, intrinsic results and ``additional_tools``.
"""

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal, cast

from ...base.tools.content import (
    convert_ir_content_blocks_to_p,
)
from ....types.ir import (
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from llm_rosetta.conversion_warnings import report_loss
from ...base.helpers.opaque import CARRIER_KEY
from llm_rosetta.ir.primitives.tool_group import (
    GENERATED_GROUP_MARKER,
    generated_group_id,
    tool_call_group,
)
from ...base.tools import (
    make_intrinsic_tool_call,
    mark_tool_error,
    prefixed_tool_call_id,
    sanitize_tool_call_id,
    warn_tool_error,
    warn_tool_error_dropped,
)

logger = logging.getLogger(__name__)


# Native OpenAI Responses server tools for client-declared intrinsic kinds.
# Responses native server-tool types → canonical intrinsic kind.
_RESPONSES_NATIVE_TYPE_TO_KIND: dict[str, str] = {
    "web_search": "web_search",
    "web_search_preview": "web_search",
    "code_interpreter": "code_interpreter",
    "file_search": "file_search",
    "image_generation": "image_generation",
}

#: Kinds this wire can be handed from another format — the native tool each is
#: written as.  Only kinds whose payload can be built without client-supplied
#: configuration are here; measured against the live API:
#:
#:   image_generation     200 with ``{"type": "image_generation"}``
#:   code_interpreter     needs ``container``; synthesised as ``{"type": "auto"}``
#:   file_search          needs ``vector_store_ids``
#:   mcp                  needs ``server_label``
#:   computer_use_preview needs ``environment``
#:   shell, tool_search   model-gated (400 "not supported with <model>")
#:
#: A kind in the second group would be *invented* rather than translated, so it
#: stays source-only: recognised on the way in and restored same-format from
#: its preserved native payload.
_RESPONSES_INTRINSIC_TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {"type": "web_search"},
    "code_interpreter": {"type": "code_interpreter", "container": {"type": "auto"}},
    "image_generation": {"type": "image_generation"},
}

#: Responses input item type that carries tool definitions inline (Codex).
ADDITIONAL_TOOLS_ITEM_TYPE = "additional_tools"

#: Result item types whose tool_type is not the "function" default.
_RESULT_ITEM_TOOL_TYPES: dict[str, Literal["custom", "mcp"]] = {
    "custom_tool_call_output": "custom",
    "mcp_call_output": "mcp",
}

_INTRINSIC_RESULT_TYPES: dict[str, str] = {
    "code_interpreter_call_output": "code_interpreter",
    "web_search_call_output": "web_search",
    "file_search_call_output": "file_search",
    "shell_call_output": "shell",
    "computer_call_output": "computer_use",
}


# ==================== Orphaned Tool Call Fix ====================


def _synthesize_passthrough_tool(
    provider_tool: dict[str, Any], tool_type: str
) -> ToolDefinition:
    """Build a passthrough IR tool for non-standard provider tool types.

    Synthesizes a minimal JSON Schema so cross-provider degradation
    produces "a function that accepts one text input" instead of an
    empty schema.  Appends format-constraint hints to the description.
    """
    synth_params: dict[str, Any] = {}
    if provider_tool.get("name"):
        synth_params = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Free-form text input",
                }
            },
            "required": ["input"],
        }
    desc = provider_tool.get("description", "")
    fmt = provider_tool.get("format")
    if fmt:
        fmt_type = fmt.get("type", "unknown")
        fmt_syntax = fmt.get("syntax", "")
        hint = f"[Output format: {fmt_type}"
        if fmt_syntax:
            hint += f", syntax: {fmt_syntax}"
        hint += "]"
        desc = f"{desc}\n\n{hint}" if desc else hint
    return cast(
        ToolDefinition,
        {
            "type": "function",
            "name": provider_tool.get("name", tool_type),
            "description": desc,
            "parameters": synth_params,
            "_passthrough": dict(provider_tool),
            "metadata": {"provider_type": tool_type},
            "required_parameters": synth_params.get("required", []),
        },
    )


def _declared_child_name(child: Any) -> str | None:
    """The name a container child declares, in either wire spelling."""
    if not isinstance(child, dict):
        return None
    name = child.get("name") or (
        child.get("function", {}).get("name")
        if isinstance(child.get("function"), dict)
        else None
    )
    return name if isinstance(name, str) and name else None


def _declared_child_names(children: Any) -> list[str]:
    """Every name the container's children declare, in declaration order."""
    return [name for name in (_declared_child_name(c) for c in children) if name]


def _flattened_child(
    child: Any, index: int, group_name: str
) -> list[ToolDefinition] | None:
    """Convert one container child, or skip it with a log line.

    Returns ``None`` for anything that cannot become a tool: a non-dict entry,
    a nested container, an unnamed child, or a child whose own conversion
    declined.  Nested namespaces are one level only — no real-world use case
    exists, and the OpenAI function-name constraint ``^[a-zA-Z0-9_-]+$``
    prohibits hierarchical separators.
    """
    if not isinstance(child, dict):
        report_loss(
            f"Skipping non-dict child at index {index} in group {group_name!r}",
            subject=group_name,
        )
        return None
    if child.get("type", "function") == "namespace":
        report_loss(
            f"Skipping nested namespace {child.get('name', '')!r} inside "
            f"{group_name!r} — only one level of nesting is supported",
            subject=str(child.get("name", "")),
        )
        return None
    if not _declared_child_name(child):
        report_loss(
            f"Skipping unnamed child at index {index} in group {group_name!r}",
            subject=group_name,
        )
        return None
    from .tool_ops import OpenResponsesToolOps  # lazy: tool_ops imports this module

    converted = OpenResponsesToolOps.p_tool_definition_to_ir(child)
    if converted is None:
        return None
    return converted if isinstance(converted, list) else [converted]


def _tag_child_group(
    item: ToolDefinition,
    *,
    group_name: str,
    group_desc: str,
    generated: bool,
    defer_loading: bool,
) -> ToolDefinition:
    """Write the group onto one converted child, and mark it when generated."""
    tagged = dict(item)
    # The group is IR-level, not provider detail: the client addressed this
    # tool by ``(name, group)``, and the value has to survive until the target
    # leg can decide whether it is able to carry a container.
    tagged["tool_group"] = group_name
    if group_desc:
        tagged["tool_group_description"] = group_desc
    if not (generated or defer_loading):
        return cast(ToolDefinition, tagged)
    # ``metadata`` is otherwise left exactly as the child conversion produced
    # it; only what this pass adds is written here.
    meta = dict(item.get("metadata") or {})
    if generated:
        # Marks the name as ours, so the passes that must treat a nameless
        # container differently from a named one can tell — and so a
        # per-request warning can report it.
        meta[GENERATED_GROUP_MARKER] = True
    if defer_loading:
        meta["defer_loading"] = True
    tagged["metadata"] = meta
    return cast(ToolDefinition, tagged)


def _flatten_namespace_tool(
    provider_tool: dict[str, Any],
) -> list[ToolDefinition]:
    """Expand a ``type: "namespace"`` container into individual IR tools.

    Iterates the children, converts each via ``p_tool_definition_to_ir``, and
    tags every result with the group it came from, for future round-trip.

    A container carrying no name of its own is given a generated one rather
    than the empty string: ``""`` qualifies nothing, so its children would
    keep one bare upstream name and could not be told apart — by the upstream,
    or by the client that would have to address them.  See
    :func:`~llm_rosetta.ir.primitives.tool_group.generated_group_id`.
    """
    declared = provider_tool.get("name", "")
    # Blank reads as no name.  The qualification ladder joins the two with an
    # underscore and providers match tool names against ``^[a-zA-Z0-9_-]+$``,
    # so a container called ``" "`` would otherwise ship ``" _wait"`` and a
    # 400.  A non-string name reads as absent for the same reason.
    #
    # An inner space (``"ta sks"``) survives this and still builds an illegal
    # name; that wider case is not this pass's to fix.
    group_name = declared.strip() if isinstance(declared, str) else ""
    group_desc = provider_tool.get("description", "")
    children = provider_tool.get("tools", [])
    if not children:
        return []

    generated = not group_name
    if generated:
        group_name = generated_group_id(group_desc, _declared_child_names(children))

    results: list[ToolDefinition] = []
    for index, child in enumerate(children):
        converted = _flattened_child(child, index, group_name)
        if converted is None:
            continue
        results.extend(
            _tag_child_group(
                item,
                group_name=group_name,
                group_desc=group_desc,
                generated=generated,
                defer_loading=bool(child.get("defer_loading")),
            )
            for item in converted
        )

    return results


def function_call_item_id(tool_call_id: str, provider_metadata: Any = None) -> str:
    """The Responses item id for a function call.

    A body and a stream both name the item, and they must agree: the id the
    source carried when there is one, otherwise the call id in the Responses
    API's own ``fc_`` spelling (a ``call_`` prefix is replaced, any other id
    scheme gets the prefix added).

    Args:
        tool_call_id: The call id the turn carries.
        provider_metadata: The IR tool call's provider metadata, which may hold
            the item id the source response used.

    Returns:
        The item id.
    """
    metadata = provider_metadata if isinstance(provider_metadata, Mapping) else {}
    item_id = metadata.get("responses_item_id")
    if item_id:
        return str(item_id)
    if tool_call_id.startswith("fc_"):
        return tool_call_id
    if tool_call_id.startswith("call_"):
        return "fc_" + tool_call_id[5:]
    return "fc_" + tool_call_id


def custom_call_input_text(tool_input: Any) -> str:
    """The plain-text ``input`` a custom tool call carries.

    A custom tool takes text, not JSON, so the IR wraps it as ``{"input": str}``
    when the text is not itself a JSON object; both the body writer and the
    stream writer must unwrap it the same way.

    Args:
        tool_input: The IR tool call's ``tool_input``.

    Returns:
        The text the wire carries.
    """
    if isinstance(tool_input, dict) and list(tool_input.keys()) == ["input"]:
        return str(tool_input["input"])
    return json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)


def _build_function_call_item(
    ir_tool_call: ToolCallPart,
    tool_call_id: str,
    tool_name: str,
    arguments: str,
) -> dict:
    """Build a ``function_call`` item with correct ``fc_`` prefixed ID.

    Recovers the Responses API item ID from provider_metadata when
    available, otherwise converts ``call_`` prefix to ``fc_`` or adds
    the prefix for other ID schemes (e.g. Anthropic ``toolu_``).
    """
    metadata = ir_tool_call.get("provider_metadata") or {}
    item_id = function_call_item_id(tool_call_id, metadata)
    item = {
        "type": "function_call",
        "id": item_id,
        "call_id": tool_call_id,
        "name": tool_name,
        "arguments": arguments,
        "status": "completed",
    }
    # Flattened upstream; the client dispatches on (name, group).  The
    # first-class call field is read first, with the legacy metadata spelling
    # as a fallback.
    namespace = tool_call_group(ir_tool_call)
    if namespace:
        item["namespace"] = namespace
    # Ferry the full provider_metadata so a foreign-converter carrier
    # (e.g. a Google thought_signature) survives a Responses hop.
    if metadata:
        item[CARRIER_KEY] = metadata
    return item


_INTRINSIC_KIND_TO_ITEM: dict[str, str] = {
    "web_search": "web_search_call",
    "code_interpreter": "code_interpreter_call",
    "file_search": "file_search_call",
    "shell": "shell_call",
    "computer_use": "computer_call",
}

_ITEM_TO_INTRINSIC_KIND: dict[str, str] = {
    "shell_call": "shell",
    "computer_call": "computer_use",
    "code_interpreter_call": "code_interpreter",
    "web_search_call": "web_search",
    "file_search_call": "file_search",
}

#: Server result item type per intrinsic kind, derived from the call mapping so
#: the two cannot drift.  A result for one of these kinds is its own flat
#: ``*_call_output`` item; a web search has no separate output item (its payload
#: rides on the call item's ``action``), so it is absent here and a web-search
#: result is dropped on the write leg exactly as ``ir_tool_result_to_p`` drops
#: it.
_INTRINSIC_KIND_TO_RESULT_ITEM: dict[str, str] = {
    kind: f"{item_type}_output"
    for kind, item_type in _INTRINSIC_KIND_TO_ITEM.items()
    if item_type != "web_search_call"
}

#: Keys on a native server item that describe its identity, not its payload.
#: Read by the fallback below, which is reachable only for the recognised
#: kinds that have no entry in ``_SERVER_ITEM_INPUT_KEYS`` (``shell_call``,
#: ``computer_call``, ``file_search_call``); an item type this converter does
#: not recognise raises before reaching it.
_SERVER_ITEM_STRUCTURAL_KEYS = frozenset({"type", "id", "call_id", "status"})

#: Payload keys lifted into ``tool_input`` for server items whose write path
#: reads them by name (``code_interpreter_call``).  ``web_search_call`` is
#: handled separately because its payload is nested under ``action``.
_SERVER_ITEM_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "code_interpreter_call": ("code", "outputs", "container_id"),
}

#: The ``web_search_call.action.type`` the writer emits when it has nothing
#: better to go on.  Any other type is carried on ``tool_input["action_type"]``
#: so the writer rebuilds the same discriminator.
_DEFAULT_WEB_SEARCH_ACTION = "search"


def _server_item_tool_input(item: dict[str, Any], item_type: str) -> dict[str, Any]:
    """Build an IR ``tool_input`` dict from a native server item's payload.

    A native server item (``web_search_call``, ``code_interpreter_call``, …)
    carries its payload under type-specific keys, unlike the ``arguments``
    JSON string a ``function_call`` uses.  Reading ``arguments`` alone loses
    the query / the code.  ``web_search_call`` nests its payload under
    ``action``; that action's own fields are lifted, with its discriminator
    moved to ``action_type`` when it is not the default, so the writer can
    rebuild whichever action shape came in (``search``, ``open_page``,
    ``find`` — see the ``Action*`` types on the OpenAI side).

    Args:
        item: Native OpenAI Responses server item dict.
        item_type: The item's wire ``type`` (e.g. ``"web_search_call"``).

    Returns:
        A JSON-serializable ``tool_input`` dict; empty when the item carries
        no payload.
    """
    if item_type == "web_search_call":
        action = item.get("action")
        if not isinstance(action, dict):
            return {}
        lifted = {key: value for key, value in action.items() if key != "type"}
        action_type = action.get("type")
        if action_type and action_type != _DEFAULT_WEB_SEARCH_ACTION:
            lifted["action_type"] = action_type
        return lifted

    keys = _SERVER_ITEM_INPUT_KEYS.get(item_type)
    if keys is not None:
        return {key: item[key] for key in keys if item.get(key) is not None}

    # Any other recognised server item: use its own payload fields rather
    # than the (absent) ``arguments`` string.
    return {
        key: value
        for key, value in item.items()
        if key not in _SERVER_ITEM_STRUCTURAL_KEYS
    }


#: Server output item types this format recognises as intrinsic calls.
_SERVER_ITEM_TYPES: frozenset[str] = frozenset(_ITEM_TO_INTRINSIC_KIND)


def _p_server_item_to_ir(
    provider_tool_call: dict[str, Any], item_type: str
) -> ToolCallPart:
    """Read a native server item into an intrinsic ``ToolCallPart``.

    The canonical kind comes from the item type, not the wire name, so the
    definition side and ``intrinsic_kind`` agree.  The item's ``status`` is kept
    so the write leg can restore it (#981).
    """
    intrinsic_kind = _ITEM_TO_INTRINSIC_KIND.get(item_type, item_type)
    extra: dict[str, Any] = {}
    status = provider_tool_call.get("status")
    if status is not None:
        extra["responses_status"] = status
    return cast(
        ToolCallPart,
        make_intrinsic_tool_call(
            tool_call_id=provider_tool_call.get(
                "call_id", provider_tool_call.get("id", "")
            ),
            # Canonical kind, not the raw wire type: the definition side and
            # ``intrinsic_kind`` already speak the canonical name, so a raw
            # ``web_search_call`` here disagrees with both.
            tool_name=intrinsic_kind,
            tool_input=_server_item_tool_input(provider_tool_call, item_type),
            intrinsic_kind=intrinsic_kind,
            extra_metadata=extra or None,
        ),
    )


_SERVER_ITEM_ID_PREFIX: dict[str, str] = {
    "web_search_call": "ws_",
    "code_interpreter_call": "ci_",
    "file_search_call": "fs_",
}


def _build_server_item(
    item_type: str,
    *,
    item_id: str,
    intrinsic_kind: str,
    fields: dict[str, Any],
    arguments: str,
    status: str,
) -> dict[str, Any]:
    """Build a native server item from IR tool input.

    A server item identifies itself with ``id`` — not the ``call_id`` /
    ``arguments`` pair a client ``function_call`` uses — and carries its
    payload under type-specific keys: ``action`` for a web search, the
    ``code`` / ``container_id`` siblings for the interpreter.  The API
    requires ``id`` on every server item and rejects ``arguments`` / ``call_id``
    on one (verified live 2026-10-13: replaying a written
    ``code_interpreter_call`` without ``id`` answers
    ``400 Missing required parameter: 'input[2].id'``).
    """
    # The API requires a server item's id to carry its own prefix (verified
    # live: ``Expected an ID that begins with 'ws'``); a foreign id is prefixed.
    id_prefix = _SERVER_ITEM_ID_PREFIX.get(item_type)
    if id_prefix:
        item_id = prefixed_tool_call_id(item_id, id_prefix)
    if item_type == "web_search_call":
        action: dict[str, Any] = {
            "type": fields.get("action_type") or _DEFAULT_WEB_SEARCH_ACTION
        }
        # Whatever the read side lifted goes back verbatim, so a multi-query
        # search or an `open_page`/`find` action survives the round trip
        # instead of being flattened to `{"type": "search", "query": …}`.
        action.update({k: v for k, v in fields.items() if k != "action_type"})
        item: dict[str, Any] = {
            "type": "web_search_call",
            "id": item_id,
            "status": status,
        }
        if len(action) > 1:
            item["action"] = action
        return item

    if item_type == "function_call":
        return {
            "type": "function_call",
            "call_id": item_id,
            "name": intrinsic_kind,
            "arguments": arguments,
        }

    item = {"type": item_type, "id": item_id}
    if item_type == "code_interpreter_call":
        if fields.get("code") is not None:
            item["code"] = fields["code"]
        for key in ("container_id", "outputs"):
            if fields.get(key) is not None:
                item[key] = fields[key]
    elif item_type == "file_search_call":
        # The wire item carries ``queries`` (a list) and ``results``; the
        # singular ``query`` this branch used to emit is rejected live
        # (``400 Missing required parameter: 'input[1].queries'``), so emit the
        # payload the read side lifted, and only when it is present — a
        # streamed item has none at ``added`` time, matching the web-search
        # branch's "no action until there is one".
        if fields.get("queries"):
            item["queries"] = fields["queries"]
        if fields.get("results") is not None:
            item["results"] = fields["results"]
    else:
        # shell_call / computer_call: the read side lifted the payload keys, so
        # carry them back verbatim, and re-add ``call_id`` — the published item
        # schema requires it, and the read side keeps the wire ``call_id`` as
        # the IR's ``tool_call_id``.  The item's own ``id`` is not lifted, so
        # ``id`` mirrors the call id here.
        item.update(fields)
        item["call_id"] = item_id
    return item


def _server_item_renderable(item_type: str, fields: Mapping[str, Any]) -> bool:
    """Whether the IR carries the payload the API requires for this server item.

    Used on the history leg: a replayed turn carries the item's own payload, so
    the native item can be re-emitted; a foreign request that lacks it is
    degraded to a plain function call so the request stays valid.  Required
    fields verified live 2026-10-13 (a ``code_interpreter_call`` without
    ``container_id`` and a ``file_search_call`` without complete ``results``
    are both rejected).

    Args:
        item_type: Native server item type.
        fields: IR tool input for the call.

    Returns:
        True when the item can be emitted natively on the history leg.
    """
    if item_type == "code_interpreter_call":
        return fields.get("container_id") is not None
    if item_type == "file_search_call":
        results = fields.get("results")
        return (
            bool(fields.get("queries"))
            and isinstance(results, list)
            and all(
                isinstance(result, dict) and result.get("filename")
                for result in results
            )
        )
    # shell_call / computer_call: no live replay of either shape exists and the
    # published item schema is unverified here, so keep the ``function_call``
    # degradation rather than emit a native item on the schema's word alone.
    # The IR carries what a replay would need (``tool_call_id`` and
    # ``responses_status``); the caution is about the wire shape, not a field.
    return False


def _warn_dropped_intrinsic_result(
    ir_tool_result: ToolResultPart, kwargs: dict[str, Any]
) -> None:
    """Report an ``is_error`` flag on an intrinsic result that has no item.

    Args:
        ir_tool_result: IR tool result part.
        kwargs: Forwarded caller kwargs (``warnings``, etc.).
    """
    if ir_tool_result.get("is_error"):
        warn_tool_error_dropped(
            kwargs.get("warnings"),
            format_name="OpenAI Responses",
            tool_call_id=sanitize_tool_call_id(ir_tool_result.get("tool_call_id", "")),
        )


def _build_intrinsic_result_item(
    ir_tool_result: ToolResultPart,
    result_item_type: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build a native intrinsic result item for the response leg.

    On the response leg the call stays native (e.g. ``shell_call``), so its
    result must pair with the matching native output type (``shell_call_output``).

    Args:
        ir_tool_result: IR tool result part.
        result_item_type: Native output item type (e.g. ``shell_call_output``).
        kwargs: Forwarded caller kwargs (``warnings``, etc.).
    """
    result_content = ir_tool_result.get("result") or ir_tool_result.get("content", "")

    if isinstance(result_content, list):
        from .content_ops import OpenResponsesContentOps

        output = convert_ir_content_blocks_to_p(result_content, OpenResponsesContentOps)
    elif isinstance(result_content, dict):
        output = json.dumps(result_content)
    elif isinstance(result_content, str):
        output = result_content
    else:
        output = str(result_content)

    call_id = sanitize_tool_call_id(ir_tool_result["tool_call_id"])

    if ir_tool_result.get("is_error"):
        output = mark_tool_error(output, text_block_type="input_text")
        warn_tool_error(
            kwargs.get("warnings"),
            format_name="OpenAI Responses",
            tool_call_id=call_id,
        )

    return {
        "type": result_item_type,
        "call_id": call_id,
        "output": output,
        "status": "completed",
    }


def _wire_tool_call_group(
    provider_tool_call: Mapping[str, Any], carried: Mapping[str, Any]
) -> str | None:
    """Read the group a Responses tool call carries, wire field first.

    The group is spelled ``namespace`` on the wire; *carried* is the ferried
    provider metadata, which may still hold that spelling from a foreign hop.
    The wire value wins when both are present.  A missing, empty, or non-string
    value reads as no group.
    """
    for candidate in (provider_tool_call.get("namespace"), carried.get("namespace")):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _custom_tool_call_to_ir(provider_tool_call: dict[str, Any]) -> ToolCallPart:
    """Parse a ``custom_tool_call`` item into an IR tool call part.

    Custom tools carry plain text ``input`` instead of JSON ``arguments``.
    IR requires ``tool_input`` to be a dict, so the text is wrapped as
    ``{"input": str}`` — unless it happens to parse as a JSON object, in
    which case it is kept structured so other converters can read fields.
    """
    input_str = provider_tool_call.get("input", "")
    try:
        parsed_input = json.loads(input_str) if input_str else {}
    except (json.JSONDecodeError, TypeError):
        parsed_input = {"input": input_str}
    if not isinstance(parsed_input, dict):
        parsed_input = {"input": parsed_input}

    part = ToolCallPart(
        type="tool_call",
        tool_call_id=provider_tool_call.get(
            "call_id", provider_tool_call.get("id", "")
        ),
        tool_name=provider_tool_call.get("name", ""),
        tool_input=parsed_input,
        tool_type="custom",
    )
    # Keep it: (name, group) is what the request leg maps back.  The group is
    # first-class on the call; the legacy metadata spelling, if a foreign
    # carrier brought one in, is folded into that field rather than duplicated.
    restored = provider_tool_call.get(CARRIER_KEY)
    pm: dict[str, Any] = dict(restored) if isinstance(restored, dict) else {}
    group = _wire_tool_call_group(provider_tool_call, pm)
    pm.pop("namespace", None)
    if group:
        part["tool_group"] = group
    if pm:
        part["provider_metadata"] = pm
    return part


# ==================== additional_tools extraction ====================


def _collect_additional_tools(
    input_items: list[Any],
    warnings: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect raw tool dicts from ``additional_tools`` input items.

    Returns provider-format tool dicts exactly as declared.  Namespace
    containers and name collisions are handled downstream by
    ``p_tool_definition_to_ir`` (namespace flattening) and
    :func:`~llm_rosetta.ir.passes.qualify_tool_names` (collision resolution).

    Args:
        input_items: The raw Responses ``input`` value.
        warnings: A list to collect the losses in, for a caller outside a
            conversion; inside one they go to its sink.

    Returns:
        A ``(tools, warnings)`` pair.
    """
    tools: list[dict[str, Any]] = []
    collected: list[str] = []

    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != ADDITIONAL_TOOLS_ITEM_TYPE:
            continue
        for entry in item.get("tools") or []:
            if not isinstance(entry, dict):
                collected.append(
                    report_loss(
                        "Non-dict entry in additional_tools — skipped",
                        subject="additional_tools",
                        warnings=warnings,
                    )
                )
                continue
            tools.append(entry)

    return tools, collected


def strip_additional_tools_items(input_items: list[Any]) -> list[Any]:
    """Drop ``additional_tools`` items from a Responses ``input`` list.

    Called once their tool definitions have been harvested.  Leaving them
    in place would route them through the passthrough path, which strands
    a content-less assistant message.
    """
    return [
        item
        for item in input_items
        if not (
            isinstance(item, dict) and item.get("type") == ADDITIONAL_TOOLS_ITEM_TYPE
        )
    ]


def harvest_additional_tools(
    input_items: Any,
    warnings: list[str] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Extract ``additional_tools`` definitions and drop the spent items.

    Tool definitions are returned as raw provider-format dicts.  Namespace
    flattening and name-collision resolution are handled downstream by
    the normal tool conversion pipeline.

    Args:
        input_items: The raw Responses ``input`` value (list or otherwise).
        warnings: A list to collect the losses in, for a caller outside a
            conversion.  Inside one the report goes to the conversion's sink,
            which is where a caller that installed its own expects it.

    Returns:
        A ``(tools, input_items)`` pair.
    """
    if not isinstance(input_items, list):
        return [], input_items

    stripped = strip_additional_tools_items(input_items)
    if len(stripped) == len(input_items):
        return [], input_items

    nested_tools, nested_warnings = _collect_additional_tools(input_items, warnings)
    return nested_tools, stripped
    return nested_tools, stripped
