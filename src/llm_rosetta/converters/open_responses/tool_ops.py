"""
LLM-Rosetta - OpenAI Responses Tool Operations

OpenAI Responses API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.

Note: Responses API uses flat items (function_call, function_call_output)
instead of nested tool_calls within messages. Tool definitions use a flat
format with type/name/description/parameters at the top level.
"""

import json
import logging
from collections.abc import Mapping
from typing import Any, Literal, cast

from ..base.tools.content import (
    convert_content_blocks_to_ir,
    convert_ir_content_blocks_to_p,
)
from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig
from ..base import BaseToolOps, IntrinsicLeg
from ..base.helpers.opaque import CARRIER_KEY
from ..base.helpers.tool_group import (
    GENERATED_GROUP_MARKER,
    generated_group_id,
    tool_call_group,
)
from ..base.tools import (
    allowed_tool_names,
    extract_part_ids,
    get_native_definition,
    intrinsic_kind_of,
    ir_strict,
    make_intrinsic_tool_definition,
    log_orphan_warnings,
    make_intrinsic_tool_call,
    mark_tool_error,
    provider_allowed_tools_to_ir,
    sanitize_schema,
    prefixed_tool_call_id,
    sanitize_tool_call_id,
    warn_tool_error,
    warn_tool_error_dropped,
    write_ir_strict,
)

logger = logging.getLogger(__name__)

# Native OpenAI Responses server tools for client-declared intrinsic kinds.
# Responses native server-tool types → canonical intrinsic kind.
_RESPONSES_NATIVE_TYPE_TO_KIND: dict[str, str] = {
    "web_search": "web_search",
    "web_search_preview": "web_search",
    "code_interpreter": "code_interpreter",
    "file_search": "file_search",
}

_RESPONSES_INTRINSIC_TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {"type": "web_search"},
    "code_interpreter": {"type": "code_interpreter", "container": {"type": "auto"}},
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


def fix_orphaned_tool_calls(
    items: list[dict[str, Any]],
    *,
    placeholder: str = "[No output available yet]",
) -> list[dict[str, Any]]:
    """Fix mismatched function_calls and outputs in OpenAI Responses format.

    The OpenAI Responses API **strictly requires** bidirectional pairing
    between function_call and function_call_output items:

    1. Every ``function_call`` item (identified by ``call_id``) must have a
       matching ``function_call_output`` (**orphaned function_call**).
    2. Every ``function_call_output`` must have a preceding ``function_call``
       with the same ``call_id`` (**orphaned function_call_output**).

    Violations of either rule cause a 400 error.  Anthropic enforces the same
    strict pairing.  Only Google Gemini is lenient about both cases.

    This function handles both directions:

    - **Orphaned function_calls**: injects a synthetic
      ``function_call_output`` with *placeholder* content.
    - **Orphaned function_call_outputs**: removes output items whose
      ``call_id`` does not appear in any ``function_call`` item.

    The original list is **not** modified; a new list is returned.

    Args:
        items: OpenAI Responses format input items list.
        placeholder: Output string for injected synthetic results.

    Returns:
        A new items list with orphaned function_calls/outputs fixed.
    """
    known_call_ids = extract_part_ids(items, "function_call", "call_id")
    answered_ids = extract_part_ids(items, "function_call_output", "call_id")

    if not known_call_ids and not answered_ids:
        return items

    patched: list[dict[str, Any]] = []
    orphaned_call_ids: list[str] = []
    orphaned_output_ids: list[str] = []

    for item in items:
        itype = item.get("type")
        call_id = item.get("call_id")

        # Remove orphaned outputs
        if (
            itype == "function_call_output"
            and call_id
            and call_id not in known_call_ids
        ):
            orphaned_output_ids.append(call_id)
            continue

        patched.append(item)

        # Inject synthetic outputs for orphaned function_calls
        if itype == "function_call" and call_id and call_id not in answered_ids:
            orphaned_call_ids.append(call_id)
            patched.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": placeholder,
                }
            )

    log_orphan_warnings(
        logger,
        orphaned_call_ids,
        orphaned_output_ids,
        "function_call",
        "function_call_output",
    )
    return patched


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
        logger.warning(
            "Skipping non-dict child at index %d in group %r", index, group_name
        )
        return None
    if child.get("type", "function") == "namespace":
        logger.warning(
            "Skipping nested namespace %r inside %r — only one level of "
            "nesting is supported",
            child.get("name", ""),
            group_name,
        )
        return None
    if not _declared_child_name(child):
        logger.warning(
            "Skipping unnamed child at index %d in group %r", index, group_name
        )
        return None
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
        # per-request warning can report it, which the flatten cannot: this
        # conversion is process-cached.
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
    :func:`~llm_rosetta.converters.base.helpers.tool_group.generated_group_id`.
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
    item_id = metadata.get("responses_item_id")
    if not item_id:
        if tool_call_id and tool_call_id.startswith("fc_"):
            item_id = tool_call_id
        elif tool_call_id and tool_call_id.startswith("call_"):
            item_id = "fc_" + tool_call_id[5:]
        else:
            item_id = "fc_" + tool_call_id
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
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect raw tool dicts from ``additional_tools`` input items.

    Returns provider-format tool dicts exactly as declared.  Namespace
    containers and name collisions are handled downstream by
    ``p_tool_definition_to_ir`` (namespace flattening) and
    :func:`~llm_rosetta.capabilities.qualify_tool_names` (collision resolution).
    """
    tools: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != ADDITIONAL_TOOLS_ITEM_TYPE:
            continue
        for entry in item.get("tools") or []:
            if not isinstance(entry, dict):
                warnings.append("Non-dict entry in additional_tools — skipped")
                continue
            tools.append(entry)

    return tools, warnings


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
    warnings: list[str],
) -> tuple[list[dict[str, Any]], Any]:
    """Extract ``additional_tools`` definitions and drop the spent items.

    Tool definitions are returned as raw provider-format dicts.  Namespace
    flattening and name-collision resolution are handled downstream by
    the normal tool conversion pipeline.

    Args:
        input_items: The raw Responses ``input`` value (list or otherwise).
        warnings: Conversion warning sink, extended in place.

    Returns:
        A ``(tools, input_items)`` pair.
    """
    if not isinstance(input_items, list):
        return [], input_items

    stripped = strip_additional_tools_items(input_items)
    if len(stripped) == len(input_items):
        return [], input_items

    nested_tools, nested_warnings = _collect_additional_tools(input_items)
    warnings.extend(nested_warnings)
    return nested_tools, stripped


class OpenResponsesToolOps(BaseToolOps):
    """OpenAI Responses API tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → OpenAI Responses tool definition.

        Responses API uses a flat format:
        ``{"type": "function", "name": "...", "description": "...", "parameters": {...}}``

        Non-function passthrough tools (e.g. ``web_search``) stored in
        ``_passthrough`` are returned as-is.

        Args:
            ir_tool: IR tool definition.

        Returns:
            OpenAI Responses tool definition dict.
        """
        if ir_tool.get("type") == "intrinsic":
            stored = get_native_definition(ir_tool, "openai_responses")
            if stored is not None:
                return dict(stored)
            kind = intrinsic_kind_of(ir_tool)
            native = _RESPONSES_INTRINSIC_TOOLS.get(kind)
            if native is None:
                logger.warning(
                    "OpenAI Responses has no server tool for intrinsic kind %r;"
                    " dropping",
                    kind,
                )
                return {}
            return dict(native)

        # Passthrough tools (web_search, etc.) go back as-is except for the
        # name, since a rename must follow the tool upstream or nothing else
        # in the request agrees on it.
        #
        # The `provider_tool.get("name")` half is required, not defensive:
        # `_synthesize_passthrough_tool` falls back to the type string as the
        # IR name, which is truthy, so testing `renamed` alone would give
        # every bare {"type": "web_search"} a `name` it never had.
        passthrough = ir_tool.get("_passthrough")
        if passthrough is not None:
            provider_tool = dict(passthrough)
            renamed = ir_tool.get("name")
            if renamed and provider_tool.get("name"):
                provider_tool["name"] = renamed
            return provider_tool

        tool_type = ir_tool.get("type", "function")

        if tool_type == "custom":
            result: dict[str, Any] = {
                "type": "custom",
                "name": ir_tool["name"],
            }
            desc = ir_tool.get("description", "")
            if desc:
                result["description"] = desc
            metadata = ir_tool.get("metadata") or {}
            fmt = metadata.get("format")
            if fmt:
                result["format"] = fmt
            return result

        parameters = ir_tool.get("parameters", {})
        if isinstance(parameters, dict):
            parameters = sanitize_schema(parameters)
        metadata = ir_tool.get("metadata") or {}
        strict = ir_strict(ir_tool)
        result: dict[str, Any] = {
            "type": "function",
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
            "parameters": parameters,
            # Responses defaults the flag, so a source that said nothing gets
            # the wire default rather than an absent field.
            "strict": False if strict is None else strict,
        }
        output_schema = metadata.get("output_schema")
        if output_schema is not None:
            result["output_schema"] = output_schema
        return result

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition]:
        """OpenAI Responses tool definition → IR ToolDefinition.

        Handles both flat format (Responses API native) and nested format
        (with ``function`` key).  ``type: "namespace"`` containers are
        expanded into individual child tools via ``_flatten_namespace_tool``.
        A tool whose type is outside the IR set (e.g. ``web_search``) is
        stored as passthrough, keeping its original type in
        ``metadata["provider_type"]`` so ``ir_tool_definition_to_p`` can
        restore it unmodified.  Everything else is an IR type already —
        ``function``, ``mcp`` or ``custom`` — and keeps it.  Notably a Codex
        ``custom`` tool such as ``apply_patch`` is *not* downgraded here; a
        provider that cannot accept one is served later and separately, by
        :func:`~llm_rosetta.capabilities.downgrade_custom_tools`, which
        records itself under ``metadata["_downgraded_from"]``.

        Args:
            provider_tool: OpenAI Responses tool definition dict.

        Returns:
            IR ToolDefinition, or list of ToolDefinitions for namespace
            containers.
        """
        _IR_ALLOWED_TYPES = {"function", "mcp", "custom", "intrinsic"}

        # Handle nested format ({"type": "function", "function": {...}})
        if "function" in provider_tool and isinstance(provider_tool["function"], dict):
            func = provider_tool["function"]
            result: dict[str, Any] = {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
        else:
            tool_type = provider_tool.get("type", "function")
            if tool_type == "namespace":
                return _flatten_namespace_tool(provider_tool)
            kind = _RESPONSES_NATIVE_TYPE_TO_KIND.get(tool_type)
            if kind is not None:
                return cast(
                    ToolDefinition,
                    make_intrinsic_tool_definition(
                        kind,
                        description=provider_tool.get("description", ""),
                        native=provider_tool,
                        native_base="openai_responses",
                    ),
                )
            # Non-function tools outside the IR type set (e.g. web_search or
            # Codex custom apply_patch) are stored as passthrough to avoid
            # lossy conversion. IR ``type`` is forced to "function" to
            # satisfy validation; ``ir_tool_definition_to_p`` restores the
            # original payload on the outbound leg.
            if tool_type != "function" and tool_type not in _IR_ALLOWED_TYPES:
                return _synthesize_passthrough_tool(provider_tool, tool_type)

            # Flat format (Responses API native).
            # Custom tools use "schema" instead of "parameters".
            params = provider_tool.get("parameters", {})
            if tool_type != "function" and not params:
                params = provider_tool.get("schema", {})
            # No downgrade to make: every type outside the IR set has already
            # returned above as a passthrough, so what reaches here is an IR
            # type and stays one.
            result = {
                "type": tool_type,
                "name": provider_tool.get("name", ""),
                "description": provider_tool.get("description", ""),
                "parameters": params,
            }

        # Extract required_parameters from JSON Schema if available
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        meta: dict[str, Any] = {}
        fmt = provider_tool.get("format")
        if fmt:
            meta["format"] = fmt
        output_schema = provider_tool.get("output_schema")
        if output_schema is not None:
            meta["output_schema"] = output_schema
        result["metadata"] = meta
        # The strict flag is a first-class IR field now; metadata["strict"] was
        # where it used to live and is still read by ``ir_strict``.
        write_ir_strict(result, provider_tool.get("strict"))
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(
        ir_tool_choice: ToolChoice,
        *,
        allowed_tools: dict[str, Any] | None = None,
        tool_types: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str | dict:
        """IR ToolChoice → OpenAI Responses tool_choice parameter.

        Mapping:
        - ``mode:"none"`` → ``"none"``
        - ``mode:"auto"`` → ``"auto"``
        - ``mode:"any"`` → ``"required"``
        - ``mode:"required"`` → ``"required"``
        - ``mode:"tool"`` → ``{"type":"function","function":{"name":"..."}}``

        Also supports legacy ``type`` field for backward compatibility.

        Args:
            ir_tool_choice: IR tool choice.
            allowed_tools: Legacy copy of the wire object, retrieved from
                ``provider_extensions``.  Only used when the IR carries no
                ``allowed_tools`` of its own, so IR built before that field
                existed still round-trips.
            **kwargs: ``warnings`` (list[str] | None) — optional warning sink
                for a restriction naming a tool the request does not declare.
            tool_types: Declared tool name → IR type, used to spell each
                allow-list entry.  An entry's ``type`` must match how the tool
                was declared — the API answers "Tool choice 'custom' not found
                in 'tools' parameter" for a function-shaped entry naming a
                custom tool, and the reverse — so it is derived from the
                declaration rather than carried in the IR, where a downgraded
                custom tool would leave it stale.  A name the map does not
                know is written as ``function``.

        Returns:
            OpenAI tool_choice value (string or dict).
        """
        mode = ir_tool_choice.get("mode") or ir_tool_choice.get("type")
        wire_mode = {
            "any": "required",
            "required": "required",
            "none": "none",
            "auto": "auto",
        }.get(str(mode), "auto")

        # The IR states the restriction as tool identities, so rebuild the
        # wire object around the IR mode.
        names = allowed_tool_names(ir_tool_choice)
        if names:
            declared_types = tool_types or {}
            warnings = kwargs.get("warnings")
            if warnings is not None and tool_types is not None:
                # The upstream refuses a restriction naming a tool it was not
                # sent ("Tool choice 'x' not found in 'tools' parameter"), so
                # the client is told here as well as by the naming pass — a
                # library caller driving the converter directly has no naming
                # pass to hear it from.
                undeclared = [n for n in names if n not in declared_types]
                if undeclared:
                    warnings.append(
                        "tool_choice.allowed_tools names "
                        f"{', '.join(repr(n) for n in undeclared)}, which the "
                        "request does not declare; the provider rejects a "
                        "restriction naming a tool it was not sent"
                    )
            return {
                "type": "allowed_tools",
                "tools": [
                    {"type": declared_types.get(name, "function"), "name": name}
                    for name in names
                ],
                "mode": wire_mode,
            }

        # Legacy ``provider_extensions`` passthrough, re-emitted verbatim with
        # its ``mode`` refreshed from the IR mode.
        if (
            isinstance(allowed_tools, dict)
            and allowed_tools.get("type") == "allowed_tools"
        ):
            out = dict(allowed_tools)
            out["mode"] = wire_mode
            return out

        # Support both "mode" and legacy "type" field
        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode in ("any", "required"):
            return "required"
        elif mode in ("tool", "function"):
            tool_name = ir_tool_choice.get("tool_name")
            if not tool_name and "function" in ir_tool_choice:
                tool_name = cast(dict, ir_tool_choice)["function"].get("name")
            if tool_name:
                if ir_tool_choice.get("tool_type") == "custom":
                    return {"type": "custom", "name": tool_name}
                return {"type": "function", "name": tool_name}
            return "required"

        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(
        provider_tool_choice: Any,
        *,
        extensions: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolChoice:
        """OpenAI Responses tool_choice → IR ToolChoice.

        Mapping:
        - ``"none"`` → ``mode:"none"``
        - ``"auto"`` → ``mode:"auto"``
        - ``"required"`` → ``mode:"any"``
        - ``{"type":"function","function":{"name":"..."}}`` → ``mode:"tool"``
        - ``{"type":"allowed_tools","tools":[...],"mode":...}`` → the same
          mode, plus ``allowed_tools`` identities.  Entries carry a name, and
          the ``namespace`` beside it when the source echoed a container.
        - ``{"type":"namespace","name":...,"tools":[...]}`` → ``mode:"auto"``,
          reported on *warnings*.  The IR ``ToolChoice`` has no group slot, so
          a forced container child cannot be carried and the call is no longer
          forced (parent issue #953).

        Args:
            provider_tool_choice: OpenAI tool_choice value.
            extensions: Unused; kept so older call sites still pass it.
            warnings: Optional warning sink.  Receives one entry when a
                namespace container form is dropped.

        Returns:
            IR ToolChoice.
        """
        if (
            isinstance(provider_tool_choice, dict)
            and provider_tool_choice.get("type") == "allowed_tools"
        ):
            return provider_allowed_tools_to_ir(
                provider_tool_choice,
                warnings=warnings,
                selector="tool_choice.allowed_tools",
            )

        if isinstance(provider_tool_choice, str):
            if provider_tool_choice == "none":
                return cast(ToolChoice, {"mode": "none", "tool_name": ""})
            elif provider_tool_choice == "auto":
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})
            elif provider_tool_choice == "required":
                return cast(ToolChoice, {"mode": "any", "tool_name": ""})
            return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        if isinstance(provider_tool_choice, dict):
            if provider_tool_choice.get("type") == "custom":
                return cast(
                    ToolChoice,
                    {
                        "mode": "tool",
                        "tool_name": provider_tool_choice.get("name", ""),
                        "tool_type": "custom",
                    },
                )
            if provider_tool_choice.get("type") == "function":
                tool_name = provider_tool_choice.get("name", "")
                if not tool_name:
                    func = provider_tool_choice.get("function", {})
                    tool_name = func.get("name", "")
                return cast(ToolChoice, {"mode": "tool", "tool_name": tool_name})
            if provider_tool_choice.get("type") == "namespace":
                if warnings is not None:
                    warnings.append(
                        "Dropped a namespace tool_choice the IR cannot "
                        "represent; fell back to 'auto', so the call is no "
                        "longer forced."
                    )
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

        return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

    # ==================== Tool Call ====================

    @staticmethod
    def stream_intrinsic_item(
        kind: str,
        item_id: str,
        tool_input: dict[str, Any] | None = None,
        *,
        status: str = "in_progress",
    ) -> dict[str, Any] | None:
        """Build a native server item for a streamed intrinsic call.

        The streaming counterpart of ``_build_server_item``: it lets the
        writer re-emit the provider's own server item type for an intrinsic
        ``ToolCallStartEvent`` instead of downgrading it to a client
        ``function_call``.  Returns ``None`` for a kind this dialect cannot
        render as a server item, so the caller drops the event rather than
        emitting a client call.
        """
        item_type = _INTRINSIC_KIND_TO_ITEM.get(kind)
        if item_type is None:
            return None
        return _build_server_item(
            item_type,
            item_id=item_id,
            intrinsic_kind=kind,
            fields=tool_input or {},
            arguments="",
            status=status,
        )

    @staticmethod
    def stream_result_item(
        kind: str, call_id: str, result: Any
    ) -> dict[str, Any] | None:
        """Build a server result item for a streamed ``ToolResultEvent``.

        The streaming counterpart of the result half of ``ir_tool_result_to_p``:
        it produces the provider's own ``*_call_output`` item.  Returns ``None``
        for a kind with no separate output item (``web_search`` — its result
        rides on the call item) or one this dialect cannot render, so the caller
        drops the event rather than re-encoding it as a client
        ``function_call_output``.
        """
        item_type = _INTRINSIC_KIND_TO_RESULT_ITEM.get(kind)
        if item_type is None:
            return None
        output: Any = result
        if isinstance(result, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_ir_content_blocks_to_p(result, OpenResponsesContentOps)
        elif isinstance(result, dict):
            output = json.dumps(result)
        elif result is None:
            output = ""
        return {"type": item_type, "call_id": call_id, "output": output}

    @classmethod
    def intrinsic_leg_from_flags(
        cls, kwargs: Any, *, default: IntrinsicLeg
    ) -> IntrinsicLeg:
        """Map this writer's ``for_history`` flag onto :data:`IntrinsicLeg`.

        The Responses write path says which leg it is on — ``message_ops``
        passes ``for_history=True``, the response writer says nothing — so the
        same hook pair answers differently per leg without a second predicate.
        A caller that says nothing takes the template's default, which is what
        this writer answered before the hooks existed.

        Args:
            kwargs: The keyword arguments the write template was called with.
            default: The leg when ``for_history`` is absent.

        Returns:
            ``"history"`` for a history write, else ``"response"``.
        """
        if "for_history" in kwargs:
            return "history" if kwargs["for_history"] else "response"
        return default

    @classmethod
    def intrinsic_renderable(
        cls, kind: str, fields: Mapping[str, Any], *, leg: IntrinsicLeg
    ) -> bool:
        """Whether a server item can be written for *kind* on this leg.

        A kind with no native item type (``google_search`` arriving from
        another lane, …) has one: it is written as a client ``function_call``
        on both legs.  A native item the IR cannot complete — the API requires
        a ``container_id`` on a code interpreter and queries with complete
        results on a file search, verified live 2026-10-13 — is still written
        natively on the response leg (the turn the provider just returned) and
        degraded on the history leg, where replaying it would be rejected.

        Args:
            kind: Canonical intrinsic kind (``intrinsic_kind_of``).
            fields: The IR ``tool_input`` for the call.
            leg: Which leg the part is written on.

        Returns:
            True when the call is written natively (see
            :meth:`ir_intrinsic_call_to_p`).
        """
        item_type = _INTRINSIC_KIND_TO_ITEM.get(kind)
        if item_type is None:
            return False
        # A `web_search_call` has a valid replay shape for history (an
        # `action`, no `call_id`/`arguments`), so it is the one kind whose
        # payload the API does not require on a replay.
        if item_type == "web_search_call" or leg == "response":
            return True
        return _server_item_renderable(item_type, fields if fields else {})

    @classmethod
    def intrinsic_call_degrades_on_history(cls, ir_tool_call: ToolCallPart) -> bool:
        """Whether the history-leg writer degrades this call to a client call.

        The result writer pairs with the call that is actually written, so it
        must consult the same decision.  See :meth:`intrinsic_renderable`.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.

        Returns:
            True when the call is written as a client ``function_call``.
        """
        fields = ir_tool_call.get("tool_input")
        return not cls.intrinsic_renderable(
            intrinsic_kind_of(ir_tool_call),
            fields if isinstance(fields, dict) else {},
            leg="history",
        )

    @classmethod
    def ir_intrinsic_call_to_p(
        cls,
        ir_tool_call: ToolCallPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """IR intrinsic ToolCallPart → Responses native server item.

        The shape is a client ``function_call`` on neither leg for an
        unrenderable kind: :meth:`intrinsic_renderable` decides, and the
        template's ``non_native_intrinsic_call_to_p`` writes the degradation,
        so the call and its result cannot disagree about which shape was
        written.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            Native server item, or ``None`` when the call is degraded.
        """
        kind = intrinsic_kind_of(ir_tool_call)
        tool_input = ir_tool_call.get("tool_input")
        fields = tool_input if isinstance(tool_input, dict) else {}
        if not cls.intrinsic_renderable(kind, fields, leg=leg):
            return None
        # A history part keys off the kind only.  It is deliberately not gated
        # by the target shim (which gates *definitions*): a history part is
        # context, and its shape here is independent of whether the same kind
        # is currently declared.  See the note in pipeline.py / #839.
        item_type = _INTRINSIC_KIND_TO_ITEM[kind]
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )
        tool_call_id = sanitize_tool_call_id(
            ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
        )
        # A web search restores the status the read side lifted; every other
        # server call the IR knows is complete.
        status = (ir_tool_call.get("provider_metadata") or {}).get(
            "responses_status"
        ) or "completed"
        return _build_server_item(
            item_type,
            item_id=tool_call_id,
            intrinsic_kind=kind,
            fields=fields,
            arguments=arguments,
            status=status,
        )

    @classmethod
    def non_native_intrinsic_call_to_p(
        cls,
        ir_tool_call: ToolCallPart,
        *,
        leg: IntrinsicLeg,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write an intrinsic call as the client ``function_call`` it degrades to.

        A Responses ``function_call`` is the only shape left for a server call
        this dialect cannot replay (see :meth:`intrinsic_renderable`), and it
        is the shape the matching result pairs with, so it carries the
        canonical kind as its name and the IR input as its arguments.

        Args:
            ir_tool_call: IR tool call part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            A client ``function_call`` item.
        """
        tool_input = ir_tool_call.get("tool_input")
        return {
            "type": "function_call",
            "call_id": sanitize_tool_call_id(
                ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
            ),
            "name": intrinsic_kind_of(ir_tool_call),
            "arguments": (
                json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else str(tool_input)
            ),
        }

    @classmethod
    def ir_intrinsic_result_to_p(
        cls,
        ir_tool_result: ToolResultPart,
        *,
        leg: IntrinsicLeg = "history",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """IR intrinsic ToolResultPart → Responses native server result item.

        Only the response leg has a native result item: on the history leg a
        native server call accepts no output item (its ``*_call_output`` type
        is rejected for a replayed call, and a ``function_call_output`` with a
        server call's id pairs with no client call), so the hook declines and
        the template's ``non_native_intrinsic_result_to_p`` decides between
        dropping the result and pairing it with a degraded call.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: Extra arguments.

        Returns:
            Native ``*_call_output`` item on the response leg, ``None``
            otherwise.
        """
        if leg != "response":
            return None
        result_item_type = _INTRINSIC_KIND_TO_RESULT_ITEM.get(
            intrinsic_kind_of(ir_tool_result)
        )
        if result_item_type is None:
            return None
        return _build_intrinsic_result_item(ir_tool_result, result_item_type, kwargs)

    @classmethod
    def non_native_intrinsic_result_to_p(
        cls,
        ir_tool_result: ToolResultPart,
        *,
        leg: IntrinsicLeg,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write (or drop) an intrinsic result with no native item.

        A result is written as a client ``function_call_output`` only on the
        **history** leg, and only for a call that was degraded to a client
        ``function_call``: every other case answers ``{}`` — a ``web_search``
        result rides on its call item, and a native server call pairs with no
        output item on either leg.  On the **response** leg a server result is
        never handed to the client as a client output — the client did not run
        the call — so it is dropped with a warning.

        Args:
            ir_tool_result: IR tool result part with ``tool_type == "intrinsic"``.
            leg: Which leg the part is written on.
            **kwargs: ``degraded_intrinsic_calls`` (dict[str, bool]) — intrinsic
                call id → whether the call writer degrades it, so a result
                pairs with the call that is actually written.

        Returns:
            A client ``function_call_output`` for a degraded call, else ``{}``.
        """
        kind = intrinsic_kind_of(ir_tool_result)
        if _INTRINSIC_KIND_TO_ITEM.get(kind) == "web_search_call":
            # Its payload rides on the call item; no separate output item on
            # either leg.
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            return {}
        if leg != "history":
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            warnings = kwargs.get("warnings")
            if isinstance(warnings, list):
                warnings.append(
                    f"OpenAI Responses has no server result item for kind "
                    f"{kind!r}; dropped the result of tool_call_id "
                    f"{ir_tool_result.get('tool_call_id')!r}"
                )
            return {}
        degraded = kwargs.get("degraded_intrinsic_calls")
        call_id = sanitize_tool_call_id(ir_tool_result.get("tool_call_id", ""))
        if degraded is not None and not degraded.get(call_id, True):
            _warn_dropped_intrinsic_result(ir_tool_result, kwargs)
            return {}
        return cast(dict[str, Any], cls.ir_client_result_to_p(ir_tool_result, **kwargs))

    @classmethod
    def ir_client_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → OpenAI Responses client tool call item.

        Converts to function_call or mcp_call depending on tool_type/tool_name.

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            OpenAI Responses tool call item dict.
        """
        tool_type = ir_tool_call.get("tool_type", "function")
        tool_call_id = sanitize_tool_call_id(
            ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
        )
        tool_name = ir_tool_call.get("tool_name", ir_tool_call.get("name", ""))
        tool_input = ir_tool_call.get("tool_input", ir_tool_call.get("arguments", {}))

        # Serialize tool_input
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        is_mcp = tool_type == "mcp" or (tool_name and tool_name.startswith("mcp://"))
        if is_mcp:
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": ir_tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "function":
            return _build_function_call_item(
                ir_tool_call, tool_call_id, tool_name, arguments
            )
        elif tool_type == "custom":
            # Custom tool calls use plain text 'input' instead of JSON
            # 'arguments'.  If tool_input has a single "input" key, unwrap
            # it to plain text; otherwise JSON-serialize the dict.
            if isinstance(tool_input, dict) and list(tool_input.keys()) == ["input"]:
                input_str = str(tool_input["input"])
            else:
                input_str = (
                    json.dumps(tool_input)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )
            item: dict[str, Any] = {
                "type": "custom_tool_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "input": input_str,
            }
            # Same round-trip as a function_call above.
            metadata = ir_tool_call.get("provider_metadata") or {}
            namespace = tool_call_group(ir_tool_call)
            if namespace:
                item["namespace"] = namespace
            if metadata:
                item[CARRIER_KEY] = metadata
            return item
        else:
            # Default to function_call
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": f"{tool_type}_{tool_name}",
                "arguments": arguments,
            }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """OpenAI Responses tool call item → IR ToolCallPart.

        Handles function_call, mcp_call, custom_tool_call, and server
        tool item types (shell_call, computer_call, code_interpreter_call,
        web_search_call, file_search_call) which map to intrinsic.

        Args:
            provider_tool_call: OpenAI Responses tool call item dict.

        Returns:
            IR ToolCallPart.
        """
        item_type = provider_tool_call.get("type")

        # Parse arguments
        arguments = provider_tool_call.get("arguments", {})
        if isinstance(arguments, dict):
            tool_input = arguments
        elif isinstance(arguments, str):
            try:
                tool_input = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                tool_input = {"input": arguments}
        else:
            tool_input = {}

        if item_type == "function_call":
            # Responses API has both 'id' (item ID, fc_ prefix) and
            # 'call_id' (correlation ID, call_ prefix). Store call_id as
            # tool_call_id for correlation, preserve 'id' in provider_metadata
            # for lossless round-trip.
            call_id = provider_tool_call.get("call_id")
            item_id = provider_tool_call.get("id", "")
            if not call_id:
                # Fallback: derive call_ prefix from fc_ prefix
                if item_id.startswith("fc_"):
                    call_id = "call_" + item_id[3:]
                else:
                    call_id = item_id
            part = ToolCallPart(
                type="tool_call",
                tool_call_id=call_id,
                tool_name=provider_tool_call.get("name", ""),
                tool_input=tool_input,
                tool_type="function",
            )
            # Keep it: (name, group) is what the request leg maps back.  The
            # group is first-class on the call now; a legacy metadata spelling
            # (a carrier from a foreign hop) folds into it rather than staying
            # a second copy.
            restored = provider_tool_call.get(CARRIER_KEY)
            pm: dict[str, Any] = dict(restored) if isinstance(restored, dict) else {}
            if item_id and item_id != call_id:
                pm["responses_item_id"] = item_id
            group = _wire_tool_call_group(provider_tool_call, pm)
            pm.pop("namespace", None)
            if group:
                part["tool_group"] = group
            if pm:
                part["provider_metadata"] = pm
            return part
        elif item_type == "mcp_call":
            # MCP call may use server/tool fields or name field
            server = provider_tool_call.get("server", "")
            tool = provider_tool_call.get("tool", provider_tool_call.get("name", ""))
            tool_name = f"mcp://{server}/{tool}" if server and tool else tool

            return ToolCallPart(
                type="tool_call",
                tool_call_id=provider_tool_call.get("id", ""),
                tool_name=tool_name,
                tool_input=tool_input,
                tool_type="mcp",
            )
        elif item_type in _SERVER_ITEM_TYPES:
            return _p_server_item_to_ir(provider_tool_call, item_type)
        elif item_type == "custom_tool_call":
            return _custom_tool_call_to_ir(provider_tool_call)
        else:
            raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

    # ==================== Tool Result ====================

    @classmethod
    def ir_client_result_to_p(
        cls, ir_tool_result: ToolResultPart, **kwargs: Any
    ) -> dict:
        """IR ToolResultPart → OpenAI Responses tool call output item.

        Emits ``custom_tool_call_output`` when the context indicates the
        tool call was custom, otherwise ``function_call_output``.

        The Responses API has no tool error field, so an ``is_error`` flag is
        re-encoded as a trailing ``input_text`` block (see
        ``base.tools.error_marker``).

        An intrinsic result reaches this path only when its call was written as
        a client ``function_call``; the hook pair below decides that.

        Args:
            ir_tool_result: IR tool result part.
            **kwargs: ``warnings`` (list[str]) receives a note when the error
                flag had to be re-encoded or dropped.
        """
        result_content = ir_tool_result.get("result") or ir_tool_result.get(
            "content", ""
        )

        if isinstance(result_content, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_ir_content_blocks_to_p(
                result_content, OpenResponsesContentOps
            )
        elif isinstance(result_content, dict):
            output = json.dumps(result_content)
        elif isinstance(result_content, str):
            output = result_content
        else:
            output = str(result_content)

        # Sanitize here to match the ID registered by ir_tool_call_to_p / streaming start.
        call_id = sanitize_tool_call_id(ir_tool_result["tool_call_id"])

        if ir_tool_result.get("is_error"):
            # Both function_call_output and custom_tool_call_output accept a
            # list of input_* content blocks, so the marker rides along as one.
            output = mark_tool_error(output, text_block_type="input_text")
            warn_tool_error(
                kwargs.get("warnings"),
                format_name="OpenAI Responses",
                tool_call_id=call_id,
            )

        # The IR part wins: it survives request history, where the context's
        # tool-type map is empty because only streaming populates it.
        tool_type = ir_tool_result.get("tool_type")
        if tool_type is None:
            ctx = kwargs.get("context")
            tool_type = ctx.get_tool_type(call_id) if ctx is not None else "function"

        if tool_type == "custom":
            return {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": output,
                "status": "completed",
            }
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
            "status": "completed",
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """OpenAI Responses tool call output → IR ToolResultPart.

        Handles function_call_output, custom_tool_call_output, and
        mcp_call_output.

        Args:
            provider_tool_result: OpenAI Responses tool result item dict.

        Returns:
            IR ToolResultPart.
        """
        output = provider_tool_result.get("output", "")
        # String outputs are opaque tool data, even when they contain JSON.
        # Only an actual list represents multimodal content blocks.
        if isinstance(output, list):
            from .content_ops import OpenResponsesContentOps

            output = convert_content_blocks_to_ir(output, OpenResponsesContentOps)

        part = ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("call_id", ""),
            result=output,
        )
        # Only a boolean is copied: the IR field is typed ``bool``, so a
        # malformed value would fail IR validation and take the request down.
        is_error = provider_tool_result.get("is_error")
        if isinstance(is_error, bool):
            part["is_error"] = is_error
        # Record the non-default types so the outbound leg can emit the
        # same item kind.  The context it would otherwise consult is only
        # populated while streaming a response, never by parsing history.
        item_type = provider_tool_result.get("type", "")
        tool_type = _RESULT_ITEM_TOOL_TYPES.get(item_type)
        if tool_type is not None:
            part["tool_type"] = tool_type
        intrinsic_kind = _INTRINSIC_RESULT_TYPES.get(item_type)
        if intrinsic_kind is not None:
            part["tool_type"] = "intrinsic"
            part["intrinsic_kind"] = intrinsic_kind
        return part

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → OpenAI Responses tool call config fields.

        Mapping:
        - ``disable_parallel`` → ``parallel_tool_calls`` (inverted)
        - ``max_calls`` → ``max_tool_calls``

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "disable_parallel" in ir_tool_config:
            result["parallel_tool_calls"] = not ir_tool_config["disable_parallel"]

        if "max_calls" in ir_tool_config:
            result["max_tool_calls"] = ir_tool_config["max_calls"]

        return result

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """OpenAI Responses tool call config → IR ToolCallConfig.

        Mapping:
        - ``parallel_tool_calls`` → ``disable_parallel`` (inverted)
        - ``max_tool_calls`` → ``max_calls``

        Args:
            provider_tool_config: Dict with OpenAI tool config fields.

        Returns:
            IR ToolCallConfig.
        """
        result: dict[str, Any] = {}

        if isinstance(provider_tool_config, dict):
            parallel = provider_tool_config.get("parallel_tool_calls")
            if parallel is not None:
                result["disable_parallel"] = not parallel

            max_calls = provider_tool_config.get("max_tool_calls")
            if max_calls is not None:
                result["max_calls"] = max_calls

        return cast(ToolCallConfig, result)


# Backward-compatible alias (deprecated): the OpenAI Responses profile reuses
# the same tool operations as the vendor-neutral Open Responses base.
OpenAIResponsesToolOps = OpenResponsesToolOps
