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
from typing import Any, cast

from ..base.helpers.tool_content import (
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
from ..base import BaseToolOps
from ..base.helpers import (
    extract_part_ids,
    log_orphan_warnings,
    sanitize_schema,
    sanitize_tool_call_id,
)

logger = logging.getLogger(__name__)

#: Responses input item type that carries tool definitions inline (Codex).
ADDITIONAL_TOOLS_ITEM_TYPE = "additional_tools"


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


def _flatten_namespace_tool(
    provider_tool: dict[str, Any],
) -> list[ToolDefinition]:
    """Expand a ``type: "namespace"`` container into individual IR tools.

    Iterates child tools, converts each via ``p_tool_definition_to_ir``,
    and tags every result with namespace metadata for future round-trip.

    Nested namespaces (namespace inside namespace) are skipped with a
    warning — no real-world use case exists, and the OpenAI function-name
    constraint ``^[a-zA-Z0-9_-]+$`` prohibits hierarchical separators.
    """
    namespace_name = provider_tool.get("name", "")
    namespace_desc = provider_tool.get("description", "")
    children = provider_tool.get("tools", [])
    if not children:
        return []

    results: list[ToolDefinition] = []
    for i, child in enumerate(children):
        if not isinstance(child, dict):
            logger.warning(
                "Skipping non-dict child at index %d in namespace %r",
                i,
                namespace_name,
            )
            continue

        child_type = child.get("type", "function")
        if child_type == "namespace":
            logger.warning(
                "Skipping nested namespace %r inside %r — "
                "only one level of nesting is supported",
                child.get("name", ""),
                namespace_name,
            )
            continue

        child_name = child.get("name") or (
            child.get("function", {}).get("name")
            if isinstance(child.get("function"), dict)
            else None
        )
        if not child_name:
            logger.warning(
                "Skipping unnamed child at index %d in namespace %r",
                i,
                namespace_name,
            )
            continue

        converted = OpenAIResponsesToolOps.p_tool_definition_to_ir(child)
        if converted is None:
            continue

        items = converted if isinstance(converted, list) else [converted]
        for item in items:
            meta = dict(item.get("metadata", {}))
            meta["namespace"] = namespace_name
            if namespace_desc:
                meta["namespace_description"] = namespace_desc
            if child.get("defer_loading"):
                meta["defer_loading"] = True
            item_copy = dict(item)
            item_copy["metadata"] = meta
            results.append(cast(ToolDefinition, item_copy))

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
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": tool_call_id,
        "name": tool_name,
        "arguments": arguments,
        "status": "completed",
    }


# ==================== additional_tools extraction ====================


def _collect_additional_tools(
    input_items: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect raw tool dicts from ``additional_tools`` input items.

    Returns provider-format tool dicts exactly as declared.  Namespace
    containers and name collisions are handled downstream by
    ``p_tool_definition_to_ir`` (namespace flattening) and
    ``_dedup_ir_tool_names`` (collision resolution).
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


class OpenAIResponsesToolOps(BaseToolOps):
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
        # Return passthrough tools as-is (web_search, etc.)
        passthrough = ir_tool.get("_passthrough")
        if passthrough is not None:
            return dict(passthrough)

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
        result = {
            "type": "function",
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
            "parameters": parameters,
            "strict": False,
        }
        return result

    @staticmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition]:
        """OpenAI Responses tool definition → IR ToolDefinition.

        Handles both flat format (Responses API native) and nested format
        (with ``function`` key).  ``type: "namespace"`` containers are
        expanded into individual child tools via ``_flatten_namespace_tool``.
        Non-function tool types without a ``name`` field (e.g.
        ``web_search``) are stored as passthrough so they can be
        round-tripped without modification.  Named non-function tools (e.g.
        Codex ``"custom"`` ``apply_patch``) are downgraded to IR
        ``type: "function"`` so the request passes IR validation; this
        mirrors the existing downgrade in ``openai_chat/tool_ops.py``.  The
        original provider type is retained in ``metadata["provider_type"]``
        for diagnostics.

        Args:
            provider_tool: OpenAI Responses tool definition dict.

        Returns:
            IR ToolDefinition, or list of ToolDefinitions for namespace
            containers.
        """
        _IR_ALLOWED_TYPES = {"function", "mcp", "custom"}

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
            # Downgrade unknown provider tool types (e.g. Codex "custom") to
            # IR "function" so the request passes IR validation.
            ir_type = tool_type if tool_type in _IR_ALLOWED_TYPES else "function"
            result = {
                "type": ir_type,
                "name": provider_tool.get("name", ""),
                "description": provider_tool.get("description", ""),
                "parameters": params,
            }
            if ir_type != tool_type:
                result["_downgraded_from"] = tool_type

        # Extract required_parameters from JSON Schema if available
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        downgraded_from = result.pop("_downgraded_from", None)
        meta: dict[str, Any] = (
            {"provider_type": downgraded_from} if downgraded_from else {}
        )
        fmt = provider_tool.get("format")
        if fmt:
            meta["format"] = fmt
        result["metadata"] = meta
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> str | dict:
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

        Returns:
            OpenAI tool_choice value (string or dict).
        """
        # Support both "mode" and legacy "type" field
        mode = ir_tool_choice.get("mode") or ir_tool_choice.get("type")

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
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """OpenAI Responses tool_choice → IR ToolChoice.

        Mapping:
        - ``"none"`` → ``mode:"none"``
        - ``"auto"`` → ``mode:"auto"``
        - ``"required"`` → ``mode:"any"``
        - ``{"type":"function","function":{"name":"..."}}`` → ``mode:"tool"``

        Args:
            provider_tool_choice: OpenAI tool_choice value.

        Returns:
            IR ToolChoice.
        """
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

        return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

    # ==================== Tool Call ====================

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → OpenAI Responses tool call item.

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
            return {
                "type": "custom_tool_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "input": input_str,
            }
        elif tool_type == "web_search":
            return {
                "type": "function_web_search",
                "call_id": tool_call_id,
                "query": tool_input.get("query", "")
                if isinstance(tool_input, dict)
                else "",
                "arguments": arguments,
            }
        elif tool_type == "code_interpreter":
            return {
                "type": "code_interpreter_call",
                "call_id": tool_call_id,
                "code": tool_input.get("code", "")
                if isinstance(tool_input, dict)
                else "",
                "arguments": arguments,
            }
        elif tool_type == "file_search":
            return {
                "type": "file_search_call",
                "call_id": tool_call_id,
                "query": tool_input.get("query", "")
                if isinstance(tool_input, dict)
                else "",
                "arguments": arguments,
            }
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

        Handles function_call, mcp_call, shell_call, computer_call,
        and code_interpreter_call item types.

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
            if item_id and item_id != call_id:
                part["provider_metadata"] = {"responses_item_id": item_id}
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
        elif item_type in ("shell_call", "computer_call", "code_interpreter_call"):
            tool_type_map = {
                "shell_call": "code_interpreter",
                "computer_call": "computer_use",
                "code_interpreter_call": "code_interpreter",
            }
            return cast(
                ToolCallPart,
                {
                    "type": "tool_call",
                    "tool_call_id": provider_tool_call.get(
                        "call_id", provider_tool_call.get("id", "")
                    ),
                    "tool_name": provider_tool_call.get("name", item_type),
                    "tool_input": tool_input,
                    "tool_type": tool_type_map.get(item_type, "function"),
                },
            )
        elif item_type == "custom_tool_call":
            # custom_tool_call uses plain text 'input' instead of JSON
            # 'arguments'.  Wrap as {"input": str} for IR compatibility
            # (tool_input must be dict).  If the input happens to be valid
            # JSON, parse it so cross-provider converters can inspect fields.
            input_str = provider_tool_call.get("input", "")
            try:
                parsed_input = json.loads(input_str) if input_str else {}
            except (json.JSONDecodeError, TypeError):
                parsed_input = {"input": input_str}
            # Ensure tool_input is always a dict
            if not isinstance(parsed_input, dict):
                parsed_input = {"input": parsed_input}
            return ToolCallPart(
                type="tool_call",
                tool_call_id=provider_tool_call.get(
                    "call_id", provider_tool_call.get("id", "")
                ),
                tool_name=provider_tool_call.get("name", ""),
                tool_input=parsed_input,
                tool_type="custom",
            )
        else:
            raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → OpenAI Responses tool call output item.

        Emits ``custom_tool_call_output`` when the context indicates the
        tool call was custom, otherwise ``function_call_output``.
        """
        result_content = ir_tool_result.get("result") or ir_tool_result.get(
            "content", ""
        )

        if isinstance(result_content, list):
            from .content_ops import OpenAIResponsesContentOps

            output = convert_ir_content_blocks_to_p(
                result_content, OpenAIResponsesContentOps
            )
        elif isinstance(result_content, dict):
            output = json.dumps(result_content)
        elif isinstance(result_content, str):
            output = result_content
        else:
            output = str(result_content)

        # Sanitize here to match the ID registered by ir_tool_call_to_p / streaming start.
        # Non-streaming ctx may be None (falls back to "function"), which is correct.
        call_id = sanitize_tool_call_id(ir_tool_result["tool_call_id"])
        ctx = kwargs.get("context")
        tool_type = ctx.get_tool_type(call_id) if ctx is not None else "function"

        if tool_type == "custom":
            return {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": output,
            }
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
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
        # Try to parse JSON output
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
                output = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Normalize provider-specific content blocks to IR format
        if isinstance(output, list):
            from .content_ops import OpenAIResponsesContentOps

            output = convert_content_blocks_to_ir(output, OpenAIResponsesContentOps)

        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("call_id", ""),
            result=output,
            is_error=provider_tool_result.get("is_error", False),
        )

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
