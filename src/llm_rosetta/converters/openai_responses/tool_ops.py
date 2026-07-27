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
from ..base.helpers import extract_part_ids, log_orphan_warnings, sanitize_schema

logger = logging.getLogger(__name__)

# IR ToolDefinition.type is restricted to Literal["function", "mcp"];
# see types/ir/tools.py.
_IR_ALLOWED_TOOL_TYPES = frozenset({"function", "mcp"})

# Responses API item types handled by the shared "extended" tool-call parser.
_EXTENDED_CALL_ITEM_TYPES = frozenset(
    {"shell_call", "computer_call", "code_interpreter_call"}
)

# Mapping from Responses extended item type → IR tool_type.
_EXTENDED_CALL_TYPE_MAP = {
    "shell_call": "code_interpreter",
    "computer_call": "computer_use",
    "code_interpreter_call": "code_interpreter",
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

        # After #177, all non-function tools entering IR are coerced to
        # type="function" and carry _passthrough (handled above).  Only
        # genuine function tools reach here.
        parameters = ir_tool.get("parameters", {})
        if isinstance(parameters, dict):
            parameters = sanitize_schema(parameters)
        result: dict[str, Any] = {
            "type": "function",
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
            "parameters": parameters,
            "strict": False,
        }
        return result

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """OpenAI Responses tool definition → IR ToolDefinition.

        Handles both flat format (Responses API native) and nested format
        (with ``function`` key).  Non-function tool types without a ``name``
        field (e.g. ``web_search``) are stored as passthrough so they can be
        round-tripped without modification.  Named non-function tools (e.g.
        Codex ``"custom"`` ``apply_patch``) are downgraded to IR
        ``type: "function"`` so the request passes IR validation; this
        mirrors the existing downgrade in ``openai_chat/tool_ops.py``.  The
        original provider type is retained in ``metadata["provider_type"]``
        for diagnostics.

        Args:
            provider_tool: OpenAI Responses tool definition dict.

        Returns:
            IR ToolDefinition.
        """
        # Handle nested format ({"type": "function", "function": {...}})
        if "function" in provider_tool and isinstance(provider_tool["function"], dict):
            result = OpenAIResponsesToolOps._p_nested_definition_to_ir(
                provider_tool["function"]
            )
        else:
            tool_type = provider_tool.get("type", "function")
            # Non-function tools outside the IR type set (e.g. web_search or
            # Codex custom apply_patch) are stored as passthrough to avoid
            # lossy conversion. IR ``type`` is forced to "function" to
            # satisfy validation; ``ir_tool_definition_to_p`` restores the
            # original payload on the outbound leg.
            if tool_type != "function" and tool_type not in _IR_ALLOWED_TOOL_TYPES:
                return OpenAIResponsesToolOps._p_extended_definition_to_ir(
                    provider_tool, tool_type
                )
            result = OpenAIResponsesToolOps._p_flat_definition_to_ir(
                provider_tool, tool_type
            )

        OpenAIResponsesToolOps._finalize_ir_tool_definition(result)
        return cast(ToolDefinition, result)

    @staticmethod
    def _p_nested_definition_to_ir(func: dict[str, Any]) -> dict[str, Any]:
        """Nested ``{"function": {...}}`` provider tool → intermediate IR dict."""
        return {
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }

    @staticmethod
    def _p_flat_definition_to_ir(
        provider_tool: dict[str, Any], tool_type: str
    ) -> dict[str, Any]:
        """Flat Responses-native tool → intermediate IR dict (with downgrade tag)."""
        # Custom tools use "schema" instead of "parameters".
        params = provider_tool.get("parameters", {})
        if tool_type != "function" and not params:
            params = provider_tool.get("schema", {})
        # Downgrade unknown provider tool types (e.g. Codex "custom") to
        # IR "function" so the request passes IR validation.
        ir_type = tool_type if tool_type in _IR_ALLOWED_TOOL_TYPES else "function"
        result: dict[str, Any] = {
            "type": ir_type,
            "name": provider_tool.get("name", ""),
            "description": provider_tool.get("description", ""),
            "parameters": params,
        }
        if ir_type != tool_type:
            result["_downgraded_from"] = tool_type
        return result

    @staticmethod
    def _p_extended_definition_to_ir(
        provider_tool: dict[str, Any], tool_type: str
    ) -> ToolDefinition:
        """Non-function extended tool (e.g. Codex custom) → passthrough IR."""
        # Synthesize a minimal JSON Schema for cross-provider degradation.
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
        desc = OpenAIResponsesToolOps._decorate_extended_description(
            provider_tool.get("description", ""), provider_tool.get("format")
        )
        result: dict[str, Any] = {
            "type": "function",
            "name": provider_tool.get("name", tool_type),
            "description": desc,
            "parameters": synth_params,
            "_passthrough": dict(provider_tool),
            "metadata": {"provider_type": tool_type},
            "required_parameters": (
                synth_params.get("required", []) if synth_params else []
            ),
        }
        return cast(ToolDefinition, result)

    @staticmethod
    def _decorate_extended_description(desc: str, fmt: Any) -> str:
        """Append format-hint suffix to *desc* when *fmt* metadata is present."""
        if not fmt:
            return desc
        fmt_type = fmt.get("type", "unknown")
        fmt_syntax = fmt.get("syntax", "")
        hint = f"[Output format: {fmt_type}"
        if fmt_syntax:
            hint += f", syntax: {fmt_syntax}"
        hint += "]"
        return f"{desc}\n\n{hint}" if desc else hint

    @staticmethod
    def _finalize_ir_tool_definition(result: dict[str, Any]) -> None:
        """In-place: fill required_parameters and metadata for a flat/nested IR tool."""
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        downgraded_from = result.pop("_downgraded_from", None)
        result["metadata"] = (
            {"provider_type": downgraded_from} if downgraded_from else {}
        )

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
                # Responses API format: {"type": "function", "name": "..."}
                # (NOT Chat Completions format which nests under "function" key)
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
            if provider_tool_choice.get("type") == "function":
                # Support both Responses format {"name": "..."} and
                # Chat Completions format {"function": {"name": "..."}}
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
        tool_call_id = ir_tool_call.get("tool_call_id", ir_tool_call.get("id", ""))
        tool_name = ir_tool_call.get("tool_name", ir_tool_call.get("name", ""))
        tool_input = ir_tool_call.get("tool_input", ir_tool_call.get("arguments", {}))

        # Serialize tool_input
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        # MCP call detected by either scheme prefix or explicit tool_type.
        if (tool_name and tool_name.startswith("mcp://")) or tool_type == "mcp":
            return OpenAIResponsesToolOps._build_p_mcp_call(
                tool_call_id, tool_name, arguments, ir_tool_call
            )
        if tool_type == "function":
            return OpenAIResponsesToolOps._build_p_function_call(
                tool_call_id, tool_name, arguments, ir_tool_call
            )
        if tool_type == "custom":
            return OpenAIResponsesToolOps._build_p_custom_call(
                tool_call_id, tool_name, tool_input
            )
        if tool_type == "web_search":
            return OpenAIResponsesToolOps._build_p_query_call(
                "function_web_search", tool_call_id, tool_input, arguments, "query"
            )
        if tool_type == "code_interpreter":
            return OpenAIResponsesToolOps._build_p_query_call(
                "code_interpreter_call", tool_call_id, tool_input, arguments, "code"
            )
        if tool_type == "file_search":
            return OpenAIResponsesToolOps._build_p_query_call(
                "file_search_call", tool_call_id, tool_input, arguments, "query"
            )
        # Default to function_call
        return {
            "type": "function_call",
            "call_id": tool_call_id,
            "name": f"{tool_type}_{tool_name}",
            "arguments": arguments,
        }

    @staticmethod
    def _build_p_mcp_call(
        tool_call_id: str,
        tool_name: str,
        arguments: str,
        ir_tool_call: ToolCallPart,
    ) -> dict:
        """Build a Responses API ``mcp_call`` item."""
        return {
            "type": "mcp_call",
            "id": tool_call_id,
            "name": tool_name,
            "arguments": arguments,
            "server_label": ir_tool_call.get("server_name", "default"),
            "status": "calling",
        }

    @staticmethod
    def _build_p_function_call(
        tool_call_id: str,
        tool_name: str,
        arguments: str,
        ir_tool_call: ToolCallPart,
    ) -> dict:
        """Build a Responses API ``function_call`` item."""
        metadata = ir_tool_call.get("provider_metadata") or {}
        item_id = metadata.get("responses_item_id") or (
            OpenAIResponsesToolOps._derive_function_item_id(tool_call_id)
        )
        return {
            "type": "function_call",
            "id": item_id,
            "call_id": tool_call_id,
            "name": tool_name,
            "arguments": arguments,
            "status": "completed",
        }

    @staticmethod
    def _derive_function_item_id(tool_call_id: str) -> str:
        """Ensure the Responses-API-required ``fc_`` prefix on function item IDs."""
        # The API requires 'id' to start with 'fc_' prefix. Convert
        # Chat-Completions/Anthropic prefixes appropriately.
        if tool_call_id and tool_call_id.startswith("fc_"):
            return tool_call_id
        if tool_call_id and tool_call_id.startswith("call_"):
            return "fc_" + tool_call_id[5:]
        # Other prefixes (e.g. toolu_ from Anthropic)
        return "fc_" + tool_call_id

    @staticmethod
    def _build_p_custom_call(
        tool_call_id: str, tool_name: str, tool_input: Any
    ) -> dict:
        """Build a Responses API ``custom_tool_call`` item."""
        # Custom tool calls use plain text 'input' instead of JSON 'arguments'.
        # If tool_input has a single "input" key, unwrap it to plain text;
        # otherwise JSON-serialize the dict.
        if isinstance(tool_input, dict) and list(tool_input.keys()) == ["input"]:
            input_str = str(tool_input["input"])
        elif isinstance(tool_input, dict):
            input_str = json.dumps(tool_input)
        else:
            input_str = str(tool_input)
        return {
            "type": "custom_tool_call",
            "call_id": tool_call_id,
            "name": tool_name,
            "input": input_str,
        }

    @staticmethod
    def _build_p_query_call(
        item_type: str,
        tool_call_id: str,
        tool_input: Any,
        arguments: str,
        field_key: str,
    ) -> dict:
        """Build a Responses API extended-tool call (web_search / code_interpreter / file_search)."""
        field_value = (
            tool_input.get(field_key, "") if isinstance(tool_input, dict) else ""
        )
        return {
            "type": item_type,
            "call_id": tool_call_id,
            field_key: field_value,
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
        tool_input = OpenAIResponsesToolOps._parse_tool_arguments(
            provider_tool_call.get("arguments", {})
        )

        if item_type == "function_call":
            return OpenAIResponsesToolOps._p_function_call_to_ir(
                provider_tool_call, tool_input
            )
        if item_type == "mcp_call":
            return OpenAIResponsesToolOps._p_mcp_call_to_ir(
                provider_tool_call, tool_input
            )
        if item_type in _EXTENDED_CALL_ITEM_TYPES:
            return OpenAIResponsesToolOps._p_extended_call_to_ir(
                provider_tool_call, tool_input, item_type
            )
        if item_type == "custom_tool_call":
            return OpenAIResponsesToolOps._p_custom_call_to_ir(provider_tool_call)
        raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict:
        """Parse Responses API ``arguments`` field into a dict tool_input."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                return {"input": arguments}
        return {}

    @staticmethod
    def _p_function_call_to_ir(
        provider_tool_call: dict[str, Any], tool_input: dict
    ) -> ToolCallPart:
        """Responses API ``function_call`` item → IR ToolCallPart."""
        # Responses API has both 'id' (item ID, fc_ prefix) and
        # 'call_id' (correlation ID, call_ prefix). Store call_id as
        # tool_call_id for correlation, preserve 'id' in provider_metadata
        # for lossless round-trip.
        call_id = provider_tool_call.get("call_id")
        item_id = provider_tool_call.get("id", "")
        if not call_id:
            # Fallback: derive call_ prefix from fc_ prefix
            call_id = "call_" + item_id[3:] if item_id.startswith("fc_") else item_id
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

    @staticmethod
    def _p_mcp_call_to_ir(
        provider_tool_call: dict[str, Any], tool_input: dict
    ) -> ToolCallPart:
        """Responses API ``mcp_call`` item → IR ToolCallPart."""
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

    @staticmethod
    def _p_extended_call_to_ir(
        provider_tool_call: dict[str, Any], tool_input: dict, item_type: str
    ) -> ToolCallPart:
        """Responses API shell/computer/code_interpreter call → IR ToolCallPart."""
        return cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": provider_tool_call.get(
                    "call_id", provider_tool_call.get("id", "")
                ),
                "tool_name": provider_tool_call.get("name", item_type),
                "tool_input": tool_input,
                "tool_type": _EXTENDED_CALL_TYPE_MAP.get(item_type, "function"),
            },
        )

    @staticmethod
    def _p_custom_call_to_ir(provider_tool_call: dict[str, Any]) -> ToolCallPart:
        """Responses API ``custom_tool_call`` item → IR ToolCallPart."""
        # custom_tool_call uses plain text 'input' instead of JSON 'arguments'.
        # Wrap as {"input": str} for IR compatibility (tool_input must be dict).
        # If the input happens to be valid JSON, parse it so cross-provider
        # converters can inspect fields.
        input_str = provider_tool_call.get("input", "")
        try:
            parsed_input = json.loads(input_str) if input_str else {}
        except (json.JSONDecodeError, TypeError):
            parsed_input = {"input": input_str}
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

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → OpenAI Responses function_call_output item.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            OpenAI Responses function_call_output item dict.
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

        return {
            "type": "function_call_output",
            "call_id": ir_tool_result["tool_call_id"],
            "output": output,
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """OpenAI Responses function_call_output → IR ToolResultPart.

        Handles both function_call_output and mcp_call_output.

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
