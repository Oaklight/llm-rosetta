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

from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig
from ..base import BaseToolOps
from ..base.tools import sanitize_schema

logger = logging.getLogger(__name__)


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
    # --- Pass 1: collect known call_ids and answered ids ---
    known_call_ids: set[str] = set()
    answered_ids: set[str] = set()
    for item in items:
        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            if call_id:
                known_call_ids.add(call_id)
        elif item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            if call_id:
                answered_ids.add(call_id)

    # Fast path: nothing to fix
    if not known_call_ids and not answered_ids:
        return items

    # --- Pass 2: walk items, inject/remove as needed ---
    patched: list[dict[str, Any]] = []
    orphaned_call_ids: list[str] = []
    orphaned_output_ids: list[str] = []

    for item in items:
        # Remove orphaned outputs (output without preceding function_call)
        if item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            if call_id and call_id not in known_call_ids:
                orphaned_output_ids.append(call_id)
                continue  # skip this item

        patched.append(item)

        # Inject synthetic outputs for orphaned function_calls
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        if call_id and call_id not in answered_ids:
            orphaned_call_ids.append(call_id)
            patched.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": placeholder,
                }
            )

    if orphaned_call_ids:
        logger.warning(
            "Fixed %d orphaned function_call(s) by injecting synthetic outputs: %s",
            len(orphaned_call_ids),
            ", ".join(orphaned_call_ids),
        )
    if orphaned_output_ids:
        logger.warning(
            "Removed %d orphaned function_call_output(s) with no matching call: %s",
            len(orphaned_output_ids),
            ", ".join(orphaned_output_ids),
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

        if ir_tool.get("type", "function") == "function":
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

        # Non-function tool types: wrap as custom
        raw_schema = ir_tool.get("parameters", {})
        return {
            "type": "custom",
            "name": f"{ir_tool['type']}_{ir_tool['name']}",
            "description": ir_tool.get("description", ""),
            "schema": sanitize_schema(raw_schema)
            if isinstance(raw_schema, dict)
            else raw_schema,
        }

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """OpenAI Responses tool definition → IR ToolDefinition.

        Handles both flat format (Responses API native) and nested format
        (with ``function`` key).  Non-function tool types without a ``name``
        field (e.g. ``web_search``) are stored as passthrough so they can be
        round-tripped without modification.

        Args:
            provider_tool: OpenAI Responses tool definition dict.

        Returns:
            IR ToolDefinition.
        """
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
            # Non-function tools without a name (e.g. web_search) should be
            # stored as passthrough to avoid lossy conversion.
            if tool_type != "function" and "name" not in provider_tool:
                result = {
                    "type": tool_type,
                    "name": tool_type,
                    "description": "",
                    "parameters": {},
                    "_passthrough": dict(provider_tool),
                }
                result["metadata"] = {}
                result["required_parameters"] = []
                return cast(ToolDefinition, result)

            # Flat format (Responses API native)
            # Custom tools use "schema" instead of "parameters"
            params = provider_tool.get("parameters", {})
            if tool_type != "function" and not params:
                params = provider_tool.get("schema", {})
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

        result["metadata"] = {}
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
                return {"type": "function", "function": {"name": tool_name}}
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
                func = provider_tool_choice.get("function", {})
                return cast(
                    ToolChoice, {"mode": "tool", "tool_name": func.get("name", "")}
                )

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

        # Detect MCP call
        if tool_name and tool_name.startswith("mcp://"):
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": ir_tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "mcp":
            return {
                "type": "mcp_call",
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "server_label": ir_tool_call.get("server_name", "default"),
                "status": "calling",
            }
        elif tool_type == "function":
            return {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
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
            return ToolCallPart(
                type="tool_call",
                tool_call_id=provider_tool_call.get(
                    "call_id", provider_tool_call.get("id", "")
                ),
                tool_name=provider_tool_call.get("name", ""),
                tool_input=tool_input,
                tool_type="function",
            )
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
        else:
            raise ValueError(f"Unsupported OpenAI Responses item type: {item_type}")

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
        return {
            "type": "function_call_output",
            "call_id": ir_tool_result["tool_call_id"],
            "output": str(result_content),
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
