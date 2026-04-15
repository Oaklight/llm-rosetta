"""
LLM-Rosetta - OpenAI Chat Tool Operations

OpenAI Chat Completions API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Also provides ``fix_orphaned_tool_calls`` — a module-level utility that
fixes bidirectional mismatches between tool_calls and tool results
(OpenAI Chat API rejects both orphaned calls and orphaned results with 400).

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.
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


def _collect_openai_tool_ids(
    messages: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Collect tool_call_ids from assistant messages and answered ids from tool messages.

    Returns:
        Tuple of (known_call_ids, answered_ids).
    """
    known_call_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        known_call_ids.add(tc_id)
        elif msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                answered_ids.add(tc_id)
    return known_call_ids, answered_ids


def fix_orphaned_tool_calls(
    messages: list[dict[str, Any]],
    *,
    placeholder: str = "[No output available yet]",
) -> list[dict[str, Any]]:
    """Fix mismatched tool_calls and tool results in OpenAI Chat format.

    The OpenAI Chat Completions API **strictly requires** bidirectional
    pairing between tool calls and tool results:

    1. Every ``tool_call_id`` in an assistant message must have a matching
       ``role: "tool"`` result message (**orphaned tool_call**).
    2. Every ``role: "tool"`` result message must have a preceding assistant
       message containing the matching ``tool_call_id`` (**orphaned tool_result**).

    Violations of either rule cause a 400 error.  Anthropic enforces the same
    strict pairing.  Only Google Gemini is lenient about both cases.

    This function handles both directions:

    - **Orphaned tool_calls**: injects a synthetic ``role: "tool"`` message
      with *placeholder* content immediately after the assistant message.
    - **Orphaned tool_results**: removes ``role: "tool"`` messages whose
      ``tool_call_id`` does not appear in any preceding assistant
      ``tool_calls``.

    The original list is **not** modified; a new list is returned.

    Args:
        messages: OpenAI Chat format messages list.
        placeholder: Content string for injected synthetic tool results.

    Returns:
        A new messages list with orphaned tool_calls/results fixed.
    """
    # --- Pass 1: collect known tool_call_ids and answered ids ---
    known_call_ids, answered_ids = _collect_openai_tool_ids(messages)

    # Fast path: nothing to fix
    if not known_call_ids and not answered_ids:
        return messages

    # --- Pass 2: walk messages, inject/remove as needed ---
    patched: list[dict[str, Any]] = []
    orphaned_call_ids: list[str] = []
    orphaned_result_ids: list[str] = []

    for msg in messages:
        # Remove orphaned tool results (result without preceding tool_call)
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id not in known_call_ids:
                orphaned_result_ids.append(tc_id)
                continue  # skip this message
        patched.append(msg)

        # Inject synthetic results for orphaned tool_calls
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            tc_id = tc.get("id")
            if tc_id and tc_id not in answered_ids:
                orphaned_call_ids.append(tc_id)
                patched.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": placeholder,
                    }
                )

    if orphaned_call_ids:
        logger.warning(
            "Fixed %d orphaned tool_call(s) by injecting synthetic results: %s",
            len(orphaned_call_ids),
            ", ".join(orphaned_call_ids),
        )
    if orphaned_result_ids:
        logger.warning(
            "Removed %d orphaned tool_result(s) with no matching tool_call: %s",
            len(orphaned_result_ids),
            ", ".join(orphaned_result_ids),
        )

    return patched


class OpenAIChatToolOps(BaseToolOps):
    """OpenAI Chat Completions tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → OpenAI Chat tool definition.

        Converts flat IR format to nested OpenAI ``{"type":"function","function":{...}}``
        format.  Unsupported JSON Schema keywords (e.g. ``propertyNames``) are
        recursively stripped from ``parameters``.

        Args:
            ir_tool: IR tool definition.

        Returns:
            OpenAI Chat tool definition dict.
        """
        if ir_tool.get("type", "function") == "function":
            func_def: dict[str, Any] = {
                "name": ir_tool["name"],
                "description": ir_tool.get("description", ""),
            }
            parameters = ir_tool.get("parameters")
            if parameters and isinstance(parameters, dict):
                func_def["parameters"] = sanitize_schema(parameters)
            elif parameters:
                func_def["parameters"] = parameters
            return {"type": "function", "function": func_def}

        # Non-function tool types (e.g. Codex apply_patch with type="custom"):
        # wrap as Chat Completions function, preserving the original name so
        # the client can match tool_call responses back to the tool.
        raw_params = ir_tool.get("parameters", {})
        params = (
            sanitize_schema(raw_params) if isinstance(raw_params, dict) else raw_params
        )
        return {
            "type": "function",
            "function": {
                "name": ir_tool["name"],
                "description": ir_tool.get("description", ""),
                "parameters": params,
            },
        }

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """OpenAI Chat tool definition → IR ToolDefinition.

        Converts nested OpenAI format to flat IR format.

        Args:
            provider_tool: OpenAI Chat tool definition dict.

        Returns:
            IR ToolDefinition.
        """
        func = provider_tool.get("function", {})
        result: dict[str, Any] = {
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }

        # Extract required_parameters from JSON Schema if available
        parameters = func.get("parameters", {})
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        result["metadata"] = {}
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> str | dict:
        """IR ToolChoice → OpenAI Chat tool_choice parameter.

        Mapping:
        - ``mode:"none"`` → ``"none"``
        - ``mode:"auto"`` → ``"auto"``
        - ``mode:"any"`` → ``"required"``
        - ``mode:"tool"`` → ``{"type":"function","function":{"name":"..."}}``

        Args:
            ir_tool_choice: IR tool choice.

        Returns:
            OpenAI tool_choice value (string or dict).
        """
        mode = ir_tool_choice.get("mode", "auto")

        if mode == "none":
            return "none"
        elif mode == "auto":
            return "auto"
        elif mode == "any":
            return "required"
        elif mode == "tool":
            tool_name = ir_tool_choice.get("tool_name")
            if tool_name:
                return {"type": "function", "function": {"name": tool_name}}
            return "required"

        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """OpenAI Chat tool_choice → IR ToolChoice.

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
        """IR ToolCallPart → OpenAI Chat tool call.

        Converts ``tool_input`` dict to JSON string ``arguments``.

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            OpenAI Chat tool call dict.
        """
        tool_input = ir_tool_call.get("tool_input", {})
        arguments = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        return {
            "id": ir_tool_call["tool_call_id"],
            "type": "function",
            "function": {
                "name": ir_tool_call["tool_name"],
                "arguments": arguments,
            },
        }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """OpenAI Chat tool call → IR ToolCallPart.

        Parses JSON string ``arguments`` back to dict.

        Args:
            provider_tool_call: OpenAI Chat tool call dict.

        Returns:
            IR ToolCallPart.
        """
        func = provider_tool_call.get("function", {})
        arguments_str = func.get("arguments", "{}")

        try:
            tool_input = json.loads(arguments_str) if arguments_str else {}
        except (json.JSONDecodeError, TypeError):
            tool_input = {"raw_arguments": arguments_str}

        return ToolCallPart(
            type="tool_call",
            tool_call_id=provider_tool_call.get("id", ""),
            tool_name=func.get("name", ""),
            tool_input=tool_input,
            tool_type="function",
        )

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → OpenAI Chat tool role message.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            OpenAI tool role message dict.
        """
        result = ir_tool_result.get("result", "")

        if isinstance(result, (list, dict)):
            content = json.dumps(result)
        elif result is not None:
            content = str(result)
        else:
            content = ""

        return {
            "role": "tool",
            "tool_call_id": ir_tool_result["tool_call_id"],
            "content": content,
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """OpenAI Chat tool role message → IR ToolResultPart.

        Args:
            provider_tool_result: OpenAI tool role message dict.

        Returns:
            IR ToolResultPart.
        """
        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_call_id", ""),
            result=provider_tool_result.get("content", ""),
        )

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → OpenAI Chat tool call config fields.

        Mapping:
        - ``disable_parallel`` → ``parallel_tool_calls`` (inverted)

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "disable_parallel" in ir_tool_config:
            result["parallel_tool_calls"] = not ir_tool_config["disable_parallel"]

        # max_calls is not supported by OpenAI Chat
        return result

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """OpenAI Chat tool call config → IR ToolCallConfig.

        Mapping:
        - ``parallel_tool_calls`` → ``disable_parallel`` (inverted)

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

        return cast(ToolCallConfig, result)
