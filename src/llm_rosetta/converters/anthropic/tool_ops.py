"""
LLM-Rosetta - Anthropic Tool Operations

Anthropic Messages API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.
"""

from typing import Any, Dict

from ...types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.tools import ToolCallConfig
from ..base import BaseToolOps


class AnthropicToolOps(BaseToolOps):
    """Anthropic Messages API tool conversion operations.

    All methods are static and stateless. Handles tool definitions,
    calls, results, choice strategies, and call configurations.

    Key differences from OpenAI:
    - Tool call arguments are Dict (not JSON string)
    - Tool definitions use ``input_schema`` (not ``parameters``)
    - Tool choice uses ``any`` instead of ``required``
    - ``disable_parallel_tool_use`` is part of tool_choice
    """

    # ==================== Tool Definition ====================

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> dict:
        """IR ToolDefinition → Anthropic tool definition.

        Converts flat IR format to Anthropic's flat format with ``input_schema``.

        Args:
            ir_tool: IR tool definition.

        Returns:
            Anthropic tool definition dict.
        """
        result: Dict[str, Any] = {
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
            "input_schema": ir_tool.get("parameters", {}),
        }
        return result

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        """Anthropic tool definition → IR ToolDefinition.

        Converts Anthropic format to flat IR format.

        Args:
            provider_tool: Anthropic tool definition dict.

        Returns:
            IR ToolDefinition.
        """
        parameters = provider_tool.get("input_schema", {})
        result: Dict[str, Any] = {
            "type": "function",
            "name": provider_tool.get("name", ""),
            "description": provider_tool.get("description", ""),
            "parameters": parameters,
        }

        # Extract required_parameters from JSON Schema if available
        if isinstance(parameters, dict) and "required" in parameters:
            result["required_parameters"] = parameters["required"]
        else:
            result["required_parameters"] = []

        result["metadata"] = {}
        return result

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(
        ir_tool_choice: ToolChoice, **kwargs: Any
    ) -> Dict[str, Any]:
        """IR ToolChoice → Anthropic tool_choice parameter.

        Mapping:
        - ``mode:"none"`` → ``{"type": "none"}`` (not officially supported)
        - ``mode:"auto"`` → ``{"type": "auto"}``
        - ``mode:"any"`` → ``{"type": "any"}``
        - ``mode:"tool"`` → ``{"type": "tool", "name": "..."}``

        Args:
            ir_tool_choice: IR tool choice.

        Returns:
            Anthropic tool_choice dict.
        """
        mode = ir_tool_choice.get("mode", "auto")
        result: Dict[str, Any] = {}

        if mode == "none":
            result["type"] = "none"
        elif mode == "auto":
            result["type"] = "auto"
        elif mode == "any":
            result["type"] = "any"
        elif mode == "tool":
            result["type"] = "tool"
            tool_name = ir_tool_choice.get("tool_name")
            if tool_name:
                result["name"] = tool_name

        return result

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Anthropic tool_choice → IR ToolChoice.

        Mapping:
        - ``{"type": "auto"}`` → ``mode:"auto"``
        - ``{"type": "any"}`` → ``mode:"any"``
        - ``{"type": "tool", "name": "..."}`` → ``mode:"tool"``

        Args:
            provider_tool_choice: Anthropic tool_choice dict.

        Returns:
            IR ToolChoice.
        """
        if isinstance(provider_tool_choice, dict):
            choice_type = provider_tool_choice.get("type", "auto")
            if choice_type == "auto":
                return {"mode": "auto", "tool_name": ""}
            elif choice_type == "any":
                return {"mode": "any", "tool_name": ""}
            elif choice_type == "tool":
                tool_name = provider_tool_choice.get("name", "")
                return {"mode": "tool", "tool_name": tool_name}
            elif choice_type == "none":
                return {"mode": "none", "tool_name": ""}

        return {"mode": "auto", "tool_name": ""}

    # ==================== Tool Call ====================

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> dict:
        """IR ToolCallPart → Anthropic tool_use content block.

        Anthropic tool call arguments are Dict (not JSON string).

        Args:
            ir_tool_call: IR tool call part.

        Returns:
            Anthropic tool_use content block dict.
        """
        tool_type = ir_tool_call.get("tool_type", "function")
        tool_input = ir_tool_call.get("tool_input", {})

        if tool_type == "web_search":
            return {
                "type": "server_tool_use",
                "id": ir_tool_call["tool_call_id"],
                "name": "web_search",
                "input": tool_input,
            }

        return {
            "type": "tool_use",
            "id": ir_tool_call["tool_call_id"],
            "name": ir_tool_call["tool_name"],
            "input": tool_input,
        }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Anthropic tool_use/server_tool_use → IR ToolCallPart.

        Handles both ``tool_use`` and ``server_tool_use`` block types.

        Args:
            provider_tool_call: Anthropic tool call content block dict.

        Returns:
            IR ToolCallPart.
        """
        block_type = provider_tool_call.get("type", "tool_use")
        tool_name = provider_tool_call.get("name", "")

        if block_type == "server_tool_use":
            tool_type = "web_search" if tool_name == "web_search" else "function"
        else:
            tool_type = "function"

        return ToolCallPart(
            type="tool_call",
            tool_call_id=provider_tool_call.get("id", ""),
            tool_name=tool_name,
            tool_input=provider_tool_call.get("input", {}),
            tool_type=tool_type,
        )

    # ==================== Tool Result ====================

    @staticmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> dict:
        """IR ToolResultPart → Anthropic tool_result content block.

        Args:
            ir_tool_result: IR tool result part.

        Returns:
            Anthropic tool_result content block dict.
        """
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": ir_tool_result["tool_call_id"],
            "content": ir_tool_result.get("result", ""),
        }

        is_error = ir_tool_result.get("is_error")
        if is_error is not None:
            result["is_error"] = is_error

        return result

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Anthropic tool_result → IR ToolResultPart.

        Args:
            provider_tool_result: Anthropic tool_result content block dict.

        Returns:
            IR ToolResultPart.
        """
        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_use_id", ""),
            result=provider_tool_result.get("content", ""),
            is_error=provider_tool_result.get("is_error", False),
        )

    # ==================== Tool Config ====================

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> dict:
        """IR ToolCallConfig → Anthropic tool call config fields.

        Anthropic handles ``disable_parallel_tool_use`` as part of
        the ``tool_choice`` parameter.

        Mapping:
        - ``disable_parallel`` → ``disable_parallel_tool_use`` in tool_choice

        Args:
            ir_tool_config: IR tool call config.

        Returns:
            Dict of fields to merge into tool_choice.
        """
        result: Dict[str, Any] = {}

        if "disable_parallel" in ir_tool_config:
            result["disable_parallel_tool_use"] = ir_tool_config["disable_parallel"]

        # max_calls is not supported by Anthropic
        return result

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Anthropic tool call config → IR ToolCallConfig.

        Extracts ``disable_parallel_tool_use`` from tool_choice dict.

        Args:
            provider_tool_config: Dict with Anthropic tool config fields.

        Returns:
            IR ToolCallConfig.
        """
        result: Dict[str, Any] = {}

        if isinstance(provider_tool_config, dict):
            disable_parallel = provider_tool_config.get("disable_parallel_tool_use")
            if disable_parallel is not None:
                result["disable_parallel"] = disable_parallel

        return result
