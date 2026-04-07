"""
LLM-Rosetta - Anthropic Tool Operations

Anthropic Messages API tool conversion operations.
Handles bidirectional conversion of tool definitions, calls, results,
choice strategies, and call configurations.

Also provides ``fix_orphaned_tool_calls`` — a module-level utility that
fixes bidirectional mismatches between tool_use and tool_result blocks
(Anthropic rejects both orphaned calls and orphaned results with 400).

Self-contained: does not depend on utils/ToolCallConverter or utils/ToolConverter.
"""

import json
import logging
from typing import Any, cast

from ..base.tool_content import (
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
from ..base.tools import sanitize_schema

logger = logging.getLogger(__name__)


# ==================== Orphaned Tool Call Fix ====================


def fix_orphaned_tool_calls(
    messages: list[dict[str, Any]],
    *,
    placeholder: str = "[No output available yet]",
) -> list[dict[str, Any]]:
    """Fix mismatched tool_use and tool_result blocks in Anthropic format.

    The Anthropic Messages API **strictly requires** bidirectional pairing
    between ``tool_use`` (in assistant messages) and ``tool_result`` (in user
    messages):

    1. Every ``tool_use`` block must have a corresponding ``tool_result``
       block with the same ``id`` / ``tool_use_id`` (**orphaned tool_use**).
    2. Every ``tool_result`` block must reference a preceding ``tool_use``
       block (**orphaned tool_result**).

    Violations of either rule cause a 400 error.  Only Google Gemini is
    lenient about both cases.

    This function handles both directions:

    - **Orphaned tool_use**: injects a synthetic ``tool_result`` block with
      *placeholder* content into the next user message (or creates one).
    - **Orphaned tool_result**: removes ``tool_result`` blocks whose
      ``tool_use_id`` does not appear in any preceding ``tool_use`` block.

    The original list is **not** modified; a new list is returned.

    Args:
        messages: Anthropic Messages format messages list.
        placeholder: Content string for injected synthetic tool results.

    Returns:
        A new messages list with orphaned tool_use/results fixed.
    """
    # --- Pass 1: collect known tool_use ids and answered ids ---
    known_use_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                use_id = block.get("id")
                if use_id:
                    known_use_ids.add(use_id)
            elif block.get("type") == "tool_result":
                use_id = block.get("tool_use_id")
                if use_id:
                    answered_ids.add(use_id)

    # Fast path: nothing to fix
    if not known_use_ids and not answered_ids:
        return messages

    # --- Pass 2: walk messages, inject/remove as needed ---
    patched: list[dict[str, Any]] = []
    orphaned_use_ids: list[str] = []
    orphaned_result_ids: list[str] = []

    for msg in messages:
        msg_role = msg.get("role")
        content = msg.get("content", [])

        if msg_role == "user" and isinstance(content, list):
            # Filter out orphaned tool_result blocks
            filtered_content: list[Any] = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") not in known_use_ids
                ):
                    orphaned_result_ids.append(block.get("tool_use_id", "?"))
                    continue
                filtered_content.append(block)
            if filtered_content != content:
                if not filtered_content:
                    # All content was orphaned results; skip this message
                    continue
                msg = {**msg, "content": filtered_content}

        patched.append(msg)

        if msg_role == "assistant" and isinstance(content, list):
            # Collect orphaned tool_use ids in this assistant message
            pending_ids: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    use_id = block.get("id")
                    if use_id and use_id not in answered_ids:
                        pending_ids.append(use_id)

            if pending_ids:
                orphaned_use_ids.extend(pending_ids)
                # Inject synthetic tool_result blocks in a user message
                synthetic_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": uid,
                        "content": placeholder,
                    }
                    for uid in pending_ids
                ]
                patched.append({"role": "user", "content": synthetic_results})

    if orphaned_use_ids:
        logger.warning(
            "Fixed %d orphaned tool_use(s) by injecting synthetic results: %s",
            len(orphaned_use_ids),
            ", ".join(orphaned_use_ids),
        )
    if orphaned_result_ids:
        logger.warning(
            "Removed %d orphaned tool_result(s) with no matching tool_use: %s",
            len(orphaned_result_ids),
            ", ".join(orphaned_result_ids),
        )

    return patched


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
        result: dict[str, Any] = {
            "name": ir_tool["name"],
            "description": ir_tool.get("description", ""),
        }
        raw_schema = ir_tool.get("parameters", {})
        schema = (
            sanitize_schema(raw_schema) if isinstance(raw_schema, dict) else raw_schema
        )
        # Anthropic requires input_schema to have "type"; default to object
        if isinstance(schema, dict) and "type" not in schema:
            schema = {"type": "object", **schema}
        result["input_schema"] = schema
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
        result: dict[str, Any] = {
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
        return cast(ToolDefinition, result)

    # ==================== Tool Choice ====================

    @staticmethod
    def ir_tool_choice_to_p(
        ir_tool_choice: ToolChoice, **kwargs: Any
    ) -> dict[str, Any]:
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
        result: dict[str, Any] = {}

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
                return cast(ToolChoice, {"mode": "auto", "tool_name": ""})
            elif choice_type == "any":
                return cast(ToolChoice, {"mode": "any", "tool_name": ""})
            elif choice_type == "tool":
                tool_name = provider_tool_choice.get("name", "")
                return cast(ToolChoice, {"mode": "tool", "tool_name": tool_name})
            elif choice_type == "none":
                return cast(ToolChoice, {"mode": "none", "tool_name": ""})

        return cast(ToolChoice, {"mode": "auto", "tool_name": ""})

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
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": ir_tool_result["tool_call_id"],
        }

        content = ir_tool_result.get("result", "")
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            from .content_ops import AnthropicContentOps

            result["content"] = convert_ir_content_blocks_to_p(
                content, AnthropicContentOps
            )
        elif content is not None:
            result["content"] = json.dumps(content)
        else:
            result["content"] = ""

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
        content = provider_tool_result.get("content", "")
        if isinstance(content, list):
            from .content_ops import AnthropicContentOps

            content = convert_content_blocks_to_ir(content, AnthropicContentOps)

        return ToolResultPart(
            type="tool_result",
            tool_call_id=provider_tool_result.get("tool_use_id", ""),
            result=content,
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
        result: dict[str, Any] = {}

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
        result: dict[str, Any] = {}

        if isinstance(provider_tool_config, dict):
            disable_parallel = provider_tool_config.get("disable_parallel_tool_use")
            if disable_parallel is not None:
                result["disable_parallel"] = disable_parallel

        return cast(ToolCallConfig, result)
