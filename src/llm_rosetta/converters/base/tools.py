"""
LLM-Rosetta - Base Tool Operations

Abstract base class for tool conversion operations plus shared orphan-fixing
utilities (``extract_part_ids``, ``log_orphan_warnings``, IR-level fixers).

Schema sanitization lives in the sibling ``schema`` module.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from ...types.ir import (
    Message,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ...types.ir.request import IRRequest
from ...types.ir.tools import ToolCallConfig

# Re-export for backward compatibility — callers import sanitize_schema
# from this module via ``from ..base.tools import sanitize_schema``.
from .schema import sanitize_schema  # noqa: F401

logger = logging.getLogger(__name__)


# ==================== Orphaned Tool Call Fix (IR level) ====================


def extract_part_ids(parts: Sequence[Any], part_type: str, id_key: str) -> set[str]:
    """Extract IDs from content parts/items matching a given type.

    Generic helper reused across IR, Anthropic, and OpenAI Responses
    orphaned-tool-call fixers.

    Args:
        parts: List of dicts (content parts, blocks, or flat items).
        part_type: Value of the ``type`` key to match.
        id_key: Key whose value is collected as an ID.

    Returns:
        Set of extracted ID strings.
    """
    return {
        part[id_key]
        for part in parts
        if isinstance(part, dict) and part.get("type") == part_type and part.get(id_key)
    }


def log_orphan_warnings(
    log: logging.Logger,
    call_ids: Sequence[str],
    result_ids: Sequence[str],
    call_label: str = "tool_call",
    result_label: str = "tool_result",
) -> None:
    """Log warnings for orphaned tool calls and results.

    Shared by all per-format ``fix_orphaned_tool_calls`` functions.

    Args:
        log: Logger instance to emit warnings on.
        call_ids: IDs of orphaned calls that were patched.
        result_ids: IDs of orphaned results that were removed.
        call_label: Human-readable label for call items (e.g. "tool_use").
        result_label: Human-readable label for result items.
    """
    if call_ids:
        log.warning(
            "Fixed %d orphaned %s(s) by injecting synthetic results: %s",
            len(call_ids),
            call_label,
            ", ".join(call_ids),
        )
    if result_ids:
        log.warning(
            "Removed %d orphaned %s(s) with no matching %s: %s",
            len(result_ids),
            result_label,
            call_label,
            ", ".join(result_ids),
        )


def _collect_ir_tool_ids(
    messages: Sequence[Message],
) -> tuple[set[str], set[str]]:
    """Collect all tool_call IDs and answered (tool_result) IDs from IR messages."""
    known_call_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg in messages:
        content = msg.get("content", [])
        role = msg.get("role")
        if role == "assistant":
            known_call_ids |= extract_part_ids(content, "tool_call", "tool_call_id")
        elif role == "tool":
            answered_ids |= extract_part_ids(content, "tool_result", "tool_call_id")
    return known_call_ids, answered_ids


def fix_orphaned_tool_calls_ir(
    messages: Sequence[Message],
    *,
    placeholder: str = "[No output available yet]",
) -> list[Message]:
    """Fix mismatched tool_calls and tool results at IR level.

    Both the OpenAI Chat Completions API and the Responses API **strictly
    require** bidirectional pairing between tool calls and tool results:

    1. Every ``tool_call_id`` in an assistant message must have a matching
       ``role: "tool"`` result message (**orphaned tool_call**).
    2. Every ``role: "tool"`` result message must have a preceding assistant
       message containing the matching ``tool_call_id``
       (**orphaned tool_result**).

    Anthropic enforces the same strict pairing; only Google Gemini is
    lenient.  This function patches IR messages so that downstream
    converters produce valid output for any target provider.

    This function handles both directions:

    - **Orphaned tool_calls**: injects a synthetic ``role: "tool"`` IR
      message with *placeholder* content immediately after the assistant
      message.
    - **Orphaned tool_results**: removes ``role: "tool"`` messages whose
      ``tool_call_id`` does not appear in any preceding assistant
      ``tool_call`` content part.

    The original iterable is **not** modified; a new list is returned.

    Args:
        messages: IR messages (any iterable of Message dicts).
        placeholder: Content string for injected synthetic tool results.

    Returns:
        A new messages list with orphaned tool_calls/results fixed.
    """
    msg_list = list(messages)

    known_call_ids, answered_ids = _collect_ir_tool_ids(msg_list)

    # Fast path: nothing to fix
    if not known_call_ids and not answered_ids:
        return msg_list

    patched: list[Message] = []
    orphaned_call_ids: list[str] = []
    orphaned_result_ids: list[str] = []

    for msg in msg_list:
        role = msg.get("role")
        content = msg.get("content", [])

        # Remove orphaned tool results (result without preceding tool_call)
        if role == "tool":
            result_ids = extract_part_ids(content, "tool_result", "tool_call_id")
            if result_ids and result_ids.isdisjoint(known_call_ids):
                orphaned_result_ids.extend(result_ids)
                continue

        patched.append(msg)

        # Inject synthetic results for orphaned tool_calls
        if role == "assistant":
            unanswered = (
                extract_part_ids(content, "tool_call", "tool_call_id") - answered_ids
            )
            for tc_id in unanswered:
                orphaned_call_ids.append(tc_id)
                patched.append(
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": tc_id,
                                "result": placeholder,
                            }
                        ],
                    }
                )

    log_orphan_warnings(logger, orphaned_call_ids, orphaned_result_ids)
    return patched


# ==================== Orphaned Tool Config Fix (IR level) ====================


def strip_orphaned_tool_config(ir_request: IRRequest) -> list[str]:
    """Strip ``tool_choice`` and ``tool_config`` when no tools are defined.

    Codex CLI context compaction can remove all tool definitions from a
    request while keeping ``tool_choice`` (e.g. ``"auto"``).  This produces
    an invalid request that upstream APIs reject with *"tool_choice is set
    but no tools are provided"*.

    This is part of the same problem family as
    :func:`fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing)
    and ``_reorder_tool_messages`` (tool message ordering) — all stem from
    Codex context compaction breaking request structural integrity.

    The request dict is modified **in-place**.

    Args:
        ir_request: IR request dict (mutated in-place).

    Returns:
        List of warning strings for each stripped field.
    """
    tools = ir_request.get("tools")
    has_tools = bool(tools)

    if has_tools:
        return []

    # Cast to plain dict for mutation — IRRequest is a TypedDict at
    # type-check time but a regular dict at runtime.
    request_dict = cast(dict[str, Any], ir_request)

    warnings: list[str] = []
    for field in ("tool_choice", "tool_config"):
        if field in request_dict:
            value = request_dict.pop(field)
            warnings.append(
                f"Stripped orphaned '{field}' (value: {value!r}) — "
                "no tool definitions present in request"
            )
            logger.warning(
                "Stripped orphaned '%s' from IR request — "
                "no tool definitions present (Codex context compaction workaround)",
                field,
            )

    return warnings


class BaseToolOps(ABC):
    """工具转换操作的抽象基类
    Abstract base class for tool conversion operations

    统一处理工具生命周期的所有阶段：定义 → 选择 → 调用 → 结果。
    Uniformly handles all stages of the tool lifecycle: definition → choice → call → result.
    """

    # ==================== 工具定义转换 Tool definition conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_definition_to_p(ir_tool: ToolDefinition, **kwargs: Any) -> Any:
        """IR ToolDefinition → Provider Tool Definition
        将IR工具定义转换为Provider工具定义

        处理工具的基本信息：名称、描述、参数schema等。
        Handles basic tool information: name, description, parameter schema, etc.

        Args:
            ir_tool: IR格式的工具定义
            **kwargs: 额外参数

        Returns:
            Provider格式的工具定义
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_definition_to_ir(
        provider_tool: Any, **kwargs: Any
    ) -> ToolDefinition | list[ToolDefinition] | None:
        """Provider Tool Definition → IR ToolDefinition

        Args:
            provider_tool: Provider tool definition.
            **kwargs: Extra arguments.

        Returns:
            IR tool definition(s), or None if the entry cannot be converted
            (e.g. provider-specific built-in tools with no function schema).
        """
        pass

    # ==================== 工具选择转换 Tool choice conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        """IR ToolChoice → Provider Tool Choice Config
        将IR工具选择转换为Provider工具选择配置

        处理工具选择策略：none、auto、any、specific tool等。
        Handles tool choice strategies: none, auto, any, specific tool, etc.

        Args:
            ir_tool_choice: IR格式的工具选择
            **kwargs: 额外参数

        Returns:
            Provider格式的工具选择配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice
        将Provider工具选择配置转换为IR工具选择

        Args:
            provider_tool_choice: Provider格式的工具选择配置
            **kwargs: 额外参数

        Returns:
            IR格式的工具选择
        """
        pass

    # ==================== 工具调用转换 Tool call conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        """IR ToolCallPart → Provider Tool Call
        将IR工具调用部分转换为Provider工具调用

        处理工具调用请求：调用ID、工具名称、输入参数等。
        Handles tool call requests: call ID, tool name, input parameters, etc.

        Args:
            ir_tool_call: IR格式的工具调用部分
            **kwargs: 额外参数

        Returns:
            Provider格式的工具调用
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart
        将Provider工具调用转换为IR工具调用部分

        Args:
            provider_tool_call: Provider格式的工具调用
            **kwargs: 额外参数

        Returns:
            IR格式的工具调用部分
        """
        pass

    # ==================== 工具结果转换 Tool result conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_result_to_p(ir_tool_result: ToolResultPart, **kwargs: Any) -> Any:
        """IR ToolResultPart → Provider Tool Result
        将IR工具结果部分转换为Provider工具结果

        处理工具执行结果：结果数据、错误信息、状态等。
        Handles tool execution results: result data, error information, status, etc.

        Args:
            ir_tool_result: IR格式的工具结果部分
            **kwargs: 额外参数

        Returns:
            Provider格式的工具结果
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart
        将Provider工具结果转换为IR工具结果部分

        Args:
            provider_tool_result: Provider格式的工具结果
            **kwargs: 额外参数

        Returns:
            IR格式的工具结果部分
        """
        pass

    # ==================== 工具配置转换 Tool configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_tool_config_to_p(ir_tool_config: ToolCallConfig, **kwargs: Any) -> Any:
        """IR ToolCallConfig → Provider Tool Call Config
        将IR工具调用配置转换为Provider工具调用配置

        处理工具调用的控制参数：并行调用、最大调用数等。
        Handles tool call control parameters: parallel calls, max call count, etc.

        Args:
            ir_tool_config: IR格式的工具调用配置
            **kwargs: 额外参数

        Returns:
            Provider格式的工具调用配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        """Provider Tool Call Config → IR ToolCallConfig
        将Provider工具调用配置转换为IR工具调用配置

        Args:
            provider_tool_config: Provider格式的工具调用配置
            **kwargs: 额外参数

        Returns:
            IR格式的工具调用配置
        """
        pass
