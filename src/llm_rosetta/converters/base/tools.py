"""
LLM-Rosetta - Base Tool Operations
工具转换操作的抽象基类
Abstract base class for tool conversion operations

处理所有工具相关的转换：
- 工具定义：函数签名、参数schema
- 工具调用：调用请求、参数传递
- 工具结果：执行结果、错误处理
- 工具配置：选择策略、调用配置
Handles all tool-related conversions:
- Tool definitions: function signatures, parameter schemas
- Tool calls: call requests, parameter passing
- Tool results: execution results, error handling
- Tool configurations: choice strategies, call configurations
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

logger = logging.getLogger(__name__)

# ==================== Schema sanitization utilities ====================

# JSON Schema keywords not supported by OpenAI / Vertex AI compatible
# endpoints.  These are valid per the JSON Schema spec but upstream servers
# (e.g. Vertex AI's OpenAI-compatible layer) reject them with Pydantic
# ``extra='forbid'`` validation errors.
UNSUPPORTED_SCHEMA_KEYS: set[str] = {
    "propertyNames",
    "const",
    "$schema",
    "$comment",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "ref",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "deprecated",
    "readOnly",
    "writeOnly",
    "examples",
}

# Keys that hold definition maps (consumed for $ref resolution, then removed).
_DEFS_KEYS: set[str] = {"$defs", "definitions"}


def _flatten_combination(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``anyOf``/``oneOf`` nullable patterns into a simple typed schema.

    Vertex AI's OpenAI-compatible layer does not support ``anyOf``/``oneOf``
    at all.  The most common pattern is a nullable union like
    ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``, which we convert to
    ``{"type": "string", "nullable": true}``.

    For single-variant unions we unwrap directly.  For multi-type (non-null)
    unions we keep only the first non-null variant (lossy but safe).

    ``allOf`` with a single element is simply unwrapped.

    Args:
        schema: A schema dict that may contain ``anyOf``/``oneOf``/``allOf``.

    Returns:
        A new dict with combination keywords resolved.
    """
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue

        non_null = [v for v in variants if v.get("type") != "null"]
        has_null = len(non_null) < len(variants)

        # Preserve sibling metadata (description, title, etc.)
        base: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("anyOf", "oneOf", "allOf")
        }

        if len(non_null) == 1:
            # Common nullable pattern: merge the single real type
            base.update(non_null[0])
        elif len(non_null) > 1:
            # Multiple non-null types: pick the first (lossy but avoids rejection)
            base.update(non_null[0])
        # else: all variants are null → just mark nullable

        if has_null:
            base["nullable"] = True

        return base

    # allOf with a single element: unwrap
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        base = {k: v for k, v in schema.items() if k != "allOf"}
        base.update(all_of[0])
        return base

    return schema


def _resolve_ref(ref: str, defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve a JSON Schema ``$ref`` pointer against collected definitions.

    Only local definition references (``#/$defs/Name`` or
    ``#/definitions/Name``) are supported.  Unresolvable refs return an
    empty dict so the caller can proceed without crashing.

    Args:
        ref: The ``$ref`` string value.
        defs: Merged definitions from ``$defs`` and ``definitions``.

    Returns:
        The referenced schema dict, or ``{}`` if unresolvable.
    """
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            return defs.get(name, {})
    return {}


def sanitize_schema(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]] | None = None,
    extra_strip_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords.

    Also resolves ``$ref`` references by inlining the referenced definition,
    and flattens ``anyOf``/``oneOf``/``allOf`` combination keywords into
    simple typed schemas, as required by Vertex AI's OpenAI-compatible layer
    which does not support these constructs at all.

    Args:
        schema: A JSON Schema dict (or sub-schema).
        defs: Collected ``$defs``/``definitions`` from the top-level schema.
            Populated automatically on the first call if the schema contains
            definition maps.
        extra_strip_keys: Additional provider-specific keys to strip
            (e.g. ``{"additionalProperties"}`` for Google GenAI).

    Returns:
        A new dict with unsupported keys removed at every level.
    """
    # On first call, collect $defs/definitions for $ref resolution.
    if defs is None:
        defs = {}
        for key in _DEFS_KEYS:
            d = schema.get(key)
            if isinstance(d, dict):
                defs.update(d)

    strip_keys = UNSUPPORTED_SCHEMA_KEYS | (extra_strip_keys or set())

    # Resolve $ref: inline the referenced definition (merge siblings).
    ref = schema.get("$ref")
    if isinstance(ref, str) and defs:
        resolved = _resolve_ref(ref, defs)
        if resolved:
            # Siblings of $ref (e.g. description) are kept; $ref itself is
            # replaced by the resolved definition's content.
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            merged.update(resolved)
            return sanitize_schema(merged, defs, extra_strip_keys)

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in strip_keys or key in _DEFS_KEYS:
            continue
        if key == "$ref":
            # Unresolvable $ref — drop it to avoid upstream rejection.
            continue
        if isinstance(value, dict):
            result[key] = sanitize_schema(value, defs, extra_strip_keys)
        elif isinstance(value, list):
            result[key] = [
                sanitize_schema(item, defs, extra_strip_keys)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    # Flatten combination keywords (anyOf/oneOf/allOf) into simple types.
    if result.keys() & {"anyOf", "oneOf", "allOf"}:
        result = _flatten_combination(result)

    return result


# ==================== Orphaned Tool Call Fix (IR level) ====================


def _collect_ir_tool_ids(
    messages: Sequence[Message],
) -> tuple[set[str], set[str]]:
    """Collect all tool_call IDs and answered (tool_result) IDs from IR messages."""
    known_call_ids: set[str] = set()
    answered_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "tool_call":
                    tc_id = part.get("tool_call_id")
                    if tc_id:
                        known_call_ids.add(tc_id)
        elif msg.get("role") == "tool":
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tc_id = part.get("tool_call_id")
                    if tc_id:
                        answered_ids.add(tc_id)
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

    # --- Walk messages, inject/remove as needed ---
    patched: list[Message] = []
    orphaned_call_ids: list[str] = []
    orphaned_result_ids: list[str] = []

    for msg in msg_list:
        # Remove orphaned tool results (result without preceding tool_call)
        if msg.get("role") == "tool":
            content = msg.get("content", [])
            # Check if ALL tool_result parts in this message are orphaned
            result_ids_in_msg = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tc_id = part.get("tool_call_id")
                    if tc_id:
                        result_ids_in_msg.append(tc_id)
            if result_ids_in_msg and all(
                rid not in known_call_ids for rid in result_ids_in_msg
            ):
                orphaned_result_ids.extend(result_ids_in_msg)
                continue  # skip this message entirely

        patched.append(msg)

        # Inject synthetic results for orphaned tool_calls
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_call":
                continue
            tc_id = part.get("tool_call_id")
            if tc_id and tc_id not in answered_ids:
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
