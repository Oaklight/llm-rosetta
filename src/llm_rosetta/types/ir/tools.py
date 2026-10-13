"""
LLM-Rosetta - IR Tool Types

IR工具相关类型定义
IR tool-related type definitions
"""

import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

# ============================================================================
# 工具定义类型 Tool definition types
# ============================================================================


class ToolDefinition(TypedDict):
    """工具定义
    Tool definition

    统一了各provider的工具定义格式：
    - OpenAI Chat: {"type": "function", "function": {...}}
    - OpenAI Responses: {"type": "function", "name": "...", ...}
    - Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    - Google: {"function_declarations": [{"name": "...", ...}]}

    Source converter contract:
    Provider tool types that fall outside the ``type`` Literal below
    MUST be coerced to a recognized IR type at the provider→IR boundary
    so that runtime IR validation (``validate_ir_request``) accepts the
    result.  Provider-specific information may be retained in ``metadata``
    or in the ``_passthrough`` extension for round-tripping.

    ``"custom"`` is a first-class IR type supported natively by both
    OpenAI Chat Completions and Responses APIs.  Custom tools accept
    free-form text input (not JSON); ``parameters`` carries a synthesized
    ``{"input": string}`` schema for providers that require JSON Schema.
    Custom tool format info (text/grammar) is stored in ``metadata``.

    ``"intrinsic"`` represents provider-hosted built-in tools (web search,
    code execution, file search, etc.).  The specific kind is stored in
    ``metadata["intrinsic_kind"]`` (e.g. ``"web_search"``,
    ``"code_execution"``).  Intrinsic tools pass through in same-format
    conversions and are stripped in cross-format conversions.

    ``tool_group`` names the group a tool belongs to, when the source
    addressed it that way.  A client may declare tools inside a named
    container and call them by ``(name, group)``; the group is part of the
    tool's identity, so two tools that are otherwise identical are still two
    tools when their groups differ.  Absent means the tool was declared on
    its own, which is every tool from a format without containers.

    The word ``namespace`` is reserved for the wire field the Responses API
    family uses for this concept (``{"type": "namespace", ...}`` and
    ``function_call.namespace``).  It appears only where a provider body is
    read or written, never in the IR — see
    ``converters/open_responses/tool_ops.py``.  The vendor-neutral Open
    Responses spec has no grouping concept at all, so there is no neutral
    spelling to borrow and the IR states the concept in its own terms.
    """

    type: Literal[
        "function",
        "mcp",
        "custom",
        "intrinsic",
    ]
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    required_parameters: NotRequired[list[str]]
    tool_group: NotRequired[str]
    """Group this tool belongs to, as named by the source.  Present means the
    tool was declared inside a container; an empty value means that container
    had no name of its own, which reads as "cannot be qualified" rather than
    as top-level — see ``build_tool_name_map`` and ``dedupe_tool_definitions``.
    Handing that case a generated id instead is tracked on #953."""
    tool_group_description: NotRequired[str]
    """Description carried by the group itself, when it had one."""
    metadata: NotRequired[dict[str, Any]]
    cache_hint: NotRequired[dict[str, Any]]


# ============================================================================
# 工具选择配置 Tool choice configuration
# ============================================================================


class ToolChoice(TypedDict):
    """工具选择配置
    Tool choice configuration

    统一了各provider的工具选择策略：
    - none: 不使用工具
    - auto: 自动决定是否使用工具
    - any: 必须使用某个工具（Anthropic的"any"）
    - tool: 使用指定的工具（需要tool_name）
    """

    mode: Literal["none", "auto", "any", "tool"]
    tool_name: NotRequired[str]  # 当mode为"tool"时必需
    tool_type: NotRequired[str]  # "function" (default) or "custom"


# ============================================================================
# 工具调用配置 Tool call configuration
# ============================================================================


class ToolCallConfig(TypedDict, total=False):
    """工具调用配置（少见参数）
    Tool call configuration (less common parameters)

    这些参数不是所有provider都支持，放在这里但主要通过provider_extensions使用：
    - disable_parallel: 禁用并行工具调用
    - max_calls: 最大工具调用数
    """

    disable_parallel: bool  # Anthropic: disable_parallel_tool_use
    max_calls: int  # OpenAI Responses: max_tool_calls


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 工具定义类型 Tool definition types
    "ToolDefinition",
    # 工具选择配置 Tool choice configuration
    "ToolChoice",
    # 工具调用配置 Tool call configuration
    "ToolCallConfig",
]
