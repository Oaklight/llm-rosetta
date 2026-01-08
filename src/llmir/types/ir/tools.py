"""
LLMIR - IR Tool Types

IR工具相关类型定义
IR tool-related type definitions
"""

from typing import Any, Dict, Iterable, Literal

from typing_extensions import TypedDict

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
    """

    type: Literal[
        "function",
        "mcp",
        # 未来陆续支持 Future supports
        # "web_search",
        # "code_interpreter",
        # "file_search",
    ]
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required_parameters: Iterable[str]
    metadata: Dict[str, Any]


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
    tool_name: str  # 当mode为"tool"时必需


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
