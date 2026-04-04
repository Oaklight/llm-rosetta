"""
LLM-Rosetta - IR Extension Types

IR扩展项类型定义，用于特殊场景的扩展功能
IR extension type definitions for special scenario extensions
"""

import sys
from typing import Any, Literal, TypeGuard, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict

# ============================================================================
# 扩展项类型（用于特殊场景） Extension item types (for special scenarios)
# ============================================================================


class SystemEvent(TypedDict):
    """
    系统级事件，用于记录会话状态变化。
    System-level events, used to record session state changes.

    Examples:
        - 会话开始/结束 Session start/end
        - 会话暂停/恢复 Session pause/resume
        - 超时警告 Timeout warning
        - 错误事件 Error event
    """

    type: Required[Literal["system_event"]]
    event_type: Required[
        Literal[
            "session_start",
            "session_pause",
            "session_resume",
            "session_timeout",
            "session_end",
            "error",
            "warning",
        ]
    ]
    timestamp: Required[str]  # ISO 8601格式 ISO 8601 format
    event_data: NotRequired[dict[str, Any]]
    message: NotRequired[str]


class BatchMarker(TypedDict):
    """
    批次标记，用于标记一组相关的操作。
    Batch marker, used to mark a group of related operations.

    Examples:
        - 并行工具调用的开始/结束 Start/end of parallel tool calls
        - 部分结果的进度跟踪 Progress tracking of partial results
    """

    type: Required[Literal["batch_marker"]]
    batch_id: Required[str]
    batch_type: Required[Literal["start", "end", "partial"]]
    total_items: NotRequired[int]
    completed_items: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


class SessionControl(TypedDict):
    """
    会话控制指令，用于控制工具调用的执行。
    Session control instructions, used to control the execution of tool calls.

    Examples:
        - 取消工具调用 Cancel tool call
        - 修改工具调用参数 Modify tool call parameters
        - 暂停/恢复工具执行 Pause/resume tool execution
    """

    type: Required[Literal["session_control"]]
    control_type: Required[
        Literal[
            "cancel_tool",
            "modify_tool",
            "pause_tool",
            "resume_tool",
        ]
    ]
    target_id: Required[str]  # 目标tool_call_id Target tool_call_id
    reason: NotRequired[str]
    new_input: NotRequired[dict[str, Any]]  # 用于modify_tool Used for modify_tool


class ToolChainNode(TypedDict):
    """
    工具链节点，用于表示工具调用的依赖关系。
    Tool chain node, used to represent dependencies between tool calls.

    支持DAG结构，一个工具的输出可以作为另一个工具的输入。
    Supports DAG structure, the output of one tool can be used as the input of another.

    Examples:
        - 搜索 → 总结 Search → Summarize
        - 数据获取 → 分析 → 可视化 Data acquisition → Analysis → Visualization
    """

    type: Required[Literal["tool_chain_node"]]
    node_id: Required[str]
    tool_call: Required[dict[str, Any]]  # ToolCallPart
    depends_on: NotRequired[list[str]]  # 依赖的节点ID列表 List of dependent node IDs
    auto_execute: NotRequired[bool]  # 是否自动执行 Whether to auto execute


# 扩展项联合类型 Extension item union type
ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
]


# ============================================================================
# 类型守卫函数 Type guard functions
# ============================================================================

# 扩展项类型映射表 Extension item type mapping table
EXTENSION_TYPE_MAP: dict[str, type[ExtensionItem]] = {
    "system_event": SystemEvent,
    "batch_marker": BatchMarker,
    "session_control": SessionControl,
    "tool_chain_node": ToolChainNode,
}


def is_extension_type(
    item: Any, extension_class: type[ExtensionItem]
) -> TypeGuard[ExtensionItem]:
    """
    通用的扩展项类型检查函数，类似isinstance但针对TypedDict优化
    Generic extension item type checking function, similar to isinstance but optimized for TypedDict

    Args:
        item: 要检查的扩展项 Extension item to check
        extension_class: 目标类型类 Target type class

    Returns:
        是否匹配指定类型 Whether it matches the specified type

    Examples:
        >>> event = {"type": "system_event", "event_type": "session_start", "timestamp": "2024-01-01T00:00:00Z"}
        >>> is_extension_type(event, SystemEvent)  # True
        >>> is_extension_type(event, BatchMarker)  # False
    """
    if not isinstance(item, dict):
        return False

    # 检查type字段
    item_type = item.get("type")
    if not item_type:
        return False

    # 获取期望的类型值
    expected_type = None
    if extension_class == SystemEvent:
        expected_type = "system_event"
    elif extension_class == BatchMarker:
        expected_type = "batch_marker"
    elif extension_class == SessionControl:
        expected_type = "session_control"
    elif extension_class == ToolChainNode:
        expected_type = "tool_chain_node"
    else:
        return False

    if item_type != expected_type:
        return False

    # 基本的必需字段检查
    if extension_class == SystemEvent:
        return all(key in item for key in ["event_type", "timestamp"])
    elif extension_class == BatchMarker:
        return all(key in item for key in ["batch_id", "batch_type"])
    elif extension_class == SessionControl:
        return all(key in item for key in ["control_type", "target_id"])
    elif extension_class == ToolChainNode:
        return all(key in item for key in ["node_id", "tool_call"])

    return True


def is_extension_item(item: Any) -> TypeGuard[ExtensionItem]:
    """
    判断是否是ExtensionItem
    Determine if it is an ExtensionItem

    Args:
        item: 要检查的项目 Item to check

    Returns:
        是否是扩展项 Whether it is an extension item

    Examples:
        >>> event = {"type": "system_event", "event_type": "session_start", "timestamp": "2024-01-01T00:00:00Z"}
        >>> is_extension_item(event)  # True
        >>> is_extension_item({"type": "unknown"})  # False
    """
    if not isinstance(item, dict):
        return False

    item_type = item.get("type")
    if item_type not in EXTENSION_TYPE_MAP:
        return False

    # 使用对应的类型检查函数
    extension_class = EXTENSION_TYPE_MAP[item_type]
    return is_extension_type(item, extension_class)


def get_extension_type(item: Any) -> type[ExtensionItem] | None:
    """
    获取扩展项的具体类型
    Get the specific type of extension item

    Args:
        item: 扩展项 Extension item

    Returns:
        对应的类型类，如果无法确定则返回None
        Corresponding type class, None if cannot be determined

    Examples:
        >>> event = {"type": "system_event", "event_type": "session_start", "timestamp": "2024-01-01T00:00:00Z"}
        >>> get_extension_type(event)  # SystemEvent
    """
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if item_type in EXTENSION_TYPE_MAP:
        extension_class = EXTENSION_TYPE_MAP[item_type]
        # 验证是否真的匹配这个类型
        if is_extension_type(item, extension_class):
            return extension_class

    return None


def isinstance_extension(item: Any, *extension_types: type[ExtensionItem]) -> bool:
    """
    类似isinstance的函数，支持多个扩展项类型检查
    isinstance-like function supporting multiple extension type checking

    Args:
        item: 要检查的扩展项 Extension item to check
        *extension_types: 一个或多个类型类 One or more type classes

    Returns:
        是否匹配任一指定类型 Whether it matches any of the specified types

    Examples:
        >>> event = {"type": "system_event", "event_type": "session_start", "timestamp": "2024-01-01T00:00:00Z"}
        >>> isinstance_extension(event, SystemEvent)  # True
        >>> isinstance_extension(event, SystemEvent, BatchMarker)  # True
        >>> isinstance_extension(event, BatchMarker)  # False
    """
    for extension_type in extension_types:
        if is_extension_type(item, extension_type):
            return True
    return False


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 扩展项类型 Extension item types
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    "ExtensionItem",
    # 类型守卫函数 Type guard functions
    "is_extension_item",
    "is_extension_type",
    "get_extension_type",
    "isinstance_extension",
    # 映射表 Mapping tables
    "EXTENSION_TYPE_MAP",
]
