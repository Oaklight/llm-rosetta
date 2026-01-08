"""
LLMIR - IR Helper Functions

IR辅助函数，用于处理IR消息和内容
IR helper functions for processing IR messages and content
"""

from typing import Any, List, Union

from .messages import Message, ToolMessage, create_tool_message
from .parts import TextPart, ToolCallPart
from .type_guards import is_part_type

# ============================================================================
# 内容提取函数 Content extraction functions
# ============================================================================


def extract_text_content(message: Message) -> str:
    """从消息中提取所有文本内容
    Extract all text content from message

    Args:
        message: IR格式的消息 IR format message

    Returns:
        拼接后的文本内容 Concatenated text content
    """
    texts = []
    for part in message.get("content", []):
        if is_part_type(part, TextPart):
            text = part.get("text", "")
            if text is not None:  # 确保text不是None Ensure text is not None
                texts.append(str(text))
    return "".join(texts)


def extract_tool_calls(
    message: Message, limit: Union[int, None] = None
) -> List[ToolCallPart]:
    """从消息中提取工具调用
    Extract tool calls from message

    Args:
        message: IR格式的消息 IR format message
        limit: 限制返回的工具调用数量。None表示返回所有，1表示只返回第一个
               Limit the number of tool calls returned. None means return all, 1 means return only the first

    Returns:
        工具调用部分的列表 List of tool call parts

    Examples:
        >>> # 获取所有工具调用 Get all tool calls
        >>> all_calls = extract_tool_calls(message)
        >>>
        >>> # 只获取第一个工具调用 Get only the first tool call
        >>> first_call = extract_tool_calls(message, limit=1)
        >>> if first_call:
        >>>     tool_call = first_call[0]
        >>>
        >>> # 获取前3个工具调用 Get the first 3 tool calls
        >>> first_three = extract_tool_calls(message, limit=3)
    """
    tool_calls = [
        part  # type: ignore
        for part in message.get("content", [])
        if is_part_type(part, ToolCallPart)
    ]

    if limit is not None:
        return tool_calls[:limit]
    return tool_calls


# ============================================================================
# 消息创建函数 Message creation functions
# ============================================================================


def create_tool_result_message(
    tool_call_id: str, result: Any, is_error: bool = False, **metadata
) -> ToolMessage:
    """创建工具结果消息
    Create tool result message

    Args:
        tool_call_id: 工具调用ID Tool call ID
        result: 工具执行结果 Tool execution result
        is_error: 是否为错误结果 Whether it is an error result
        **metadata: 额外的元数据 Additional metadata

    Returns:
        IR格式的工具结果消息，使用tool角色
        IR format tool result message with tool role
    """
    return create_tool_message(
        tool_call_id=tool_call_id, result=result, is_error=is_error, **metadata
    )


# ============================================================================
# 导出的主要函数 Main Exported Functions
# ============================================================================

__all__ = [
    # 内容提取函数 Content extraction functions
    "extract_text_content",
    "extract_tool_calls",
    # 消息创建函数 Message creation functions
    "create_tool_result_message",
]
