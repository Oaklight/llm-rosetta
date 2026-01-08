"""
LLMIR - IR Message Types

IR消息类型定义，包含独立角色的TypedDict
IR message type definitions with independent role TypedDicts
"""

from typing import Any, Dict, Iterable, Literal, Union

from typing_extensions import NotRequired, Required, TypedDict

from .parts import (
    AssistantContentPart,
    ContentPart,
    SystemContentPart,
    ToolContentPart,
    UserContentPart,
)

# ============================================================================
# 消息元数据 Message metadata
# ============================================================================


class StreamingMetadata(TypedDict, total=False):
    """流式传输的元数据
    Metadata for streaming transmission
    """

    is_streaming: bool
    is_final: bool
    chunk_index: int


class MessageMetadata(TypedDict, total=False):
    """消息的元数据，用于存储额外信息
    Metadata of the message, used to store extra information
    """

    message_id: str
    timestamp: str
    streaming: StreamingMetadata
    custom: Dict[str, Any]


# ============================================================================
# 基础消息类型 Base message type
# ============================================================================


class BaseMessage(TypedDict):
    """
    基础消息类型，所有角色消息的共同基础。
    Base message type, common foundation for all role messages.
    """

    content: Required[Iterable[ContentPart]]
    metadata: NotRequired[MessageMetadata]


# ============================================================================
# 独立角色消息类型 Independent role message types
# ============================================================================


class SystemMessage(TypedDict):
    """
    系统消息类型，用于系统指令。
    System message type for system instructions.

    role字段预填写为"system"，使用时不需要手动指定角色。
    The role field is pre-filled as "system", no need to manually specify the role when using.

    内容限制：只允许文本内容。
    Content restrictions: Only text content is allowed.
    """

    role: Required[Literal["system"]]
    content: Required[Iterable[SystemContentPart]]
    metadata: NotRequired[MessageMetadata]


class UserMessage(TypedDict):
    """
    用户消息类型，用于用户输入。
    User message type for user input.

    role字段预填写为"user"，使用时不需要手动指定角色。
    The role field is pre-filled as "user", no need to manually specify the role when using.

    内容限制：文本、图像，未来支持文件、音频。
    Content restrictions: Text, images, future support for files and audio.
    """

    role: Required[Literal["user"]]
    content: Required[Iterable[UserContentPart]]
    metadata: NotRequired[MessageMetadata]


class AssistantMessage(TypedDict):
    """
    助手消息类型，用于AI助手的响应。
    Assistant message type for AI assistant responses.

    role字段预填写为"assistant"，使用时不需要手动指定角色。
    The role field is pre-filled as "assistant", no need to manually specify the role when using.

    内容限制：文本、工具调用、推理、引用，未来支持音频、图像。
    Content restrictions: Text, tool calls, reasoning, citations, future support for audio and images.
    """

    role: Required[Literal["assistant"]]
    content: Required[Iterable[AssistantContentPart]]
    metadata: NotRequired[MessageMetadata]


class ToolMessage(TypedDict):
    """
    工具消息类型，用于工具调用的结果。
    Tool message type for tool call results.

    这是新增的独立角色，替代了之前将工具结果放在user消息中的做法。
    This is a new independent role, replacing the previous practice of putting tool results in user messages.

    role字段预填写为"tool"，使用时不需要手动指定角色。
    The role field is pre-filled as "tool", no need to manually specify the role when using.

    内容限制：只允许工具结果内容。
    Content restrictions: Only tool result content is allowed.
    """

    role: Required[Literal["tool"]]
    content: Required[Iterable[ToolContentPart]]
    metadata: NotRequired[MessageMetadata]


# ============================================================================
# 统一消息类型 Unified message type
# ============================================================================


# 统一的消息类型，支持所有角色
# Unified message type supporting all roles
Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# 为了向后兼容，也提供传统的Message定义
# For backward compatibility, also provide traditional Message definition
class LegacyMessage(TypedDict):
    """
    传统的消息类型定义，为了向后兼容。
    Legacy message type definition for backward compatibility.

    推荐使用具体的角色消息类型（SystemMessage, UserMessage等）。
    Recommend using specific role message types (SystemMessage, UserMessage, etc.).
    """

    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[Iterable[ContentPart]]
    metadata: NotRequired[MessageMetadata]


# ============================================================================
# 类型守卫函数 Type guard functions
# ============================================================================


def is_system_message(message: Message) -> bool:
    """判断是否是系统消息
    Determine if it is a system message
    """
    return message.get("role") == "system"


def is_user_message(message: Message) -> bool:
    """判断是否是用户消息
    Determine if it is a user message
    """
    return message.get("role") == "user"


def is_assistant_message(message: Message) -> bool:
    """判断是否是助手消息
    Determine if it is an assistant message
    """
    return message.get("role") == "assistant"


def is_tool_message(message: Message) -> bool:
    """判断是否是工具消息
    Determine if it is a tool message
    """
    return message.get("role") == "tool"


def is_message(item: Any) -> bool:
    """判断是否是Message
    Determine if it is a Message
    """
    return isinstance(item, dict) and "role" in item and "content" in item


# ============================================================================
# 辅助函数 Helper functions
# ============================================================================


def create_system_message(text: str, **metadata) -> SystemMessage:
    """创建系统消息
    Create system message

    Args:
        text: 系统指令文本 System instruction text
        **metadata: 额外的元数据 Additional metadata

    Returns:
        系统消息，role字段自动设置为"system"
        System message with role field automatically set to "system"
    """
    message: SystemMessage = {
        "role": "system",
        "content": [{"type": "text", "text": text}],
    }
    if metadata:
        message["metadata"] = MessageMetadata(**metadata)
    return message


def create_user_message(text: str, **metadata) -> UserMessage:
    """创建用户消息
    Create user message

    Args:
        text: 用户输入文本 User input text
        **metadata: 额外的元数据 Additional metadata

    Returns:
        用户消息，role字段自动设置为"user"
        User message with role field automatically set to "user"
    """
    message: UserMessage = {
        "role": "user",
        "content": [{"type": "text", "text": text}],
    }
    if metadata:
        message["metadata"] = MessageMetadata(**metadata)
    return message


def create_assistant_message(text: str, **metadata) -> AssistantMessage:
    """创建助手消息
    Create assistant message

    Args:
        text: 助手响应文本 Assistant response text
        **metadata: 额外的元数据 Additional metadata

    Returns:
        助手消息，role字段自动设置为"assistant"
        Assistant message with role field automatically set to "assistant"
    """
    message: AssistantMessage = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }
    if metadata:
        message["metadata"] = MessageMetadata(**metadata)
    return message


def create_tool_message(
    tool_call_id: str, result: Any, is_error: bool = False, **metadata
) -> ToolMessage:
    """创建工具消息
    Create tool message

    Args:
        tool_call_id: 工具调用ID Tool call ID
        result: 工具执行结果 Tool execution result
        is_error: 是否为错误结果 Whether it is an error result
        **metadata: 额外的元数据 Additional metadata

    Returns:
        工具消息，role字段自动设置为"tool"
        Tool message with role field automatically set to "tool"
    """
    message: ToolMessage = {
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": tool_call_id,
                "result": result,
                "is_error": is_error,
            }
        ],
    }
    if metadata:
        message["metadata"] = MessageMetadata(**metadata)
    return message


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 元数据类型 Metadata types
    "MessageMetadata",
    "StreamingMetadata",
    # 基础消息类型 Base message type
    "BaseMessage",
    # 独立角色消息类型 Independent role message types
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    # 统一消息类型 Unified message type
    "Message",
    "LegacyMessage",
    # 类型守卫函数 Type guard functions
    "is_system_message",
    "is_user_message",
    "is_assistant_message",
    "is_tool_message",
    "is_message",
    # 辅助函数 Helper functions
    "create_system_message",
    "create_user_message",
    "create_assistant_message",
    "create_tool_message",
]
