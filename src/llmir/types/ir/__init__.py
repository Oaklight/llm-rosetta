"""
LLMIR - IR (Intermediate Representation) Types

统一的IR类型导出入口
Unified IR types export entry point

这个模块重新组织了IR类型定义：
- parts.py: 内容部分类型（ContentPart及其子类型）
- messages.py: 消息类型（独立角色的TypedDict）
- tools.py: 工具相关类型（工具定义、选择、配置）
- generation.py: 生成控制配置类型（温度、top_p等生成参数）
- request.py: 请求参数类型（基于SDK body structures）
- response.py: 响应类型（扩展项和响应统计）
- helpers.py: 辅助函数（内容提取、消息创建等）

This module reorganizes IR type definitions:
- parts.py: Content part types (ContentPart and its subtypes)
- messages.py: Message types (independent role TypedDicts)
- tools.py: Tool-related types (tool definition, choice, configuration)
- generation.py: Generation control configuration types (temperature, top_p, etc.)
- request.py: Request parameter types (based on SDK body structures)
- response.py: Response types (extension items and response statistics)
- helpers.py: Helper functions (content extraction, message creation, etc.)
"""

# ============================================================================
# 从各模块导入类型 Import types from modules
# ============================================================================

# 生成配置类型 Generation configuration types
# ============================================================================
# 向后兼容类型定义 Backward compatibility type definitions
# ============================================================================
from typing import Iterable, Union

from .configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)

# 扩展项类型 Extension types
from .extensions import (
    BatchMarker,
    ExtensionItem,
    SessionControl,
    SystemEvent,
    ToolChainNode,
    is_extension_item,
)

# 辅助函数 Helper functions
from .helpers import (
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

# 消息类型 Message types
from .messages import (
    AssistantMessage,
    BaseMessage,
    LegacyMessage,
    Message,
    MessageMetadata,
    StreamingMetadata,
    SystemMessage,
    ToolMessage,
    UserMessage,
    create_assistant_message,
    create_system_message,
    create_tool_message,
    create_user_message,
    is_assistant_message,
    is_message,
    is_system_message,
    is_tool_message,
    is_user_message,
)

# 内容部分类型 Content part types
from .parts import (
    AssistantContentPart,
    AudioPart,
    CitationPart,
    ContentPart,
    FileData,
    FilePart,
    ImageData,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    SystemContentPart,
    TextPart,
    ToolCallPart,
    ToolContentPart,
    ToolResultPart,
    UserContentPart,
)

# 请求类型 Request types
from .request import IRRequest

# 响应类型 Response types
from .response import (
    ChoiceInfo,
    FinishReason,
    IRResponse,
    UsageInfo,
)

# 流式事件类型 Stream event types
from .stream import (
    FinishEvent,
    IRStreamEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)

# 工具类型 Tool types
from .tools import ToolCallConfig, ToolChoice, ToolDefinition

# 类型守卫 Type guards
from .type_guards import TYPE_CLASS_MAP, get_part_type, is_part_type, isinstance_part

# 为了向后兼容，定义旧的类型别名
# For backward compatibility, define old type aliases
IRInput = Iterable[Union[Message, ExtensionItem]]
IRInputSimple = Iterable[Message]


# 向后兼容的类型守卫函数
# Backward compatibility type guard functions
def is_text_part(part):
    """向后兼容的文本部分检查函数"""
    return is_part_type(part, TextPart)


def is_image_part(part):
    """向后兼容的图像部分检查函数"""
    return is_part_type(part, ImagePart)


def is_tool_call_part(part):
    """向后兼容的工具调用部分检查函数"""
    return is_part_type(part, ToolCallPart)


def is_tool_result_part(part):
    """向后兼容的工具结果部分检查函数"""
    return is_part_type(part, ToolResultPart)


def is_file_part(part):
    """向后兼容的文件部分检查函数"""
    return is_part_type(part, FilePart)


def is_audio_part(part):
    """向后兼容的音频部分检查函数"""
    return is_part_type(part, AudioPart)


def is_reasoning_part(part):
    """向后兼容的推理部分检查函数"""
    return is_part_type(part, ReasoningPart)


def is_refusal_part(part):
    """向后兼容的拒绝部分检查函数"""
    return is_part_type(part, RefusalPart)


def is_citation_part(part):
    """向后兼容的引用部分检查函数"""
    return is_part_type(part, CitationPart)


# ============================================================================
# 导出所有类型 Export all types
# ============================================================================

__all__ = [
    # ========== 内容部分类型 Content part types ==========
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "RefusalPart",
    "CitationPart",
    "AudioPart",
    # 角色特定内容类型 Role-specific content types
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    # 类型守卫函数 Type guard functions
    "is_part_type",
    "isinstance_part",
    "get_part_type",
    "TYPE_CLASS_MAP",
    # ========== 消息类型 Message types ==========
    "Message",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "LegacyMessage",
    "MessageMetadata",
    "StreamingMetadata",
    # 消息类型守卫 Message type guards
    "is_message",
    "is_system_message",
    "is_user_message",
    "is_assistant_message",
    "is_tool_message",
    # 消息创建函数 Message creation functions
    "create_system_message",
    "create_user_message",
    "create_assistant_message",
    "create_tool_message",
    # ========== 工具类型 Tool types ==========
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    # ========== 生成配置类型 Generation configuration types ==========
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "CacheConfig",
    # ========== 请求类型 Request types ==========
    "IRRequest",
    # ========== 响应类型 Response types ==========
    "IRResponse",
    "ExtensionItem",
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    # 响应类型守卫 Response type guards
    "is_extension_item",
    # ========== 流式事件类型 Stream event types ==========
    "IRStreamEvent",
    "TextDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    # ========== 向后兼容类型 Backward compatibility types ==========
    "IRInput",
    "IRInputSimple",
    # ========== 向后兼容函数 Backward compatibility functions ==========
    "is_text_part",
    "is_image_part",
    "is_tool_call_part",
    "is_tool_result_part",
    "is_file_part",
    "is_audio_part",
    "is_reasoning_part",
    "is_refusal_part",
    "is_citation_part",
    # ========== 辅助函数 Helper functions ==========
    "extract_text_content",
    "extract_tool_calls",
    "create_tool_result_message",
]
