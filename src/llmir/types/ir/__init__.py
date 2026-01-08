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

# 工具类型 Tool types
from .tools import ToolCallConfig, ToolChoice, ToolDefinition

# 类型守卫 Type guards
from .type_guards import is_part_type

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
    # ========== 辅助函数 Helper functions ==========
    "extract_text_content",
    "extract_tool_calls",
    "create_tool_result_message",
]
