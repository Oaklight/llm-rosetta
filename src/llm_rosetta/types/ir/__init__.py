"""
LLM-Rosetta - IR (Intermediate Representation) Types

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
    is_extension_item,  # noqa: F401
)

# 辅助函数 Helper functions (advanced — import from .helpers directly)
from .helpers import (  # noqa: F401
    create_tool_result_message,
    extract_all_text,
    extract_text_content,
    extract_tool_calls,
)

# 验证工具 Validation utilities (internal — import from .validation directly)
from .validation import (  # noqa: F401
    ValidationError,
    validate_ir_request,
    validate_ir_response,
    validate_messages,
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
    create_assistant_message,  # noqa: F401
    create_system_message,  # noqa: F401
    create_tool_message,  # noqa: F401
    create_user_message,  # noqa: F401
    is_assistant_message,  # noqa: F401
    is_message,  # noqa: F401
    is_system_message,  # noqa: F401
    is_tool_message,  # noqa: F401
    is_user_message,  # noqa: F401
)

# 内容部分类型 Content part types
from .parts import (
    AssistantContentPart,
    AudioData,
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
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)

# 工具类型 Tool types
from .tools import ToolCallConfig, ToolChoice, ToolDefinition

# 类型守卫 Type guards (advanced/internal — import from .type_guards directly)
from .type_guards import (  # noqa: F401
    TYPE_CLASS_MAP,
    get_part_type,
    is_audio_part,
    is_citation_part,
    is_content_block_end_event,
    is_content_block_start_event,
    is_file_part,
    is_finish_event,
    is_image_part,
    is_part_type,
    is_reasoning_delta_event,
    is_reasoning_part,
    is_refusal_part,
    is_stream_end_event,
    is_stream_start_event,
    is_text_delta_event,
    is_text_part,
    is_tool_call_delta_event,
    is_tool_call_part,
    is_tool_call_start_event,
    is_tool_result_part,
    is_usage_event,
    isinstance_part,
)

# 为了向后兼容，定义旧的类型别名
# For backward compatibility, define old type aliases
from collections.abc import Sequence

IRInput = Sequence[Message | ExtensionItem]
IRInputSimple = Sequence[Message]


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
    "AudioData",
    # 角色特定内容类型 Role-specific content types
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
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
    # ========== 流式事件类型 Stream event types ==========
    "IRStreamEvent",
    "StreamStartEvent",
    "StreamEndEvent",
    "ContentBlockStartEvent",
    "ContentBlockEndEvent",
    "TextDeltaEvent",
    "ReasoningDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    # ========== 向后兼容类型 Backward compatibility types ==========
    "IRInput",
    "IRInputSimple",
]
