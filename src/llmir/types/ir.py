"""
LLMIR - IR (Intermediate Representation) Types

统一的IR类型导出入口
Unified IR types export entry point

这个模块重新组织了IR类型定义：
- ir_request.py: 请求参数类型（基于SDK body structures）
- ir_response.py: 响应消息类型（基于消息结构）
- ir.py: 统一导出入口

This module reorganizes IR type definitions:
- ir_request.py: Request parameter types (based on SDK body structures)
- ir_response.py: Response message types (based on message structures)
- ir.py: Unified export entry point
"""

# ============================================================================
# 从 ir_response 导入响应相关类型
# Import response-related types from ir_response
# ============================================================================

from .ir_response import (
    # 核心类型 Core types
    Message,
    MessageMetadata,
    StreamingMetadata,
    ContentPart,
    ExtensionItem,
    IRInput,
    IRInputSimple,
    # 内容部分类型 Content part types
    TextPart,
    ImagePart,
    ImageData,
    FilePart,
    FileData,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    # 扩展项类型 Extension item types
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
    # 类型守卫函数 Type guard functions
    is_message,
    is_extension_item,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
    # 辅助函数 Helper functions
    extract_text_content,
    extract_tool_calls,
    create_tool_result_message,
)

# ============================================================================
# 从 ir_request 导入请求相关类型
# ============================================================================

from .ir_request import (
    # 主请求类型 Main request type
    IRRequest,
    # 工具相关 Tool related
    ToolDefinition,
    ToolChoice,
    ToolCallConfig,
    # 配置类型 Configuration types
    GenerationConfig,
    ResponseFormatConfig,
    StreamConfig,
    ReasoningConfig,
    CacheConfig,
)

# ============================================================================
# 导出所有类型 Export all types
# ============================================================================

__all__ = [
    # ========== 响应相关类型 Response-related types ==========
    # 核心类型 Core types
    "Message",
    "MessageMetadata",
    "StreamingMetadata",
    "ContentPart",
    "ExtensionItem",
    "IRInput",
    "IRInputSimple",
    # 内容部分类型 Content part types
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    # 扩展项类型 Extension item types
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    # 类型守卫函数 Type guard functions
    "is_message",
    "is_extension_item",
    "is_text_part",
    "is_tool_call_part",
    "is_tool_result_part",
    # 辅助函数 Helper functions
    "extract_text_content",
    "extract_tool_calls",
    "create_tool_result_message",
    # ========== 请求相关类型 Request-related types ==========
    # 主请求类型 Main request type
    "IRRequest",
    # 工具相关 Tool related
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    # 配置类型 Configuration types
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "CacheConfig",
]