"""
LLM-Rosetta Types Package

This package contains type definitions for:
- IR (Intermediate Representation) types
- Provider-specific types (Anthropic, Google, OpenAI)
"""

from .ir import (
    # Content part types
    ContentPart,
    TextPart,
    ImagePart,
    ImageData,
    FilePart,
    FileData,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    AudioPart,
    # Role-specific content types
    SystemContentPart,
    UserContentPart,
    AssistantContentPart,
    ToolContentPart,
    # Message types
    Message,
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    LegacyMessage,
    MessageMetadata,
    StreamingMetadata,
    # Tool types
    ToolDefinition,
    ToolChoice,
    ToolCallConfig,
    # Generation configuration types
    GenerationConfig,
    ResponseFormatConfig,
    StreamConfig,
    ReasoningConfig,
    CacheConfig,
    # Request types
    IRRequest,
    # Response types
    IRResponse,
    ExtensionItem,
    UsageInfo,
    FinishReason,
    ChoiceInfo,
    # Stream event types
    IRStreamEvent,
    StreamStartEvent,
    StreamEndEvent,
    ContentBlockStartEvent,
    ContentBlockEndEvent,
    TextDeltaEvent,
    ReasoningDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    FinishEvent,
    UsageEvent,
    # Backward compatibility types
    IRInput,
    IRInputSimple,
    # Type guards
    is_message,
    is_system_message,
    is_user_message,
    is_assistant_message,
    is_tool_message,
    is_extension_item,
    is_part_type,
    isinstance_part,
    get_part_type,
    TYPE_CLASS_MAP,
    # Helper functions
    extract_text_content,
    extract_tool_calls,
    create_tool_result_message,
    create_system_message,
    create_user_message,
    create_assistant_message,
    create_tool_message,
)

__all__ = [
    # Content part types
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
    # Role-specific content types
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    # Message types
    "Message",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "LegacyMessage",
    "MessageMetadata",
    "StreamingMetadata",
    # Tool types
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    # Generation configuration types
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "CacheConfig",
    # Request types
    "IRRequest",
    # Response types
    "IRResponse",
    "ExtensionItem",
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    # Stream event types
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
    # Backward compatibility types
    "IRInput",
    "IRInputSimple",
    # Type guards
    "is_message",
    "is_system_message",
    "is_user_message",
    "is_assistant_message",
    "is_tool_message",
    "is_extension_item",
    "is_part_type",
    "isinstance_part",
    "get_part_type",
    "TYPE_CLASS_MAP",
    # Helper functions
    "extract_text_content",
    "extract_tool_calls",
    "create_tool_result_message",
    "create_system_message",
    "create_user_message",
    "create_assistant_message",
    "create_tool_message",
]
