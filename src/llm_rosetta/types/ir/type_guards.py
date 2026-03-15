"""
LLM-Rosetta - IR Type Guards

IR类型守卫系统，提供TypeGuard函数用于类型收窄
IR type guard system providing TypeGuard functions for type narrowing
"""

import sys
from typing import Any, Dict, Mapping, Type, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

from .parts import (
    AudioPart,
    CitationPart,
    ContentPart,
    FilePart,
    ImagePart,
    ReasoningPart,
    RefusalPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
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

# TypeVar for generic type checking
T = TypeVar("T", bound=Mapping[str, Any])


# ============================================================================
# ContentPart TypeGuard functions
# ============================================================================


def is_text_part(part: ContentPart) -> TypeGuard[TextPart]:
    """Check if a content part is a TextPart."""
    return isinstance(part, dict) and part.get("type") == "text"


def is_image_part(part: ContentPart) -> TypeGuard[ImagePart]:
    """Check if a content part is an ImagePart."""
    return isinstance(part, dict) and part.get("type") == "image"


def is_file_part(part: ContentPart) -> TypeGuard[FilePart]:
    """Check if a content part is a FilePart."""
    return isinstance(part, dict) and part.get("type") == "file"


def is_audio_part(part: ContentPart) -> TypeGuard[AudioPart]:
    """Check if a content part is an AudioPart."""
    return isinstance(part, dict) and part.get("type") == "audio"


def is_tool_call_part(part: ContentPart) -> TypeGuard[ToolCallPart]:
    """Check if a content part is a ToolCallPart."""
    return isinstance(part, dict) and part.get("type") == "tool_call"


def is_tool_result_part(part: ContentPart) -> TypeGuard[ToolResultPart]:
    """Check if a content part is a ToolResultPart."""
    return isinstance(part, dict) and part.get("type") == "tool_result"


def is_reasoning_part(part: ContentPart) -> TypeGuard[ReasoningPart]:
    """Check if a content part is a ReasoningPart."""
    return isinstance(part, dict) and part.get("type") == "reasoning"


def is_refusal_part(part: ContentPart) -> TypeGuard[RefusalPart]:
    """Check if a content part is a RefusalPart."""
    return isinstance(part, dict) and part.get("type") == "refusal"


def is_citation_part(part: ContentPart) -> TypeGuard[CitationPart]:
    """Check if a content part is a CitationPart."""
    return isinstance(part, dict) and part.get("type") == "citation"


# ============================================================================
# IRStreamEvent TypeGuard functions
# ============================================================================


def is_stream_start_event(event: IRStreamEvent) -> TypeGuard[StreamStartEvent]:
    """Check if a stream event is a StreamStartEvent."""
    return isinstance(event, dict) and event.get("type") == "stream_start"


def is_stream_end_event(event: IRStreamEvent) -> TypeGuard[StreamEndEvent]:
    """Check if a stream event is a StreamEndEvent."""
    return isinstance(event, dict) and event.get("type") == "stream_end"


def is_content_block_start_event(
    event: IRStreamEvent,
) -> TypeGuard[ContentBlockStartEvent]:
    """Check if a stream event is a ContentBlockStartEvent."""
    return isinstance(event, dict) and event.get("type") == "content_block_start"


def is_content_block_end_event(
    event: IRStreamEvent,
) -> TypeGuard[ContentBlockEndEvent]:
    """Check if a stream event is a ContentBlockEndEvent."""
    return isinstance(event, dict) and event.get("type") == "content_block_end"


def is_text_delta_event(event: IRStreamEvent) -> TypeGuard[TextDeltaEvent]:
    """Check if a stream event is a TextDeltaEvent."""
    return isinstance(event, dict) and event.get("type") == "text_delta"


def is_reasoning_delta_event(
    event: IRStreamEvent,
) -> TypeGuard[ReasoningDeltaEvent]:
    """Check if a stream event is a ReasoningDeltaEvent."""
    return isinstance(event, dict) and event.get("type") == "reasoning_delta"


def is_tool_call_start_event(
    event: IRStreamEvent,
) -> TypeGuard[ToolCallStartEvent]:
    """Check if a stream event is a ToolCallStartEvent."""
    return isinstance(event, dict) and event.get("type") == "tool_call_start"


def is_tool_call_delta_event(
    event: IRStreamEvent,
) -> TypeGuard[ToolCallDeltaEvent]:
    """Check if a stream event is a ToolCallDeltaEvent."""
    return isinstance(event, dict) and event.get("type") == "tool_call_delta"


def is_finish_event(event: IRStreamEvent) -> TypeGuard[FinishEvent]:
    """Check if a stream event is a FinishEvent."""
    return isinstance(event, dict) and event.get("type") == "finish"


def is_usage_event(event: IRStreamEvent) -> TypeGuard[UsageEvent]:
    """Check if a stream event is a UsageEvent."""
    return isinstance(event, dict) and event.get("type") == "usage"


# ============================================================================
# Generic type checking (backward compatible)
# ============================================================================


def is_part_type(part: Any, part_class: Type[T]) -> bool:
    """
    通用的类型检查函数，类似isinstance但针对TypedDict优化
    Generic type checking function, similar to isinstance but optimized for TypedDict

    Args:
        part: 要检查的内容部分 Content part to check
        part_class: 目标类型类 Target type class

    Returns:
        是否匹配指定类型 Whether it matches the specified type

    Examples:
        >>> part = {"type": "text", "text": "hello"}
        >>> is_part_type(part, TextPart)  # True
        >>> is_part_type(part, ToolCallPart)  # False
    """
    if not isinstance(part, dict):
        return False

    part_type = part.get("type")
    if not part_type:
        return False

    expected_type = _TYPE_STRING_MAP.get(part_class)
    if expected_type is None:
        return False

    return part_type == expected_type


# Internal mapping from TypedDict class to expected "type" string
_TYPE_STRING_MAP: Dict[type, str] = {
    TextPart: "text",
    ImagePart: "image",
    FilePart: "file",
    ToolCallPart: "tool_call",
    ToolResultPart: "tool_result",
    ReasoningPart: "reasoning",
    RefusalPart: "refusal",
    CitationPart: "citation",
    AudioPart: "audio",
    # Stream events
    StreamStartEvent: "stream_start",
    StreamEndEvent: "stream_end",
    ContentBlockStartEvent: "content_block_start",
    ContentBlockEndEvent: "content_block_end",
    TextDeltaEvent: "text_delta",
    ReasoningDeltaEvent: "reasoning_delta",
    ToolCallStartEvent: "tool_call_start",
    ToolCallDeltaEvent: "tool_call_delta",
    FinishEvent: "finish",
    UsageEvent: "usage",
}

# ============================================================================
# 类型映射表 Type mapping table
# ============================================================================

# 类型字符串到类型类的映射
TYPE_CLASS_MAP: Dict[str, Type[ContentPart]] = {
    "text": TextPart,
    "image": ImagePart,
    "file": FilePart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "reasoning": ReasoningPart,
    "refusal": RefusalPart,
    "citation": CitationPart,
    "audio": AudioPart,
}


def get_part_type(part: Any) -> Union[Type[ContentPart], None]:
    """
    获取内容部分的具体类型
    Get the specific type of content part
    """
    if not isinstance(part, dict):
        return None

    part_type = part.get("type")
    if part_type in TYPE_CLASS_MAP:
        return TYPE_CLASS_MAP[part_type]

    return None


def isinstance_part(part: Any, *part_types: Type[ContentPart]) -> bool:
    """
    类似isinstance的函数，支持多个类型检查
    isinstance-like function supporting multiple type checking
    """
    for part_type in part_types:
        if is_part_type(part, part_type):
            return True
    return False


# ============================================================================
# 导出的主要函数 Main Exported Functions
# ============================================================================

__all__ = [
    # ContentPart TypeGuard functions
    "is_text_part",
    "is_image_part",
    "is_file_part",
    "is_audio_part",
    "is_tool_call_part",
    "is_tool_result_part",
    "is_reasoning_part",
    "is_refusal_part",
    "is_citation_part",
    # IRStreamEvent TypeGuard functions
    "is_stream_start_event",
    "is_stream_end_event",
    "is_content_block_start_event",
    "is_content_block_end_event",
    "is_text_delta_event",
    "is_reasoning_delta_event",
    "is_tool_call_start_event",
    "is_tool_call_delta_event",
    "is_finish_event",
    "is_usage_event",
    # Generic functions
    "is_part_type",
    "isinstance_part",
    "get_part_type",
    # Mapping tables
    "TYPE_CLASS_MAP",
]
