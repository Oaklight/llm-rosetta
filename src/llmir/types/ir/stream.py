"""
LLMIR - IR Stream Event Types

IR流式事件类型定义，用于支持SSE chunk级别的实时转换
IR stream event type definitions for supporting SSE chunk-level real-time conversion

包含以下事件类型：
- TextDeltaEvent: 文本增量事件
- ToolCallStartEvent: 工具调用开始事件
- ToolCallDeltaEvent: 工具调用增量事件
- FinishEvent: 完成事件
- UsageEvent: 使用统计事件

Contains the following event types:
- TextDeltaEvent: Text delta event
- ToolCallStartEvent: Tool call start event
- ToolCallDeltaEvent: Tool call delta event
- FinishEvent: Finish event
- UsageEvent: Usage statistics event
"""

from typing import Literal, Union

from typing_extensions import NotRequired, Required, TypedDict

from .response import FinishReason, UsageInfo

# ============================================================================
# Stream event types
# ============================================================================


class TextDeltaEvent(TypedDict):
    """Text content delta event.

    Emitted when a new text fragment is received from the model.
    """

    type: Required[Literal["text_delta"]]
    text: Required[str]
    choice_index: NotRequired[int]


class ToolCallStartEvent(TypedDict):
    """Tool call start event.

    Emitted when a new tool call begins, providing the tool call ID and name.
    """

    type: Required[Literal["tool_call_start"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    choice_index: NotRequired[int]


class ToolCallDeltaEvent(TypedDict):
    """Tool call arguments delta event.

    Emitted when a new fragment of tool call arguments JSON string is received.
    """

    type: Required[Literal["tool_call_delta"]]
    tool_call_id: Required[str]
    arguments_delta: Required[str]  # JSON string fragment
    choice_index: NotRequired[int]


class FinishEvent(TypedDict):
    """Finish event.

    Emitted when the model finishes generating for a choice.
    """

    type: Required[Literal["finish"]]
    finish_reason: Required[FinishReason]
    choice_index: NotRequired[int]


class UsageEvent(TypedDict):
    """Usage statistics event.

    Emitted when token usage statistics are available (typically at the end of stream).
    """

    type: Required[Literal["usage"]]
    usage: Required[UsageInfo]


# ============================================================================
# Union type
# ============================================================================

IRStreamEvent = Union[
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    FinishEvent,
    UsageEvent,
]

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TextDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    "IRStreamEvent",
]
