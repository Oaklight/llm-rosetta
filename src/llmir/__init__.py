"""
LLMIR

一个用于在不同LLM provider之间转换消息格式的库
A library for converting message formats between different LLM providers
"""

from .auto_detect import (
    ProviderType,
    convert,
    detect_provider,
    get_converter_for_provider,
)
from .converters import (
    AnthropicConverter,
    BaseConverter,
    GoogleConverter,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
)
from .types.ir import (
    ContentPart,
    ExtensionItem,
    IRInput,
    IRInputSimple,
    Message,
    ToolChoice,
    ToolDefinition,
    is_extension_item,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)

__version__ = "0.1.0"

__all__ = [
    # 核心类型 Core types
    "IRInput",
    "IRInputSimple",
    "Message",
    "ContentPart",
    "ExtensionItem",
    "ToolDefinition",
    "ToolChoice",
    "ProviderType",
    # 类型守卫函数 Type guard functions
    "is_message",
    "is_extension_item",
    "is_text_part",
    "is_tool_call_part",
    "is_tool_result_part",
    # 转换器 Converters
    "BaseConverter",
    "AnthropicConverter",
    "GoogleConverter",
    "OpenAIChatConverter",
    "OpenAIResponsesConverter",
    # 自动检测和转换 Auto-detection and conversion
    "detect_provider",
    "get_converter_for_provider",
    "convert",
]
