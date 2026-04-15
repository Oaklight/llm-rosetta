"""
LLM-Rosetta

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
    GoogleGenAIConverter,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
)
from . import tool_ops
from .converters.base.context import ConversionContext, StreamContext

__version__ = "0.5.0"

__all__ = [
    # 转换器 Converters
    "BaseConverter",
    "OpenAIChatConverter",
    "AnthropicConverter",
    "GoogleGenAIConverter",
    "GoogleConverter",
    "OpenAIResponsesConverter",
    # 转换上下文 Conversion context
    "ConversionContext",
    "StreamContext",
    # 工具定义便利 API Tool definition convenience API
    "tool_ops",
    # 自动检测和转换 Auto-detection and conversion
    "detect_provider",
    "get_converter_for_provider",
    "convert",
    "ProviderType",
]
