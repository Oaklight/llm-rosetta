"""
LLM-Rosetta

一个用于在不同LLM provider之间转换消息格式的库
A library for converting message formats between different LLM providers
"""

from .auto_detect import (
    ProviderType,
    convert,
    convert_response,
    detect_provider,
    get_converter_for_provider,
)
from .pipeline import ConversionError, ConversionPipeline
from .converters import (
    AnthropicConverter,
    BaseConverter,
    GoogleConverter,
    GoogleGenerateConverter,
    GoogleGenAIConverter,  # deprecated alias
    GoogleInteractionsConverter,
    OpenAIChatConverter,
    OpenAIResponsesConverter,
)
from . import tool_ops
from .converters.base.context import ConversionContext, StreamContext
from .shims import (
    ProviderShim,
    Transform,
    apply_transforms,
    get_shim,
    list_shims,
    register_shim,
    rename_field,
    resolve_base,
    set_defaults,
    strip_fields,
    unregister_shim,
)

__version__ = "0.12.0"

__all__ = [
    # Converters
    "BaseConverter",
    "OpenAIChatConverter",
    "AnthropicConverter",
    "GoogleGenerateConverter",
    "GoogleGenAIConverter",  # deprecated alias
    "GoogleConverter",
    "GoogleInteractionsConverter",
    "OpenAIResponsesConverter",
    # Conversion context
    "ConversionContext",
    "StreamContext",
    # Tool definition convenience API
    "tool_ops",
    # Auto-detection and conversion
    "detect_provider",
    "get_converter_for_provider",
    "convert",
    "convert_response",
    "ProviderType",
    # Conversion pipeline
    "ConversionPipeline",
    "ConversionError",
    # Provider shim layer
    "ProviderShim",
    "register_shim",
    "unregister_shim",
    "get_shim",
    "list_shims",
    "resolve_base",
    # Transforms
    "Transform",
    "apply_transforms",
    "strip_fields",
    "rename_field",
    "set_defaults",
]
