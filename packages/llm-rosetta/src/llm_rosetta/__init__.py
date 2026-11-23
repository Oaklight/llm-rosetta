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
from .converters.decisions.pipeline import DecisionConversionPipeline
from .converters.embeddings.pipeline import EmbeddingConversionPipeline
from .converters.rerank.pipeline import RerankConversionPipeline
from .converters import (
    AcpConverter,
    BaseConverter,
    ChatCompletionsConverter,
    CohereEmbeddingConverter,
    CohereRerankConverter,
    GoogleEmbeddingConverter,
    GoogleGenerateConverter,
    GoogleInteractionsConverter,
    JinaEmbeddingConverter,
    JinaRerankConverter,
    MessagesConverter,
    MiniMaxEmbeddingConverter,
    OpenAIEmbeddingConverter,
    OpenAIResponsesConverter,
    OpenResponsesConverter,
    VoyageEmbeddingConverter,
    VoyageRerankConverter,
)
from .converters.base.simple_converter import BaseSimpleConverter
from . import tool_ops
from .converters.base.context import ConversionContext, StreamContext
from .ir.primitives.body import Transform, apply_transforms
from .shims import (
    ConnectionConfig,
    ProviderShim,
    ToolsConfig,
    get_shim,
    list_shims,
    register_shim,
    rename_field,
    resolve_format,
    resolve_model_type,
    force_fields,
    set_defaults,
    strip_fields,
    unregister_shim,
)

__version__ = "0.14.0.dev38"


__all__ = [
    # Converters
    "BaseConverter",
    "BaseSimpleConverter",
    "ChatCompletionsConverter",
    "MessagesConverter",
    "GoogleGenerateConverter",
    "GoogleInteractionsConverter",
    "AcpConverter",
    "OpenAIResponsesConverter",
    "OpenResponsesConverter",
    # Embedding converters
    "OpenAIEmbeddingConverter",
    "JinaEmbeddingConverter",
    "VoyageEmbeddingConverter",
    "CohereEmbeddingConverter",
    "GoogleEmbeddingConverter",
    "MiniMaxEmbeddingConverter",
    # Rerank converters
    "JinaRerankConverter",
    "CohereRerankConverter",
    "VoyageRerankConverter",
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
    "DecisionConversionPipeline",
    "EmbeddingConversionPipeline",
    "RerankConversionPipeline",
    # Provider shim layer
    "ConnectionConfig",
    "ToolsConfig",
    "ProviderShim",
    "register_shim",
    "unregister_shim",
    "get_shim",
    "list_shims",
    "resolve_format",
    "resolve_model_type",
    # Transforms
    "Transform",
    "apply_transforms",
    "force_fields",
    "strip_fields",
    "rename_field",
    "set_defaults",
]
