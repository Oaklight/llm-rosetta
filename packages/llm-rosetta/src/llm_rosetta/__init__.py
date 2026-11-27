"""
LLM-Rosetta

一个用于在不同LLM provider之间转换消息格式的库
A library for converting message formats between different LLM providers
"""

from .contracts.routing import ProviderType
from .converters.registry import detect_provider, get_converter_for_provider
from .pipeline import ConversionError, ConversionPipeline, convert, convert_response
from .pipeline.decisions import DecisionConversionPipeline
from .pipeline.embeddings import EmbeddingConversionPipeline
from .pipeline.rerank import RerankConversionPipeline
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
from .ir.context import ConversionContext, StreamContext
from .ir.primitives.body import Transform, apply_transforms
from .contracts.resolver import resolve_format
from .contracts.shim import ConnectionConfig, ProviderShim, ToolsConfig

# The shim layer (the registry of provider shims and the extensions that add
# more) is optional: without it this is a plain conversion library, and a
# ``ProviderShim`` object still works wherever a shim is taken.  This is the one
# place that lets the package import without it.
try:
    from .shims import (
        force_fields,
        get_shim,
        list_shims,
        register_shim,
        rename_field,
        resolve_model_type,
        set_defaults,
        strip_fields,
        unregister_shim,
    )

    _SHIM_LAYER = True
except ImportError:
    _SHIM_LAYER = False

__version__ = "0.14.0.dev35"


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

if not _SHIM_LAYER:
    _SHIM_NAMES = {
        "force_fields",
        "get_shim",
        "list_shims",
        "register_shim",
        "rename_field",
        "resolve_model_type",
        "set_defaults",
        "strip_fields",
        "unregister_shim",
    }
    __all__ = [name for name in __all__ if name not in _SHIM_NAMES]
