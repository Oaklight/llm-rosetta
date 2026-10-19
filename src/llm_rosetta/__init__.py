"""
LLM-Rosetta

一个用于在不同LLM provider之间转换消息格式的库
A library for converting message formats between different LLM providers
"""

import warnings as _warnings

from .auto_detect import (
    ProviderType,
    convert,
    convert_response,
    detect_provider,
    get_converter_for_provider,
)
from .provider_names import normalize_provider_name
from .pipeline import ConversionError, ConversionPipeline
from .converters.decision.pipeline import DecisionConversionPipeline
from .converters.embedding.pipeline import EmbeddingConversionPipeline
from .converters.rerank.pipeline import RerankConversionPipeline
from .converters import (
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
    OpenAIEmbeddingConverter,
    OpenAIResponsesConverter,
    OpenResponsesConverter,
    VoyageEmbeddingConverter,
    VoyageRerankConverter,
)
from .converters.base.simple_converter import BaseSimpleConverter
from . import tool_ops
from .converters.base.context import ConversionContext, StreamContext
from .transforms import Transform, apply_transforms
from .shims import (
    ConnectionConfig,
    ProviderShim,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
    ToolsConfig,
    get_shim,
    list_shims,
    rebuild_shims,
    register_shim,
    rename_field,
    resolve_base,
    resolve_format,
    resolve_model_type,
    resolve_remote,
    set_defaults,
    strip_fields,
    unregister_shim,
)

__version__ = "0.14.0.dev17"


def __getattr__(name: str):
    # 0.13.0 class names (``OpenAIChatConverter`` & co.) stay importable from
    # the package root until 0.16.0 (#919).  Resolve and warn here rather than
    # delegating to ``converters``: the extra frame would attribute the
    # warning to this module instead of the caller's line.
    from . import converters as _converters

    canonical = _converters._DEPRECATED_CLASS_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_converters, canonical)
    _warnings.warn(
        f"llm_rosetta.{name} is deprecated, use llm_rosetta.{canonical} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # Bind the alias after warning so the import machinery's ``hasattr`` probe
    # and the subsequent fetch do not record two warnings for one import.
    globals()[name] = value
    return value


__all__ = [
    # Converters
    "BaseConverter",
    "BaseSimpleConverter",
    "ChatCompletionsConverter",
    "OpenAIChatConverter",  # deprecated alias
    "MessagesConverter",
    "AnthropicConverter",  # deprecated alias of MessagesConverter
    "GoogleGenerateConverter",
    "GoogleGenAIConverter",  # deprecated alias
    "GoogleConverter",
    "GoogleInteractionsConverter",
    "OpenAIResponsesConverter",
    "OpenResponsesConverter",
    # Embedding converters
    "OpenAIEmbeddingConverter",
    "JinaEmbeddingConverter",
    "VoyageEmbeddingConverter",
    "CohereEmbeddingConverter",
    "GoogleEmbeddingConverter",
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
    "normalize_provider_name",
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
    "resolve_base",
    "resolve_format",
    "resolve_model_type",
    # External shim sources
    "rebuild_shims",
    "resolve_remote",
    "ShimLoadReport",
    "ShimSourceStatus",
    "ShimSourceError",
    # Transforms
    "Transform",
    "apply_transforms",
    "strip_fields",
    "rename_field",
    "set_defaults",
]
