"""
LLM-Rosetta - Converters Package

提供各种provider之间的转换器实现
Provides converter implementations between various providers
"""

import warnings as _warnings

from .base import BaseConverter, BaseEmbeddingConverter, BaseRerankConverter
from .embeddings import (
    CohereEmbeddingConverter,
    GoogleEmbeddingConverter,
    JinaEmbeddingConverter,
    MiniMaxEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
)
from .llm.chat_completions import ChatCompletionsConverter
from .llm.google_generate import GoogleGenerateConverter
from .llm.google_interactions import GoogleInteractionsConverter
from .llm.messages import MessagesConverter
from .llm.open_responses import OpenAIResponsesConverter, OpenResponsesConverter
from .decisions import (
    EmbeddingDecisionConverter,
    LLMChatDecisionConverter,
    RerankerDecisionConverter,
    SystemOneDecisionConverter,
)
from .rerank import CohereRerankConverter, JinaRerankConverter, VoyageRerankConverter
from .registry import (
    CONVERTER_REGISTRY,
    FORMATS_BY_TYPE,
    MODEL_TYPES,
    get_converter_class,
)

__all__ = [
    "BaseConverter",
    "BaseRerankConverter",
    "BaseEmbeddingConverter",
    "CONVERTER_REGISTRY",
    "FORMATS_BY_TYPE",
    "MODEL_TYPES",
    "get_converter_class",
    "OpenAIEmbeddingConverter",
    "JinaEmbeddingConverter",
    "VoyageEmbeddingConverter",
    "CohereEmbeddingConverter",
    "GoogleEmbeddingConverter",
    "MiniMaxEmbeddingConverter",
    "ChatCompletionsConverter",
    "OpenAIChatConverter",  # deprecated alias
    "MessagesConverter",
    "AnthropicConverter",  # deprecated alias of MessagesConverter
    "GoogleGenerateConverter",
    "GoogleGenAIConverter",  # deprecated alias,
    "GoogleConverter",
    "GoogleInteractionsConverter",
    "OpenAIResponsesConverter",
    "OpenResponsesConverter",
    "JinaRerankConverter",
    "CohereRerankConverter",
    "VoyageRerankConverter",
    "SystemOneDecisionConverter",
    "LLMChatDecisionConverter",
    "RerankerDecisionConverter",
    "EmbeddingDecisionConverter",
]

# ---------------------------------------------------------------------------
# Deprecated class aliases (0.13.0 names)
# ---------------------------------------------------------------------------
# These were the canonical class names in 0.13.0.  They stay importable from
# this package until 0.16.0 (#919) but warn on access, so the module
# ``__getattr__`` serves them instead of a plain import binding.
_DEPRECATED_CLASS_ALIASES: dict[str, str] = {
    "OpenAIChatConverter": "ChatCompletionsConverter",
    "AnthropicConverter": "MessagesConverter",
    "GoogleGenAIConverter": "GoogleGenerateConverter",
    "GoogleConverter": "GoogleGenerateConverter",
}


def __getattr__(name: str):
    canonical = _DEPRECATED_CLASS_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = globals()[canonical]
    _warnings.warn(
        f"llm_rosetta.converters.{name} is deprecated, "
        f"use llm_rosetta.converters.{canonical} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # Bind the alias after warning: ``from ... import X`` probes the name with
    # ``hasattr`` before fetching it, and without the binding that probe and
    # the real fetch would each emit the same warning.
    globals()[name] = value
    return value
