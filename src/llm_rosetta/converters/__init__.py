"""
LLM-Rosetta - Converters Package

提供各种provider之间的转换器实现
Provides converter implementations between various providers
"""

from .messages import AnthropicConverter, MessagesConverter
from .base import BaseConverter, BaseEmbeddingConverter, BaseRerankConverter
from .embedding import (
    CohereEmbeddingConverter,
    GoogleEmbeddingConverter,
    JinaEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
)
from .google_generate import GoogleConverter, GoogleGenerateConverter
from .google_generate import (
    GoogleGenerateConverter as GoogleGenAIConverter,
)  # deprecated alias
from .google_interactions import GoogleInteractionsConverter
from .chat_completions import ChatCompletionsConverter, OpenAIChatConverter
from .open_responses import OpenAIResponsesConverter, OpenResponsesConverter
from .decision import (
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
