"""
LLM-Rosetta - Converters Package

提供各种provider之间的转换器实现
Provides converter implementations between various providers
"""

import sys as _sys
from types import ModuleType as _ModuleType

from .base import BaseConverter, BaseEmbeddingConverter, BaseRerankConverter
from .embedding import (
    CohereEmbeddingConverter,
    GoogleEmbeddingConverter,
    JinaEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
)
from .llm import google_generate as _google_generate
from .llm import google_interactions as _google_interactions
from .llm.chat_completions import ChatCompletionsConverter, OpenAIChatConverter
from .llm.google_generate import GoogleConverter, GoogleGenerateConverter
from .llm.google_generate import (
    GoogleGenerateConverter as GoogleGenAIConverter,
)  # deprecated alias
from .llm.google_interactions import GoogleInteractionsConverter
from .llm.messages import AnthropicConverter, MessagesConverter
from .llm.open_responses import OpenAIResponsesConverter, OpenResponsesConverter
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

# ---------------------------------------------------------------------------
# Legacy module paths (0.13.0)
# ---------------------------------------------------------------------------
# ``converters.google_generate`` and ``converters.google_interactions`` were
# importable packages in 0.13.0.  Their implementations now live under
# ``converters.llm``; register the old paths against the same module objects so
# existing imports keep resolving without a second copy of any module.  The
# other three LLM formats were introduced after 0.13.0 and have no legacy path.
for _legacy_name, _package in (
    ("google_generate", _google_generate),
    ("google_interactions", _google_interactions),
):
    _sys.modules[f"{__name__}.{_legacy_name}"] = _package
    for _attr, _value in vars(_package).items():
        if isinstance(_value, _ModuleType) and _value.__name__.startswith(
            _package.__name__ + "."
        ):
            _sys.modules[f"{__name__}.{_legacy_name}.{_attr}"] = _value
