"""
LLM-Rosetta - LLM Converters Package

The chat/LLM family: one package per wire format.

- ``chat_completions``: the vendor-neutral Chat Completions standard
- ``messages``: the vendor-neutral Messages standard
- ``open_responses``: the Open Responses spec
- ``google_generate``: Google GenAI ``generateContent``
- ``google_interactions``: Google Interactions

Non-LLM families (``embedding``, ``rerank``, ``decision``) sit beside this
package; each is keyed by ``(model_type, format)`` in
:mod:`llm_rosetta.converters.registry`.
"""

from .chat_completions import ChatCompletionsConverter, OpenAIChatConverter
from .google_generate import GoogleConverter, GoogleGenerateConverter
from .google_interactions import GoogleInteractionsConverter
from .messages import AnthropicConverter, MessagesConverter
from .open_responses import OpenAIResponsesConverter, OpenResponsesConverter

__all__ = [
    "ChatCompletionsConverter",
    "OpenAIChatConverter",  # deprecated alias
    "MessagesConverter",
    "AnthropicConverter",  # deprecated alias
    "OpenResponsesConverter",
    "OpenAIResponsesConverter",  # deprecated alias
    "GoogleGenerateConverter",
    "GoogleConverter",  # deprecated alias
    "GoogleInteractionsConverter",
]
