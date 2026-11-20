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

from .chat_completions import ChatCompletionsConverter
from .google_generate import GoogleGenerateConverter
from .google_interactions import GoogleInteractionsConverter
from .messages import MessagesConverter
from .open_responses import OpenAIResponsesConverter, OpenResponsesConverter

__all__ = [
    "ChatCompletionsConverter",
    "MessagesConverter",
    "OpenResponsesConverter",
    "OpenAIResponsesConverter",  # deprecated alias
    "GoogleGenerateConverter",
    "GoogleInteractionsConverter",
]
