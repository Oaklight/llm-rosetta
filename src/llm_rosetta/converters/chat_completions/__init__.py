"""
LLM-Rosetta - Chat Completion Converter Module

Implements the vendor-neutral Chat Completion API standard, with the OpenAI
Chat Completions API as its original profile.  The ``OpenAIChat*`` names are
retained as deprecated aliases for backward compatibility.
"""

from .config_ops import ChatCompletionsConfigOps, OpenAIChatConfigOps
from .content_ops import ChatCompletionsContentOps, OpenAIChatContentOps
from .converter import ChatCompletionsConverter, OpenAIChatConverter
from .message_ops import ChatCompletionsMessageOps, OpenAIChatMessageOps
from .tool_ops import ChatCompletionsToolOps, OpenAIChatToolOps

__all__ = [
    "ChatCompletionsConverter",
    "ChatCompletionsContentOps",
    "ChatCompletionsToolOps",
    "ChatCompletionsMessageOps",
    "ChatCompletionsConfigOps",
    # Backward-compatible aliases (deprecated)
    "OpenAIChatConverter",
    "OpenAIChatContentOps",
    "OpenAIChatToolOps",
    "OpenAIChatMessageOps",
    "OpenAIChatConfigOps",
]
