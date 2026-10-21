"""
LLM-Rosetta - Open Responses / OpenAI Responses Converter Module

Implements the vendor-neutral Open Responses spec, with the OpenAI Responses
API as a conforming profile.  The ``OpenAIResponses*`` names are retained as
deprecated aliases for backward compatibility.
"""

from .config_ops import OpenAIResponsesConfigOps, OpenResponsesConfigOps
from .content_ops import OpenAIResponsesContentOps, OpenResponsesContentOps
from .converter import OpenAIResponsesConverter, OpenResponsesConverter
from .message_ops import OpenAIResponsesMessageOps, OpenResponsesMessageOps
from .stream_context import OpenAIResponsesStreamContext, OpenResponsesStreamContext
from .tool_ops import OpenAIResponsesToolOps, OpenResponsesToolOps

__all__ = [
    "OpenResponsesConverter",
    "OpenAIResponsesConverter",
    "OpenResponsesContentOps",
    "OpenResponsesToolOps",
    "OpenResponsesMessageOps",
    "OpenResponsesConfigOps",
    "OpenResponsesStreamContext",
    # Backward-compatible aliases (deprecated)
    "OpenAIResponsesContentOps",
    "OpenAIResponsesToolOps",
    "OpenAIResponsesMessageOps",
    "OpenAIResponsesConfigOps",
    "OpenAIResponsesStreamContext",
]
