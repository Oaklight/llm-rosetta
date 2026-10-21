"""
LLM-Rosetta - Messages API Converter Module

Implements the vendor-neutral Messages API standard, with the Anthropic
Messages API as its original profile.  The ``Anthropic*`` names are retained
as deprecated aliases for backward compatibility.
"""

from .config_ops import AnthropicConfigOps, MessagesConfigOps
from .content_ops import AnthropicContentOps, MessagesContentOps
from .converter import AnthropicConverter, MessagesConverter
from .message_ops import AnthropicMessageOps, MessagesMessageOps
from .tool_ops import AnthropicToolOps, MessagesToolOps

__all__ = [
    "MessagesConverter",
    "MessagesContentOps",
    "MessagesToolOps",
    "MessagesMessageOps",
    "MessagesConfigOps",
    # Backward-compatible aliases (deprecated)
    "AnthropicConverter",
    "AnthropicContentOps",
    "AnthropicToolOps",
    "AnthropicMessageOps",
    "AnthropicConfigOps",
]
