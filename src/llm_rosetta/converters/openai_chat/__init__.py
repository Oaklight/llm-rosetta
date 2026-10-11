"""
Deprecated — use ``llm_rosetta.converters.chat_completions`` instead.

This module exists only for backward compatibility and will be removed in a
future release.  It re-exports the canonical ``ChatCompletion*`` names (plus
their deprecated ``OpenAIChat*`` aliases) and aliases the legacy submodule
paths so that existing imports such as
``from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps``
keep resolving.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "llm_rosetta.converters.openai_chat is deprecated, "
    "use llm_rosetta.converters.chat_completions instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..chat_completions import (  # noqa: F401, E402
    _constants as _constants,
    config_ops as config_ops,
    content_ops as content_ops,
    converter as converter,
    message_ops as message_ops,
    tool_ops as tool_ops,
)
from ..chat_completions import (  # noqa: F401, E402
    ChatCompletionsConfigOps,
    ChatCompletionsContentOps,
    ChatCompletionsConverter,
    ChatCompletionsMessageOps,
    ChatCompletionsToolOps,
    OpenAIChatConfigOps,
    OpenAIChatContentOps,
    OpenAIChatConverter,
    OpenAIChatMessageOps,
    OpenAIChatToolOps,
)

# Register the legacy submodule paths (e.g. ``openai_chat.tool_ops``) so that
# ``from llm_rosetta.converters.openai_chat.tool_ops import ...`` keeps working.
# The submodule objects are shared with ``chat_completions`` (same identity).
_sys.modules[__name__ + "._constants"] = _constants
_sys.modules[__name__ + ".config_ops"] = config_ops
_sys.modules[__name__ + ".content_ops"] = content_ops
_sys.modules[__name__ + ".converter"] = converter
_sys.modules[__name__ + ".message_ops"] = message_ops
_sys.modules[__name__ + ".tool_ops"] = tool_ops

__all__ = [
    "ChatCompletionsConverter",
    "OpenAIChatConverter",
    "ChatCompletionsContentOps",
    "OpenAIChatContentOps",
    "ChatCompletionsToolOps",
    "OpenAIChatToolOps",
    "ChatCompletionsMessageOps",
    "OpenAIChatMessageOps",
    "ChatCompletionsConfigOps",
    "OpenAIChatConfigOps",
]
