"""
Deprecated — use ``llm_rosetta.converters.open_responses`` instead.

This module exists only for backward compatibility and will be removed in a
future release.  It re-exports the canonical ``OpenResponses*`` names (plus
their deprecated ``OpenAIResponses*`` aliases) and aliases the legacy submodule
paths so that existing imports such as
``from llm_rosetta.converters.openai_responses.tool_ops import OpenAIResponsesToolOps``
keep resolving.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "llm_rosetta.converters.openai_responses is deprecated, "
    "use llm_rosetta.converters.open_responses instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..open_responses import (  # noqa: F401, E402
    _constants as _constants,
    config_ops as config_ops,
    content_ops as content_ops,
    converter as converter,
    message_ops as message_ops,
    stream_context as stream_context,
    tool_ops as tool_ops,
)
from ..open_responses import (  # noqa: F401, E402
    OpenAIResponsesConfigOps,
    OpenAIResponsesContentOps,
    OpenAIResponsesConverter,
    OpenAIResponsesMessageOps,
    OpenAIResponsesStreamContext,
    OpenAIResponsesToolOps,
    OpenResponsesConfigOps,
    OpenResponsesContentOps,
    OpenResponsesConverter,
    OpenResponsesMessageOps,
    OpenResponsesStreamContext,
    OpenResponsesToolOps,
)

# Register the legacy submodule paths (e.g. ``openai_responses.tool_ops``) so
# that ``from llm_rosetta.converters.openai_responses.tool_ops import ...``
# keeps working.  The submodule objects are shared with ``open_responses``
# (same identity).
_sys.modules[__name__ + "._constants"] = _constants
_sys.modules[__name__ + ".config_ops"] = config_ops
_sys.modules[__name__ + ".content_ops"] = content_ops
_sys.modules[__name__ + ".converter"] = converter
_sys.modules[__name__ + ".message_ops"] = message_ops
_sys.modules[__name__ + ".stream_context"] = stream_context
_sys.modules[__name__ + ".tool_ops"] = tool_ops

__all__ = [
    "OpenResponsesConverter",
    "OpenAIResponsesConverter",
    "OpenResponsesContentOps",
    "OpenAIResponsesContentOps",
    "OpenResponsesToolOps",
    "OpenAIResponsesToolOps",
    "OpenResponsesMessageOps",
    "OpenAIResponsesMessageOps",
    "OpenResponsesConfigOps",
    "OpenAIResponsesConfigOps",
    "OpenResponsesStreamContext",
    "OpenAIResponsesStreamContext",
]
