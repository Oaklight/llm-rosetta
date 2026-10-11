"""
Deprecated — use ``llm_rosetta.converters.messages`` instead.

This module exists only for backward compatibility and will be removed in a
future release.  It re-exports the canonical ``Messages*`` names (plus their
deprecated ``Anthropic*`` aliases) and aliases the legacy submodule paths so
that existing imports such as
``from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps``
keep resolving.

Note: the ``anthropic`` *vendor* shim lives in
``llm_rosetta.shims.providers.anthropic`` and is unaffected — only this
converter module (the former name of the neutral ``messages`` base) is
deprecated.
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "llm_rosetta.converters.anthropic is deprecated, "
    "use llm_rosetta.converters.messages instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..messages import (  # noqa: F401, E402
    _constants as _constants,
    config_ops as config_ops,
    content_ops as content_ops,
    converter as converter,
    message_ops as message_ops,
    tool_ops as tool_ops,
)
from ..messages import (  # noqa: F401, E402
    AnthropicConfigOps,
    AnthropicContentOps,
    AnthropicConverter,
    AnthropicMessageOps,
    AnthropicToolOps,
    MessagesConfigOps,
    MessagesContentOps,
    MessagesConverter,
    MessagesMessageOps,
    MessagesToolOps,
)

# Register the legacy submodule paths (e.g. ``anthropic.tool_ops``) so that
# ``from llm_rosetta.converters.anthropic.tool_ops import ...`` keeps working.
# The submodule objects are shared with ``messages`` (same identity).
_sys.modules[__name__ + "._constants"] = _constants
_sys.modules[__name__ + ".config_ops"] = config_ops
_sys.modules[__name__ + ".content_ops"] = content_ops
_sys.modules[__name__ + ".converter"] = converter
_sys.modules[__name__ + ".message_ops"] = message_ops
_sys.modules[__name__ + ".tool_ops"] = tool_ops

__all__ = [
    "MessagesConverter",
    "AnthropicConverter",
    "MessagesContentOps",
    "AnthropicContentOps",
    "MessagesToolOps",
    "AnthropicToolOps",
    "MessagesMessageOps",
    "AnthropicMessageOps",
    "MessagesConfigOps",
    "AnthropicConfigOps",
]
