"""Deprecated — moved to ``llm_rosetta.gateway.middleware.auth``.

This module re-exports the new location for backward compatibility with
0.13.0 imports and will be removed in 0.16.0 (#919).
"""

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.gateway.auth is deprecated, use llm_rosetta.gateway.middleware.auth instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.gateway.middleware.auth import *  # noqa: F401, F403, E402
from llm_rosetta.gateway.middleware.auth import api_key_context_var  # noqa: F401, E402
