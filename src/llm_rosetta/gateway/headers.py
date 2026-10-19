"""Deprecated — moved to ``llm_rosetta.gateway.middleware.headers``.

This module re-exports the new location for backward compatibility with
0.13.0 imports and will be removed in 0.16.0 (#919).
"""

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.gateway.headers is deprecated, use llm_rosetta.gateway.middleware.headers instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.gateway.middleware.headers import *  # noqa: F401, F403, E402
from llm_rosetta.gateway.middleware.headers import get_preflight_tokens_override  # noqa: F401, E402
