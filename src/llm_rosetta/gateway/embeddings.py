"""Deprecated — moved to ``llm_rosetta.gateway.pipelines.embeddings``.

This module re-exports the new location for backward compatibility with
0.13.0 imports and will be removed in 0.16.0 (#919).
"""

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.gateway.embeddings is deprecated, use llm_rosetta.gateway.pipelines.embeddings instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.gateway.pipelines.embeddings import *  # noqa: F401, F403, E402
from llm_rosetta.gateway.pipelines.embeddings import handle_embeddings  # noqa: F401, E402
