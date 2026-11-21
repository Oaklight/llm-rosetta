"""Deprecated: the 0.13.0 routing module.  Import :mod:`llm_rosetta.contracts.routing`.

``llm_rosetta.routing`` shipped in 0.13.0 as a flat module.  The routing
contract now lives in :mod:`llm_rosetta.contracts.routing`; this module
forwards the three names 0.13.0 exported, as the same objects, for one minor
release.  Importing it warns once.  It will be removed in 0.15.0.
"""

from __future__ import annotations

import warnings

from llm_rosetta.contracts.routing import ProviderType, ResolvedRoute, Router

# Deprecated 0.13.0 path, removed in 0.15.0; the replacement is
# llm_rosetta.contracts.routing.
warnings.warn(
    "llm_rosetta.routing is deprecated and will be removed in 0.15.0; "
    "import from llm_rosetta.contracts.routing",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ProviderType", "ResolvedRoute", "Router"]
