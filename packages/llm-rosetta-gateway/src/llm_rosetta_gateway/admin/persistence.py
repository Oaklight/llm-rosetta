"""Deprecated — use :mod:`llm_rosetta.observability.persistence` instead.

This module has re-exported the observability persistence API since 0.10.0 so
admin-panel imports could stay short.  It warns on import and is removed in
0.16.0 (#919); new code imports from :mod:`llm_rosetta.observability`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "llm_rosetta_gateway.admin.persistence is deprecated, "
    "use llm_rosetta.observability.persistence instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.observability.persistence import (  # noqa: E402, F401
    DEFAULT_SUCCESS_MAX,
    PersistenceManager,
)

__all__ = ["PersistenceManager", "DEFAULT_SUCCESS_MAX"]
