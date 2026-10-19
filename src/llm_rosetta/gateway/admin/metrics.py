"""Deprecated — use :mod:`llm_rosetta.observability.metrics` instead.

This module has re-exported the observability metrics API since 0.10.0 so
admin-panel imports could stay short.  It warns on import and is removed in
0.16.0 (#919); new code imports from :mod:`llm_rosetta.observability`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "llm_rosetta.gateway.admin.metrics is deprecated, "
    "use llm_rosetta.observability.metrics instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.observability.metrics import (  # noqa: E402, F401
    MetricsCollector,
    _Bucket,
    _ProviderStats,
    _RollingWindow,
)

__all__ = ["MetricsCollector"]
