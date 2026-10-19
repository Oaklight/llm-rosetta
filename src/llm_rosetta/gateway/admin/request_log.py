"""Deprecated — use :mod:`llm_rosetta.observability.request_log` instead.

This module has re-exported the observability request-log API since 0.10.0 so
admin-panel imports could stay short.  It warns on import and is removed in
0.16.0 (#919); new code imports from :mod:`llm_rosetta.observability`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "llm_rosetta.gateway.admin.request_log is deprecated, "
    "use llm_rosetta.observability.request_log instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.observability.request_log import (  # noqa: E402, F401
    RequestLog,
    RequestLogEntry,
)

__all__ = ["RequestLog", "RequestLogEntry"]
