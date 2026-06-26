"""Request log for the gateway admin panel.

.. deprecated::
    This module re-exports from :mod:`llm_rosetta.observability.request_log`
    for backward compatibility.  Import directly from
    ``llm_rosetta.observability`` instead.
"""

from __future__ import annotations

from llm_rosetta.observability.request_log import (  # noqa: F401
    RequestLog,
    RequestLogEntry,
)

__all__ = ["RequestLog", "RequestLogEntry"]
