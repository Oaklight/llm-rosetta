"""Request log for the gateway admin panel.

This module re-exports from :mod:`llm_rosetta.observability.request_log`
for backward compatibility.  New code should import directly from
``llm_rosetta.observability``.
"""

from __future__ import annotations

from llm_rosetta.observability.request_log import (  # noqa: F401
    RequestLog,
    RequestLogEntry,
)

__all__ = ["RequestLog", "RequestLogEntry"]
