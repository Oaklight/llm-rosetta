"""In-process metrics collector for the gateway admin panel.

This module re-exports from :mod:`llm_rosetta.observability.metrics`
for backward compatibility.  New code should import directly from
``llm_rosetta.observability``.
"""

from __future__ import annotations

from llm_rosetta.observability.metrics import (  # noqa: F401
    MetricsCollector,
    _Bucket,
    _ProviderStats,
    _RollingWindow,
)

__all__ = ["MetricsCollector"]
