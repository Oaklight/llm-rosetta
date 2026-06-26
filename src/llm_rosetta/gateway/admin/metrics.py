"""In-process metrics collector for the gateway admin panel.

.. deprecated::
    This module re-exports from :mod:`llm_rosetta.observability.metrics`
    for backward compatibility.  Import directly from
    ``llm_rosetta.observability`` instead.
"""

from __future__ import annotations

from llm_rosetta.observability.metrics import (  # noqa: F401
    MetricsCollector,
    _Bucket,
    _ProviderStats,
    _RollingWindow,
)

__all__ = ["MetricsCollector", "_RollingWindow", "_ProviderStats", "_Bucket"]
