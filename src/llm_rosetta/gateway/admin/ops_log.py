"""Ops log for the gateway admin panel.

Convenience re-exports from :mod:`llm_rosetta.observability.ops_log`
so admin-package consumers can use short imports.  The canonical
module is ``llm_rosetta.observability``.
"""

from __future__ import annotations

from llm_rosetta.observability.ops_log import OpsLog, OpsLogEntry  # noqa: F401

__all__ = ["OpsLog", "OpsLogEntry"]
