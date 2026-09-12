"""Ops log for the gateway admin panel.

This module re-exports from :mod:`llm_rosetta.observability.ops_log`
for backward compatibility.  New code should import directly from
``llm_rosetta.observability``.
"""

from __future__ import annotations

from llm_rosetta.observability.ops_log import OpsLog, OpsLogEntry  # noqa: F401

__all__ = ["OpsLog", "OpsLogEntry"]
