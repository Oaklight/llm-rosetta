"""Backward-compatibility shim for ``llm_rosetta.converters.base.schema``.

This module moved to ``llm_rosetta.converters.base.helpers.schema`` in
v0.6.11.  It is re-exported here so the old import path keeps working.

New code should import from ``llm_rosetta.converters.base.helpers`` instead.
"""

from .helpers.schema import sanitize_schema

__all__ = ["sanitize_schema"]
