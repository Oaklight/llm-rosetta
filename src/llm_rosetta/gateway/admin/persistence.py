"""SQLite-based persistence for gateway admin data.

.. deprecated::
    This module re-exports from :mod:`llm_rosetta.observability.persistence`
    for backward compatibility.  Import directly from
    ``llm_rosetta.observability`` instead.
"""

from __future__ import annotations

from llm_rosetta.observability.persistence import (  # noqa: F401
    DEFAULT_ERROR_MAX,
    DEFAULT_SUCCESS_MAX,
    PersistenceManager,
)

__all__ = ["PersistenceManager", "DEFAULT_SUCCESS_MAX", "DEFAULT_ERROR_MAX"]
