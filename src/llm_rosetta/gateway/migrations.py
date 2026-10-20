"""Deprecated — use :mod:`llm_rosetta.gateway.maintenance.migrations`.

``llm_rosetta.gateway.migrations`` was importable in v0.13.0.  The
implementation moved with the maintenance package; this module keeps the old
dotted path resolving, with a warning, until 0.16.0 (#919) — the same window
the admin compatibility shims use for paths that shipped in 0.13.0.

The other six modules that moved into the package did not exist in v0.13.0,
so they have no compatibility path.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "llm_rosetta.gateway.migrations is deprecated, "
    "use llm_rosetta.gateway.maintenance.migrations instead",
    DeprecationWarning,
    stacklevel=2,
)

from llm_rosetta.gateway.maintenance.migrations import (  # noqa: E402, F401
    CURRENT_VERSION,
    migrate,
)

__all__ = ["CURRENT_VERSION", "migrate"]
