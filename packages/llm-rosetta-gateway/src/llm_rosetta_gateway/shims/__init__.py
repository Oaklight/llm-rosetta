"""Gateway-side shim sources — extension-index installs and the catalogue.

The built-in identity cards live in :mod:`llm_rosetta.shims`.  This package
owns what only a gateway needs: installing shim vendors from a source's
extension index, and the catalogue of sources offered in Settings.

Nothing here is imported by the translation core, so ``import
llm_rosetta.shims`` no longer pulls in the HTTP client these modules use.
"""

from ._types import (
    ShimArchiveMismatch,
    ShimArchiveUnusable,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
)
from .catalog import legacy_shim_migration_advice, source_package_advice
from .sources import rebuild_shims

__all__ = [
    "ShimArchiveMismatch",
    "ShimArchiveUnusable",
    "ShimLoadReport",
    "ShimSourceError",
    "ShimSourceStatus",
    "legacy_shim_migration_advice",
    "rebuild_shims",
    "source_package_advice",
]
