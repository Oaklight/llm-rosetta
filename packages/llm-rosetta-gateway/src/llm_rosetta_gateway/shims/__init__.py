"""Gateway-side shim sources — external packages, catalogs and removals.

The built-in identity cards live in :mod:`llm_rosetta.shims`.  This package
owns what only a gateway needs: loading shims from a directory or a remote
archive, the catalog of standalone packages offered in Settings, and the
manifest of built-ins with a scheduled removal.

Nothing here is imported by the translation core, so ``import
llm_rosetta.shims`` no longer pulls in the HTTP client these modules use.
"""

from .catalog import (
    SHIM_SOURCE_PACKAGES,
    ShimSourcePackage,
    canonical_shim_for,
    find_legacy_source_package,
    find_source_package,
    get_source_package,
    legacy_shim_migration_advice,
    source_package_advice,
)
from .removals import (
    SHIM_REMOVALS,
    RemovalReference,
    ShimRemoval,
    ShimRemovalAdvisory,
    detect_shim_removals,
    parse_version,
    removals_due,
    scan_removal_references,
)
from .sources import (
    ShimArchiveMismatch,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
    rebuild_shims,
    resolve_remote,
)

__all__ = [
    "SHIM_REMOVALS",
    "SHIM_SOURCE_PACKAGES",
    "canonical_shim_for",
    "find_legacy_source_package",
    "legacy_shim_migration_advice",
    "source_package_advice",
    "RemovalReference",
    "ShimArchiveMismatch",
    "ShimLoadReport",
    "ShimRemoval",
    "ShimRemovalAdvisory",
    "ShimSourceError",
    "ShimSourcePackage",
    "ShimSourceStatus",
    "detect_shim_removals",
    "find_source_package",
    "get_source_package",
    "parse_version",
    "rebuild_shims",
    "removals_due",
    "resolve_remote",
    "scan_removal_references",
]
