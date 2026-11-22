"""Gateway-side shim sources — manifest installs, directories and the catalogue.

The built-in identity cards live in :mod:`llm_rosetta.shims`.  This package
owns what only a gateway needs: installing shim vendors from a source's
manifest, loading a local shim directory, and the catalogue of manifest
sources offered in Settings.

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
from .manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    ShimManifest,
    ShimManifestEntry,
    ShimManifestError,
    read_shim_manifest,
)
from .sources import rebuild_shims

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "ShimArchiveMismatch",
    "ShimArchiveUnusable",
    "ShimLoadReport",
    "ShimManifest",
    "ShimManifestEntry",
    "ShimManifestError",
    "ShimSourceError",
    "ShimSourceStatus",
    "legacy_shim_migration_advice",
    "read_shim_manifest",
    "rebuild_shims",
    "source_package_advice",
]
