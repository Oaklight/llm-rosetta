"""Scanning a local shim directory."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

from llm_rosetta.api import load_providers_from_dir

from .manifest import (
    MANIFEST_FILENAME,
    ShimManifestError,
    core_version_problem,
    read_shim_manifest,
)
from ._types import (
    ShimSourceStatus,
    logger,
)


# ---------------------------------------------------------------------------
# Local directories
# ---------------------------------------------------------------------------


def _load_local_dir(directory: str | os.PathLike[str]) -> ShimSourceStatus:
    """Scan one local directory, reporting failure instead of raising."""
    return _scan_dir(Path(directory), kind="path", location=str(directory))


def _scan_dir(
    directory: Path,
    *,
    kind: Literal["path", "remote"],
    location: str,
    name: str | None = None,
    from_cache: bool = False,
    stale: bool = False,
    fetched_at: float | None = None,
    resolved_ref: str | None = None,
    refuse: Collection[str] = (),
    only: Collection[str] | None = None,
) -> ShimSourceStatus:
    """Run the loader over *directory* and fold the result into a status.

    *refuse* names shims that must not be registered from this directory —
    the names core already provides, for a manifest install (core wins a
    clash).

    *only*, when given, is the complete set of names that may register — a
    manifest install's release record.  Any other name is refused with a
    WARNING.

    A source that ships ``shims.yaml`` is loaded exactly as it declares: entries
    the manifest does not list are never read.  A source without one is scanned
    as before.
    """
    base: dict[str, Any] = {
        "kind": kind,
        "location": location,
        "name": name,
        "resolved_dir": str(directory),
        "from_cache": from_cache,
        "stale": stale,
        "fetched_at": fetched_at,
        "resolved_ref": resolved_ref,
    }
    if not directory.is_dir():
        return ShimSourceStatus(
            **base,
            ok=False,
            error=f"not a directory: {directory}",
        )

    try:
        manifest = read_shim_manifest(directory)
    except ShimManifestError as exc:
        # A manifest that cannot be honoured is not a reason to scan the tree it
        # meant to bound: that fallback would parse exactly what it excluded.
        return ShimSourceStatus(
            **base,
            ok=False,
            error=f"{MANIFEST_FILENAME}: {exc}",
        )

    if manifest is not None and manifest.core:
        # A package that tracks its branch can bring a shape this core cannot
        # read; the floor it declares is checked before anything is loaded, so
        # the failure names the version rather than surfacing as a bad leaf.
        problem = core_version_problem(manifest.core)
        if problem:
            return ShimSourceStatus(
                **base,
                ok=False,
                error=f"{MANIFEST_FILENAME}: {problem}",
            )

    refused = frozenset(refuse)
    allowed = None if only is None else frozenset(only)

    def selected(shim_name: str) -> bool:
        if allowed is not None and shim_name not in allowed:
            logger.warning(
                "Shim source %s: %r is not in the release record; refused",
                location,
                shim_name,
            )
            return False
        if shim_name in refused:
            logger.warning(
                "Shim source %s: %r is a core shim; core wins and this "
                "source's copy is ignored",
                location,
                shim_name,
            )
            return False
        return True

    only_paths: set[str] | None = None
    if manifest is not None:
        only_paths = {entry.path for entry in manifest.entries if selected(entry.name)}

    try:
        shims = load_providers_from_dir(
            directory,
            on_error="warn",
            only_paths=only_paths,
            keep_names=selected,
        )
    except Exception as exc:
        return ShimSourceStatus(
            **base,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    names = tuple(s.name for s in shims)
    if manifest is not None:
        declared = set(manifest.names)
        for shim_name in names:
            if shim_name not in declared:
                logger.warning(
                    "Shim source %s: %s declares no shim named %r, but the "
                    "entry's path loads one; fix the manifest",
                    location,
                    MANIFEST_FILENAME,
                    shim_name,
                )
    return ShimSourceStatus(
        **base,
        ok=True,
        shim_count=len(names),
        shim_names=names,
        manifest_shims=manifest.names if manifest is not None else (),
    )
