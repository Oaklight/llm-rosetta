"""Scanning a local shim directory, with the include/exclude selection."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Literal

from llm_rosetta.shims.providers import load_providers_from_dir

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
    ref: str | None = None,
    token_env: str | None = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> ShimSourceStatus:
    """Run the loader over *directory* and fold the result into a status.

    A source that ships ``shims.yaml`` is loaded exactly as it declares: entries
    the manifest does not list are never read.  A source without one is scanned
    as before, and the selection still applies — by name, against each parsed
    leaf.

    Where the manifest names the paths, the selection is resolved *before* the
    scan: entries it drops are removed from the allowed paths, so their
    ``provider.yaml`` is not opened.  The name predicate is the second gate,
    and the only one a manifest-less source has.
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
        "ref": ref,
        "token_env": token_env,
        "include": include,
        "exclude": exclude,
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

    keep = _selection_predicate(include, exclude)
    # What the source ships beyond what it loads is known only here: ``keep`` is
    # asked about every leaf the scan parses before the leaf is dropped, so a
    # refused name is a shipped shim the selection declined.  Without that
    # record an ``exclude`` that matches nothing cannot be told from one that
    # matches a shim the selection already dropped — both leave no trace.
    declined: list[str] = []

    def selected(shim_name: str) -> bool:
        keep_it = keep(shim_name)
        if not keep_it:
            declined.append(shim_name)
        return keep_it

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
    declined_names = tuple(n for n in dict.fromkeys(declined) if n not in names)
    unmatched, unmatched_excludes, emptied_includes = _selection_report(
        location, names, declined_names, include=include, exclude=exclude
    )
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
        unmatched=unmatched,
        unmatched_excludes=unmatched_excludes,
        emptied_includes=emptied_includes,
        manifest_shims=manifest.names if manifest is not None else (),
        declined=declined_names,
    )


def _selection_predicate(
    include: Collection[str], exclude: Collection[str]
) -> Callable[[str], bool]:
    """Build the name predicate an ``include``/``exclude`` pair declares.

    An empty ``include`` selects everything, and ``exclude`` wins over
    ``include``: an operator who names both meant to drop, not to keep.  Both
    sides are ``fnmatch`` patterns, so an exact name and ``vendor_*`` are
    spelled the same way.
    """

    def keep(shim_name: str) -> bool:
        if include and not any(fnmatch(shim_name, p) for p in include):
            return False
        return not any(fnmatch(shim_name, p) for p in exclude)

    return keep


def _unmatched_patterns(
    patterns: Collection[str], shipped: Collection[str]
) -> tuple[str, ...]:
    """The selection *patterns* that matched none of *shipped*.

    Both directions are read against everything the source ships — the shims it
    loaded and the ones the selection declined — because that is the set a
    pattern can match.  A name dropped by the selection is by construction
    absent from the loaded set, so matching against the loaded set alone would
    call every ``exclude`` that did its job unmatched.
    """
    return tuple(
        pattern
        for pattern in patterns
        if not any(fnmatch(shipped_name, pattern) for shipped_name in shipped)
    )


def _selection_report(
    location: str,
    names: tuple[str, ...],
    declined: tuple[str, ...],
    *,
    include: Collection[str],
    exclude: Collection[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Measure both selection directions and log what matched nothing.

    A pattern is matched against everything the source ships — the shims it
    loaded and the ones the selection declined — because that is the set a
    pattern can hit.  An ``include`` whose every hit the ``exclude`` then drops
    matches every pattern and still leaves the source empty, so it is reported
    on its own: the operator believes a shim is running and nothing loads.

    Returns:
        ``(unmatched, unmatched_excludes, emptied_includes)``.
    """
    shipped = names + declined
    unmatched = _unmatched_patterns(include, shipped)
    unmatched_excludes = _unmatched_patterns(exclude, shipped)
    emptied_includes = (
        tuple(pattern for pattern in include if pattern not in unmatched)
        if include and not names
        else ()
    )
    for pattern in unmatched:
        logger.warning(
            "Shim source %s: include %r matched no shim; it loads: %s",
            location,
            pattern,
            ", ".join(names) or "nothing",
        )
    for pattern in unmatched_excludes:
        logger.warning(
            "Shim source %s: exclude %r matched no shim; it ships: %s",
            location,
            pattern,
            ", ".join(shipped) or "nothing",
        )
    for pattern in emptied_includes:
        logger.warning(
            "Shim source %s: include %r selected only shims the exclude "
            "drops; it loads: nothing",
            location,
            pattern,
        )
    return unmatched, unmatched_excludes, emptied_includes
