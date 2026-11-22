"""Install: downloading a release and placing it in the shim cache.

:func:`install_from_manifests` is the entry point the registry rebuild calls.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import shutil
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_rosetta.contracts.shim_manifest import (
    VendorRelease,
    requirement_problem,
)

from .._archive import _extract_archive
from .._cache import _atomic_replace, _file_lock, _read_stamp, _write_stamp
from .._legs import _is_unreachable
from .._types import (
    HTTP_TIMEOUT,
    MAX_ARCHIVE_BYTES,
    HttpGet,
    ShimArchiveMismatch,
    ShimSourceError,
    ShimSourceStatus,
    ShimSourceUnreachable,
    logger,
)
from .freshness import (
    _load_manifest,
    _ManifestState,
)
from .plan import (
    InstallSelector,
    ManifestPlan,
    ManifestSourceSpec,
    _dist_root,
    _headers_for,
    _manifest_legs,
    _scan_root,
    _source_root,
)
from .selection import (
    _choose_release,
    _installed_release,
    _keeps,
    _no_release_error,
    _select_vendors,
    _skip_reason,
    _unmatched_rows,
)


def install_from_manifests(
    plan: ManifestPlan,
    *,
    cache_root: str | os.PathLike[str],
    http_get: HttpGet,
    now: Callable[[], float],
    ttl: int,
    proxy: str | None = None,
    force_refresh: bool = False,
    core_names: Collection[str] = (),
    core_version: str | None = None,
) -> list[ShimSourceStatus]:
    """Fetch every planned manifest and install the selected vendors.

    The registry is not touched: each returned status carries the directory
    to scan, and :func:`scan_manifest_status` loads it.

    Args:
        plan: The sources and the selectors.
        cache_root: The shim cache directory.
        http_get: ``(url, *, headers, timeout, proxy) -> bytes``.
        now: Clock.
        ttl: Seconds a fetched manifest is reused without a refetch.
        proxy: Upstream proxy URL.
        force_refresh: Refetch every manifest regardless of its age.
        core_names: The shim names core registers; a vendor shipping one is
            not downloaded (the scan step re-checks against the live set).
        core_version: The running core's version, for ``promoted_to``;
            :data:`llm_rosetta.api.__version__` by default.

    Returns:
        One status per selected vendor, or per failed source.
    """
    if core_version is None:
        from llm_rosetta.api import __version__ as core_version
    statuses: list[ShimSourceStatus] = []
    claimed: dict[str, str] = {}
    matched: set[str] = set()
    overridden: dict[str, str] = {}
    answered: list[tuple[str, ManifestSourceSpec]] = []
    for spec in plan.sources:
        root = _source_root(cache_root, spec)
        root.mkdir(parents=True, exist_ok=True)
        with _file_lock(root.parent / f"{root.name}.lock"):
            state = _load_manifest(
                spec,
                root,
                http_get=http_get,
                now=now,
                ttl=ttl,
                proxy=proxy,
                force=force_refresh,
                selectors=plan.selectors,
            )
            if state.manifest is None:
                statuses.append(_unanswered_source(spec, state))
                continue
            source_id = spec.key or state.manifest.source
            answered.append((source_id, spec))
            for vendor, selector in _select_vendors(
                state.manifest,
                source_id,
                plan.selectors,
                claimed,
                matched,
                root,
                overridden,
            ):
                status = _install_vendor(
                    spec,
                    source_id,
                    root,
                    state,
                    vendor,
                    selector,
                    http_get=http_get,
                    now=now,
                    proxy=proxy,
                    core_names=core_names,
                    core_version=core_version,
                )
                if state.stale_reason and status.ok and not status.skipped:
                    status = replace(
                        status, stale=True, error=status.error or state.stale_reason
                    )
                statuses.append(status)
    statuses.extend(
        _unmatched_rows(plan.selectors, matched, answered, claimed, overridden)
    )
    return statuses


def _unanswered_source(
    spec: ManifestSourceSpec, state: _ManifestState
) -> ShimSourceStatus:
    """The failed row for a source with no manifest at all."""
    return ShimSourceStatus(
        kind="remote",
        location=spec.url,
        name=spec.key,
        ok=False,
        error=state.error or "manifest unavailable",
        unreachable=state.unreachable,
        source=spec.key,
    )


def _install_vendor(
    spec: ManifestSourceSpec,
    source_id: str,
    root: Path,
    state: _ManifestState,
    vendor: str,
    selector: InstallSelector,
    *,
    http_get: HttpGet,
    now: Callable[[], float],
    proxy: str | None,
    core_names: Collection[str],
    core_version: str,
) -> ShimSourceStatus:
    """Decide on one vendor and install it when needed."""
    assert state.manifest is not None
    releases = state.manifest.vendors.get(vendor, ())
    dropped = state.manifest.dropped.get(vendor)
    base = ShimSourceStatus(
        kind="remote",
        location=f"{spec.url}#{vendor}",
        name=f"{source_id}:{vendor}",
        source=source_id,
        vendor=vendor,
        fetched_at=state.fetched_at,
    )

    chosen, passed_over, pin_problem = _choose_release(
        releases, selector, source_id, vendor
    )

    if chosen is not None:
        skip = _skip_reason(chosen, selector, source_id, core_names, core_version)
        if skip:
            return replace(base, skipped=skip)

    vendor_dir = root / vendor
    stamp_path = root / f"{vendor}.stamp.json"
    stamp = _read_stamp(stamp_path)
    installed = _installed_release(stamp, vendor) if vendor_dir.is_dir() else None
    usable = installed is not None and requirement_problem(installed.requires) is None
    if installed is not None and installed.version not in {r.version for r in releases}:
        logger.warning(
            "Shim source %s: manifest no longer lists %s %s; keeping the install",
            source_id,
            vendor,
            installed.version,
        )

    if dropped is not None:
        # Every record of the vendor is invalid: keep the last good install
        # (stale, like an unreachable source) and never uninstall for it.
        why = f"no valid release: {dropped}"
        kept = installed if usable else None
        logger.warning(
            "Shim source %s: vendor %r has %s; %s",
            source_id,
            vendor,
            why,
            f"keeping the installed {kept.version}" if kept else "it is not installed",
        )
        if kept is None:
            return replace(base, skipped=why)
        return _kept_or_failed(base, vendor_dir, kept, stamp, state, why)

    if chosen is None:
        kept = installed if usable else None
        if not pin_problem and kept is not None:
            return _installed_status(base, vendor_dir, kept, stamp, state)
        error = _no_release_error(vendor, selector, pin_problem, passed_over)
        return _kept_or_failed(base, vendor_dir, kept, stamp, state, error)

    if usable and installed is not None and _keeps(installed, chosen, selector):
        return _installed_status(base, vendor_dir, installed, stamp, state)

    try:
        archive_url = _download_release(
            spec, state, chosen, vendor_dir, http_get=http_get, proxy=proxy
        )
    except ShimSourceError as exc:
        mismatch = isinstance(exc, ShimArchiveMismatch)
        logger.warning(
            "Shim source %s: cannot install %s %s: %s%s",
            source_id,
            vendor,
            chosen.version,
            exc,
            f"; keeping {installed.version}" if usable and installed else "",
        )
        if usable and installed is not None:
            status = _installed_status(base, vendor_dir, installed, stamp, state)
            return replace(
                status,
                stale=True,
                error=str(exc),
                checksum_ok=False if mismatch else None,
            )
        return replace(
            base,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            unreachable=isinstance(exc, ShimSourceUnreachable),
            checksum_ok=False if mismatch else None,
        )

    record = {
        **chosen.to_dict(),
        "vendor": vendor,
        "source": source_id,
        "archive_url": archive_url,
        "installed_at": now(),
    }
    _write_stamp(stamp_path, record)
    if installed is not None:
        logger.info(
            "Shim source %s: %s upgraded %s -> %s",
            source_id,
            vendor,
            installed.version,
            chosen.version,
        )
    return replace(
        base,
        resolved_dir=str(_scan_root(vendor_dir)),
        resolved_ref=chosen.commit,
        checksum_ok=True,
        archive_url=archive_url,
        leg="primary" if archive_url.startswith(_dist_root(spec.url)) else "mirror",
        release=chosen.to_dict(),
    )


def _kept_or_failed(
    base: ShimSourceStatus,
    vendor_dir: Path,
    installed: VendorRelease | None,
    stamp: Mapping[str, Any],
    state: _ManifestState,
    error: str,
) -> ShimSourceStatus:
    """The installed release kept (stale, with *error*), or a failed row."""
    if installed is None:
        return replace(base, ok=False, error=error)
    status = _installed_status(base, vendor_dir, installed, stamp, state)
    return replace(status, stale=True, error=error)


def _installed_status(
    base: ShimSourceStatus,
    vendor_dir: Path,
    installed: VendorRelease,
    stamp: Mapping[str, Any],
    state: _ManifestState,
) -> ShimSourceStatus:
    """The status for a vendor served from its existing install."""
    archive_url = stamp.get("archive_url")
    return replace(
        base,
        resolved_dir=str(_scan_root(vendor_dir)),
        resolved_ref=installed.commit,
        from_cache=True,
        stale=state.stale,
        checksum_ok=True,
        archive_url=archive_url,
        release=installed.to_dict(),
    )


def _download_release(
    spec: ManifestSourceSpec,
    state: _ManifestState,
    release: VendorRelease,
    vendor_dir: Path,
    *,
    http_get: HttpGet,
    proxy: str | None,
) -> str:
    """Download, verify and install *release* into *vendor_dir*.

    The archive is resolved against the dist root of each manifest URL, the
    one that served the manifest first.  A digest mismatch on one URL moves on
    to the next (a stale CDN copy), but never installs.

    Returns:
        The archive URL that was installed.

    Raises:
        ShimArchiveMismatch: Every URL that answered served other bytes.
        ShimSourceUnreachable: No URL answered.
        ShimSourceError: Another failure (HTTP error, unusable archive).
    """
    failures: list[str] = []
    mismatch = False
    all_unreachable = True
    for manifest_url in _manifest_legs(spec, state.leg_url):
        url = _dist_root(manifest_url) + posixpath.normpath(release.archive)
        try:
            data = bytes(
                http_get(
                    url,
                    headers=_headers_for(spec, url),
                    timeout=HTTP_TIMEOUT,
                    proxy=proxy,
                )
            )
        except Exception as exc:  # noqa: BLE001 - the next URL is tried
            all_unreachable = all_unreachable and _is_unreachable(exc)
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        all_unreachable = False
        if len(data) > MAX_ARCHIVE_BYTES:
            failures.append(f"{url}: archive exceeds {MAX_ARCHIVE_BYTES} bytes")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != release.sha256:
            mismatch = True
            failures.append(
                f"{url}: sha256 mismatch: expected {release.sha256}, got {digest}"
            )
            continue
        try:
            _install_tree(data, vendor_dir, url)
        except ShimSourceError as exc:
            failures.append(f"{url}: {exc}")
            continue
        return url
    detail = "; ".join(failures)
    if mismatch:
        raise ShimArchiveMismatch(detail)
    if all_unreachable:
        raise ShimSourceUnreachable(detail)
    raise ShimSourceError(detail)


def _install_tree(data: bytes, vendor_dir: Path, url: str) -> None:
    """Extract *data* next to *vendor_dir* and swap it in atomically."""
    tmp_root = vendor_dir.parent / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"extract-{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir()
        # ``providers`` as the declared path keeps a ``providers/`` root from
        # being stripped as the archive's single top-level directory.
        _extract_archive(
            data, tmp_dir, archive="tar.gz", url=url, declared_path="providers"
        )
        _atomic_replace(tmp_dir, vendor_dir)
    except OSError as exc:
        raise ShimSourceError(f"cannot install locally: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
