"""Install: downloading a release wheel and placing it in the extensions directory.

:func:`install_from_manifests` is the entry point the registry rebuild calls.
A release is a wheel of one extension (``llm_rosetta_ext/<vendor>/``); core's
:func:`~llm_rosetta.api.install_wheel` checks and extracts it into the
extensions directory, which :func:`~llm_rosetta.api.configure_extensions_dir`
points at the gateway's data directory.  The install stamp stays in the source
cache: it records which release was installed and from where.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
from collections.abc import Callable, Collection, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_rosetta.api import (
    ExtensionInstallError,
    InstalledExtension,
    extensions_dir,
    install_wheel,
    installed_extensions,
    missing_dependencies,
    remove_extension,
)
from llm_rosetta.contracts.shim_manifest import (
    VendorRelease,
    requirement_problem,
)

from .._cache import _file_lock, _read_stamp, _write_stamp
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
    extensions_root: str | os.PathLike[str] | None = None,
) -> list[ShimSourceStatus]:
    """Fetch every planned manifest and install the selected vendors.

    The registry is not touched: each returned status carries the extension
    id, and :func:`scan_manifest_status` reads the outcome of loading it.

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
        extensions_root: The extensions directory the wheels install into;
            :func:`llm_rosetta.api.extensions_dir` by default.

    Returns:
        One status per selected vendor, or per failed source.
    """
    if core_version is None:
        from llm_rosetta.api import __version__ as core_version
    dest = Path(extensions_root) if extensions_root is not None else extensions_dir()
    have = {item.id: item for item in installed_extensions(dest)}
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
                    dest=dest,
                    have=have,
                )
                if state.stale_reason and status.ok and not status.skipped:
                    status = replace(
                        status, stale=True, error=status.error or state.stale_reason
                    )
                statuses.append(status)
    statuses.extend(
        _unmatched_rows(plan.selectors, matched, answered, claimed, overridden)
    )
    return _with_dependency_notes(statuses, dest)


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
    dest: Path,
    have: Mapping[str, InstalledExtension],
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

    stamp_path = root / f"{vendor}.stamp.json"
    stamp = _read_stamp(stamp_path)
    # Installed means: the stamp records a release and the extensions
    # directory really holds that version of the extension.
    present = have.get(vendor)
    installed = (
        _installed_release(stamp, vendor)
        if present is not None and present.version == stamp.get("version")
        else None
    )
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
        return _kept_or_failed(base, present, kept, stamp, state, why)

    if chosen is None:
        kept = installed if usable else None
        if not pin_problem and kept is not None:
            return _installed_status(base, present, kept, stamp, state)
        error = _no_release_error(vendor, selector, pin_problem, passed_over)
        return _kept_or_failed(base, present, kept, stamp, state, error)

    if usable and installed is not None and _keeps(installed, chosen, selector):
        return _installed_status(base, present, installed, stamp, state)

    try:
        archive_url, fresh = _download_release(
            spec, state, chosen, vendor, dest, http_get=http_get, proxy=proxy
        )
    except ShimSourceError as exc:
        kept = installed if usable else None
        logger.warning(
            "Shim source %s: cannot install %s %s: %s%s",
            source_id,
            vendor,
            chosen.version,
            exc,
            f"; keeping {kept.version}" if kept else "",
        )
        return _install_failed(base, present, kept, stamp, state, exc)

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
        resolved_dir=str(fresh.path),
        extension=vendor,
        resolved_ref=chosen.commit,
        checksum_ok=True,
        archive_url=archive_url,
        leg="primary" if archive_url.startswith(_dist_root(spec.url)) else "mirror",
        release=chosen.to_dict(),
    )


def _install_failed(
    base: ShimSourceStatus,
    present: InstalledExtension | None,
    kept: VendorRelease | None,
    stamp: Mapping[str, Any],
    state: _ManifestState,
    exc: ShimSourceError,
) -> ShimSourceStatus:
    """The row for a download or install that failed: keep the last good one."""
    mismatch = isinstance(exc, ShimArchiveMismatch)
    if kept is not None:
        status = _installed_status(base, present, kept, stamp, state)
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


def _kept_or_failed(
    base: ShimSourceStatus,
    present: InstalledExtension | None,
    installed: VendorRelease | None,
    stamp: Mapping[str, Any],
    state: _ManifestState,
    error: str,
) -> ShimSourceStatus:
    """The installed release kept (stale, with *error*), or a failed row."""
    if installed is None or present is None:
        return replace(base, ok=False, error=error)
    status = _installed_status(base, present, installed, stamp, state)
    return replace(status, stale=True, error=error)


def _installed_status(
    base: ShimSourceStatus,
    present: InstalledExtension | None,
    installed: VendorRelease,
    stamp: Mapping[str, Any],
    state: _ManifestState,
) -> ShimSourceStatus:
    """The status for a vendor served from its existing install."""
    archive_url = stamp.get("archive_url")
    return replace(
        base,
        resolved_dir=str(present.path) if present is not None else None,
        extension=base.vendor,
        resolved_ref=installed.commit,
        from_cache=True,
        stale=state.stale,
        checksum_ok=True,
        archive_url=archive_url,
        release=installed.to_dict(),
    )


def _with_dependency_notes(
    statuses: list[ShimSourceStatus], dest: Path
) -> list[ShimSourceStatus]:
    """Name, per installed vendor, the required extensions that are not installed.

    Read once every install of the run is done, so the order the vendors were
    installed in does not decide the answer.  Dependencies are reported, never
    installed on the extension's behalf.
    """
    if not any(status.extension for status in statuses):
        return statuses
    by_id = {item.id: item for item in installed_extensions(dest)}
    out: list[ShimSourceStatus] = []
    for status in statuses:
        item = by_id.get(status.extension or "")
        if item is None or item.manifest is None or not status.ok or status.skipped:
            out.append(status)
            continue
        missing = missing_dependencies(item.manifest, by_id)
        if not missing:
            out.append(status)
            continue
        names = ", ".join(repr(name) for name in missing)
        out.append(
            replace(
                status,
                note=(
                    f"extension {item.id!r} requires extension(s) {names}, which "
                    f"{'is' if len(missing) == 1 else 'are'} not installed"
                ),
            )
        )
    return out


def _download_release(
    spec: ManifestSourceSpec,
    state: _ManifestState,
    release: VendorRelease,
    vendor: str,
    dest: Path,
    *,
    http_get: HttpGet,
    proxy: str | None,
) -> tuple[str, InstalledExtension]:
    """Download, verify and install the wheel of *release* into *dest*.

    The wheel is resolved against the dist root of each manifest URL, the one
    that served the manifest first.  A digest mismatch on one URL moves on to
    the next (a stale CDN copy), but never installs.  Core's
    :func:`~llm_rosetta.api.install_wheel` then checks the wheel itself (a safe
    zip, one extension, a valid manifest, a surface this core runs) before it
    writes anything.

    Returns:
        The wheel URL that was installed and the installed extension.

    Raises:
        ShimArchiveMismatch: Every URL that answered served other bytes.
        ShimSourceUnreachable: No URL answered.
        ShimSourceError: Another failure (HTTP error, a wheel core refuses).
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
            installed = _install_wheel(data, release, vendor, dest)
        except ShimSourceError as exc:
            failures.append(f"{url}: {exc}")
            continue
        return url, installed
    detail = "; ".join(failures)
    if mismatch:
        raise ShimArchiveMismatch(detail)
    if all_unreachable:
        raise ShimSourceUnreachable(detail)
    raise ShimSourceError(detail)


def _install_wheel(
    data: bytes, release: VendorRelease, vendor: str, dest: Path
) -> InstalledExtension:
    """Install *data* with core's installer and check it is the vendor's wheel."""
    try:
        installed = install_wheel(data, sha256=release.sha256, dest=dest)
    except (ExtensionInstallError, OSError) as exc:
        raise ShimSourceError(f"cannot install the wheel: {exc}") from exc
    if installed.id != vendor:
        # The record names a vendor and the digest vouches for the bytes, so a
        # wheel of another extension is a publisher error.  Take it back out.
        remove_extension(installed.id, dest)
        raise ShimSourceError(
            f"the wheel holds extension {installed.id!r}, not {vendor!r}"
        )
    return installed
