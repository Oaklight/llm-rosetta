"""Status: read-only views of a manifest source for the registry rebuild and
the admin page.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_rosetta.contracts.shim_manifest import (
    SourceManifest,
    select_release,
    version_key,
)

from .._types import (
    HttpGet,
    ShimSourceStatus,
    logger,
)
from .freshness import _fetch_manifest_once
from .plan import ManifestSourceSpec


def scan_manifest_status(
    status: ShimSourceStatus, core_names: Collection[str]
) -> ShimSourceStatus:
    """Load one installed vendor into the registry being built.

    *core_names* is the authoritative set (built-ins and plugins, just
    registered): those names are refused by name (core wins) and the vendor's
    other shims load; a vendor all of whose shims core registers is skipped.
    When the archive ships no ``shims.yaml``, only the names its release record
    lists may register, and any other is refused with a WARNING.
    """
    from .._scan import _scan_dir
    from ..manifest import MANIFEST_FILENAME

    if not status.ok or status.skipped or not status.resolved_dir:
        return status
    record_shims = (status.release or {}).get("shims") or ()
    clash = sorted(set(record_shims) & set(core_names))
    if clash and set(clash) == set(record_shims):
        logger.warning(
            "Shim source %s: vendor %r is ignored; core already provides %s",
            status.source,
            status.vendor,
            ", ".join(clash),
        )
        return replace(status, skipped="core provides " + ", ".join(clash))
    scanned = _scan_dir(
        Path(status.resolved_dir),
        kind="remote",
        location=status.location,
        name=status.name,
        from_cache=status.from_cache,
        stale=status.stale,
        fetched_at=status.fetched_at,
        resolved_ref=status.resolved_ref,
        refuse=core_names,
        only=(
            None
            if (Path(status.resolved_dir) / MANIFEST_FILENAME).is_file()
            else frozenset(record_shims)
        ),
    )
    return replace(
        scanned,
        leg=status.leg,
        archive_url=status.archive_url,
        checksum_ok=status.checksum_ok,
        error=scanned.error or status.error,
        source=status.source,
        vendor=status.vendor,
        release=status.release,
    )


def check_manifest_updates(
    statuses: Sequence[ShimSourceStatus],
    *,
    http_get: HttpGet,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Compare each installed vendor with its manifest's newest installable version.

    One manifest fetch per source.  A vendor counts as having an update only
    when a newer version is installable on this core — the rule the install
    itself follows.

    Returns:
        One entry per vendor status: ``location``, ``current`` and ``head``
        (versions), ``update`` and ``error``.
    """
    from ..catalog import SHIM_SOURCES

    fallbacks = {s.manifest_url: s.fallback_url for s in SHIM_SOURCES}
    manifests: dict[str, SourceManifest | str] = {}
    out: list[dict[str, Any]] = []
    for status in statuses:
        current = (status.release or {}).get("version")
        entry: dict[str, Any] = {
            "location": status.location,
            "current": current,
            "head": None,
            "update": False,
            "error": None,
        }
        url = status.location.split("#", 1)[0]
        if url not in manifests:
            spec = ManifestSourceSpec(
                key=None, url=url, fallback_url=fallbacks.get(url)
            )
            manifests[url] = _fetch_manifest_once(spec, http_get=http_get, proxy=proxy)
        manifest = manifests[url]
        if isinstance(manifest, str):
            entry["error"] = manifest
        elif status.vendor not in manifest.vendors:
            entry["error"] = f"{status.vendor} is no longer in the manifest"
        else:
            chosen, _ = select_release(manifest.vendors[status.vendor])
            if chosen is None:
                entry["error"] = "no version is installable on this core"
            else:
                entry["head"] = chosen.version
                entry["update"] = bool(current) and version_key(
                    chosen.version
                ) > version_key(str(current))
        out.append(entry)
    return out
