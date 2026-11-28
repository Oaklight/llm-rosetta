"""Status: read-only views of a manifest source for the registry rebuild and
the admin page.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import Any

from llm_rosetta.api import ExtensionIndex, choose_release, version_key

from .._types import (
    HttpGet,
    ShimSourceStatus,
    logger,
)
from .freshness import _fetch_manifest_once
from .plan import ManifestSourceSpec
from .selection import _provided_by


def scan_manifest_status(
    status: ShimSourceStatus, core_names: Collection[str]
) -> ShimSourceStatus:
    """Report how one installed vendor's extension loaded.

    The extension is already in the extensions directory, and the registry
    rebuild that calls this has just run ``load_providers()``, which loads every
    installed extension through the one loader and records its outcome.  This
    function reads that outcome (:func:`~llm_rosetta.api.list_extensions`) and
    folds it into the vendor's status: the loaded shims, or the state and
    reason it was skipped or failed.

    *core_names* is the live registry.  A vendor all of whose shims are
    registered by something other than an extension is skipped (core wins), as
    it is not installed; when only some clash, the loader refuses the
    extension as a whole and the reason says which names are held.
    """
    from llm_rosetta.api import list_extensions

    if not status.ok or status.skipped or not status.extension:
        return status
    record_shims = (status.release or {}).get("shims") or ()
    items = list_extensions()
    by_extension = {item.id: item for item in items}
    held_by_extensions = {name for item in items for name in item.shims}
    clash = sorted((set(record_shims) & set(core_names)) - held_by_extensions)
    if clash and set(clash) == set(record_shims):
        reason = f"already provided by {_provided_by(clash)}: " + ", ".join(clash)
        logger.warning(
            "Shim source %s: vendor %r is ignored; %s",
            status.source,
            status.vendor,
            reason,
        )
        return replace(status, skipped=reason)

    item = by_extension.get(status.extension)
    if item is None:
        return replace(
            status,
            ok=False,
            error=(
                f"extension {status.extension!r} was installed but not found "
                "by the loader; is the extensions directory on the import path?"
            ),
        )
    base = replace(status, state=item.state, resolved_dir=str(item.path))
    if item.state == "skipped":
        return replace(base, skipped=item.reason or "skipped")
    if item.state != "loaded":
        return replace(base, ok=False, error=item.reason or "failed to load")
    if record_shims and set(item.shims) != set(record_shims):
        logger.warning(
            "Shim source %s: vendor %r registered %s, but its release record lists %s",
            status.source,
            status.vendor,
            ", ".join(item.shims) or "nothing",
            ", ".join(record_shims),
        )
    return replace(
        base,
        shim_names=tuple(item.shims),
        shim_count=len(item.shims),
        manifest_shims=tuple(item.manifest.shims) if item.manifest else (),
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
    manifests: dict[str, ExtensionIndex | str] = {}
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
        elif status.vendor not in manifest:
            entry["error"] = f"{status.vendor} is no longer in the index"
        else:
            chosen, _ = choose_release(manifest[status.vendor], prefer_stable=True)
            if chosen is None:
                entry["error"] = "no version is installable on this core"
            else:
                entry["head"] = chosen.version
                entry["update"] = bool(current) and version_key(
                    chosen.version
                ) > version_key(str(current))
        out.append(entry)
    return out
