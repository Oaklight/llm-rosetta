"""What each manifest source offers, for the admin panel (#1660 S9).

The install path (:mod:`.manifest_sources`) only looks at the vendors a
selector picks.  The panel needs the other half too: every vendor of every
source the gateway reads, with its newest version, the version this core
would install, whether the installed release's ``requires`` matches this
core, and the selector (if any) that picks it.  This module builds that view
from the same cached manifest the installer reads and the status rows the
last rebuild reported; it installs nothing.

A manifest is read through the installer's own cache and lock, so opening the
panel and a rebuild never race on one file, and a fresh cache costs no
network call.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.shim_manifest import (
    SourceManifest,
    VendorRelease,
    requirement_problem,
    select_release,
    version_key,
)

from ._cache import _file_lock
from ._types import HttpGet, ShimSourceStatus
from .manifest_sources import (
    InstallSelector,
    ManifestSourceSpec,
    _load_manifest,
    _source_root,
    parse_source_entry,
)


def _requires(release: VendorRelease | None) -> dict[str, Any] | None:
    if release is None:
        return None
    return {"surface": release.requires.surface, "ir": release.requires.ir}


def _problem(release: VendorRelease | None) -> str | None:
    if release is None:
        return None
    return requirement_problem(release.requires)


def _release_from_status(status: ShimSourceStatus | None) -> Mapping[str, Any]:
    return (status.release or {}) if status is not None else {}


def release_requires_problem(release: Mapping[str, Any]) -> str | None:
    """``requirement_problem`` for a release record kept as a plain mapping."""
    from llm_rosetta.contracts.shim_manifest import ShimRequirement

    requires = release.get("requires") or {}
    try:
        req = ShimRequirement(int(requires["surface"]), int(requires["ir"]))
    except (KeyError, TypeError, ValueError):
        return "the installed release names no readable requirement"
    return requirement_problem(req)


def _picker(
    vendor: str,
    tags: Collection[str],
    source_id: str,
    selectors: Sequence[InstallSelector],
) -> InstallSelector | None:
    """The first selector that picks *vendor*, as the installer reads them."""
    return next(
        (s for s in selectors if s.applies_to(source_id) and s.matches(vendor, tags)),
        None,
    )


def _presets_picking(
    vendor: str,
    tags: Collection[str],
    source_id: str,
    selectors: Sequence[InstallSelector],
) -> list[str]:
    """The ``shims.presets`` keys whose mapped selector picks *vendor*."""
    keys = [
        s.origin.split(":", 1)[1]
        for s in selectors
        if s.origin.startswith("presets:")
        and s.applies_to(source_id)
        and s.matches(vendor, tags)
    ]
    return list(dict.fromkeys(keys))


def vendor_view(
    vendor: str,
    releases: Sequence[VendorRelease],
    *,
    source_id: str,
    selectors: Sequence[InstallSelector],
    status: ShimSourceStatus | None,
    core_names: Collection[str] = (),
    claimed_by: str | None = None,
    logo: str | None = None,
) -> dict[str, Any]:
    """One vendor card's data.

    Args:
        vendor: The vendor name.
        releases: Its releases, in any order.
        source_id: The source the manifest belongs to.
        selectors: The ``shims.install`` selectors in force (mapped presets
            included).
        status: The vendor's status row from the last rebuild, when there is
            one.
        core_names: The shim names core registers.
        claimed_by: The earlier source that provides the same vendor, when a
            selector picks it there first.
        logo: The catalogue's mark for the vendor, if any.

    Returns:
        A JSON-safe mapping (see the keys below).
    """
    ordered = sorted(releases, key=lambda r: version_key(r.version), reverse=True)
    newest = ordered[0] if ordered else None
    installable, _passed = select_release(ordered)
    # The record that would be installed speaks for maturity and tags, as it
    # does for the tag selectors; with nothing installable, the newest does.
    record = installable or newest
    tags = tuple(record.tags) if record is not None else ()
    selector = _picker(vendor, tags, source_id, selectors)
    installed = _release_from_status(status)
    installed_problem = release_requires_problem(installed) if installed else None
    return {
        "vendor": vendor,
        "logo": logo,
        "versions": [r.version for r in ordered],
        "newest": newest.version if newest else None,
        "newest_requires": _requires(newest),
        "newest_problem": _problem(newest),
        "installable": installable.version if installable else None,
        "installed": installed.get("version"),
        "installed_requires": installed.get("requires"),
        "installed_problem": installed_problem,
        "requires_ok": (installed_problem is None) if installed else None,
        "update_available": _has_update(installed, installable),
        **_described(installed, record, tags),
        "core_provides": sorted(set(record.shims if record else ()) & set(core_names)),
        "selected_by": selector.raw if selector is not None else None,
        "selected_origin": selector.origin if selector is not None else None,
        # Every legacy preset that selects the vendor, not only the first
        # selector: a preset behind an explicit entry still installs it when
        # the entry is removed, so the panel must refuse that removal.
        "selected_presets": _presets_picking(vendor, tags, source_id, selectors),
        "claimed_by": claimed_by,
        **_status_fields(status, installed, selector),
    }


def _has_update(
    installed: Mapping[str, Any], installable: VendorRelease | None
) -> bool:
    """Whether the release this core would install is newer than the installed one."""
    current = installed.get("version")
    return bool(
        current
        and installable is not None
        and version_key(installable.version) > version_key(str(current))
    )


def _described(
    installed: Mapping[str, Any], record: VendorRelease | None, tags: Sequence[str]
) -> dict[str, Any]:
    """What the source says about the vendor: the installed record first."""
    return {
        "maturity": installed.get("maturity") or (record.maturity if record else None),
        "since": installed.get("since") or (record.since if record else None),
        "tags": list(installed.get("tags") or tags),
        "conformance": (
            dict(record.conformance) if record and record.conformance else None
        ),
        "promoted_to": record.promoted_to if record else None,
        "shims": list(record.shims) if record else [],
    }


def _status_fields(
    status: ShimSourceStatus | None,
    installed: Mapping[str, Any],
    selector: InstallSelector | None,
) -> dict[str, Any]:
    """The card's state word and what the last rebuild reported."""
    if status is not None and status.skipped:
        state = "skipped"
    elif status is not None and not status.ok:
        state = "failed"
    elif installed.get("version"):
        state = "installed"
    elif selector is not None:
        state = "selected"
    else:
        state = "available"
    return {
        "state": state,
        "installed_shims": list(status.shim_names) if status is not None else [],
        "skipped": status.skipped if status is not None else None,
        "error": status.error if status is not None and not status.ok else None,
        "stale": bool(status.stale) if status is not None else False,
    }


def _configured_entry(
    spec: ManifestSourceSpec, raw_sources: Sequence[Any]
) -> Any | None:
    """The ``shims.sources`` entry that declares *spec*, as written."""
    for entry in raw_sources:
        try:
            parsed = parse_source_entry(entry)
        except ValueError:
            continue
        if parsed.url == spec.url:
            return entry
    return None


def _catalogue_logos(key: str | None) -> dict[str, str]:
    from .catalog import get_shim_source

    source = get_shim_source(key) if key else None
    if source is None:
        return {}
    return {v["name"]: v["logo"] for v in source.to_dict()["vendors"] if v.get("logo")}


def source_catalogue(
    specs: Sequence[ManifestSourceSpec],
    *,
    raw_sources: Sequence[Any],
    selectors: Sequence[InstallSelector],
    statuses: Sequence[ShimSourceStatus],
    cache_root: str | Path,
    ttl: int,
    http_get: HttpGet,
    proxy: str | None = None,
    refresh: bool = False,
    core_names: Collection[str] = (),
    now: Callable[[], float] = time.time,
) -> list[dict[str, Any]]:
    """Every source with every vendor it publishes.

    Args:
        specs: The sources to read, in ``shims.sources`` order.
        raw_sources: ``shims.sources`` as written; a spec without an entry
            there is implied (the defaults, or a mapped legacy preset).
        selectors: The ``shims.install`` selectors in force.
        statuses: The last rebuild's status rows.
        cache_root: The shim cache directory.
        ttl: Seconds a cached manifest is reused.
        http_get: ``(url, *, headers, timeout, proxy) -> bytes``.
        proxy: Upstream proxy URL.
        refresh: Refetch every manifest regardless of its age.
        core_names: The shim names core registers.
        now: Clock.

    Returns:
        One mapping per source: ``id``, ``key``, ``label``, ``url``,
        ``entry`` (the config entry, or ``None`` when implied), ``ok``,
        ``error``, ``stale``, ``generated_at``, ``repo`` and ``vendors``.
    """
    from .catalog import get_shim_source

    by_vendor = {
        (s.source, s.vendor): s for s in statuses if getattr(s, "vendor", None)
    }
    claimed: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for spec in specs:
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
                force=refresh,
                selectors=selectors,
            )
        catalogue = get_shim_source(spec.key) if spec.key else None
        entry = _configured_entry(spec, raw_sources)
        manifest: SourceManifest | None = state.manifest
        source_id = spec.key or (manifest.source if manifest else None) or spec.url
        logos = _catalogue_logos(spec.key)
        vendors: list[dict[str, Any]] = []
        releases = dict(manifest.vendors) if manifest is not None else {}
        for vendor in sorted(releases):
            view = vendor_view(
                vendor,
                releases[vendor],
                source_id=source_id,
                selectors=selectors,
                status=by_vendor.get((source_id, vendor)),
                core_names=core_names,
                claimed_by=claimed.get(vendor),
                logo=logos.get(vendor),
            )
            if view["selected_by"] and vendor not in claimed:
                claimed[vendor] = source_id
            vendors.append(view)
        out.append(
            {
                "id": source_id,
                "key": spec.key,
                "label": catalogue.label if catalogue else source_id,
                "url": spec.url,
                "fallback_url": spec.fallback_url,
                "repo": (manifest.repo if manifest else None)
                or (catalogue.repo if catalogue else None),
                "entry": entry,
                "configured": entry is not None,
                "ok": manifest is not None,
                "error": state.error,
                "stale": state.stale,
                "fetched_at": state.fetched_at,
                "generated_at": manifest.generated_at if manifest else None,
                "vendors": vendors,
            }
        )
    return out


def _specs_and_selectors(config: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """The sources to show, the selectors in force, and ``shims.sources`` as written.

    With no source in the plan, the catalogue's default sources are listed as
    implied, so the panel can offer their vendors before any is selected.
    """
    from .catalog import DEFAULT_SOURCES, get_shim_source
    from .manifest_sources import source_spec_from_catalogue

    shims = getattr(config, "shims", None)
    plan = getattr(shims, "manifests", None)
    specs = list(getattr(plan, "sources", ()) or ())
    selectors = list(getattr(plan, "selectors", ()) or ())
    raw_sources = list(getattr(shims, "sources", ()) or ())
    if not specs:
        specs = [
            source_spec_from_catalogue(source)
            for source in (get_shim_source(k) for k in DEFAULT_SOURCES)
            if source is not None
        ]
    return specs, selectors, raw_sources


def _core_names(report: Any) -> set[str]:
    """The registered names no declared source loaded: core and plugins."""
    from llm_rosetta.api import list_shims

    from_sources: set[str] = set()
    for status in getattr(report, "sources", []) or []:
        from_sources.update(getattr(status, "shim_names", ()) or ())
    return {shim.name for shim in list_shims()} - from_sources


def catalogue_payload(
    config: Any,
    report: Any,
    *,
    refresh: bool = False,
    http_get: HttpGet | None = None,
) -> dict[str, Any]:
    """The admin source catalogue for the served *config* and load *report*.

    Args:
        config: The served gateway config (its ``shims`` section is read).
        report: The last :class:`ShimLoadReport`, or ``None``.
        refresh: Refetch every manifest regardless of its age.
        http_get: Injectable getter; the shim loader's default otherwise.

    Returns:
        ``{"core", "sources", "install"}``; see :func:`source_catalogue`.
    """
    from ..shim_sources import _cache_root
    from .sources import _default_http_get

    shims = getattr(config, "shims", None)
    specs, selectors, raw_sources = _specs_and_selectors(config)
    sources = source_catalogue(
        specs,
        raw_sources=raw_sources,
        selectors=selectors,
        statuses=list(getattr(report, "sources", []) or []),
        cache_root=getattr(shims, "cache_root", None) or _cache_root(None, None),
        ttl=int(getattr(shims, "remote_ttl", 24 * 3600) or 0),
        http_get=http_get or _default_http_get,
        proxy=getattr(config, "proxy", None),
        refresh=refresh,
        core_names=_core_names(report),
    )
    return {
        "core": core_versions(),
        "sources": sources,
        "install": list(getattr(shims, "install", ()) or ()),
    }


def core_versions() -> dict[str, Any]:
    """The levels a release's ``requires`` is checked against."""
    from llm_rosetta.api import __version__

    return {"surface": SURFACE_VERSION, "ir": IR_VERSION, "version": __version__}


__all__ = [
    "catalogue_payload",
    "core_versions",
    "release_requires_problem",
    "source_catalogue",
    "vendor_view",
]
