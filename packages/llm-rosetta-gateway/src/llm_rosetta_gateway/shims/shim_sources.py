"""Bridge between the gateway config and declarative shim sources.

The top-level ``shims`` config section declares the manifest sources to read, the
vendors to install from them, and extra local directories to load provider
shims from.  This module resolves that section
against the config file location and drives
:func:`llm_rosetta.shims.rebuild_shims`.

Shims must be registered **before** :class:`~llm_rosetta_gateway.config.GatewayConfig`
is constructed: its provider-type resolution looks shims up in the global
registry.  The CLI therefore calls :func:`apply_shim_sources` between
``load_config`` and ``GatewayConfig``; the admin reload path does the same
before rebuilding the config.

Programmatic embedders holding a raw config dict should call
:func:`apply_shim_sources` before ``GatewayConfig(raw)``; ``create_app`` also
applies the section automatically when none was supplied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import ShimLoadReport, rebuild_shims
from .catalog import DEFAULT_SOURCES, get_shim_source
from .manifest_sources.plan import (
    InstallSelector,
    ManifestPlan,
    ManifestSourceSpec,
    parse_selector,
    parse_source_entry,
    source_spec_from_catalogue,
)
from .moved import MovedReference, moved_references
from ._types import DEFAULT_CACHE_DIRNAME, REMOTE_TTL_SECONDS, HttpGet

logger = logging.getLogger("llm-rosetta-gateway")


def shims_signature(raw: Mapping[str, Any] | None) -> str:
    """Return a stable hash of the ``shims`` section.

    Used by the admin reload path to skip a full shim rebuild (and any remote
    fetch) when an unrelated part of the config changed.  Key order inside the
    section does not affect the result.

    The hash covers the section **as the document spells it**, not the parsed
    :class:`ShimSources`: the loader stores this value and the reload path
    recomputes it from the document it has just read, so both sides must hash
    the same kind of value.  A parsed section has already anchored its paths
    and expanded its defaults, which would make the two disagree.

    *raw* is the whole document: the moved vendors its providers name are
    part of the hash (see :mod:`.shims.moved`).
    """
    section = raw.get("shims") if isinstance(raw, Mapping) else None
    # The vendors that moved out of core are installed for the providers that
    # name them (#1660), so a provider edit that adds one changes what a
    # rebuild loads even when the section itself did not change.
    moved = sorted({ref.vendor for ref in moved_references(raw)})
    if not section and not moved:
        return ""
    payload: Any = {"section": section, "moved": moved} if moved else section
    try:
        blob = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(section)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_paths(raw_paths: Any, config_dir: str | None) -> list[str]:
    """Validate and anchor the ``shims.paths`` entries."""
    if not isinstance(raw_paths, list):
        raise ValueError("config: 'shims.paths' must be an array")
    dirs: list[str] = []
    for entry in raw_paths:
        if not isinstance(entry, str) or not entry:
            raise ValueError("config: 'shims.paths' entries must be non-empty strings")
        dirs.append(_anchor(entry, config_dir))
    return dirs


def _source_specs(raw_sources: Any, on_unknown: str) -> list[ManifestSourceSpec]:
    """Parse ``shims.sources``; an unknown id is skipped or refused per *on_unknown*."""
    if not isinstance(raw_sources, list):
        raise ValueError("config: 'shims.sources' must be an array")
    specs: list[ManifestSourceSpec] = []
    for entry in raw_sources:
        try:
            spec = parse_source_entry(entry)
        except ValueError as exc:
            if on_unknown == "skip" and isinstance(entry, str) and "://" not in entry:
                logger.warning("%s; it is skipped", exc)
                continue
            raise
        if all(spec.url != other.url for other in specs):
            specs.append(spec)
    return specs


def _resolve_manifest_plan(
    section: Mapping[str, Any], *, on_unknown: str
) -> ManifestPlan:
    """Build the manifest plan from ``shims.sources`` and ``shims.install``.

    ``shims.install`` selectors read ``shims.sources`` (default
    :data:`~llm_rosetta_gateway.shims.catalog.DEFAULT_SOURCES` when only
    ``install`` is declared).
    """
    raw_install = section.get("install") or []
    if not isinstance(raw_install, list):
        raise ValueError("config: 'shims.install' must be an array")
    selectors: list[InstallSelector] = [parse_selector(x) for x in raw_install]
    raw_sources = section.get("sources")
    if raw_sources is None:
        raw_sources = list(DEFAULT_SOURCES) if selectors else []
    specs = _source_specs(raw_sources, on_unknown)
    _check_qualifiers(selectors, specs, on_unknown)
    return ManifestPlan(sources=tuple(specs), selectors=tuple(selectors))


def _check_qualifiers(
    selectors: list[InstallSelector],
    specs: list[ManifestSourceSpec],
    on_unknown: str,
) -> None:
    """Refuse or warn about a ``<source>:`` qualifier no declared source has."""
    known = {spec.key for spec in specs}
    has_url_source = any(spec.key is None for spec in specs)
    for selector in selectors:
        if selector.source and selector.source not in known and not has_url_source:
            message = (
                f"config: 'shims.install' entry {selector.raw!r} names source "
                f"{selector.source!r}, which 'shims.sources' does not declare"
            )
            if on_unknown == "raise":
                raise ValueError(message)
            logger.warning("%s; it matches nothing", message)


#: ``shims`` keys 0.14.x read and 0.15.0 refuses, with what replaced them.
RETIRED_KEYS: dict[str, str] = {
    "presets": "declare the vendor in 'shims.install' (for example "
    "'argonne:alcf'); 'llm-rosetta-migrate --config <path>' rewrites the key",
    "include": "it selected shims of a preset; use 'shims.install' selectors and "
    "'shims.disabled'; 'llm-rosetta-migrate --config <path>' rewrites it",
    "exclude": "it selected shims of a preset; use '!<source>:<vendor>' in "
    "'shims.install' and 'shims.disabled'; 'llm-rosetta-migrate --config <path>' "
    "rewrites it",
    "remotes": "an explicit archive URL has no manifest equivalent; publish a "
    "manifest source ('shims.sources') or point 'shims.paths' at a local copy",
}


def _refuse_retired_keys(section: Mapping[str, Any]) -> None:
    """Refuse a ``shims`` key the gateway stopped reading in 0.15.0.

    Every retired key the section carries is named in one error, so an operator
    does not meet them one restart at a time.  A key holding nothing (``null``,
    ``[]``, ``{}``) is not a declaration and passes: refusing it would stop a
    gateway for a value that selects and installs nothing.
    """
    found = [
        f"'shims.{key}' was removed in 0.15.0: {advice}"
        for key, advice in RETIRED_KEYS.items()
        if section.get(key)
    ]
    if found:
        raise ValueError("config: " + "; ".join(found))


def _validate_knobs(section: Mapping[str, Any]) -> None:
    """Validate the scalar ``shims`` knobs in place."""
    cache_dir = section.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise ValueError("config: 'shims.cache_dir' must be a string")
    remote_ttl = section.get("remote_ttl")
    if remote_ttl is not None:
        try:
            int(remote_ttl)
        except (TypeError, ValueError):
            raise ValueError("config: 'shims.remote_ttl' must be an integer") from None
    strict = section.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError("config: 'shims.strict' must be a boolean")


@dataclass(frozen=True)
class ShimSources:
    """The ``shims`` section, parsed and validated once.

    One parser feeds the config loader, :func:`apply_shim_sources` and the
    admin save path, so a document a load accepts is a document an apply
    accepts too — they cannot drift apart the way a loader that kept the raw
    mapping and an apply that validated it did (#1330).

    Attributes:
        paths: Local shim directories, anchored to the config file.
        disabled: Shim names to keep out of the registry.
        cache_root: Where fetched manifests and installs are cached, anchored the same way.
        remote_ttl: Seconds a fetched manifest may be reused without a refetch.
        strict: Whether an unknown source or a failed source stops the load.
        declared: Whether the document carried a non-empty ``shims``
            section.  Only this makes the object truthy — a parsed empty
            section still resolves a cache root, which is not a declaration.
        sources: ``shims.sources`` as written.
        install: ``shims.install`` as written.
        manifests: The manifest sources to read and the vendor selectors,
            including the ones the moved vendors add.
        moved: The places in the document that name a shim or a quota probe
            of a vendor that moved out of core (#1660), when the document was
            passed to the parser.
        auto_install: The moved vendors the parser added a
            ``<source>:<vendor>`` selector for, because no declared selector
            named them.  Makes the object truthy: an apply must run even for
            a document with no ``shims`` section.
    """

    paths: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()
    cache_root: str | None = None
    remote_ttl: int = REMOTE_TTL_SECONDS
    strict: bool = False
    declared: bool = False
    sources: tuple[Any, ...] = ()
    install: tuple[str, ...] = ()
    manifests: ManifestPlan = ManifestPlan()
    moved: tuple[MovedReference, ...] = ()
    auto_install: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Whether there is anything to apply: a declared section or a moved vendor."""
        return self.declared or bool(self.auto_install)


def _add_moved_installs(
    sources: ShimSources, refs: list[MovedReference], *, explicit_sources: bool = False
) -> ShimSources:
    """Add a ``<source>:<vendor>`` selector for each moved vendor *refs* name.

    A vendor that a declared ``<vendor>`` selector already names (with no
    qualifier or the right one) is left to it; a tag or ``*`` selector that
    happens to cover it is not known before the manifest is read, so the
    added selector stays, and the install path does not report it as
    unmatched when another selector installed the vendor.

    The catalogue source (``contrib``) is added only when the document has no
    ``shims.sources`` of its own.  An explicit list is the operator's choice
    of where to fetch from — a mirror, or no egress at all — so only the
    selector is added: a declared source that publishes the vendor (a mirror
    whose manifest calls itself ``contrib``) still serves it, and when none
    does, the apply's WARNING says the vendor lives in ``contrib``.
    """
    if not refs:
        return sources
    specs = list(sources.manifests.sources)
    selectors = list(sources.manifests.selectors)
    added: list[str] = []
    for ref in refs:
        if ref.vendor in added:
            continue
        if any(
            s.kind == "vendor"
            and not s.negate
            and s.value == ref.vendor
            and s.applies_to(ref.source)
            for s in selectors
        ):
            continue
        catalogue = get_shim_source(ref.source)
        if catalogue is None:
            continue
        spec = source_spec_from_catalogue(catalogue)
        if not explicit_sources and all(spec.url != other.url for other in specs):
            specs.append(spec)
        selectors.append(
            parse_selector(f"{ref.source}:{ref.vendor}", origin=f"moved:{ref.vendor}")
        )
        added.append(ref.vendor)
    return replace(
        sources,
        manifests=ManifestPlan(sources=tuple(specs), selectors=tuple(selectors)),
        moved=tuple(refs),
        auto_install=tuple(added),
    )


def parse_shims_section(
    section: Mapping[str, Any] | None,
    config_path: str | None = None,
    *,
    on_unknown: str | None = None,
    document: Mapping[str, Any] | None = None,
) -> ShimSources:
    """Parse and validate a ``shims`` section.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, the anchoring base for relative
            ``paths`` and ``cache_dir`` entries.
        on_unknown: ``"raise"`` or ``"skip"`` for a ``shims.sources`` id this
            build does not ship.  ``None`` reads it from ``shims.strict``: a
            strict document raises, a tolerant one skips with a warning.
        document: The whole config document.  When given, every vendor that
            moved out of core and that a provider names is added to the
            install selection (:func:`_add_moved_installs`), with or without
            a ``shims`` section.

    Returns:
        The parsed section.  Falsy when the document carried none and needs
        no moved vendor.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    config_dir = os.path.dirname(config_path) if config_path else None
    refs = moved_references(document) if document is not None else []
    if not section:
        return _add_moved_installs(
            ShimSources(cache_root=_cache_root(None, config_dir)), refs
        )
    if not isinstance(section, Mapping):
        raise ValueError("config: 'shims' must be an object")

    mode = on_unknown
    if mode is None:
        mode = "raise" if section.get("strict") is True else "skip"
    _refuse_retired_keys(section)
    dirs = _resolve_paths(section.get("paths") or [], config_dir)
    manifests = _resolve_manifest_plan(section, on_unknown=mode)
    _validate_knobs(section)
    parsed = ShimSources(
        sources=tuple(section.get("sources") or ()),
        install=tuple(section.get("install") or ()),
        manifests=manifests,
        paths=tuple(dirs),
        disabled=tuple(resolve_disabled(section)),
        cache_root=_cache_root(section.get("cache_dir"), config_dir),
        remote_ttl=_int_setting(section, "remote_ttl", REMOTE_TTL_SECONDS),
        strict=bool(section.get("strict", False)),
        declared=True,
    )
    return _add_moved_installs(
        parsed, refs, explicit_sources=section.get("sources") is not None
    )


def resolve_source_config(
    section: Mapping[str, Any] | None,
    config_path: str | None,
    *,
    on_unknown: str = "raise",
) -> tuple[list[str], str]:
    """Resolve a ``shims`` section into absolute dirs and the cache root.

    A thin view over :func:`parse_shims_section`, kept for callers that want
    the two values rather than the object.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, used as the anchoring base.
        on_unknown: What to do with a ``shims.sources`` id this build does not
            ship; defaults to raising, because every caller of this view is a
            path where the caller can still correct what it sent.

    Returns:
        ``(dirs, cache_root)``.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    sources = parse_shims_section(section, config_path, on_unknown=on_unknown)
    return list(sources.paths), sources.cache_root or _cache_root(None, None)


def resolve_disabled(section: Mapping[str, Any] | None) -> list[str]:
    """Return the shim names ``shims.disabled`` keeps out of the registry.

    The switch is a declaration, not a deletion: the shim's source stays
    declared, so enabling the name again puts the same definition back.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).

    Returns:
        The disabled names, in declaration order.

    Raises:
        ValueError: If the value is not an array of strings.
    """
    if not isinstance(section, Mapping):
        return []
    raw = section.get("disabled") or []
    if not isinstance(raw, list) or any(not isinstance(n, str) for n in raw):
        raise ValueError("config: 'shims.disabled' must be an array of strings")
    return [name for name in raw if name]


def describe_shim_sources(report: ShimLoadReport | None) -> list[str]:
    """One line per declared remote source, naming the leg it loaded from.

    Local ``paths`` entries are left out: they have no legs to choose between.
    The lines are what the startup output prints under the banner, next to the
    matching log line.

    Args:
        report: The applied load report, or ``None`` when no section applied.

    Returns:
        The lines, in declaration order.
    """
    if report is None:
        return []
    lines: list[str] = []
    for status in report.sources:
        if status.kind != "remote":
            continue
        where = status.location
        if status.vendor or status.skipped:
            version = (status.release or {}).get("version")
            where = f"{status.name}" + (f" {version}" if version else "")
            if status.skipped:
                lines.append(f"{where}: skipped ({status.skipped})")
                continue
        if not status.ok:
            lines.append(f"{where}: failed")
            continue
        leg = status.leg or "primary"
        origin = "cache" if status.from_cache else "network"
        if status.stale:
            origin += ", stale"
        lines.append(f"{where}: loaded via {leg} ({origin})")
    return lines


def apply_shim_sources(
    raw: Mapping[str, Any],
    config_path: str | None = None,
    *,
    strict: bool | None = None,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
    force_refresh: bool = False,
) -> ShimLoadReport:
    """Apply ``raw['shims']`` to the global shim registry.

    Always performs a full rebuild, so calling it also *drops* shims whose
    source was removed from the config.  Callers that want to avoid a rebuild
    when nothing changed should compare :func:`shims_signature` first.

    Args:
        raw: The parsed config mapping.
        config_path: Path to the config file (for path anchoring).
        strict: Override ``shims.strict``; ``None`` reads it from the section.
        http_get: Injectable HTTP getter (tests).
        proxy: Upstream proxy; defaults to ``server.proxy``.
        force_refresh: Fetch every manifest regardless of its age.

    Returns:
        The :class:`ShimLoadReport` for the rebuild.
    """
    section = raw.get("shims") if isinstance(raw, Mapping) else None
    if proxy is None and isinstance(raw, Mapping):
        server = raw.get("server")
        if isinstance(server, Mapping):
            proxy = server.get("proxy")
    return apply_shim_section(
        section,
        config_path,
        strict=strict,
        http_get=http_get,
        proxy=proxy,
        force_refresh=force_refresh,
        document=raw if isinstance(raw, Mapping) else None,
    )


def apply_shim_section(
    section: ShimSources | Mapping[str, Any] | None,
    config_path: str | None = None,
    *,
    strict: bool | None = None,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
    force_refresh: bool = False,
    document: Mapping[str, Any] | None = None,
) -> ShimLoadReport:
    """Apply a parsed ``shims`` section to the registry.

    Accepts the parsed object (what the config loader stores) or a raw
    mapping, which is parsed with the same validator the loader uses — so a
    section that reached this far has already been checked, and the two
    paths cannot disagree about its shape.

    A *strict* argument given here also governs the parse: an unknown source
    id is refused when it is ``True``, skipped when it is ``False``, and read
    from ``shims.strict`` when it is ``None``.  A section that was already
    parsed keeps the verdict the loader reached.

    A raw *section* is parsed with *document* (the whole config), so the
    vendors that moved out of core and that a provider names are installed
    too; a parsed section already carries them.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    if isinstance(section, ShimSources):
        sources = section
    else:
        sources = parse_shims_section(
            section,
            config_path,
            on_unknown=(None if strict is None else ("raise" if strict else "skip")),
            document=document,
        )
    if strict is None:
        strict = sources.strict
    report = rebuild_shims(
        list(sources.paths),
        cache_root=sources.cache_root or _cache_root(None, None),
        strict=bool(strict),
        http_get=http_get,
        ttl=sources.remote_ttl,
        proxy=proxy,
        disabled=list(sources.disabled),
        manifests=sources.manifests,
        force_refresh=force_refresh,
    )
    if report.sources:
        logger.info(
            "Shim sources: %d source(s), %d shim(s), %d failure(s)",
            len(report.sources),
            report.total_shims,
            len(report.failures()),
        )
    _report_moved(sources, report)
    return report


#: Moved vendors whose automatic install was already announced in this process.
_ANNOUNCED_MOVED: set[str] = set()


def _report_moved(sources: ShimSources, report: ShimLoadReport) -> None:
    """Announce each automatic install once, and warn about what is missing.

    INFO, once per vendor per process: the config names a shim that moved out
    of core and the gateway installs its vendor without a config edit.
    WARNING, on every apply that leaves one missing: the moved vendors and the
    shims or probes the config names that are not registered (a source that
    could not be reached on a cold cache, or a manifest without the vendor),
    so the operator sees why those providers are disabled.
    """
    if not sources.moved:
        return
    from llm_rosetta.api import get_shim, list_shims, retired_names

    for vendor in sources.auto_install:
        if vendor in _ANNOUNCED_MOVED:
            continue
        _ANNOUNCED_MOVED.add(vendor)
        ref = next(r for r in sources.moved if r.vendor == vendor)
        logger.info(
            "Config names %r (%s), whose vendor %r moved from core to the %r "
            "shim source; installing it from there (shims.install += %r is "
            "implied, no config edit needed)",
            ref.name,
            ref.where,
            vendor,
            ref.source,
            f"{ref.source}:{vendor}",
        )
    probes = {p.id for shim in list_shims() for p in shim.quota_probes}
    missing: list[MovedReference] = []
    for ref in sources.moved:
        if retired_names.moved_probe_vendor(ref.name) is not None:
            if ref.name not in probes:
                missing.append(ref)
            continue
        name = retired_names.replacement_for(ref.name, retired_names.SHIM) or ref.name
        if get_shim(name) is None:
            missing.append(ref)
    if not missing:
        return
    vendors = sorted({ref.vendor for ref in missing})
    homes = sorted({ref.source for ref in missing})
    declared = {spec.key for spec in sources.manifests.sources}
    elsewhere = [home for home in homes if home not in declared]
    failures = "; ".join(
        f"{status.location}: {status.error}" for status in report.failures()
    )
    logger.warning(
        "Shim vendor(s) %s moved from core to the %s shim source and could not "
        "be installed; unavailable: %s. Providers that use them are disabled "
        "until the source answers%s%s",
        ", ".join(vendors),
        ", ".join(homes),
        ", ".join(f"{ref.name!r} ({ref.where})" for ref in missing),
        f" (source failures: {failures})" if failures else "",
        (
            f". The declared shims.sources do not include {', '.join(map(repr, elsewhere))}, "
            "where they are published; add it, or a mirror that publishes them"
        )
        if elsewhere
        else "",
    )


def _anchor(path: str, config_dir: str | None) -> str:
    """Anchor a relative path to the config directory."""
    if os.path.isabs(path) or not config_dir:
        return path
    return os.path.join(config_dir, path)


def _cache_root(cache_dir: str | None, config_dir: str | None) -> str:
    """Resolve the manifest and install cache directory (config-adjacent)."""
    if cache_dir:
        return _anchor(cache_dir, config_dir)
    if config_dir:
        return os.path.join(config_dir, DEFAULT_CACHE_DIRNAME)
    return str(Path("~/.cache/llm-rosetta/shims").expanduser())


def _int_setting(section: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer setting, falling back to *default*."""
    value = section.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("config: 'shims.%s' must be an integer; using %d", key, default)
        return default


__all__ = [
    "ShimSources",
    "apply_shim_section",
    "apply_shim_sources",
    "describe_shim_sources",
    "parse_shims_section",
    "resolve_source_config",
    "shims_signature",
]
