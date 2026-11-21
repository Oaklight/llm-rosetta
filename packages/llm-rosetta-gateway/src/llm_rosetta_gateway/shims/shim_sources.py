"""Bridge between the gateway config and declarative shim sources.

The top-level ``shims`` config section declares extra local directories and
remote archives to load provider shims from.  This module resolves that section
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import (
    SHIM_SOURCE_PACKAGES,
    ShimLoadReport,
    get_source_package,
    rebuild_shims,
)
from .catalog import (
    DEFAULT_SOURCES,
    get_shim_source,
    package_repo_keys,
    repo_key,
)
from .manifest_sources import (
    InstallSelector,
    ManifestPlan,
    ManifestSourceSpec,
    parse_selector,
    parse_source_entry,
    source_spec_from_catalogue,
)
from .moved import MovedReference, moved_references
from .removals import ShimRemovalAdvisory, detect_shim_removals
from .sources import (
    DEFAULT_CACHE_DIRNAME,
    REMOTE_TTL_SECONDS,
    HttpGet,
    normalize_subpath,
    normalize_mirror,
)

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
    and expanded its presets, which would make the two disagree.

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


def _known_preset_keys() -> str:
    """The preset keys this build ships, for an error or a warning."""
    return ", ".join(entry.key for entry in SHIM_SOURCE_PACKAGES)


def _unknown_preset_error(key: str) -> ValueError:
    """The one error for a preset key this build does not ship.

    One constructor for both the parse path and the already-parsed path, so
    the two cannot word the same refusal differently.
    """
    return ValueError(
        f"config: unknown shim preset {key!r}; known presets: {_known_preset_keys()}"
    )


def _resolve_presets(
    raw_presets: list[Any],
    *,
    include: Mapping[str, tuple[str, ...]] | None = None,
    exclude: Mapping[str, tuple[str, ...]] | None = None,
    on_unknown: str = "raise",
) -> list[dict[str, Any]]:
    """Expand ``shims.presets`` keys into their catalogue remotes.

    Duplicates collapse to the first mention, so the same archive is not
    listed twice.  A key that carries a selection in ``shims.include`` /
    ``shims.exclude`` takes it with it: the preset key is the selection's only
    name, so folding the two here is what keeps the config free of a second
    lookup the operator would have to keep in step.

    Args:
        raw_presets: The declared keys.
        include: The ``shims.include`` selections, keyed by preset.
        exclude: The ``shims.exclude`` selections, keyed by preset.
        on_unknown: ``"raise"`` (the admin save path, where the operator can
            correct the body) or ``"skip"`` (startup, where a key this build
            does not ship must not keep the gateway from running — it may come
            from a newer build, and the panel that would fix it is part of the
            same process).

    Raises:
        ValueError: On a non-string key, or an unknown key when *on_unknown*
            is ``"raise"``.
    """
    include = include or {}
    exclude = exclude or {}
    remotes: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for key in raw_presets:
        if not isinstance(key, str) or not key:
            raise ValueError(
                "config: 'shims.presets' entries must be non-empty strings"
            )
        package = get_source_package(key)
        if package is None:
            if on_unknown == "skip":
                logger.warning(
                    "config: unknown shim preset %r is skipped; this build ships: %s",
                    key,
                    _known_preset_keys(),
                )
                continue
            raise _unknown_preset_error(key)
        remote = package.remote
        url = remote.get("url")
        if url and url not in seen_urls:
            if include.get(key):
                remote = {**remote, "include": list(include[key])}
            if exclude.get(key):
                remote = {**remote, "exclude": list(exclude[key])}
            remotes.append(remote)
            seen_urls.add(url)
    return remotes


def _resolve_selection_map(
    value: Any, *, where: str, on_unknown: str
) -> dict[str, tuple[str, ...]]:
    """Parse one keyed selection map — ``shims.include`` or ``shims.exclude``.

    The key is a preset key, so a selection cannot attach itself to a source the
    operator did not name, and an unknown key is refused where the save path can
    still correct it (the rule ``shims.presets`` already follows).

    Args:
        value: The declared mapping, or ``None``.
        where: The config key, for the error and the log line.
        on_unknown: ``"raise"`` or ``"skip"`` for a key this build does not ship.

    Returns:
        The selected patterns per preset key, in declaration order.

    Raises:
        ValueError: On a non-mapping value, a non-string key, an unknown key
            when *on_unknown* is ``"raise"``, or a malformed pattern list.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config: '{where}' must be an object keyed by preset")
    selections: dict[str, tuple[str, ...]] = {}
    for key, patterns in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"config: '{where}' keys must be preset names")
        if get_source_package(key) is None:
            if on_unknown == "skip":
                logger.warning(
                    "config: '%s' names unknown preset %r and is skipped; "
                    "this build ships: %s",
                    where,
                    key,
                    _known_preset_keys(),
                )
                continue
            raise ValueError(
                f"config: '{where}' names unknown shim preset {key!r}; "
                f"known presets: {_known_preset_keys()}"
            )
        if isinstance(patterns, str):
            raise ValueError(
                f"config: '{where}.{key}' must be an array of non-empty shim names"
            )
        if not isinstance(patterns, (list, tuple)) or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise ValueError(
                f"config: '{where}.{key}' must be an array of non-empty shim names"
            )
        selections[key] = tuple(patterns)
    return selections


def _warn_selection_without_preset(
    raw_presets: list[Any],
    include: Mapping[str, tuple[str, ...]],
    exclude: Mapping[str, tuple[str, ...]],
) -> None:
    """Warn about a selection keyed to a preset the document does not declare.

    Not an error — the document may be mid-edit, and the key is meaningful the
    moment the preset is added — but silently doing nothing is exactly the
    failure a subset selection exists to avoid.
    """
    declared = {key for key in raw_presets if isinstance(key, str)}
    for key in (*include, *exclude):
        if key not in declared:
            logger.warning(
                "config: 'shims' selects from preset %r, which 'shims.presets' "
                "does not declare; the selection has no effect",
                key,
            )


@dataclass
class PresetRewrite:
    """``shims.presets`` spelled as ``shims.install`` entries (#1660).

    Attributes:
        install: The entries that replace the presets, in declaration order:
            the preset's own selector (``contrib:tag:tier:free``), or one
            ``<source>:<vendor>`` per vendor an ``include`` keeps, plus an
            exclusion ``!<source>:<vendor>`` per vendor an ``exclude`` drops
            whole.
        disabled: Shim names to add to ``shims.disabled``: what a selection
            dropped from a vendor it keeps.  The vendor is the install unit,
            the shim the enable unit, so a partial selection becomes a switch.
        sources: The source keys the entries need in a declared
            ``shims.sources`` list.
        converted: The preset keys this rewrite replaces.
        unconvertible: ``{key: why}`` for a key that stays a preset: unknown
            to this build, with no manifest source, pinned by an explicit
            ``shims.remotes`` entry (the remote path is the operator's choice),
            or one whose selection converts to nothing (an ``include`` that
            matches none of its vendors) — that is reported, never deleted.
    """

    install: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    unconvertible: dict[str, str] = field(default_factory=dict)
    #: Sentences for the operator that are not refusals (a source left
    #: undeclared because a declared URL may already be it).
    notes: list[str] = field(default_factory=list)


def _vendor_of_shim(shim: str, vendors: Sequence[str]) -> str | None:
    """The package vendor a shim belongs to (``alcf_metis--…`` → ``alcf``)."""
    head = shim.split("--", 1)[0]
    best = None
    for vendor in vendors:
        if (head == vendor or head.startswith(vendor + "_")) and (
            best is None or len(vendor) > len(best)
        ):
            best = vendor
    return best


def presets_as_install(
    presets: Sequence[Any],
    include: Mapping[str, Sequence[str]] | None = None,
    exclude: Mapping[str, Sequence[str]] | None = None,
    remotes: Any = None,
) -> PresetRewrite:
    """Spell legacy ``shims.presets`` as explicit ``shims.install`` entries.

    The one home of the rewrite: ``llm-rosetta-migrate --config`` edits the
    file with it, and the admin panel runs it before removing one vendor a
    preset selects.

    A preset with no selection becomes its mapped selector.  An ``include``
    becomes one explicit ``<source>:<vendor>`` per vendor that keeps a shim;
    an ``exclude`` keeps the selector and adds ``!<source>:<vendor>`` for each
    vendor it drops whole.  Shims a selection drops from a vendor it keeps go
    to ``shims.disabled``.  The shim lists are the catalogue's, so a shim a
    package added after this build installs enabled.

    Args:
        presets: ``shims.presets`` as written.
        include: ``shims.include`` keyed by preset.
        exclude: ``shims.exclude`` keyed by preset.
        remotes: ``shims.remotes`` as written (a remote that pins a preset's
            repository keeps that preset on the remote path).

    Returns:
        The rewrite.
    """
    include = include or {}
    exclude = exclude or {}
    keys = [k for k in presets if isinstance(k, str) and k]
    pinned = _pinned_preset_keys(keys, remotes)
    out = PresetRewrite()
    for key in dict.fromkeys(keys):
        package = get_source_package(key)
        why = _unconvertible(package, key in pinned)
        if why or package is None or package.source is None:
            out.unconvertible[key] = why or "unknown preset"
            continue
        entries, disabled = _preset_entries(
            package, tuple(include.get(key) or ()), tuple(exclude.get(key) or ())
        )
        if not entries:
            out.unconvertible[key] = (
                f"converts to nothing: no vendor of {package.label} keeps a shim "
                "under its include/exclude; fix the selection or drop the preset"
            )
            continue
        out.install.extend(e for e in entries if e not in out.install)
        out.disabled.extend(n for n in disabled if n not in out.disabled)
        if package.source not in out.sources:
            out.sources.append(package.source)
        out.converted.append(key)
    return out


def _unconvertible(package: Any, pinned: bool) -> str | None:
    """Why a preset key stays a preset, or ``None``."""
    if package is None:
        return "unknown preset"
    if not package.source or not package.install:
        return "no manifest source"
    if pinned:
        return "pinned by an explicit shims.remotes entry"
    return None


def _preset_entries(
    package: Any, inc: tuple[str, ...], exc: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """One preset's install entries and the shims its selection switches off."""
    from fnmatch import fnmatch

    def kept(shim: str) -> bool:
        if inc and not any(fnmatch(shim, p) for p in inc):
            return False
        return not any(fnmatch(shim, p) for p in exc)

    by_vendor: dict[str, list[str]] = {v: [] for v in package.vendors}
    for shim in package.shims:
        vendor = _vendor_of_shim(shim, package.vendors)
        if vendor is not None:
            by_vendor[vendor].append(shim)
    source = package.source
    broad = parse_selector(package.install).kind != "vendor" and not inc
    entries = [package.install] if broad else []
    disabled: list[str] = []
    for vendor, shims in by_vendor.items():
        keep = [shim for shim in shims if kept(shim)]
        if broad and shims and not keep:
            # The broad selector stays; a vendor the exclude drops whole is
            # excluded.
            entries.append(f"!{source}:{vendor}")
            continue
        if not broad:
            if not keep and (shims or inc):
                continue
            # One explicit entry per vendor that keeps a shim.
            entries.append(f"{source}:{vendor}")
        # A shim dropped from a vendor that stays is switched off.
        disabled.extend(shim for shim in shims if shim not in keep)
    return entries, disabled


def sources_to_declare(
    declared: Sequence[Any], needed: Sequence[str]
) -> tuple[list[str], list[str]]:
    """The source keys a declared ``shims.sources`` list still lacks.

    A declared entry names a source by its id (``"contrib"``) or by a manifest
    URL, as a string or as ``{"url": ...}``; the catalogue's primary and
    fallback URLs of a source count as that source.  A URL the catalogue does
    not know may be a mirror of a needed source, so a key is then not added
    (a second declaration would fetch the source twice) and is returned as
    *unsure* for the caller to report.

    Returns:
        ``(add, unsure)``.
    """
    known: set[str] = set()
    foreign = False
    by_url = {}
    for source in (get_shim_source(k) for k in DEFAULT_SOURCES):
        if source is not None:
            by_url[source.manifest_url] = source.key
            by_url[source.fallback_url] = source.key
    for entry in declared:
        url = entry.get("url") if isinstance(entry, Mapping) else entry
        if not isinstance(url, str):
            continue
        if "://" not in url:
            known.add(url)
        elif url in by_url:
            known.add(by_url[url])
        else:
            foreign = True
    missing = [k for k in needed if k not in known]
    return ([], missing) if foreign else (missing, [])


def _unsure_note(key: str) -> str:
    return (
        f"shims.sources declares a manifest URL this build does not know; "
        f"{key!r} was not added in case that URL is already it - add it if not"
    )


def rewrite_presets_section(section: Mapping[str, Any]) -> tuple[dict, PresetRewrite]:
    """The ``shims`` section with its convertible presets spelled as installs.

    Converted keys leave ``presets``, ``include`` and ``exclude``; the
    entries join ``install`` and ``disabled`` (no duplicates), and a declared
    ``sources`` list gains the source keys it lacks (an undeclared one reads
    the defaults, which name both).  A key that cannot be converted stays.

    Args:
        section: The ``shims`` mapping as written.

    Returns:
        ``(new_section, rewrite)``; *new_section* is a new mapping.
    """
    out = dict(section)
    rewrite = presets_as_install(
        section.get("presets") or [],
        section.get("include") if isinstance(section.get("include"), Mapping) else {},
        section.get("exclude") if isinstance(section.get("exclude"), Mapping) else {},
        section.get("remotes"),
    )
    if not rewrite.converted:
        return out, rewrite
    gone = set(rewrite.converted)
    left = [k for k in section.get("presets") or [] if k not in gone]
    for key in ("include", "exclude"):
        sel = section.get(key)
        if isinstance(sel, Mapping):
            out[key] = {k: v for k, v in sel.items() if k not in gone}
    out["presets"] = left
    install = list(section.get("install") or [])
    out["install"] = install + [e for e in rewrite.install if e not in install]
    disabled = list(section.get("disabled") or [])
    out["disabled"] = disabled + [n for n in rewrite.disabled if n not in disabled]
    declared = section.get("sources")
    if isinstance(declared, list):
        add, unsure = sources_to_declare(declared, rewrite.sources)
        out["sources"] = declared + add
        rewrite.notes.extend(_unsure_note(k) for k in unsure)
    for key in ("presets", "include", "exclude", "disabled"):
        if not out.get(key):
            out.pop(key, None)
    return out, rewrite


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


def _remote_url_key(url: Any) -> str:
    """Normalize a remote URL so a preset and a spelling of it still match.

    A trailing slash and a ``.git`` suffix address the same repository; the
    catalogue and an operator's config rarely spell it the same way.
    """
    if not isinstance(url, str):
        return ""
    key = url.strip().rstrip("/")
    if key.endswith(".git"):
        key = key[:-4].rstrip("/")
    return key


def _current_repo_url(url: Any) -> Any:
    """The catalogue's current URL for a renamed package repository.

    ``llm-rosetta-shim-argo`` became ``llm-rosetta-shim-argonne`` and
    ``llm-rosetta-shim-free`` became ``llm-rosetta-shim-contrib``; an explicit
    remote that still spells the old name is the same package.
    """
    if not isinstance(url, str):
        return url
    key = repo_key(url)
    for package in SHIM_SOURCE_PACKAGES:
        if key in package_repo_keys(package):
            return package.repo
    return url


def _pinned_preset_keys(raw_presets: list[Any], raw_remotes: Any) -> set[str]:
    """Preset keys an explicit ``shims.remotes`` entry pins to its repository.

    An operator who wrote a remote for a package's repository (to pin a
    ``ref``, say) keeps that package on the remote path: the manifest mapping
    must not override a pin the operator chose.
    """
    if not isinstance(raw_remotes, list):
        return set()
    explicit = {
        repo_key(entry.get("url"))
        for entry in raw_remotes
        if isinstance(entry, Mapping) and isinstance(entry.get("url"), str)
    }
    pinned: set[str] = set()
    for key in raw_presets:
        package = get_source_package(key) if isinstance(key, str) else None
        if package is not None and package_repo_keys(package) & explicit:
            pinned.add(key)
    return pinned


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
    section: Mapping[str, Any],
    raw_presets: list[Any],
    *,
    include: Mapping[str, tuple[str, ...]],
    exclude: Mapping[str, tuple[str, ...]],
    on_unknown: str,
) -> tuple[ManifestPlan, list[Any]]:
    """Build the manifest plan, and return the presets still loaded as remotes.

    ``shims.install`` selectors read ``shims.sources`` (default
    :data:`~llm_rosetta_gateway.shims.catalog.DEFAULT_SOURCES` when only
    ``install`` is declared).  A legacy ``shims.presets`` key whose package
    names a manifest source becomes one more selector on that source,
    carrying its ``shims.include`` / ``shims.exclude`` entry and its old
    remote as the fallback for a source that has never answered — so a
    config written for the per-package repositories keeps working unedited.

    Returns:
        ``(plan, legacy_presets)``.
    """
    raw_install = section.get("install") or []
    if not isinstance(raw_install, list):
        raise ValueError("config: 'shims.install' must be an array")
    selectors: list[InstallSelector] = [parse_selector(x) for x in raw_install]
    raw_sources = section.get("sources")
    if raw_sources is None:
        raw_sources = list(DEFAULT_SOURCES) if selectors else []
    specs = _source_specs(raw_sources, on_unknown)

    legacy = _map_presets(
        raw_presets,
        _pinned_preset_keys(raw_presets, section.get("remotes")),
        include=include,
        exclude=exclude,
        selectors=selectors,
        specs=specs,
    )
    _check_qualifiers(selectors, specs, on_unknown)
    return ManifestPlan(sources=tuple(specs), selectors=tuple(selectors)), legacy


def _map_presets(
    raw_presets: list[Any],
    pinned: set[str],
    *,
    include: Mapping[str, tuple[str, ...]],
    exclude: Mapping[str, tuple[str, ...]],
    selectors: list[InstallSelector],
    specs: list[ManifestSourceSpec],
) -> list[Any]:
    """Append a selector (and its source) per mapped preset; return the rest.

    Args:
        raw_presets: The declared ``shims.presets`` keys.
        pinned: Keys an explicit remote pins to the remote path.
        include: ``shims.include`` keyed by preset.
        exclude: ``shims.exclude`` keyed by preset.
        selectors: The selector list, extended in place.
        specs: The source list, extended in place.

    Returns:
        The keys that still load as a remote, in declaration order.
    """
    legacy: list[Any] = []
    mapped: set[str] = set()
    for key in raw_presets:
        package = get_source_package(key) if isinstance(key, str) else None
        source = get_shim_source(package.source) if package and package.source else None
        if key in mapped:
            continue
        if package is None or source is None or key in pinned or not package.install:
            legacy.append(key)
            continue
        mapped.add(key)
        fallback = dict(package.remote)
        if include.get(key):
            fallback["include"] = list(include[key])
        if exclude.get(key):
            fallback["exclude"] = list(exclude[key])
        selectors.append(
            parse_selector(
                package.install,
                include=tuple(include.get(key, ())),
                exclude=tuple(exclude.get(key, ())),
                origin=f"presets:{key}",
                fallback_remote=fallback,
                expected=tuple(package.vendors),
            )
        )
        spec = source_spec_from_catalogue(source)
        if all(spec.url != other.url for other in specs):
            specs.append(spec)
    return legacy


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


def _resolve_remotes(
    raw_remotes: Any, remotes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply explicit remotes on top of the preset-expanded *remotes*.

    An explicit entry with a preset's URL is **merged onto** that preset, with
    its own keys winning: that is what lets the documented ``presets: ["argo"]``
    plus ``{url: ..., ref: "v1"}`` pattern pin a ref without dropping the
    catalogue's ``path`` (and any ``token_env``).  Explicit entries keep their
    previous append-only behaviour among themselves.
    """
    if not isinstance(raw_remotes, list):
        raise ValueError("config: 'shims.remotes' must be an array")
    preset_keys = {
        _remote_url_key(remote.get("url")) for remote in remotes if remote.get("url")
    }
    for entry in raw_remotes:
        if not isinstance(entry, Mapping):
            raise ValueError("config: 'shims.remotes' entries must be objects")
        spec = dict(entry)
        # A renamed repository's old URL still names the preset's package.
        key = _remote_url_key(_current_repo_url(spec.get("url")))
        preset = None
        if key and key in preset_keys:
            preset = next(
                item for item in remotes if _remote_url_key(item.get("url")) == key
            )
        if preset is not None:
            # Layer the explicit keys over the preset instead of replacing it:
            # a pin (`ref`/`sha256`/`refresh`) must not lose `path`.
            spec = {**preset, **spec}
            remotes = [item for item in remotes if item is not preset]
            preset_keys.discard(key)
        remotes.append(spec)
    return remotes


def _validate_remote_specs(remotes: Sequence[Mapping[str, Any]]) -> None:
    """Validate the per-remote fields the config layer owns.

    The loader checks the primary URL shape and the cache paths at fetch time.
    A ``path`` and a ``mirror`` are checked here, at parse, because they are
    declarations this layer can judge without a network: startup and the admin
    save path share this parser, so a bad value is refused before it is
    persisted or served.  ``path`` is normalized by the same rule the loader
    applies (``mirror.path`` included) — refusing it here rather than at the
    fetch keeps a document a load rejected from ever being written.

    Raises:
        ValueError: If a remote's ``path`` or ``mirror`` declaration is invalid.
    """
    for index, entry in enumerate(remotes):
        for key in ("include", "exclude"):
            patterns = entry.get(key)
            if patterns is None:
                continue
            if not isinstance(patterns, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern for pattern in patterns
            ):
                raise ValueError(
                    f"config: 'shims.remotes[{index}].{key}' must be an array of "
                    "non-empty shim names"
                )
        try:
            normalize_subpath(entry.get("path"), where="'path'")
        except ValueError as exc:
            raise ValueError(
                f"config: 'shims.remotes[{index}].path' is invalid: {exc}"
            ) from None
        try:
            mirror = normalize_mirror(entry.get("mirror"))
        except ValueError as exc:
            raise ValueError(
                f"config: 'shims.remotes[{index}].mirror' is invalid: {exc}"
            ) from None
        if mirror is not None and entry.get("sha256") and mirror[1] is None:
            # The mirror is the path an attacker can force (make the primary
            # unreachable); without its own digest it would serve bytes the
            # primary's pin never approved.
            raise ValueError(
                f"config: 'shims.remotes[{index}].mirror' needs its own 'sha256' "
                "when the remote pins its primary: the two legs serve different "
                "bytes, and an unpinned mirror would bypass the pin"
            )


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
        presets: The declared catalogue keys, as written.
        remotes: The remotes to load: explicit entries, and the presets that
            still load as a remote (a package an explicit remote pins, or one
            with no manifest source), merged over a preset with the same URL.
        paths: Local shim directories, anchored to the config file.
        disabled: Shim names to keep out of the registry.
        cache_root: Where remote archives are cached, anchored the same way.
        remote_ttl: Seconds a remote archive may be reused without a refetch.
        strict: Whether an unknown preset or a failed source stops the load.
        declared: Whether the document carried a non-empty ``shims``
            section.  Only this makes the object truthy — a parsed empty
            section still resolves a cache root, which is not a declaration.
        sources: ``shims.sources`` as written.
        install: ``shims.install`` as written.
        manifests: The manifest sources to read and the vendor selectors,
            including the ones the legacy presets map to (#1660) and the
            ones the moved vendors add.
        moved: The places in the document that name a shim or a quota probe
            of a vendor that moved out of core (#1660), when the document was
            passed to the parser.
        auto_install: The moved vendors the parser added a
            ``<source>:<vendor>`` selector for, because no declared selector
            named them.  Makes the object truthy: an apply must run even for
            a document with no ``shims`` section.
    """

    presets: tuple[str, ...] = ()
    remotes: tuple[Mapping[str, Any], ...] = ()
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
        on_unknown: ``"raise"`` or ``"skip"`` for a ``presets`` key this build
            does not ship.  ``None`` reads it from ``shims.strict``: a strict
            document raises, a tolerant one skips with a warning (see
            :func:`_resolve_presets`).
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
    dirs = _resolve_paths(section.get("paths") or [], config_dir)
    raw_presets = section.get("presets") or []
    if not isinstance(raw_presets, list):
        raise ValueError("config: 'shims.presets' must be an array")
    include = _resolve_selection_map(
        section.get("include"), where="shims.include", on_unknown=mode
    )
    exclude = _resolve_selection_map(
        section.get("exclude"), where="shims.exclude", on_unknown=mode
    )
    _warn_selection_without_preset(raw_presets, include, exclude)
    manifests, legacy_presets = _resolve_manifest_plan(
        section, raw_presets, include=include, exclude=exclude, on_unknown=mode
    )
    remotes = _resolve_remotes(
        section.get("remotes") or [],
        _resolve_presets(
            legacy_presets, include=include, exclude=exclude, on_unknown=mode
        ),
    )
    _validate_remote_specs(remotes)
    _validate_knobs(section)
    parsed = ShimSources(
        sources=tuple(section.get("sources") or ()),
        install=tuple(section.get("install") or ()),
        manifests=manifests,
        presets=tuple(raw_presets),
        remotes=tuple(remotes),
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
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Resolve a ``shims`` section into absolute dirs, remotes and cache root.

    A thin view over :func:`parse_shims_section`, kept for callers that want
    the three values rather than the object.

    Args:
        section: The ``shims`` mapping (or ``None``/empty).
        config_path: Path to the config file, used as the anchoring base.
        on_unknown: What to do with a ``presets`` key this build does not ship;
            defaults to raising, because every caller of this view is a path
            where the caller can still correct what it sent.

    Returns:
        ``(dirs, remotes, cache_root)``.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape.
    """
    sources = parse_shims_section(section, config_path, on_unknown=on_unknown)
    return (
        list(sources.paths),
        [dict(remote) for remote in sources.remotes],
        sources.cache_root or _cache_root(None, None),
    )


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
        force_refresh: Fetch every remote regardless of its cache/refresh mode.

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

    A *strict* argument given here also governs the parse: an unknown preset
    is refused when it is ``True``, skipped when it is ``False``, and read from
    ``shims.strict`` when it is ``None``.  A section that was already parsed
    cannot be re-judged — the loader parsed it with the document's own
    ``strict`` — so ``strict=True`` over such a section **raises** when it
    carries a preset this build does not ship, rather than reporting success
    for a rule it never applied; ``strict=False`` accepts what the parse
    already skipped.

    A raw *section* is parsed with *document* (the whole config), so the
    vendors that moved out of core and that a provider names are installed
    too; a parsed section already carries them.

    Raises:
        ValueError: If the section or one of its fields has the wrong shape,
            or if a parsed section contradicts an explicit ``strict=True``.
    """
    if isinstance(section, ShimSources):
        sources = section
        if strict is True:
            unknown = [
                key for key in sources.presets if get_source_package(key) is None
            ]
            if unknown:
                raise _unknown_preset_error(unknown[0])
    else:
        sources = parse_shims_section(
            section,
            config_path,
            on_unknown=(None if strict is None else ("raise" if strict else "skip")),
            document=document,
        )
    if strict is None:
        strict = sources.strict
    remotes = [dict(spec) for spec in sources.remotes]
    if force_refresh:
        remotes = [dict(spec, refresh="always") for spec in remotes]

    report = rebuild_shims(
        list(sources.paths),
        remotes,
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
    """Resolve the remote-archive cache directory (config-adjacent)."""
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


def shim_removal_advisories(
    providers: Mapping[str, Any] | None,
    report: ShimLoadReport | None = None,
) -> list[ShimRemovalAdvisory]:
    """Advisories for providers still on a built-in shim that is going away.

    A shim name supplied by a successfully loaded extra source is *not* reported:
    a declared source of the same name overrides the built-in at rebuild time, so
    that provider is already on the shim that is staying.  A source that failed
    overrode nothing, so it does not suppress anything either.

    Args:
        providers: The raw ``providers`` section.  Pass the **unfiltered** one —
            a disabled provider is still worth a warning, because its config
            breaks the moment it is switched back on.
        report: The shim-source load report, when the section was applied.

    Returns:
        One advisory per provider, sorted by provider name.
    """
    supplied = {
        name
        for status in (report.sources if report else ())
        if status.ok
        for name in status.shim_names
    }
    return detect_shim_removals(providers, external_shims=supplied)


def report_shim_removals(
    providers: Mapping[str, Any] | None,
    report: ShimLoadReport | None = None,
) -> list[ShimRemovalAdvisory]:
    """Log one warning per provider on a built-in shim that is going away.

    Args:
        providers: The raw ``providers`` section (see
            :func:`shim_removal_advisories`).
        report: The shim-source load report, when the section was applied.

    Returns:
        The advisories that were logged.
    """
    advisories = shim_removal_advisories(providers, report)
    for advisory in advisories:
        preset = next(
            (
                package.key
                for package in SHIM_SOURCE_PACKAGES
                if package.repo == advisory.remote.get("url")
            ),
            None,
        )
        if preset is not None:
            logger.warning(
                "Provider %r uses built-in shim %r, removed in %s; declare the "
                "source instead: shims.presets += [%r] (or shims.remotes += [%s])",
                advisory.provider,
                advisory.shim,
                advisory.removed_in,
                preset,
                json.dumps(dict(advisory.remote)),
            )
        else:
            logger.warning(
                "Provider %r uses built-in shim %r, removed in %s; declare the "
                "source instead: shims.remotes += [%s]",
                advisory.provider,
                advisory.shim,
                advisory.removed_in,
                json.dumps(dict(advisory.remote)),
            )
    return advisories


__all__ = [
    "ShimSources",
    "apply_shim_section",
    "apply_shim_sources",
    "describe_shim_sources",
    "parse_shims_section",
    "report_shim_removals",
    "resolve_source_config",
    "shim_removal_advisories",
    "shims_signature",
]
