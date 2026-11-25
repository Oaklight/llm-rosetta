"""``shims.presets`` spelled as ``shims.install`` entries.

The gateway stopped reading ``shims.presets``, ``shims.include``,
``shims.exclude`` and the preset-derived remotes in 0.15.0.  This module is the
one place that still knows what the five legacy preset keys stood for, so
``llm-rosetta migrate`` can rewrite an old config into the manifest-source
form (:mod:`llm_rosetta_gateway.shims.manifest_sources`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..shims.catalog import DEFAULT_SOURCES, get_shim_source, get_standalone
from ..shims.manifest_sources.plan import parse_selector


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
            to this build, with no manifest source, or one whose selection converts to nothing (an ``include`` that
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
) -> PresetRewrite:
    """Spell legacy ``shims.presets`` as explicit ``shims.install`` entries.

    The one home of the rewrite: ``llm-rosetta migrate --config`` edits the
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

    Returns:
        The rewrite.
    """
    include = include or {}
    exclude = exclude or {}
    keys = [k for k in presets if isinstance(k, str) and k]
    out = PresetRewrite()
    for key in dict.fromkeys(keys):
        package = get_standalone(key)
        why = _unconvertible(package)
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


def _unconvertible(package: Any) -> str | None:
    """Why a preset key stays a preset, or ``None``."""
    if package is None:
        return "unknown preset"
    if not package.source or not package.install:
        return "no manifest source"
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


__all__ = [
    "PresetRewrite",
    "presets_as_install",
    "rewrite_presets_section",
    "sources_to_declare",
]
