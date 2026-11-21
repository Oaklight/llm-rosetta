"""Shim vendors that left core, and the installs a config needs for them (#1660).

A vendor that moved from core to a shim source repository keeps its shim
names, so a config written for the built-in shim (``type:
opencode--chat_completions``, ``quota.probe: openai_codex``) still names a
valid shim — it is only no longer built in.  This module finds those names in
a raw config document and turns them into ``<source>:<vendor>`` install
selectors, which :mod:`llm_rosetta_gateway.shim_sources` adds to the effective
``shims.install`` selection.  The operator edits nothing, and a config with no
``shims`` section at all is covered as well.

Core owns the table of moved vendors
(:data:`llm_rosetta.retired_names.MOVED_VENDORS`): the library's own lookup
errors need it, and core must not import the gateway.  Core records the
*repository* a vendor moved to; this module maps that repository to the
catalogue source (:data:`~llm_rosetta_gateway.shims.catalog.SHIM_SOURCES`)
that publishes it, so the table exists once.

The scan is pure: it reads the document and nothing else, so the config
loader, the apply path and the admin reload compute the same answer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from llm_rosetta.api import FORMATS_BY_TYPE, retired_names

from .catalog import SHIM_SOURCES, repo_key


@dataclass(frozen=True)
class MovedReference:
    """One place in a config that names a moved vendor's shim or probe.

    Attributes:
        vendor: The moved vendor.
        source: The catalogue source that publishes it (``contrib``).
        name: The shim name or the quota probe id as written.
        where: The dotted config path of the value (``providers.go.type``).
    """

    vendor: str
    source: str
    name: str
    where: str


def moved_vendor_sources() -> dict[str, str]:
    """Map each moved vendor to the catalogue source that publishes it.

    A vendor whose repository no catalogue source names is left out; a test
    keeps the two tables in step, so that does not happen in a release.
    """
    by_repo: dict[str, str] = {}
    for source in SHIM_SOURCES:
        by_repo[repo_key(source.repo)] = source.key
        for legacy in source.legacy_repos:
            by_repo.setdefault(repo_key(legacy), source.key)
    out: dict[str, str] = {}
    for row in retired_names.MOVED_VENDORS:
        key = by_repo.get(repo_key(row.repository))
        if key is not None:
            out[row.vendor] = key
    return out


def _shim_values(
    name: str, cfg: Mapping[str, Any], where: str
) -> Iterator[tuple[str, str]]:
    """``(value, path)`` for every slot of one provider entry that names a shim.

    The identity follows ``GatewayConfig``: ``shim`` wins, then ``type`` (a
    current format is not a shim), then the provider key.  Every face
    (``embeddings_shim``, an ``embeddings: {shim: ...}`` block) is a slot of
    its own.
    """
    formats = set(FORMATS_BY_TYPE.get("llm", ()))
    shim = cfg.get("shim")
    kind = cfg.get("type")
    if isinstance(shim, str) and shim:
        yield shim, f"{where}.shim"
    elif isinstance(kind, str) and kind:
        if kind not in formats:
            yield kind, f"{where}.type"
    else:
        yield name, where
    for key, value in cfg.items():
        # ``_auto_disabled`` and any other ``_``-prefixed key is the gateway's
        # own state, not a face: its ``shim`` names the slot it already saw.
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if key.endswith("_shim") and isinstance(value, str) and value:
            yield value, f"{where}.{key}"
        elif isinstance(value, Mapping) and isinstance(value.get("shim"), str):
            yield value["shim"], f"{where}.{key}.shim"


def _provider_sections(
    raw: Mapping[str, Any],
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """The unified ``providers`` section and every legacy ``<type>_providers`` one."""
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        if key == "providers" or key.endswith("_providers"):
            yield key, value


def moved_references(raw: Mapping[str, Any] | None) -> list[MovedReference]:
    """Every place *raw* names a shim or a quota probe of a moved vendor.

    A disabled provider counts: the gateway disables a provider whose shim is
    missing by itself, and that provider must be able to come back once the
    vendor is installed.

    Args:
        raw: The whole config document (not only its ``shims`` section).

    Returns:
        The references in document order.  Empty when there are none or the
        document is not a mapping.
    """
    if not isinstance(raw, Mapping):
        return []
    sources = moved_vendor_sources()
    found: list[MovedReference] = []

    def add(row: Any, name: str, where: str) -> None:
        source = sources.get(row.vendor) if row is not None else None
        if source is not None:
            found.append(MovedReference(row.vendor, source, name, where))

    for section, entries in _provider_sections(raw):
        for name, cfg in entries.items():
            if not isinstance(cfg, Mapping):
                continue
            where = f"{section}.{name}"
            for value, path in _shim_values(str(name), cfg, where):
                add(retired_names.moved_vendor(value), value, path)
            quota = cfg.get("quota")
            probe = quota.get("probe") if isinstance(quota, Mapping) else None
            if isinstance(probe, str):
                add(
                    retired_names.moved_probe_vendor(probe),
                    probe,
                    f"{where}.quota.probe",
                )
    return found


def moved_vendors_needed(raw: Mapping[str, Any] | None) -> dict[str, str]:
    """``vendor -> source`` for the moved vendors *raw* needs, in first-use order."""
    out: dict[str, str] = {}
    for ref in moved_references(raw):
        out.setdefault(ref.vendor, ref.source)
    return out


__all__ = [
    "MovedReference",
    "moved_references",
    "moved_vendor_sources",
    "moved_vendors_needed",
]
