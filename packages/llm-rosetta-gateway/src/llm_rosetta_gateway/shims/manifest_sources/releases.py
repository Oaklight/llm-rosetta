"""Releases: choosing among the wheels an extension index lists for one extension.

A release is one :class:`~llm_rosetta.api.IndexEntry`.  This module orders
them, applies the compatibility rule core states
(:func:`~llm_rosetta.api.entry_problems`) and keeps the dictionary form an
install stamp and the admin API carry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llm_rosetta.api import IndexEntry, entry_problems, version_key


def requirement_problem(entry: IndexEntry) -> str | None:
    """Why *entry* cannot run on this interpreter and core, or ``None``."""
    problems = entry_problems(entry)
    return "; ".join(problems) if problems else None


def is_prerelease(version: str) -> bool:
    """Whether *version* carries a pre-release suffix (``1.0.0-rc.1``)."""
    return "-" in version


def select_release(
    releases: Sequence[IndexEntry],
) -> tuple[IndexEntry | None, tuple[tuple[IndexEntry, str], ...]]:
    """Pick the release this core should install.

    The newest installable **stable** release wins.  A pre-release is chosen
    only when no stable release is installable, so one pre-release published
    to a source does not upgrade every gateway that reads it.

    Args:
        releases: One extension's releases, in any order.

    Returns:
        ``(chosen, skipped)``: the chosen release (``None`` when nothing is
        installable), and every release *newer* than it that was refused as
        incompatible, with the reason.  Older incompatible releases, and newer
        pre-releases passed over for a stable one, are not reported.
    """
    ordered = sorted(releases, key=lambda r: version_key(r.version), reverse=True)
    incompatible: list[tuple[IndexEntry, str]] = []
    prerelease: IndexEntry | None = None
    chosen: IndexEntry | None = None
    for release in ordered:
        problem = requirement_problem(release)
        if problem is not None:
            incompatible.append((release, problem))
        elif not is_prerelease(release.version):
            chosen = release
            break
        elif prerelease is None:
            prerelease = release
    chosen = chosen or prerelease
    if chosen is None:
        return None, tuple(incompatible)
    floor = version_key(chosen.version)
    return chosen, tuple(
        (release, why)
        for release, why in incompatible
        if version_key(release.version) > floor
    )


def release_to_dict(entry: IndexEntry) -> dict[str, Any]:
    """The JSON-safe record of *entry*, as an install stamp and the admin API hold it."""
    return {
        "version": entry.version,
        "url": entry.url,
        "sha256": entry.sha256,
        "surface": entry.surface,
        "ir": entry.ir,
        "requires_python": entry.requires_python,
        "requires_extensions": list(entry.requires_extensions),
        "name": entry.name,
        "maturity": entry.maturity,
        "tags": list(entry.tags),
        "shims": list(entry.shims),
        "logo": entry.logo,
        "commit": entry.commit,
    }


def release_from_dict(ext_id: str, record: Mapping[str, Any]) -> IndexEntry | None:
    """The release a stamp or an admin record holds, or ``None`` when unreadable."""
    try:
        return IndexEntry(
            id=ext_id,
            version=str(record["version"]),
            url=str(record["url"]),
            sha256=str(record["sha256"]),
            requires_python=record.get("requires_python"),
            surface=record.get("surface"),
            ir=record.get("ir"),
            requires_extensions=tuple(record.get("requires_extensions") or ()),
            name=record.get("name"),
            maturity=record.get("maturity"),
            tags=tuple(record.get("tags") or ()),
            shims=tuple(record.get("shims") or ()),
            logo=record.get("logo"),
            commit=record.get("commit"),
        )
    except (KeyError, TypeError, ValueError):
        return None
