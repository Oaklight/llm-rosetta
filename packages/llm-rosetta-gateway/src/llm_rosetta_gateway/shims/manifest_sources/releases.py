"""Releases: choosing among the wheels an extension index lists for one extension.

A release is one :class:`~llm_rosetta.api.IndexEntry`.  Which one to install is
core's call (:func:`~llm_rosetta.api.choose_release`); this module words the
compatibility rule (:func:`~llm_rosetta.api.entry_problems`) and keeps the
dictionary form an install stamp and the admin API carry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_rosetta.api import IndexEntry, entry_problems


def requirement_problem(entry: IndexEntry) -> str | None:
    """Why *entry* cannot run on this interpreter and core, or ``None``."""
    problems = entry_problems(entry)
    return "; ".join(problems) if problems else None


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
