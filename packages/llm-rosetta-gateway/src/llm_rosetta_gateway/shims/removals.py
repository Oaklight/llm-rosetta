"""Shims scheduled for removal, and who still uses them.

A shim that ships in core can move to a standalone package, but only if the
people depending on it are told before it goes.  This module is the single place
that knows *which* shims are going away, *when*, and *what to declare instead*.
Three consumers read it and none of them restates it:

* the gateway logs one warning per configured provider still sitting on one,
* the admin panel offers the replacement source as a one-click preset,
* ``scripts/check_shim_removals.py`` fails once the version reaches the removal,
  so the repository's own examples and docs cannot be forgotten in the move.

The manifest is a Python constant rather than a data file, and
:func:`detect_shim_removals` is deliberately pure — no I/O, no registry, no
logging — so that the answer can be produced from a raw config mapping, from the
admin payload and from CI alike, and so that it can later be a ``LOAD_TIME``
step in the maintenance framework (#1027) without dragging state into its
context.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_rosetta.provider_names import canonical_provider_name, canonical_shim_name


@dataclass(frozen=True)
class ShimRemoval:
    """One built-in shim that is going away, and what replaces it.

    Attributes:
        shim: Canonical name of the built-in shim being removed.
        removed_in: The release that removes it.
        remote: The ``shims.remotes`` entry that provides an equal shim.
        source_repo: The repository that entry points at, for humans.
        legacy_names: Older spellings of the same shim that also stop resolving.
    """

    shim: str
    removed_in: str
    remote: Mapping[str, Any]
    source_repo: str
    legacy_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (the admin payload shape)."""
        return {
            "shim": self.shim,
            "removed_in": self.removed_in,
            "remote": dict(self.remote),
            "source_repo": self.source_repo,
            "legacy_names": list(self.legacy_names),
        }


#: Every built-in shim with a removal scheduled.  Empty since 0.14.0: the
#: ALCF, AskSage and Argo shims all ship standalone, and their names are served
#: by :data:`llm_rosetta_gateway.shims.catalog.SHIM_SOURCE_PACKAGES`.  Adding an entry
#: here is still what makes the advisory, the admin preset and the CI gate know
#: about a future removal at once.
SHIM_REMOVALS: tuple[ShimRemoval, ...] = ()


@dataclass(frozen=True)
class ShimRemovalAdvisory:
    """A configured provider sitting on a shim that is going away.

    Attributes:
        provider: The provider key in the config.
        slot: Which field named the shim (``shim``, ``type`` or ``name``).
        shim: The canonical shim name it resolved to.
        removed_in: The release that removes it.
        remote: The ``shims.remotes`` entry that would replace it.
        source_repo: The repository that entry points at, for humans.
    """

    provider: str
    slot: str
    shim: str
    removed_in: str
    remote: Mapping[str, Any]
    source_repo: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (the admin payload shape)."""
        return {
            "provider": self.provider,
            "slot": self.slot,
            "shim": self.shim,
            "removed_in": self.removed_in,
            "remote": dict(self.remote),
            "source_repo": self.source_repo,
        }


def _named_shim(cfg: Mapping[str, Any], key: str) -> tuple[str, str]:
    """Return the ``(slot, shim name)`` a provider entry points at.

    Mirrors :func:`llm_rosetta_gateway.maintenance.names._corrected_value` and
    ``GatewayConfig._resolve_provider_types``: an explicit ``shim`` wins, then
    ``type``, then the provider key itself.
    """
    value = cfg.get("shim")
    if isinstance(value, str) and value:
        return "shim", canonical_shim_name(value)
    value = cfg.get("type")
    if isinstance(value, str) and value:
        canonical = canonical_provider_name(value)
        return "type", canonical if canonical != value else canonical_shim_name(value)
    return "name", canonical_shim_name(key)


def detect_shim_removals(
    providers: Mapping[str, Any] | None,
    *,
    external_shims: Iterable[str] = (),
    removals: Sequence[ShimRemoval] | None = None,
) -> list[ShimRemovalAdvisory]:
    """Advisories for providers that name a shim scheduled for removal.

    Pure: it reads *providers* and nothing else, so it can be called from the
    startup path, from an admin request and from CI with the same answer.

    Args:
        providers: The raw ``providers`` section (provider key → entry).  This
            reads a file a person edits, so an entry that is not a mapping is
            skipped rather than fatal.  A disabled provider still counts: its
            config breaks the moment it is switched back on.
        external_shims: Shim names supplied by a successfully loaded extra
            source.  One of those overrides the built-in it shadows, so it is
            not going anywhere and must not be reported.
        removals: The manifest to test against (a parameter for tests).

    Returns:
        One advisory per provider, sorted by provider name.
    """
    if not providers:
        return []
    if removals is None:
        removals = SHIM_REMOVALS
    scheduled = {entry.shim: entry for entry in removals}
    supplied = set(external_shims)
    advisories: list[ShimRemovalAdvisory] = []
    for provider in sorted(providers):
        cfg = providers[provider]
        if not isinstance(cfg, Mapping):
            continue
        slot, shim = _named_shim(cfg, provider)
        entry = scheduled.get(shim)
        if entry is None or shim in supplied:
            continue
        advisories.append(
            ShimRemovalAdvisory(
                provider=provider,
                slot=slot,
                shim=shim,
                removed_in=entry.removed_in,
                remote=dict(entry.remote),
                source_repo=entry.source_repo,
            )
        )
    return advisories


def parse_version(text: str) -> tuple[int, ...]:
    """Return the leading numeric release of *text* as a tuple.

    ``"0.16.0.dev17"`` → ``(0, 16, 0)``.  A version with no leading digits
    yields ``()``, which compares before everything else and so never reaches a
    removal by accident.

    Args:
        text: A version string, e.g. ``llm_rosetta.__version__``.

    Returns:
        The release tuple, e.g. ``(0, 16, 0)``.
    """
    match = re.match(r"\s*(\d+(?:\.\d+)*)", text or "")
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def removals_due(
    version: str | tuple[int, ...],
    *,
    removals: Sequence[ShimRemoval] | None = None,
) -> list[ShimRemoval]:
    """Return the removals whose version has been reached by *version*.

    A pre-release of the removal release counts as reaching it
    (``"0.16.0.dev1"`` → the 0.16.0 removals), because that is the first build
    that could ship the deletion.

    Args:
        version: A version string or an already-parsed release tuple.
        removals: The manifest to filter (a parameter for tests).

    Returns:
        The removals that are due, in manifest order.
    """
    current = parse_version(version) if isinstance(version, str) else tuple(version)
    if removals is None:
        removals = SHIM_REMOVALS
    return [
        entry
        for entry in removals
        if current and current >= parse_version(entry.removed_in)
    ]


@dataclass(frozen=True)
class RemovalReference:
    """One line of the repository that still spells a removed shim.

    Attributes:
        path: Repository-relative path of the file.
        line: 1-based line number.
        text: The stripped line, for the error message.
        shim: The removed shim name that line mentions.
    """

    path: str
    line: int
    text: str
    shim: str


#: Directories never worth scanning, and the two marker strings that let a
#: deliberate mention through (a migration table, a changelog entry).
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", ".mypy_cache"})
_SKIP_SUFFIXES = frozenset(
    {
        ".pyc",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".zip",
        ".gz",
    }
)
_MARKER = "shim-removal-ok"


def scan_removal_references(
    roots: Iterable[str | os.PathLike[str]],
    *,
    removals: Sequence[ShimRemoval] | None = None,
    exempt_files: Iterable[str] = (),
    marker: str = _MARKER,
) -> list[RemovalReference]:
    """Find every line under *roots* that still names a removed shim.

    Matching is on whole tokens (``acme--messages`` does not match
    ``acme--messages-extra``), and both a shim's canonical name and its legacy
    spellings count, because either one is a stale reference once the shim is
    gone.

    Args:
        roots: Files or directories to scan.  A missing root is skipped, so a
            caller can pass a list that is only partly present.
        removals: The removals to look for (a parameter for tests).
        exempt_files: Paths to skip entirely — used for the changelogs, whose
            history is not rewritten.  A file matches on its normalized path
            *or* on its basename, so relocating a scanned tree (a test's
            ``tmp_path``, a vendored copy) does not silently drop the
            exemption and turn the changelog's historical mentions into
            failures.
        marker: A string on a line that exempts that line.

    Returns:
        One reference per matching line, sorted by path then line.
    """
    if removals is None:
        removals = SHIM_REMOVALS
    names = sorted(
        {entry.shim for entry in removals}
        | {name for entry in removals for name in entry.legacy_names}
    )
    if not names:
        return []
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(name) for name in names) + r")(?![\w-])"
    )
    exempt_paths = {os.path.normpath(str(p)) for p in exempt_files}
    exempt_names = {os.path.basename(p) for p in exempt_paths}
    found: list[RemovalReference] = []

    for root in roots:
        root_path = os.path.normpath(str(root))
        if not os.path.exists(root_path):
            continue
        if os.path.isfile(root_path):
            found.extend(
                _scan_file(root_path, pattern, marker, exempt_paths, exempt_names)
            )
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            for filename in filenames:
                found.extend(
                    _scan_file(
                        os.path.join(dirpath, filename),
                        pattern,
                        marker,
                        exempt_paths,
                        exempt_names,
                    )
                )

    found.sort(key=lambda reference: (reference.path, reference.line))
    return found


def _scan_file(
    path: str,
    pattern: re.Pattern[str],
    marker: str,
    exempt_paths: set[str],
    exempt_names: set[str],
) -> list[RemovalReference]:
    """Return the matching lines of one file (empty for skipped or unreadable)."""
    if os.path.normpath(path) in exempt_paths or os.path.basename(path) in exempt_names:
        return []
    if os.path.splitext(path)[1].lower() in _SKIP_SUFFIXES:
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeDecodeError):
        return []
    references: list[RemovalReference] = []
    for number, line in enumerate(lines, start=1):
        if marker in line:
            continue
        match = pattern.search(line)
        if match is None:
            continue
        references.append(
            RemovalReference(
                path=os.path.relpath(path),
                line=number,
                text=line.strip(),
                shim=match.group(1),
            )
        )
    return references
