"""The ``shims.yaml`` manifest an external shim package ships.

A source that declares its shims does not have to be scanned: the loader reads
one file and loads exactly the set it names.  That is what makes a subset
selection honest — a directory the manifest does not list is never parsed — and
what lets the admin panel list a package's shims without fetching anything.
The manifest decides the *set*; the loader still walks directories in its own
sorted order, which carries no meaning because a duplicate name — the one case
where order would matter — is refused here.

The manifest is *inside* the archive, so it inherits the archive's pin: the list
is as fixed as the bytes.  Its paths are declared like a remote's ``path`` —
relative to the scan directory, never absolute, never escaping it — so a
manifest cannot point the loader at another location, and it carries no URLs at
all.

A manifest that is present but cannot be honoured fails its source.  Falling
back to a scan would parse exactly the tree the manifest was meant to bound,
which is the one outcome it exists to prevent.  An *absent* manifest is a
different thing: that is the legacy layout, and the source is scanned as before.

Example::

    version: 1
    shims:
      - name: llm7--chat_completions
        path: providers/llm7
      - name: ovhcloud--chat_completions
        path: providers/ovhcloud
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rosetta_gateway._vendor.yaml import load as yaml_load

#: The file a package ships at the root of its scan directory.
MANIFEST_FILENAME = "shims.yaml"

#: The only manifest version this loader understands.  A newer one is refused
#: rather than guessed at: its author may have changed what an entry means.
MANIFEST_VERSION = 1


class ShimManifestError(ValueError):
    """A manifest is present but cannot be honoured."""


@dataclass(frozen=True)
class ShimManifestEntry:
    """One declared shim: its name, and the directory that holds it.

    Attributes:
        name: The shim name, as the leaf declares it.
        path: The leaf directory, relative to the scan root and posix-spelled.
            Always non-empty, always inside the scan root.
    """

    name: str
    path: str


@dataclass(frozen=True)
class ShimManifest:
    """A parsed manifest.

    Attributes:
        version: The declared manifest version (always
            :data:`MANIFEST_VERSION` once parsed).
        entries: The declared shims, in declaration order.
        core: The oldest core this package needs, as ``">=X.Y[.Z]"``.  The
            gateway checks it before loading, because a package that tracks
            its branch can otherwise bring a shape the running core cannot
            read.  ``None`` means the package states no requirement.
    """

    version: int
    entries: tuple[ShimManifestEntry, ...]
    core: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """The declared shim names, in declaration order."""
        return tuple(entry.name for entry in self.entries)


def read_shim_manifest(root: Path) -> ShimManifest | None:
    """Read ``shims.yaml`` from *root*, if the source ships one.

    Args:
        root: The scan directory — the archive subdirectory a remote's ``path``
            names, or a local shim directory.

    Returns:
        The parsed manifest, or ``None`` when the source ships none (which
        means "scan me", the behaviour every source had before manifests).

    Raises:
        ShimManifestError: When the file exists but cannot be honoured — an
            unreadable or non-mapping document, an unknown version, a missing
            or malformed entry, a duplicate name, a path that escapes the scan
            root, or a path with no ``provider.yaml`` in it.
    """
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        document = yaml_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ShimManifestError(f"cannot be read: {exc}") from None
    if not isinstance(document, dict):
        raise ShimManifestError("must be a mapping with a 'shims' list")

    version = document.get("version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise ShimManifestError(
            f"declares version {version!r}; this loader understands "
            f"version {MANIFEST_VERSION}"
        )

    core = document.get("core")
    core = None if core is None else _normalize_core(core)

    raw = document.get("shims")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ShimManifestError("'shims' must be a list")

    entries: list[ShimManifestEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"shims[{index}]"
        if not isinstance(item, dict):
            raise ShimManifestError(f"{where} must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ShimManifestError(f"{where}.name must be a non-empty string")
        if name in seen:
            raise ShimManifestError(f"{where}.name {name!r} is declared twice")
        seen.add(name)
        leaf = _normalize_path(item.get("path"), where=f"{where}.path")
        if not leaf:
            raise ShimManifestError(
                f"{where}.path must name a directory below the scan root"
            )
        if not (root / leaf / "provider.yaml").is_file():
            raise ShimManifestError(f"{where}.path {leaf!r} holds no provider.yaml")
        entries.append(ShimManifestEntry(name=name, path=leaf))

    return ShimManifest(version=MANIFEST_VERSION, entries=tuple(entries), core=core)


#: The only requirement form a manifest may state.
_AT_LEAST = re.compile(r"^>=\s*(\d+(?:\.\d+){0,2})$")


def _normalize_core(value: Any) -> str:
    """Validate a manifest's ``core:`` floor and return it stripped."""
    if not isinstance(value, str) or _AT_LEAST.match(value.strip()) is None:
        raise ShimManifestError(
            "'core' must read '>=X.Y[.Z]' — the oldest core this package needs"
        )
    return value.strip()


def _version_tuple(value: str) -> tuple[int, int, int]:
    """The first three numeric components of a version, padded with zeros."""
    parts: list[int] = []
    for chunk in value.split(".")[:3]:
        match = re.match(r"^\d+", chunk)
        if match is None:
            break
        parts.append(int(match.group(0)))
    major, minor, patch = (parts + [0, 0, 0])[:3]
    return major, minor, patch


def core_version_problem(requirement: str, current: str | None = None) -> str | None:
    """Why *requirement* is not satisfied by the running core, or ``None``.

    Only ``>=X.Y[.Z]`` is understood.  The manifest exists to state a floor,
    and a full specifier grammar would be a second dependency to keep in step
    with the release workflow — the loader refuses anything else rather than
    guessing at it.

    Args:
        requirement: The manifest's ``core:`` value.
        current: The version to check against; the running core by default.

    Returns:
        The sentence to report, or ``None`` when the requirement holds.
    """
    if current is None:
        from llm_rosetta.api import __version__ as current_version

        current = current_version
    match = _AT_LEAST.match(requirement.strip())
    if match is None:
        return f"'core: {requirement}' is not a floor this loader understands"
    if _version_tuple(current) < _version_tuple(match.group(1)):
        return (
            f"needs llm-rosetta {requirement} but this gateway runs {current}; "
            f"upgrade the gateway, or pin an older ref for this package"
        )
    return None


def _normalize_path(value: Any, *, where: str) -> str:
    """Normalize an entry's ``path`` with the loader's own subpath rule.

    Imported here rather than at module scope: :mod:`.sources` imports this
    module, and the two are otherwise a cycle.  Using the same function is the
    point — a manifest path and a remote's ``path`` must not be judged by two
    different rules.
    """
    from ._spec import normalize_subpath

    try:
        return normalize_subpath(value, where=where)
    except ValueError as exc:
        raise ShimManifestError(str(exc)) from None
