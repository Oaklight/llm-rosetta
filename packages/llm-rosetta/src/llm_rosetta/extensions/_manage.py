"""Remove an installed extension together with the programs it installed (#1829).

:func:`plan_removal` reports what would be deleted and what is kept, so a host
can show the paths before anything happens; :func:`remove_with_programs` then
does it.  An extension is **removable** only when its folder lies in the
extensions directory: one that a pip package put into ``site-packages`` is
owned by the image, and a plan for it names the ``pip`` command instead.

The Node.js under ``<tools>/node`` and the links ``<tools>/bin/{node,npm,npx}``
belong to no single extension.  They are deleted only when no other
``<tools>/<id>/npm-<version>/`` is left that needs them.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ._dirs import ExtensionInstallError, extensions_dir, tools_dir
from ._discovery import ENTRY_POINT_GROUP, NAMESPACE, _entry_points
from ._install import (
    _owned_dist_infos,
    _require_directory,
    installed_extensions,
    remove_extension,
)

#: The links of the shared Node.js; kept in step with ``_tools._NODE_LINKS``.
_NODE_LINKS = ("node", "npm", "npx")


@dataclass(frozen=True)
class PathEntry:
    """One path a removal deletes or keeps.

    Attributes:
        path: The absolute path.
        size: Bytes held under it (a link counts as zero).
        note: Why it is kept, or what it is; ``""`` when obvious.
    """

    path: str
    size: int = 0
    note: str = ""


@dataclass(frozen=True)
class RemovalPlan:
    """What removing one extension does.

    Attributes:
        id: The extension id.
        version: Its installed version, when known.
        origin: ``"directory"`` (in the extensions directory), ``"image"``
            (elsewhere, e.g. a pip package), or ``"missing"``.
        removable: Whether :func:`remove_with_programs` may run.
        delete: The paths it deletes, in order.
        keep: The shared paths it leaves, each with the reason.
        hint: For an ``image`` extension, how to remove it; otherwise ``None``.
    """

    id: str
    version: str | None
    origin: str
    removable: bool
    delete: tuple[PathEntry, ...] = ()
    keep: tuple[PathEntry, ...] = ()
    hint: str | None = None


@dataclass(frozen=True)
class RemovalResult:
    """What a removal did.

    Attributes:
        id: The extension id.
        deleted: The paths that are gone.
        failed: ``(path, reason)`` for each path that could not be deleted.
    """

    id: str
    deleted: tuple[str, ...]
    failed: tuple[tuple[str, str], ...] = ()


def extension_origin(
    path: str | os.PathLike[str], dest: str | os.PathLike[str] | None = None
) -> str:
    """Return where an extension folder lies: ``"directory"`` or ``"image"``.

    Args:
        path: The extension's folder.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        ``"directory"`` when *path* is ``<dest>/llm_rosetta_ext/<id>`` (the one
        place ``ext remove`` can delete), ``"image"`` for any other location.
    """
    root = (Path(dest) if dest is not None else extensions_dir()) / NAMESPACE
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return "image"
    return "directory"


def _size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            with_path = Path(base, name)
            if not with_path.is_symlink():
                try:
                    total += with_path.stat().st_size
                except OSError:
                    continue
    return total


def _entry(path: Path, note: str = "") -> PathEntry:
    return PathEntry(str(path), _size(path), note)


def _dist_name(ext_id: str) -> str | None:
    for entry in _entry_points(ENTRY_POINT_GROUP):
        dist = getattr(entry, "dist", None)
        if entry.name == ext_id and dist is not None:
            return getattr(dist, "name", None)
    return None


def _image_hint(ext_id: str) -> str:
    name = _dist_name(ext_id)
    if name:
        return (
            f"Provided by the image (pip package {name}). Run `pip uninstall {name}` "
            "in the image, or rebuild the image without it."
        )
    return (
        "Provided by the image, outside the extensions directory. Remove the "
        "package that provides it, or rebuild the image without it."
    )


def _npm_users(tools: Path, leaving: str) -> list[str]:
    """The other extensions whose installed program is an npm install."""
    users: list[str] = []
    if not tools.is_dir():
        return users
    for folder in sorted(tools.iterdir()):
        if folder.name in (leaving, "node", "bin") or not folder.is_dir():
            continue
        if any(child.is_dir() for child in folder.glob("npm-*")):
            users.append(folder.name)
    return users


def _shared_links(tools: Path, ext_id: str) -> list[Path]:
    """Links in the shared ``bin`` that point into *ext_id*'s tools folder."""
    own = (tools / ext_id).resolve()
    found: list[Path] = []
    shared = tools / "bin"
    if not shared.is_dir():
        return found
    for link in sorted(shared.iterdir()):
        if not link.is_symlink():
            continue
        try:
            link.resolve().relative_to(own)
        except (ValueError, OSError):
            continue
        found.append(link)
    return found


def plan_removal(
    ext_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
    dest: str | os.PathLike[str] | None = None,
    tools: str | os.PathLike[str] | None = None,
) -> RemovalPlan:
    """Report what removing *ext_id* deletes and keeps.

    Args:
        ext_id: The extension id.
        path: The extension's folder when the caller already knows it (the
            loaded one may live outside the extensions directory); looked up in
            the extensions directory otherwise.
        dest: The extensions directory; :func:`extensions_dir` by default.
        tools: The tools directory; :func:`tools_dir` by default.

    Returns:
        The plan.  Nothing is changed.

    Raises:
        ExtensionInstallError: *ext_id* is not a plain extension id.
    """
    if not ext_id.isidentifier() or ext_id.startswith("_"):
        raise ExtensionInstallError(f"{ext_id!r} is not an extension id")
    target = Path(dest) if dest is not None else extensions_dir()
    version = None
    for item in installed_extensions(target):
        if item.id == ext_id:
            version = item.version
            path = path or item.path
    if path is None:
        return RemovalPlan(ext_id, None, "missing", False)
    if extension_origin(path, target) == "image":
        return RemovalPlan(ext_id, version, "image", False, hint=_image_hint(ext_id))
    delete = [_entry(Path(path))]
    delete += [_entry(d) for d in _owned_dist_infos(target, ext_id)]
    tools_root = Path(tools) if tools is not None else tools_dir()
    own = tools_root / ext_id
    keep: list[PathEntry] = []
    if own.is_dir():
        delete += [_entry(link) for link in _shared_links(tools_root, ext_id)]
        delete.append(_entry(own))
        node = tools_root / "node"
        users = _npm_users(tools_root, ext_id)
        if node.is_dir() and users:
            keep.append(
                PathEntry(
                    str(node),
                    0,
                    f"still used by: {', '.join(users)}",
                )
            )
        elif node.is_dir():
            delete.append(_entry(node))
            delete += [
                _entry(link)
                for name in _NODE_LINKS
                if (link := tools_root / "bin" / name).is_symlink()
            ]
    return RemovalPlan(ext_id, version, "directory", True, tuple(delete), tuple(keep))


def _delete(path: Path) -> str | None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def remove_with_programs(
    ext_id: str,
    *,
    path: str | os.PathLike[str] | None = None,
    dest: str | os.PathLike[str] | None = None,
    tools: str | os.PathLike[str] | None = None,
) -> RemovalResult:
    """Remove *ext_id* and the programs it installed, as :func:`plan_removal` says.

    The extension goes first, through :func:`remove_extension`; the programs
    follow, so a failure leaves the extension gone rather than half there.

    Args:
        ext_id: The extension id.
        path: See :func:`plan_removal`.
        dest: The extensions directory; :func:`extensions_dir` by default.
        tools: The tools directory; :func:`tools_dir` by default.

    Returns:
        The paths deleted and the ones that failed.

    Raises:
        ExtensionInstallError: The extension is not removable (not installed in
            the extensions directory), with the reason.
    """
    plan = plan_removal(ext_id, path=path, dest=dest, tools=tools)
    if not plan.removable:
        raise ExtensionInstallError(
            plan.hint or f"{ext_id!r} is not installed in the extensions directory"
        )
    target = Path(dest) if dest is not None else extensions_dir()
    _require_directory(target)
    deleted: list[str] = []
    failed: list[tuple[str, str]] = []
    if not remove_extension(ext_id, target):
        raise ExtensionInstallError(f"{ext_id!r} is not installed in {target}")
    # The extension folder and its owned dist-info went with remove_extension; the
    # entries after them are the programs and the shared links.
    removed = {
        e.path
        for e in plan.delete
        if e.path == plan.delete[0].path or e.path.endswith(".dist-info")
    }
    deleted += [e.path for e in plan.delete if e.path in removed]
    for entry in plan.delete:
        if entry.path in removed:
            continue
        reason = _delete(Path(entry.path))
        if reason is None:
            deleted.append(entry.path)
        else:
            failed.append((entry.path, reason))
    return RemovalResult(ext_id, tuple(deleted), tuple(failed))
