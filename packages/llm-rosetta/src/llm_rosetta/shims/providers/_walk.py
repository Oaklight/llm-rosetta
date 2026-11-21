"""Walking a providers tree: type, group and lane folders and the directory entry point."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any


from llm_rosetta.contracts.shim import ProviderShim, VendorManifest
from ._leaf import _load_single_provider
from ._constants import (
    _MAX_LANE_DEPTH,
    _TYPE_DIRS,
    logger,
)


def _path_selected(directory: Path, root: Path, only: Collection[str] | None) -> bool:
    """Whether *directory* may be scanned under an ``only_paths`` selection.

    A path selects that directory **and everything under it**, so a vendor
    directory is a valid selection and a leaf inside one is too.  A directory
    that is merely an ancestor of a selection is entered as well: the loader
    must walk through ``vendor/`` to reach ``vendor/llm/x``, and skipping the
    vendor would skip the leaf.  An empty collection selects nothing, which is
    the difference between "declares no shims" and ``None`` ("declares no
    restriction").

    Args:
        directory: The directory about to be scanned.
        root: The scan root the selection's paths are relative to.
        only: The selected relative posix paths, or ``None`` for everything.
            Callers normalize them; a ``./`` prefix or a trailing slash will
            not match.

    Returns:
        Whether the directory may be entered.
    """
    if only is None:
        return True
    if not only:
        return False
    try:
        rel = directory.relative_to(root).as_posix()
    except ValueError:
        # Not under the root: the selection says nothing about it, so the
        # caller's own rules decide rather than this one silently dropping it.
        return True
    for path in only:
        if rel == path or rel.startswith(f"{path}/") or path.startswith(f"{rel}/"):
            return True
    return False


def _load_provider_tolerant(
    directory: Path,
    group: str | None,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
    expected_type: str | None = None,
    manifest: VendorManifest | None = None,
    keep_names: Callable[[str], bool] | None = None,
) -> ProviderShim | None:
    """Load one provider directory, optionally downgrading errors to a warning.

    With ``on_error="warn"`` a malformed ``provider.yaml`` or a failing
    ``transforms.py`` in *directory* is logged and skipped instead of aborting
    the whole scan.
    """
    try:
        return _load_single_provider(
            directory,
            group=group,
            _builtin=builtin,
            root=root,
            quota_probes=quota_probes,
            expected_type=expected_type,
            manifest=manifest,
            keep_names=keep_names,
        )
    except Exception as exc:
        if on_error == "warn":
            logger.warning("Skipping shim directory %s: %s", directory, exc)
            return None
        raise


def _load_type_folder(
    type_dir: Path,
    group: str | None,
    expected_type: str,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
    manifest: VendorManifest | None = None,
    only_paths: Collection[str] | None = None,
    keep_names: Callable[[str], bool] | None = None,
) -> list[ProviderShim]:
    """Load the leaves under a model-type directory.

    The directory itself may be the leaf (``vendor/embeddings/provider.yaml``)
    or group format directories (``vendor/llm/chat_completions/provider.yaml``).
    Every leaf is checked against the type the directory names.
    """
    if (type_dir / "provider.yaml").exists():
        shim = _load_provider_tolerant(
            type_dir,
            group,
            builtin=builtin,
            root=root,
            on_error=on_error,
            quota_probes=quota_probes,
            expected_type=expected_type,
            manifest=manifest,
            keep_names=keep_names,
        )
        return [shim] if shim is not None else []

    child_group = f"{group}.{type_dir.name}" if group else type_dir.name
    loaded: list[ProviderShim] = []
    for sub in sorted(type_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        if not _path_selected(sub, root, only_paths):
            continue
        if (sub / "provider.yaml").exists():
            shim = _load_provider_tolerant(
                sub,
                child_group,
                builtin=builtin,
                root=root,
                on_error=on_error,
                quota_probes=quota_probes,
                expected_type=expected_type,
                manifest=manifest,
                keep_names=keep_names,
            )
            if shim is not None:
                loaded.append(shim)
    return loaded


def _lane_holds_shims(
    directory: Path,
    root: Path,
    only_paths: Collection[str] | None,
    depth: int = 0,
) -> bool:
    """Whether a lane directory has a shim somewhere below it.

    A lane is a named instance of one vendor, and it holds type directories
    of its own rather than a leaf directly.  The loader enters one only when
    something loadable sits below it: a model-type directory, or a leaf
    ``provider.yaml``.  Without that test a stray ``docs/`` or ``assets/``
    directory would be walked for nothing.

    Args:
        directory: The candidate lane directory.
        root: The scan root the ``only_paths`` selection is relative to.
        only_paths: The selected relative paths, or ``None`` for everything.
        depth: How far below the first lane this call already is.

    Returns:
        Whether the directory should be scanned as a lane.
    """
    if depth >= _MAX_LANE_DEPTH:
        # A shim one level past the cap is the one shape that loads as nothing
        # while looking fine: the tree is reachable, the leaves are valid, and
        # only the count is lower.  Say so rather than returning quietly.
        if any(
            child.is_dir()
            and not child.name.startswith(("_", "."))
            and (child.name in _TYPE_DIRS or (child / "provider.yaml").exists())
            for child in directory.iterdir()
        ):
            logger.warning(
                "Shim lane below %s is not read: the loader enters at most %d "
                "lane levels, and a shim sits one level deeper. Flatten the "
                "tree, or file it under the vendor with a single lane level.",
                directory,
                _MAX_LANE_DEPTH,
            )
        return False
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        if not _path_selected(sub, root, only_paths):
            continue
        if sub.name in _TYPE_DIRS or (sub / "provider.yaml").exists():
            return True
        if _lane_holds_shims(sub, root, only_paths, depth + 1):
            return True
    return False


def _load_group_folder(
    group_dir: Path,
    group: str | None,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
    manifest: VendorManifest | None = None,
    only_paths: Collection[str] | None = None,
    keep_names: Callable[[str], bool] | None = None,
    lane_depth: int = 0,
) -> list[ProviderShim]:
    """Load every leaf provider inside a *group_dir*.

    A child is a leaf (``<group>/<leaf>/provider.yaml``), a model-type
    directory (``<group>/llm/...``, ``<group>/embeddings/...``), or a lane — a
    named instance of the same vendor that holds type directories of its own.
    A child that is none of those is not a provider directory and is skipped.
    *manifest* is the vendor manifest read from *group_dir* itself, if it has
    one; a lane inherits it, because the credential and the identity belong to
    the vendor rather than to one instance.
    """
    child_group = f"{group}.{group_dir.name}" if group else group_dir.name
    loaded: list[ProviderShim] = []
    for sub in sorted(group_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        if not _path_selected(sub, root, only_paths):
            continue
        expected_type = _TYPE_DIRS.get(sub.name)
        if expected_type is not None:
            loaded.extend(
                _load_type_folder(
                    sub,
                    child_group,
                    expected_type,
                    builtin=builtin,
                    root=root,
                    on_error=on_error,
                    quota_probes=quota_probes,
                    manifest=manifest,
                    only_paths=only_paths,
                    keep_names=keep_names,
                )
            )
            continue
        if (sub / "provider.yaml").exists():
            shim = _load_provider_tolerant(
                sub,
                child_group,
                builtin=builtin,
                root=root,
                on_error=on_error,
                quota_probes=quota_probes,
                manifest=manifest,
                keep_names=keep_names,
            )
            if shim is not None:
                loaded.append(shim)
            continue
        # A lane: a named instance of this vendor, holding type directories of
        # its own.  Entered only when something loadable sits below, and only
        # within the depth cap.
        if lane_depth < _MAX_LANE_DEPTH and _lane_holds_shims(sub, root, only_paths):
            loaded.extend(
                _load_group_folder(
                    sub,
                    child_group,
                    builtin=builtin,
                    root=root,
                    on_error=on_error,
                    quota_probes=quota_probes,
                    manifest=manifest,
                    only_paths=only_paths,
                    keep_names=keep_names,
                    lane_depth=lane_depth + 1,
                )
            )
    return loaded
