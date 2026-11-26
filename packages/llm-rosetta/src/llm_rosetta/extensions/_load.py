"""Loading the discovered extensions: manifest, compatibility, kind handlers.

Each extension is handled alone.  Whatever goes wrong with one (no manifest, an
invalid one, a core it cannot run on, a kind handler that refuses it) is
recorded as that extension's :class:`ExtensionStatus` and logged; the others
load as if it were absent.

This module manages packages only.  What a ``provides`` kind means is the
business of the layer that owns it: that layer registers a handler with
:func:`register_kind_handler` (the shim layer registers ``"shims"``), and a
kind with no load-time work (commands, transports) needs none.  Nothing here
imports the shim layer or the converters.
"""

from __future__ import annotations

import importlib
import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from llm_rosetta._vendor.yaml import load as yaml_load
from llm_rosetta.contracts.extension import (
    MANIFEST_FILENAME,
    ExtensionManifest,
    ExtensionManifestInvalid,
    compatibility_problems,
    parse_extension_manifest,
)
from llm_rosetta.contracts.shim import ProviderShim

from ._dirs import activate_if_present
from ._discovery import (
    ENTRY_POINT_GROUP,
    NAMESPACE,
    LEGACY_ENTRY_POINT_GROUP,
    FoundExtension,
    discover,
    retired_entry_points,
)

logger = logging.getLogger("llm_rosetta.shims.providers")

LOADED = "loaded"
SKIPPED = "skipped"
FAILED = "failed"


@dataclass(frozen=True)
class ExtensionStatus:
    """What happened to one extension at load time.

    Attributes:
        id: The extension id (the manifest's, or the finder's hint when the
            manifest could not be read).
        path: The extension's folder.
        source: How it was found: ``"entry point"`` or ``"namespace"``.
        state: ``"loaded"``, ``"skipped"`` (it cannot run here) or ``"failed"``.
        reason: Why it was skipped or failed; ``None`` when loaded.
        manifest: The parsed manifest, when it could be read.
        shims: The shim names this extension registered.
    """

    id: str
    path: Path
    source: str
    state: str
    reason: str | None = None
    manifest: ExtensionManifest | None = None
    shims: tuple[str, ...] = ()


class KindLoadError(Exception):
    """A kind handler refused an extension; the message is the reason shown."""


#: What a handler receives (the manifest and the extension folder) and what it
#: hands back: the objects it registered, which :func:`load_extensions` returns.
KindHandler = Callable[[ExtensionManifest, Path], Sequence[ProviderShim]]

_KIND_HANDLERS: dict[str, KindHandler] = {}


def register_kind_handler(kind: str, handler: KindHandler) -> None:
    """Register the code that loads one ``provides`` kind of an extension.

    The handler runs once per extension that lists the kind, after the manifest
    and the compatibility checks.  It raises :class:`KindLoadError` (or any
    exception) to fail the extension, and must undo what it registered before it
    does, so a half-loaded extension leaves nothing behind.

    Args:
        kind: The ``provides`` key, e.g. ``"shims"``.
        handler: ``handler(manifest, folder)``; the later registration of a
            kind replaces the earlier.
    """
    _KIND_HANDLERS[kind] = handler


_STATUS: list[ExtensionStatus] = []
_STATUS_LOCK = threading.Lock()


def list_extensions() -> tuple[ExtensionStatus, ...]:
    """Return the status of every extension the last load found.

    Returns:
        One record per discovered folder, in discovery order.
    """
    with _STATUS_LOCK:
        return tuple(_STATUS)


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SUBMODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def import_extension_module(ext_id: str, submodule: str = "") -> ModuleType:
    """Import a module of an installed extension, by extension id.

    The loader reads manifests and shims without importing extension code; a
    host that needs the code (a transport, say) asks for it here, by name.  The
    module is ``llm_rosetta_ext.<ext_id>[.<submodule>]``.

    Args:
        ext_id: The extension id.
        submodule: A dotted path below the extension folder; empty for the
            extension package itself.

    Returns:
        The imported module.

    Raises:
        ValueError: *ext_id* or *submodule* is not a plain module name.
        ImportError: The module does not exist or fails to import.
    """
    if not _ID_RE.match(ext_id):
        raise ValueError(f"{ext_id!r} is not an extension id")
    if submodule and not _SUBMODULE_RE.match(submodule):
        raise ValueError(f"{submodule!r} is not a module path")
    name = f"{NAMESPACE}.{ext_id}" + (f".{submodule}" if submodule else "")
    return importlib.import_module(name)


def _read_manifest(folder: Path) -> ExtensionManifest:
    path = folder / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        document = yaml_load(handle.read())
    return parse_extension_manifest(document)


def _load_one(
    found: FoundExtension, seen_ids: dict[str, Path]
) -> tuple[ExtensionStatus, list[ProviderShim]]:
    def status(state: str, reason: str | None = None, **kw) -> ExtensionStatus:
        return ExtensionStatus(
            id=kw.pop("id", found.hint),
            path=found.path,
            source=found.source,
            state=state,
            reason=reason,
            **kw,
        )

    if found.problem is not None:
        return status(FAILED, found.problem), []
    try:
        manifest = _read_manifest(found.path)
    except ExtensionManifestInvalid as exc:
        return status(FAILED, str(exc)), []
    except Exception as exc:  # noqa: BLE001 - one extension must not stop the rest
        return status(FAILED, f"{type(exc).__name__}: {exc}"), []

    def result(state: str, reason: str | None = None, shims=()):
        return status(
            state, reason, id=manifest.id, manifest=manifest, shims=tuple(shims)
        )

    if manifest.id != found.path.name:
        return (
            result(
                FAILED,
                f"plugin.yaml names id {manifest.id!r} but the folder is "
                f"{found.path.name!r}; an extension is imported under its id",
            ),
            [],
        )
    if manifest.id in seen_ids:
        return (
            result(
                FAILED,
                f"id {manifest.id!r} is already provided by {seen_ids[manifest.id]}",
            ),
            [],
        )
    seen_ids[manifest.id] = found.path
    problems = compatibility_problems(manifest)
    if problems:
        return result(SKIPPED, "; ".join(problems)), []
    if not manifest.shims:
        return result(LOADED), []

    handler = _KIND_HANDLERS.get("shims")
    if handler is None:
        return result(SKIPPED, "no shim layer is loaded to register its shims"), []
    try:
        loaded = list(handler(manifest, found.path))
    except KindLoadError as exc:
        return result(FAILED, str(exc)), []
    except Exception as exc:  # noqa: BLE001 - name the extension, keep going
        return result(FAILED, f"{type(exc).__name__}: {exc}"), []
    return result(LOADED, shims=sorted(shim.name for shim in loaded)), loaded


def _report_legacy_entry_points() -> None:
    for entry in retired_entry_points():
        logger.error(
            "Entry point %r from %s in the group %r is not loaded: the group was "
            "retired in 0.14.0. Publish the shims as an extension (group %r, a "
            "folder with a plugin.yaml imported as llm_rosetta_ext.<id>).",
            entry.name,
            entry.distribution or "an unknown distribution",
            LEGACY_ENTRY_POINT_GROUP,
            ENTRY_POINT_GROUP,
        )


def load_extensions() -> list[ProviderShim]:
    """Find the installed extensions and register their shims.

    Returns:
        The shims the loaded extensions registered.  The status of every
        extension, loaded or not, is available from :func:`list_extensions`.
    """
    _report_legacy_entry_points()
    activate_if_present()
    seen_ids: dict[str, Path] = {}
    statuses: list[ExtensionStatus] = []
    registered: list[ProviderShim] = []
    for found in discover():
        item, shims = _load_one(found, seen_ids)
        statuses.append(item)
        registered.extend(shims)
        if item.state == LOADED:
            logger.info("Loaded extension %s from %s", item.id, item.path)
        else:
            logger.warning(
                "Extension %s (%s) is %s: %s",
                item.id,
                found.path,
                item.state,
                item.reason,
            )
    with _STATUS_LOCK:
        _STATUS[:] = statuses
    return registered
