"""Loading the discovered extensions: manifest, compatibility, shim registration.

Each extension is handled alone.  Whatever goes wrong with one (no manifest, an
invalid one, a core it cannot run on, a shim it declares but does not provide,
a name another owner already holds) is recorded as that extension's
:class:`ExtensionStatus` and logged; the others load as if it were absent.  The
shims an extension registered before it failed are removed again, so a
half-loaded extension never leaves a half-registered vendor behind.

Names the registry already holds are never replaced: a core built-in wins over
an extension, and the first extension to register a name wins over a later one.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

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
    LEGACY_ENTRY_POINT_GROUP,
    FoundExtension,
    discover,
    legacy_entry_point_names,
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


_STATUS: list[ExtensionStatus] = []
_STATUS_LOCK = threading.Lock()


def list_extensions() -> tuple[ExtensionStatus, ...]:
    """Return the status of every extension the last load found.

    Returns:
        One record per discovered folder, in discovery order.
    """
    with _STATUS_LOCK:
        return tuple(_STATUS)


def _read_manifest(folder: Path) -> ExtensionManifest:
    path = folder / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        document = yaml_load(handle.read())
    return parse_extension_manifest(document)


def _registered_names() -> set[str]:
    from llm_rosetta.shims.provider_shim import list_shims

    return {shim.name for shim in list_shims()}


def _unregister(names: set[str]) -> None:
    from llm_rosetta.shims.provider_shim import unregister_shim

    for name in names:
        unregister_shim(name)


def _label(manifest: ExtensionManifest) -> str:
    return f"{manifest.id} {manifest.version}"


def _load_one(
    found: FoundExtension, seen_ids: dict[str, Path]
) -> tuple[ExtensionStatus, list[ProviderShim]]:
    from llm_rosetta.shims._loader import (
        _PLUGIN_SOURCES,
        load_providers_from_dir,
    )

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

    declared = set(manifest.shims)
    # A reload replaces what this extension registered last time, so its own
    # earlier names are not "held by another owner".  Names another owner holds
    # stay protected.
    own = {
        name
        for name in _registered_names()
        if (_PLUGIN_SOURCES.get(name) or "").partition(" ")[0] == manifest.id
    }
    _unregister(own)
    for name in own:
        _PLUGIN_SOURCES.pop(name, None)
    before = _registered_names()
    taken = set(before)
    try:
        loaded = load_providers_from_dir(
            found.path.parent,
            on_error="raise",
            only_paths=(found.path.name,),
            keep_names=lambda name: name in declared and name not in taken,
        )
    except Exception as exc:  # noqa: BLE001 - name the extension, keep going
        _unregister(_registered_names() - before)
        return result(FAILED, f"{type(exc).__name__}: {exc}"), []

    provided = {shim.name for shim in loaded}
    missing = sorted(declared - provided)
    if missing:
        _unregister(_registered_names() - before)
        held = sorted(name for name in missing if name in taken)
        detail = (
            f"; already registered by another owner: {', '.join(held)}" if held else ""
        )
        return (
            result(
                FAILED,
                f"declares shims it does not register: {', '.join(missing)}{detail}",
            ),
            [],
        )
    for shim in loaded:
        _PLUGIN_SOURCES.setdefault(shim.name, _label(manifest))
    return result(LOADED, shims=sorted(provided)), loaded


def _report_legacy_entry_points() -> None:
    for name in legacy_entry_point_names():
        logger.error(
            "Entry point %r in the group %r is not loaded: the group was retired "
            "in 0.14.0. Publish the shims as an extension (group %r, a folder "
            "with a plugin.yaml imported as llm_rosetta_ext.<id>).",
            name,
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
