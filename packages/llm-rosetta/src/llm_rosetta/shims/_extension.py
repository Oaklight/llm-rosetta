"""The shim layer's handler for the ``shims`` kind of an extension.

The extension loader manages packages and knows no shim; this module is what
the shim layer registers with it (:func:`llm_rosetta.extensions.register_kind_handler`).
It registers the shims an extension declares, records which extension owns each,
and undoes both when the extension cannot be loaded whole.

Names the registry already holds are never replaced: a core built-in wins over
an extension, and the first extension to register a name wins over a later one.
"""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.contracts.extension import ExtensionManifest
from llm_rosetta.contracts.shim import ProviderShim
from llm_rosetta.extensions import KindLoadError

from ._loader import _PLUGIN_SOURCES, load_providers_from_dir
from .provider_shim import list_shims, unregister_shim


def _registered_names() -> set[str]:
    return {shim.name for shim in list_shims()}


def _unregister(names: set[str]) -> None:
    for name in names:
        unregister_shim(name)


def _own_names(extension_id: str) -> set[str]:
    return {
        name
        for name in _registered_names()
        if (_PLUGIN_SOURCES.get(name) or "").partition(" ")[0] == extension_id
    }


def forget_extension_shims(manifest: ExtensionManifest) -> None:
    """Unregister every shim the extension *manifest* registered.

    The extension loader calls this when a later kind of the same extension
    fails, so the extension is not left half loaded.
    """
    own = _own_names(manifest.id)
    _unregister(own)
    for name in own:
        _PLUGIN_SOURCES.pop(name, None)


def load_extension_shims(
    manifest: ExtensionManifest, folder: Path
) -> list[ProviderShim]:
    """Register the shims *manifest* declares, from *folder*.

    Args:
        manifest: The extension's parsed manifest.
        folder: The extension's folder.

    Returns:
        The shims it registered.

    Raises:
        KindLoadError: A declared shim is not provided, or another owner holds
            the name.  Anything the loader raises is passed through.
    """
    declared = set(manifest.shims)
    # A reload replaces what this extension registered last time, so its own
    # earlier names are not "held by another owner".  Names another owner holds
    # stay protected.
    forget_extension_shims(manifest)
    before = _registered_names()
    taken = set(before)
    try:
        loaded = load_providers_from_dir(
            folder.parent,
            on_error="raise",
            only_paths=(folder.name,),
            keep_names=lambda name: name in declared and name not in taken,
        )
    except Exception:
        _unregister(_registered_names() - before)
        raise

    provided = {shim.name for shim in loaded}
    missing = sorted(declared - provided)
    if missing:
        _unregister(_registered_names() - before)
        held = sorted(name for name in missing if name in taken)
        detail = (
            f"; already registered by another owner: {', '.join(held)}" if held else ""
        )
        raise KindLoadError(
            f"declares shims it does not register: {', '.join(missing)}{detail}"
        )
    label = f"{manifest.id} {manifest.version}"
    for shim in loaded:
        _PLUGIN_SOURCES.setdefault(shim.name, label)
    return loaded
