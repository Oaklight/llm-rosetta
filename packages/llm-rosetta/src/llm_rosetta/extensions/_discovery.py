"""Finding extension folders: the ``llm_rosetta_ext`` namespace and entry points.

Discovery reads directories and package metadata and imports no extension
code.  A namespace package has no module body, and the entry point's value is
resolved to a folder with :func:`importlib.util.find_spec`, which never runs the
extension's ``__init__``.  The same extension reached by both routes is one
folder and is returned once.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

logger = logging.getLogger("llm_rosetta.shims.providers")

#: The top-level namespace every extension is imported under
#: (``llm_rosetta_ext.<id>``).  It is a PEP 420 namespace: no distribution owns
#: it, so any directory on ``sys.path`` can contribute a folder to it.
NAMESPACE = "llm_rosetta_ext"

#: The entry point group an installed extension may declare.  The entry point's
#: name is the extension id and its value is ``llm_rosetta_ext.<id>``.
ENTRY_POINT_GROUP = "llm_rosetta.extensions"

#: The 0.13.0 group, retired in 0.14.0.  Its entry points are reported, not run.
LEGACY_ENTRY_POINT_GROUP = "llm_rosetta.shim_providers"


@dataclass(frozen=True)
class FoundExtension:
    """One folder discovery found.

    Attributes:
        hint: The id the finder expects (the folder name or the entry point
            name); the manifest decides the real id.
        path: The extension's folder.
        source: ``"entry point"`` or ``"namespace"``.
        problem: Why the folder could not be used, when discovery already knows.
    """

    hint: str
    path: Path
    source: str
    problem: str | None = None


def _entry_points(group: str) -> list[metadata.EntryPoint]:
    found = metadata.entry_points()
    if hasattr(found, "select"):
        return list(found.select(group=group))
    return list(found.get(group, ()))


def legacy_entry_point_names() -> list[str]:
    """Names of entry points left in the retired ``shim_providers`` group."""
    return [entry.name for entry in _entry_points(LEGACY_ENTRY_POINT_GROUP)]


def _from_entry_points() -> list[FoundExtension]:
    found: list[FoundExtension] = []
    for entry in _entry_points(ENTRY_POINT_GROUP):
        module = entry.value.partition(":")[0].strip()
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError) as exc:
            found.append(
                FoundExtension(
                    entry.name,
                    Path(module),
                    "entry point",
                    f"cannot locate {module!r}: {type(exc).__name__}: {exc}",
                )
            )
            continue
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if not locations:
            found.append(
                FoundExtension(
                    entry.name,
                    Path(module),
                    "entry point",
                    f"{module!r} is not a package folder",
                )
            )
            continue
        found.append(FoundExtension(entry.name, Path(locations[0]), "entry point"))
    return found


def _from_namespace() -> list[FoundExtension]:
    try:
        namespace = importlib.import_module(NAMESPACE)
    except ModuleNotFoundError:
        return []
    found: list[FoundExtension] = []
    for root in list(getattr(namespace, "__path__", ())):
        base = Path(root)
        if not base.is_dir():
            continue
        for folder in sorted(base.iterdir()):
            if not folder.is_dir() or folder.name.startswith(("_", ".")):
                continue
            if (folder / "plugin.yaml").is_file():
                found.append(FoundExtension(folder.name, folder, "namespace"))
            elif (folder / "provider.yaml").is_file():
                found.append(
                    FoundExtension(
                        folder.name, folder, "namespace", "has no plugin.yaml"
                    )
                )
    return found


def discover() -> list[FoundExtension]:
    """List every extension folder, entry points first, then the namespace.

    Returns:
        One record per distinct folder, in discovery order.
    """
    seen: set[str] = set()
    result: list[FoundExtension] = []
    for item in [*_from_entry_points(), *_from_namespace()]:
        key = str(item.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
