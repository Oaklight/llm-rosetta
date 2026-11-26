"""Commands an extension adds to the CLI: ``llm-rosetta <area> <verb>``.

An extension that lists ``provides.commands`` owns one *area*, its id spelled
with hyphens (``claude_cli`` is ``claude-cli``).  The manifest maps each verb to
its summary; the ``commands`` module exposes ``COMMANDS``, a mapping of verb to a
callable ``main(argv) -> int``.

This module is the CLI's side of the extension loader: it registers the
``"commands"`` kind handler, which records the area of each extension as it
loads.  The loader knows nothing of commands.

The dispatcher learns the areas, their verbs and the verbs' summaries from the
manifests alone (:func:`extension_areas` imports no extension code), so help
works without importing the extension and in a build that strips docstrings.  The
extension's code is imported only when a verb runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llm_rosetta.contracts.command import CommandError
from llm_rosetta.contracts.extension import (
    ExtensionManifest,
    command_area,
)
from llm_rosetta.extensions import (
    FAILED,
    LOADED,
    SKIPPED,
    extensions_current,
    import_extension_module,
    list_extensions,
    load_extensions,
    register_kind_handler,
)

#: The module of an extension that holds its ``COMMANDS`` mapping.
COMMANDS_MODULE = "commands"

#: The mapping that module exposes.
COMMANDS_ATTRIBUTE = "COMMANDS"


@dataclass(frozen=True)
class ExtensionArea:
    """The CLI area one extension owns.

    Attributes:
        area: The word after ``llm-rosetta`` (the id with hyphens).
        ext_id: The extension id.
        verbs: The verbs its manifest lists, in manifest order.
        summaries: The English summary of each verb.
        summary: The manifest's English summary, or ``""``.
        path: The extension's folder.
        problem: Why the area cannot run here (an incompatible extension), else
            ``None``.
    """

    area: str
    ext_id: str
    verbs: tuple[str, ...]
    summaries: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""
    path: Path | None = None
    problem: str | None = None


def _area_of(
    manifest: ExtensionManifest, folder: Path, problem: str | None = None
) -> ExtensionArea:
    return ExtensionArea(
        area=command_area(manifest.id),
        ext_id=manifest.id,
        verbs=tuple(manifest.commands),
        summaries={
            verb: summary.get("en", "") for verb, summary in manifest.commands.items()
        },
        summary=(manifest.summary or {}).get("en", ""),
        path=folder,
        problem=problem,
    )


#: The areas the loader has handed over, by area name.
_AREAS: dict[str, ExtensionArea] = {}


def _claim(manifest: ExtensionManifest, folder: Path) -> list[object]:
    """The ``commands`` kind handler: take the area of *manifest*.

    An id has one area (the id with hyphens) and the loader keeps ids unique, so
    two extensions never name the same area and there is nothing to refuse.
    """
    area = _area_of(manifest, folder)
    _AREAS[area.area] = area
    return []


def _release(manifest: ExtensionManifest) -> None:
    for name in [n for n, a in _AREAS.items() if a.ext_id == manifest.id]:
        del _AREAS[name]


register_kind_handler("commands", _claim, undo=_release)


def extension_areas() -> tuple[dict[str, ExtensionArea], list[str]]:
    """Read the CLI areas of the extensions the loader found.

    No extension code is imported.  The extensions are loaded first when nothing
    has loaded them yet (a build without the shim layer) or the extensions
    directory has changed since (a host configured it after the import).  An extension that
    cannot run here (``skipped``) keeps its area, with ``problem`` set to the
    reason; one that failed is reported and has no area.

    Returns:
        ``(areas, problems)``: the areas by area name, and one sentence for each
        folder that was left out.
    """
    if not extensions_current():
        load_extensions()
    areas: dict[str, ExtensionArea] = {}
    problems: list[str] = []
    for item in list_extensions():
        manifest = item.manifest
        if item.state == FAILED:
            if manifest is None or manifest.commands:
                problems.append(f"{item.path.name}: {item.reason}")
        elif manifest is not None and manifest.commands:
            if item.state == SKIPPED:
                area = _area_of(manifest, item.path, item.reason)
                areas.setdefault(area.area, area)
            elif item.state == LOADED:
                area = _AREAS.get(command_area(manifest.id))
                if area is not None and area.ext_id == manifest.id:
                    areas[area.area] = area
    return areas, problems


def _commands_of(ext_id: str) -> dict[str, Callable[..., object]]:
    try:
        module = import_extension_module(ext_id, COMMANDS_MODULE)
    except ImportError as exc:
        raise CommandError(
            f"the extension {ext_id!r} lists commands but its "
            f"{COMMANDS_MODULE} module cannot be imported: {exc}"
        ) from exc
    table = getattr(module, COMMANDS_ATTRIBUTE, None)
    if not isinstance(table, dict):
        raise CommandError(
            f"the extension {ext_id!r} has no {COMMANDS_ATTRIBUTE} mapping in its "
            f"{COMMANDS_MODULE} module"
        )
    return table


def resolve_extension_command(
    area: ExtensionArea, verb: str
) -> Callable[[Sequence[str]], object]:
    """Import an extension's commands module and return one verb's callable.

    Args:
        area: The area, from :func:`extension_areas`.
        verb: The verb; it must be one the manifest lists.

    Returns:
        The callable, ``main(argv) -> int``.

    Raises:
        CommandError: The verb is not listed, the module or mapping is missing,
            or the mapping does not hold a callable for the verb.
    """
    if verb not in area.verbs:
        raise CommandError(
            f"the area {area.area!r} has no command {verb!r}; "
            f"it has: {', '.join(area.verbs)}"
        )
    table = _commands_of(area.ext_id)
    target = table.get(verb)
    if not callable(target):
        raise CommandError(
            f"the extension {area.ext_id!r} lists the command {verb!r} but its "
            f"{COMMANDS_ATTRIBUTE} holds no callable for it"
        )
    return target


def area_verbs(area: ExtensionArea) -> list[tuple[str, str]]:
    """List ``(verb, summary)`` for one area, from its manifest.

    Nothing is imported, so a broken commands module cannot hide the verbs.

    Args:
        area: The area, from :func:`extension_areas`.

    Returns:
        One pair per verb, in manifest order.
    """
    return [(verb, area.summaries.get(verb, "")) for verb in area.verbs]
