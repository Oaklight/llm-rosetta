"""Commands an extension adds to the CLI: ``llm-rosetta <area> <verb>``.

An extension that lists ``provides.commands`` owns one *area*, its id spelled
with hyphens (``claude_cli`` is ``claude-cli``).  The manifest maps each verb to
its summary; the ``commands`` module exposes ``COMMANDS``, a mapping of verb to a
callable ``main(argv) -> int``.

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
    compatibility_problems,
)

from ._discovery import discover
from ._load import _read_manifest, import_extension_module

#: The module of an extension that holds its ``COMMANDS`` mapping.
COMMANDS_MODULE = "commands"

#: The mapping that module exposes.
COMMANDS_ATTRIBUTE = "COMMANDS"

__all__ = [
    "COMMANDS_ATTRIBUTE",
    "COMMANDS_MODULE",
    "ExtensionArea",
    "area_verbs",
    "extension_areas",
    "resolve_extension_command",
]


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


def _area_of(manifest: ExtensionManifest, folder: Path) -> ExtensionArea:
    problems = compatibility_problems(manifest)
    return ExtensionArea(
        area=command_area(manifest.id),
        ext_id=manifest.id,
        verbs=tuple(manifest.commands),
        summaries={
            verb: summary.get("en", "") for verb, summary in manifest.commands.items()
        },
        summary=(manifest.summary or {}).get("en", ""),
        path=folder,
        problem="; ".join(problems) if problems else None,
    )


def extension_areas() -> tuple[dict[str, ExtensionArea], list[str]]:
    """Read the CLI areas of every discovered extension from its manifest.

    No extension code is imported.  A folder whose manifest cannot be read, or
    whose folder name differs from its id, is left out and reported.

    Returns:
        ``(areas, problems)``: the areas by area name, and one sentence for each
        folder that was left out.  The first folder to claim an area keeps it.
    """
    areas: dict[str, ExtensionArea] = {}
    problems: list[str] = []
    for found in discover():
        if found.problem is not None:
            continue
        try:
            manifest = _read_manifest(found.path)
        except Exception as exc:  # noqa: BLE001 - one folder must not stop the rest
            problems.append(f"{found.path.name}: {type(exc).__name__}: {exc}")
            continue
        if not manifest.commands:
            continue
        if manifest.id != found.path.name:
            problems.append(
                f"{found.path.name}: plugin.yaml names id {manifest.id!r}; "
                "its commands are not available"
            )
            continue
        area = _area_of(manifest, found.path)
        if area.area in areas:
            problems.append(
                f"{found.path}: the area {area.area!r} is already owned by "
                f"{areas[area.area].path}"
            )
            continue
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
