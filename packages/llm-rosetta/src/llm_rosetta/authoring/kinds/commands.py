"""The commands kind: ``provides.commands`` against the commands module."""

from __future__ import annotations

import ast
from pathlib import Path

from llm_rosetta.contracts.extension import (
    COMMANDS_ATTRIBUTE,
    COMMANDS_MODULE,
    ExtensionManifest,
)

from ._static import literal_keys


def problems(folder: Path, manifest: ExtensionManifest) -> list[str]:
    """Check the commands module against ``provides.commands``.

    Args:
        folder: The extension folder.
        manifest: Its parsed manifest.

    Returns:
        One sentence per defect; empty when the manifest lists no command.
    """
    if not manifest.commands:
        return []
    return _command_problems(folder, tuple(manifest.commands))


def _command_problems(folder: Path, declared: tuple[str, ...]) -> list[str]:
    """Check ``commands.py`` against ``provides.commands``.

    The dispatcher reads a ``COMMANDS`` mapping (verb to callable) from
    ``llm_rosetta_ext.<id>.commands`` when a verb runs.  The check is static,
    so it never imports the extension.
    """
    module = next(
        (
            path
            for path in (
                folder / f"{COMMANDS_MODULE}.py",
                folder / COMMANDS_MODULE / "__init__.py",
            )
            if path.is_file()
        ),
        None,
    )
    if module is None:
        return [
            f"provides.commands lists {', '.join(declared)} but the folder has "
            "no commands module"
        ]
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{module.name} does not parse: {exc}"]
    keys = literal_keys(tree, COMMANDS_ATTRIBUTE)
    if keys is None:
        return [
            "the commands module has no COMMANDS mapping (a dict literal of verb "
            f"to callable); the dispatcher needs it to run {', '.join(declared)}"
        ]
    if keys != set(declared):
        return [
            f"COMMANDS has {sorted(keys)} but provides.commands lists "
            f"{sorted(declared)}"
        ]
    return []
