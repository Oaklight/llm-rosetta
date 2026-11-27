"""The transports kind: ``provides.transports`` against the transport module."""

from __future__ import annotations

import ast
from pathlib import Path

from llm_rosetta.contracts.extension import (
    TRANSPORT_MODULE,
    TRANSPORTS_ATTRIBUTE,
    ExtensionManifest,
)

from ._static import literal_keys


def problems(folder: Path, manifest: ExtensionManifest) -> list[str]:
    """Check the transport module against ``provides.transports``.

    Args:
        folder: The extension folder.
        manifest: Its parsed manifest.

    Returns:
        One sentence per defect; empty when the manifest lists no transport.
    """
    if not manifest.transports:
        return []
    return _transport_problems(folder, manifest.transports)


def _transport_problems(folder: Path, declared: tuple[str, ...]) -> list[str]:
    """Check the transport module against ``provides.transports``.

    The gateway reads a ``TRANSPORTS`` mapping (transport id to factory) from
    ``llm_rosetta_ext.<id>.transport`` when it builds the lane.  The check is
    static, so it never imports the extension.
    """
    module = next(
        (
            path
            for path in (
                folder / TRANSPORT_MODULE / "__init__.py",
                folder / f"{TRANSPORT_MODULE}.py",
            )
            if path.is_file()
        ),
        None,
    )
    if module is None:
        return [
            f"provides.transports lists {', '.join(declared)} but the folder has "
            "no transport module or package"
        ]
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{module.name} does not parse: {exc}"]
    keys = literal_keys(tree, TRANSPORTS_ATTRIBUTE)
    if keys is None:
        return [
            f"the transport module has no TRANSPORTS mapping (a dict literal of "
            f"transport id to factory); the gateway needs it to build "
            f"{', '.join(declared)}"
        ]
    if keys != set(declared):
        return [
            f"TRANSPORTS has {sorted(keys)} but provides.transports lists "
            f"{sorted(declared)}"
        ]
    return []
