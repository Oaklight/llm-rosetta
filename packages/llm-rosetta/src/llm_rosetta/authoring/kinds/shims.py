"""The shims kind: the shim tree against the contract and ``provides.shims``."""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.contracts.extension import ExtensionManifest

from ..tree_contract import check_tree


def tree_shims(folder: Path) -> set[str]:
    """Load the folder's shim tree off to the side and name what registered."""
    from llm_rosetta.contracts.shim import load_providers_from_dir
    from llm_rosetta.shims.provider_shim import staging_registry

    class _Discard(Exception):
        """Leaves the staging block so the live registry is not replaced."""

    names: set[str] = set()
    try:
        with staging_registry():
            loaded = load_providers_from_dir(
                folder.parent, only_paths=(folder.name,), on_error="raise"
            )
            names = {shim.name for shim in loaded}
            raise _Discard
    except _Discard:
        pass
    return names


def problems(folder: Path, manifest: ExtensionManifest) -> list[str]:
    """Check the folder's shim tree.

    The tree must follow the shim tree contract
    (:func:`~llm_rosetta.authoring.tree_contract.check_tree`, with the folder as
    the one vendor), load through the real loader, and register exactly the
    shims ``provides.shims`` lists.

    Args:
        folder: The extension folder.
        manifest: Its parsed manifest.

    Returns:
        One sentence per defect.
    """
    found = [
        f"shim tree: {finding.render(folder)}"
        for finding in check_tree(folder.parent, only=folder.name)
    ]
    try:
        registered = tree_shims(folder)
    except Exception as exc:  # noqa: BLE001 - name the loader's complaint
        return found + [f"shim tree does not load: {type(exc).__name__}: {exc}"]
    declared = set(manifest.shims)
    found.extend(
        f"provides.shims lists {name!r} but the tree does not register it"
        for name in sorted(declared - registered)
    )
    found.extend(
        f"the tree registers {name!r} but provides.shims does not list it"
        for name in sorted(registered - declared)
    )
    return found
