"""A read-only listing of the installed extensions, for the admin panel.

Built from what core reports: :func:`~llm_rosetta.api.list_extensions` (the
result of the last load: state and reason, the shims each registered) and
:func:`~llm_rosetta.api.installed_extensions` (what is on disk in the
extensions directory, which may not be loaded yet).  The external programs an
extension drives are checked with :func:`~llm_rosetta.api.check_external`; that
runs commands, so the listing blocks.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from llm_rosetta.api import (
    ExtensionInstallError,
    check_external,
    installed_extensions,
    list_extensions,
    missing_dependencies,
    tool_bin_dir,
    tools_dir,
)


class ProgramRequestError(ValueError):
    """The extension or program an install names cannot be installed.

    Attributes:
        reason: ``unknown_extension``, ``unknown_program``, ``program_required``
            or ``no_install_spec``.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


#: Where an extension was found when only the extensions directory holds it.
DIRECTORY_SOURCE = "extensions directory"


#: The three states of the extension card, from the check's own states.
_CARD_STATUS = {
    "ok": "ready",
    "missing": "missing",
    "broken": "broken",
    "too_old": "too_old",
    "unknown": "unknown",
}


def _search(ext_id: str) -> list[Any]:
    """Where ``ext install-cli`` puts the extension's entry links, if known.

    The shared ``<tools>/bin`` comes second: it holds the ``node`` link an npm
    program's script starts with.
    """
    try:
        return [tool_bin_dir(ext_id), tools_dir() / "bin"]
    except ExtensionInstallError:
        return []


def _install_view(program: Any) -> dict[str, str] | None:
    install = program.install
    if install is None:
        return None
    return {
        "version": install.version,
        "source": "npm" if install.npm is not None else "release",
    }


def _install_command(
    ext_id: str, program: Any, manifest: Any, state: str
) -> str | None:
    """The command that installs the program: only when it can and must run."""
    if program.install is None or state == "ok":
        return None
    command = f"llm-rosetta ext install-cli {ext_id}"
    if len(manifest.external) > 1:
        command += f" --program {program.name}"
    return command


def _external(ext_id: str, manifest: Any) -> list[dict[str, Any]]:
    """The status of each external program a manifest declares."""
    rows: list[dict[str, Any]] = []
    search = _search(ext_id)
    for program in manifest.external:
        try:
            result = check_external(program, search=search)
        except ValueError as exc:  # a malformed minimum is the manifest's fault
            state, found, detail = "unknown", None, str(exc)
        else:
            state, found, detail = result.state, result.found, result.detail
        rows.append(
            {
                "name": program.name,
                "state": state,
                "status": _CARD_STATUS.get(state, "unknown"),
                "found": found,
                "minimum": program.minimum,
                "detail": detail,
                "install_hint": program.install_hint,
                "install": _install_view(program),
                "install_command": _install_command(ext_id, program, manifest, state),
            }
        )
    return rows


def _record(
    ext_id: str,
    *,
    version: str | None,
    source: str,
    state: str,
    reason: str | None,
    path: Any,
    manifest: Any,
    shims: tuple[str, ...],
    present: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "id": ext_id,
        "version": version,
        "source": source,
        "state": state,
        "reason": reason,
        "path": os.fspath(path),
        "maturity": manifest.maturity if manifest else None,
        "shims": list(shims),
        "provides_shims": list(manifest.shims) if manifest else [],
        "transports": list(manifest.transports) if manifest else [],
        "requires": list(manifest.requires.extensions) if manifest else [],
        "external": _external(ext_id, manifest) if manifest else [],
        "missing_dependencies": (
            missing_dependencies(manifest, present) if manifest else []
        ),
    }


def extension_listing() -> list[dict[str, Any]]:
    """Return one record per extension, loaded or not.

    Returns:
        Records in discovery order, then the ones only the extensions
        directory holds, sorted by id.  See
        :meth:`llm_rosetta_gateway.admin_api.AdminExtensions.extension_listing`.
    """
    statuses = list(list_extensions())
    try:
        on_disk = installed_extensions()
    except ExtensionInstallError:
        on_disk = []
    present = dict.fromkeys(
        [*(item.id for item in statuses), *(item.id for item in on_disk)]
    )
    seen = {item.id for item in statuses}
    rows = [
        _record(
            item.id,
            version=item.manifest.version if item.manifest else None,
            source=item.source,
            state=item.state,
            reason=item.reason,
            path=item.path,
            manifest=item.manifest,
            shims=item.shims,
            present=present,
        )
        for item in statuses
    ]
    for item in on_disk:
        if item.id in seen:
            continue
        rows.append(
            _record(
                item.id,
                version=item.version,
                source=DIRECTORY_SOURCE,
                state="installed",
                reason=item.reason or "installed; not loaded yet (reload the shims)",
                path=item.path,
                manifest=item.manifest,
                shims=(),
                present=present,
            )
        )
    return rows


def declared_program(ext_id: str, name: str | None) -> Any:
    """Return the program *name* of extension *ext_id*, as its manifest declares it.

    The install spec on it is the manifest's: a caller cannot supply another.

    Args:
        ext_id: The extension id.
        name: The program's name; ``None`` when the extension has exactly one
            program with an install spec.

    Raises:
        ProgramRequestError: The extension or the program is unknown, the name
            is missing and ambiguous, or the program declares no install spec.
    """
    manifest = None
    for item in list_extensions():
        if item.id == ext_id and item.manifest is not None:
            manifest = item.manifest
    if manifest is None:
        try:
            on_disk = installed_extensions()
        except ExtensionInstallError:
            on_disk = []
        for item in on_disk:
            if item.id == ext_id and item.manifest is not None:
                manifest = item.manifest
    if manifest is None:
        raise ProgramRequestError("unknown_extension", f"no extension {ext_id!r}")
    if name is None:
        installable = [p for p in manifest.external if p.install is not None]
        if len(installable) != 1:
            raise ProgramRequestError(
                "program_required",
                f"{ext_id!r} declares {len(installable)} installable programs; "
                "name one",
            )
        return installable[0]
    for program in manifest.external:
        if program.name == name:
            if program.install is None:
                raise ProgramRequestError(
                    "no_install_spec", f"{name!r} declares no install spec"
                )
            return program
    raise ProgramRequestError(
        "unknown_program", f"{ext_id!r} declares no program {name!r}"
    )
