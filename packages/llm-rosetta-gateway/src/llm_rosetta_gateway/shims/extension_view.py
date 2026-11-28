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
    extension_origin,
    get_shim,
    installed_extensions,
    list_extensions,
    missing_dependencies,
    plan_removal,
    tool_bin_dir,
    tools_dir,
    version_key,
)
from llm_rosetta.contracts.extension import ExternalProgram


class ProgramRequestError(ValueError):
    """The extension or program an install names cannot be installed.

    Attributes:
        reason: ``unknown_extension``, ``unknown_program``, ``program_required``
            or ``no_install_spec``.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


#: The managed Node.js, which a shim names as ``node`` in ``needs``.  It is put in
#: place by an npm program's install, never by itself.
_NODE = ExternalProgram(
    "node",
    ("node", "--version"),
    minimum="22.0.0",
    install_hint="installed together with an npm program (llm-rosetta ext install-cli)",
)

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


def _shim_needs(names: tuple[str, ...]) -> dict[str, list[str] | None]:
    """The ``needs`` of each registered shim: ``None`` means every program."""
    found: dict[str, list[str] | None] = {}
    for name in names:
        shim = get_shim(name)
        if shim is not None:
            found[name] = None if shim.needs is None else list(shim.needs)
    return found


def _used_by(name: str, shim_needs: Mapping[str, list[str] | None]) -> list[str]:
    """The shims that need the program *name*."""
    return [
        shim for shim, needs in shim_needs.items() if needs is None or name in needs
    ]


def _external(
    ext_id: str, manifest: Any, shim_needs: Mapping[str, list[str] | None]
) -> list[dict[str, Any]]:
    """The status of each program a manifest declares, plus the managed ``node``
    when a shim names it.  Each row says which shims need it."""
    rows: list[dict[str, Any]] = []
    search = _search(ext_id)
    programs = list(manifest.external)
    if any(needs is not None and "node" in needs for needs in shim_needs.values()):
        programs.append(_NODE)
    for program in programs:
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
                "used_by": _used_by(program.name, shim_needs),
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
    origin = extension_origin(path)
    shim_needs = _shim_needs(shims or tuple(manifest.shims if manifest else ()))
    return {
        "id": ext_id,
        "version": version,
        "origin": origin,
        "remove_hint": plan_removal(ext_id, path=path).hint
        if origin == "image"
        else None,
        "shim_needs": shim_needs,
        "source": source,
        "state": state,
        "reason": reason,
        "path": os.fspath(path),
        "maturity": manifest.maturity if manifest else None,
        "shims": list(shims),
        "provides_shims": list(manifest.shims) if manifest else [],
        "transports": list(manifest.transports) if manifest else [],
        "requires": list(manifest.requires.extensions) if manifest else [],
        "external": _external(ext_id, manifest, shim_needs) if manifest else [],
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


def locate_extension(ext_id: str) -> tuple[Any, str | None] | None:
    """Return the folder and version of extension *ext_id*, or ``None``.

    The loaded extension is preferred (it may live outside the extensions
    directory); the extensions directory is read otherwise.
    """
    for item in list_extensions():
        if item.id == ext_id:
            return item.path, item.manifest.version if item.manifest else None
    try:
        on_disk = installed_extensions()
    except ExtensionInstallError:
        return None
    for item in on_disk:
        if item.id == ext_id:
            return item.path, item.version
    return None


def removal_view(ext_id: str) -> dict[str, Any] | None:
    """The removal plan of *ext_id* as JSON, or ``None`` when it is not installed."""
    found = locate_extension(ext_id)
    if found is None:
        return None
    return plan_view(plan_removal(ext_id, path=found[0]))


def plan_view(plan: Any) -> dict[str, Any]:
    """A :class:`~llm_rosetta.api.RemovalPlan` as a JSON-safe mapping."""

    def rows(entries: Any) -> list[dict[str, Any]]:
        return [{"path": e.path, "size": e.size, "note": e.note} for e in entries]

    return {
        "id": plan.id,
        "version": plan.version,
        "origin": plan.origin,
        "removable": plan.removable,
        "delete": rows(plan.delete),
        "keep": rows(plan.keep),
        "hint": plan.hint,
    }


def extension_versions() -> dict[str, str | None]:
    """The installed version of every extension, loaded or on disk."""
    found: dict[str, str | None] = {}
    for item in list_extensions():
        found[item.id] = item.manifest.version if item.manifest else None
    try:
        on_disk = installed_extensions()
    except ExtensionInstallError:
        on_disk = []
    for item in on_disk:
        found.setdefault(item.id, item.version)
    return found


def _published(
    catalogue: Mapping[str, Any], ext_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """The first readable source that publishes *ext_id*, with its vendor row."""
    for source in catalogue.get("sources", ()):
        if not source.get("ok"):
            continue
        for vendor in source.get("vendors", ()):
            if vendor.get("vendor") == ext_id:
                return source, vendor
    return None


def update_report(
    versions: Mapping[str, str | None],
    catalogue: Mapping[str, Any],
    ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare installed versions with what the manifest sources publish.

    Every extension is judged the same way, whatever put it on the host.  A
    source that could not be read is named in the error of an extension nothing
    else publishes: a failed read is never reported as "up to date".
    """
    # A source that answered from its cache after a failed refetch is as good as
    # unreadable for "is it up to date": its copy may be old.
    unreadable = [
        f"{s.get('id')}: {s.get('error') or 'unreadable'}"
        for s in catalogue.get("sources", ())
        if not s.get("ok") or (s.get("stale") and s.get("error"))
    ]
    rows: list[dict[str, Any]] = []
    for ext_id, current in versions.items():
        if ids is not None and ext_id not in ids:
            continue
        row: dict[str, Any] = {
            "id": ext_id,
            "current": current,
            "available": None,
            "newest": None,
            "source": None,
            "versions": [],
            "error": None,
        }
        hit = _published(catalogue, ext_id)
        if hit is None:
            row["status"] = "failed" if unreadable else "not_published"
            if unreadable:
                row["error"] = "; ".join(unreadable)
            rows.append(row)
            continue
        source, vendor = hit
        row.update(
            source=source.get("id"),
            newest=vendor.get("newest"),
            available=vendor.get("installable"),
            versions=list(vendor.get("versions") or ()),
        )
        if row["available"] is None:
            row["status"] = "no_compatible"
            row["error"] = vendor.get("newest_problem")
        elif current is None or version_key(row["available"]) > version_key(current):
            row["status"] = "update"
        elif source.get("stale") and source.get("error"):
            # Only an old copy of the source answered: not a confirmation.
            row["status"] = "failed"
            row["error"] = f"{source.get('id')}: {source['error']} (cached copy only)"
        else:
            row["status"] = "current"
        rows.append(row)
    return rows


def pick_release(
    catalogue: Mapping[str, Any], ext_id: str, version: str | None
) -> tuple[str, str] | str:
    """The manifest URL and version to install, or one sentence saying why not."""
    hit = _published(catalogue, ext_id)
    if hit is None:
        return f"no readable source publishes {ext_id!r}"
    source, vendor = hit
    if version is None:
        version = vendor.get("installable")
        if version is None:
            return f"no release of {ext_id!r} runs on this core"
    elif version not in (vendor.get("versions") or ()):
        return f"{source.get('id')} publishes no {ext_id} {version}"
    return str(source["url"]), version
