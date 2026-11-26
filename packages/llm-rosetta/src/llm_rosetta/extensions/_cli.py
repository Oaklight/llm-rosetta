"""The ``llm-rosetta ext`` subcommands.

``list``, ``show``, ``install``, ``remove``, ``update`` and ``doctor`` wrap the
installer, the index and the loader in this package.  ``new`` and ``check`` wrap
the scaffold in :mod:`llm_rosetta.devtools.ext_scaffold`, which hands them in as
:class:`ExtVerb` records (this package imports no dev tool).  ``list``, ``show`` and
``doctor`` take ``--json``; ``install``, ``remove`` and ``update`` take
``--dry-run``, which reports what would change and changes nothing.  Every
function here returns an exit code: ``0`` success, ``1`` a failure or a problem
found, ``2`` a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from llm_rosetta.contracts.extension import (
    ExtensionManifest,
    compatibility_problems,
)

from ._dirs import ExtensionInstallError, configure_extensions_dir, extensions_dir
from ._discovery import retired_entry_points
from ._index import (
    install_from_index,
    load_index,
    plan_updates,
    select_entry,
)
from ._install import (
    InstalledExtension,
    _url_allowed,
    check_external,
    inspect_wheel,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    missing_dependencies,
    pip_available,
    remove_extension,
)
from ._load import ExtensionStatus, list_extensions, load_extensions
from ._probe import check_import

PROG = "llm-rosetta ext"


@dataclass(frozen=True)
class ExtVerb:
    """An ``ext`` action that another layer supplies.

    Attributes:
        name: The action word (``new``).
        help: The one line the action list shows.
        add_arguments: Registers the action's flags on its parser.
        run: Runs the action from the parsed namespace; returns the exit code.
    """

    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _dump(document: Any) -> None:
    print(json.dumps(document, indent=2, ensure_ascii=False))


def _add_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        default=None,
        help="Extensions directory (default: $LLM_ROSETTA_EXTENSIONS_DIR, else "
        "the per-user data directory)",
    )


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="Print one JSON document instead of text"
    )


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Say what would change, and change nothing",
    )


def _add_index(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        default=None,
        metavar="URL_OR_FILE",
        help="The extension index (default: $LLM_ROSETTA_EXTENSION_INDEX, else the "
        "official repository's)",
    )


def _build_parser(verbs: Sequence[ExtVerb] = ()) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="Manage llm-rosetta extensions."
    )
    sub = parser.add_subparsers(dest="action", metavar="<action>")

    p = sub.add_parser("list", help="List the extensions that are installed or found")
    _add_dir(p)
    _add_json(p)

    p = sub.add_parser("show", help="Show one extension: its manifest and its state")
    p.add_argument("ext_id", metavar="id")
    _add_dir(p)
    _add_json(p)

    p = sub.add_parser(
        "install",
        help="Install an extension: a wheel file, an https URL, or an id from the index",
    )
    p.add_argument(
        "source", nargs="?", help="A wheel file, an https URL, or an extension id"
    )
    p.add_argument("--sha256", default=None, help="Expected SHA-256 of the wheel")
    p.add_argument(
        "--version", default=None, help="With an id: install exactly this version"
    )
    p.add_argument(
        "--pip",
        metavar="SPEC",
        default=None,
        help="Install with pip instead (only where pip is available)",
    )
    _add_index(p)
    _add_dir(p)
    _add_dry_run(p)

    p = sub.add_parser("remove", help="Remove an installed extension")
    p.add_argument("ext_id", metavar="id")
    _add_dir(p)
    _add_dry_run(p)

    p = sub.add_parser(
        "update", help="Install the newest compatible version found in the index"
    )
    p.add_argument("ext_id", metavar="id", nargs="?")
    _add_index(p)
    _add_dir(p)
    _add_dry_run(p)

    p = sub.add_parser("doctor", help="Check extensions and what they need")
    p.add_argument("ext_id", metavar="id", nargs="?")
    p.add_argument(
        "--no-import",
        dest="probe",
        action="store_false",
        help="Skip importing each extension in a child process",
    )
    _add_dir(p)
    _add_json(p)

    for verb in verbs:
        verb.add_arguments(sub.add_parser(verb.name, help=verb.help))
    return parser


def ext_main(argv: Sequence[str] | None = None, verbs: Sequence[ExtVerb] = ()) -> int:
    """Run ``llm-rosetta ext``.

    Args:
        argv: The arguments after ``ext``; ``sys.argv[1:]`` by default.
        verbs: The actions another layer adds (``new`` and ``check`` from the
            dev tools); an action not listed here is unknown to the parser.

    Returns:
        The exit code.
    """
    parser = _build_parser(verbs)
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        return 2
    for verb in verbs:
        if args.action == verb.name:
            return verb.run(args)
    if args.dir is not None:
        configure_extensions_dir(args.dir)
    try:
        if args.action == "list":
            return _list(args.json)
        if args.action == "show":
            return _show(args.ext_id, args.json)
        if args.action == "install":
            return _install(parser, args)
        if args.action == "remove":
            return _remove(args.ext_id, args.dry_run)
        if args.action == "update":
            return _update(args.ext_id, args.index, args.dry_run)
        return _doctor(args.ext_id, args.json, args.probe)
    except ExtensionInstallError as exc:
        _err(f"{PROG} {args.action}: {exc}")
        return 1
    finally:
        if args.dir is not None:
            configure_extensions_dir(None)


# --------------------------------------------------------------------------- #
# list, show
# --------------------------------------------------------------------------- #


def _survey() -> tuple[list[ExtensionStatus], list[InstalledExtension]]:
    """Reload the extensions and return their statuses and the installed ones."""
    load_extensions()
    statuses = list(list_extensions())
    known = {item.id for item in statuses}
    stray = [item for item in installed_extensions() if item.id not in known]
    return statuses, stray


def _rows() -> list[dict[str, Any]]:
    statuses, stray = _survey()
    rows = [
        {
            "id": item.id,
            "state": item.state,
            "source": item.source,
            "version": item.manifest.version if item.manifest else None,
            "reason": item.reason,
            "path": str(item.path),
        }
        for item in statuses
    ]
    rows += [
        {
            "id": item.id,
            "state": "not loadable",
            "source": "installed",
            "version": item.version,
            "reason": item.reason or "installed but not discovered",
            "path": str(item.path),
        }
        for item in stray
    ]
    return rows


def _list(as_json: bool = False) -> int:
    rows = _rows()
    if as_json:
        _dump({"directory": str(extensions_dir()), "extensions": rows})
        return 0
    if not rows:
        print(f"no extensions found (installation directory: {extensions_dir()})")
        return 0
    table = [
        (r["id"], r["state"], r["source"], r["version"] or "-", r["reason"] or "")
        for r in rows
    ]
    headers = ("ID", "STATE", "SOURCE", "VERSION", "REASON")
    widths = [max(len(str(r[i])) for r in [headers, *table]) for i in range(4)]
    for row in [headers, *table]:
        cells = [str(row[i]).ljust(widths[i]) for i in range(4)] + [str(row[4])]
        print("  ".join(cells).rstrip())
    return 0


def _manifest_document(manifest: ExtensionManifest) -> dict[str, Any]:
    return {
        "id": manifest.id,
        "name": manifest.name,
        "version": manifest.version,
        "maturity": manifest.maturity,
        "summary": dict(manifest.summary) if manifest.summary else None,
        "tags": list(manifest.tags),
        "homepage": manifest.homepage,
        "platforms": list(manifest.platforms),
        "requires": {
            "surface": manifest.requires.surface,
            "ir": manifest.requires.ir,
            "extensions": list(manifest.requires.extensions),
        },
        "provides": {
            "shims": list(manifest.shims),
            "transports": list(manifest.transports),
            "commands": {verb: dict(text) for verb, text in manifest.commands.items()},
        },
        "external": [
            {
                "name": program.name,
                "check": list(program.check),
                "minimum": program.minimum,
                "install_hint": program.install_hint,
            }
            for program in manifest.external
        ],
    }


def _show_problems(
    manifest: ExtensionManifest | None,
    manifests: dict[str, ExtensionManifest],
    state: str,
    reason: str | None,
) -> list[str]:
    problems: list[str] = []
    if manifest is not None:
        problems += [f"incompatible: {t}" for t in compatibility_problems(manifest)]
        problems += [
            f"requires the extension {name!r}, which is not installed"
            for name in missing_dependencies(manifest, manifests)
        ]
    if state != "loaded" and reason:
        problems.append(f"not loaded: {reason}")
    return problems


def _print_manifest(manifest: ExtensionManifest) -> None:
    print(f"  name:        {manifest.name}")
    if manifest.summary:
        print(f"  summary:     {manifest.summary.get('en', '')}")
    print(f"  maturity:    {manifest.maturity}")
    if manifest.homepage:
        print(f"  homepage:    {manifest.homepage}")
    needs = f"surface {manifest.requires.surface}, ir {manifest.requires.ir}"
    if manifest.requires.extensions:
        needs += f"; extensions {', '.join(manifest.requires.extensions)}"
    print(f"  requires:    {needs}")
    if manifest.platforms:
        print(f"  platforms:   {', '.join(manifest.platforms)}")
    for label, names in (
        ("shims", manifest.shims),
        ("transports", manifest.transports),
        ("commands", tuple(manifest.commands)),
    ):
        if names:
            print(f"  {label + ':':<12} {', '.join(names)}")
    for program in manifest.external:
        minimum = f" >= {program.minimum}" if program.minimum else ""
        print(f"  external:    {program.name}{minimum} ({' '.join(program.check)})")


def _show(ext_id: str, as_json: bool) -> int:
    statuses, stray = _survey()
    manifests = {s.id: s.manifest for s in statuses if s.manifest is not None}
    manifests.update({s.id: s.manifest for s in stray if s.manifest is not None})
    found = next((s for s in statuses if s.id == ext_id), None)
    lost = next((s for s in stray if s.id == ext_id), None)
    if found is not None:
        manifest, state, reason = found.manifest, found.state, found.reason
        source, path = found.source, found.path
    elif lost is not None:
        manifest, state, path = lost.manifest, "not loadable", lost.path
        reason = lost.reason or "installed but not discovered by the loader"
        source = "installed"
    else:
        _err(f"{PROG} show: no extension {ext_id!r} found")
        return 1
    problems = _show_problems(manifest, manifests, state, reason)
    if as_json:
        _dump(
            {
                "id": ext_id,
                "state": state,
                "source": source,
                "path": str(path),
                "reason": reason,
                "problems": problems,
                "manifest": _manifest_document(manifest) if manifest else None,
            }
        )
        return 0
    print(f"{ext_id}  {manifest.version if manifest else '-'}  [{state}]")
    print(f"  path:        {path}")
    print(f"  source:      {source}")
    if manifest is not None:
        _print_manifest(manifest)
    for text in problems:
        print(f"  problem: {text}")
    return 0


# --------------------------------------------------------------------------- #
# install / remove / update
# --------------------------------------------------------------------------- #


def _is_url(text: str) -> bool:
    return urllib.parse.urlsplit(text).scheme.lower() in ("http", "https")


def _is_id(text: str) -> bool:
    """Whether *text* names an extension rather than a file: a bare id, no such file."""
    return bool(_ID_RE.match(text)) and not os.path.exists(text)


def _said(installed: InstalledExtension) -> None:
    print(f"installed {installed.id} {installed.version} -> {installed.path}")


def _previous(ext_id: str) -> str:
    for item in installed_extensions():
        if item.id == ext_id:
            return f" (replaces {item.version or 'an unreadable version'})"
    return ""


def _install_pip(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.source is not None:
        parser.error("give a wheel or --pip SPEC, not both")
    if args.sha256 is not None:
        parser.error("--sha256 does not apply to --pip")
    if not pip_available():
        _err(
            f"{PROG} install: pip is not available in this interpreter; "
            "install a wheel file instead"
        )
        return 2
    if args.dry_run:
        print(f"would build a wheel for {args.pip!r} with pip and install it")
        return 0
    try:
        installed = install_with_pip(args.pip)
    except ExtensionInstallError as exc:
        _err(f"{PROG} install: {exc}")
        return 1
    _said(installed)
    return 0


def _install_url(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.sha256:
        parser.error("installing from a URL requires --sha256")
    if args.dry_run:
        problem = _url_allowed(args.source)
        if problem is not None:
            raise ExtensionInstallError(problem)
        print(f"would download {args.source} and install it (sha256 {args.sha256})")
        return 0
    _said(install_wheel_url(args.source, sha256=args.sha256))
    return 0


def _install_id(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.sha256 is not None:
        parser.error("--sha256 does not apply to an id: the index carries it")
    index = load_index(args.index)
    entry = select_entry(index, args.source, version=args.version)
    if args.dry_run:
        have = {item.id for item in installed_extensions()}
        for name in entry.requires_extensions:
            if name not in have:
                print(f"would install the required extension {name}")
        print(
            f"would install {entry.id} {entry.version} from {entry.url}"
            f"{_previous(entry.id)}"
        )
        return 0
    for installed in install_from_index(args.source, index, version=args.version):
        _said(installed)
    return 0


def _install_file(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            manifest = inspect_wheel(args.source, sha256=args.sha256)
            print(
                f"would install {manifest.id} {manifest.version} from {args.source}"
                f"{_previous(manifest.id)}"
            )
            return 0
        _said(install_wheel(args.source, sha256=args.sha256))
    except OSError as exc:
        _err(f"{PROG} install: cannot read {args.source}: {exc}")
        return 1
    return 0


def _install(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.pip is not None:
        return _install_pip(parser, args)
    if args.source is None:
        parser.error("give a wheel file, a URL, an extension id, or --pip SPEC")
    if args.version is not None and not _is_id(args.source):
        parser.error("--version applies to an extension id")
    if _is_url(args.source):
        return _install_url(parser, args)
    if _is_id(args.source):
        return _install_id(parser, args)
    return _install_file(args)


def _remove(ext_id: str, dry_run: bool = False) -> int:
    if dry_run:
        for item in installed_extensions():
            if item.id == ext_id:
                print(
                    f"would remove {ext_id} {item.version or ''} ({item.path})".rstrip()
                )
                return 0
        _err(f"{PROG} remove: {ext_id!r} is not installed in {extensions_dir()}")
        return 1
    if remove_extension(ext_id):
        print(f"removed {ext_id}")
        return 0
    _err(f"{PROG} remove: {ext_id!r} is not installed in {extensions_dir()}")
    return 1


def _update(ext_id: str | None, index_arg: str | None, dry_run: bool) -> int:
    installed = installed_extensions()
    if ext_id is not None:
        installed = [item for item in installed if item.id == ext_id]
        if not installed:
            _err(f"{PROG} update: {ext_id!r} is not installed in {extensions_dir()}")
            return 1
    if not installed:
        print(f"no extensions installed in {extensions_dir()}")
        return 0
    index = load_index(index_arg)
    failed = 0
    for plan in plan_updates(index, installed):
        if plan.status == "current":
            print(f"{plan.id}: up to date ({plan.current})")
        elif plan.status == "not_in_index":
            print(f"{plan.id}: not in the index ({plan.current})")
        elif plan.status == "no_compatible":
            print(f"{plan.id}: {plan.detail}")
        else:
            assert plan.target is not None
            move = f"{plan.current or '?'} -> {plan.target.version}"
            if dry_run:
                print(f"{plan.id}: would update {move}")
                continue
            try:
                install_from_index(plan.id, index, version=plan.target.version)
            except ExtensionInstallError as exc:
                _err(f"{plan.id}: update failed: {exc}")
                failed += 1
            else:
                print(f"{plan.id}: updated {move}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

_PROBLEM_STATES = ("missing", "too_old", "unknown")


def _examine(
    ext_id: str,
    manifest: ExtensionManifest | None,
    state: str,
    reason: str | None,
    manifests: dict[str, ExtensionManifest],
    probe: bool,
) -> dict[str, Any]:
    """Collect one extension's findings: its problems, programs and import check."""
    problems: list[str] = []
    external: list[dict[str, Any]] = []
    imported: dict[str, Any] | None = None
    if manifest is not None:
        problems += [
            f"incompatible: {text}" for text in compatibility_problems(manifest)
        ]
        for name in missing_dependencies(manifest, manifests):
            problems.append(f"requires the extension {name!r}, which is not installed")
        for program in manifest.external:
            try:
                status = check_external(program)
            except ValueError as exc:
                problems.append(f"external {program.name}: {exc}")
                continue
            external.append(
                {
                    "name": program.name,
                    "state": status.state,
                    "found": status.found,
                    "minimum": status.minimum,
                    "detail": status.detail,
                }
            )
            if status.state in _PROBLEM_STATES:
                detail = f" ({status.detail})" if status.detail else ""
                hint = f"; {program.install_hint}" if program.install_hint else ""
                problems.append(
                    f"external {program.name} is {status.state}{detail}{hint}"
                )
        if probe and state == "loaded":
            result = check_import(ext_id, manifest)
            imported = {
                "ok": result.ok,
                "detail": result.detail,
                "isolated": result.isolated,
            }
            if not result.ok:
                problems.append(f"import failed: {result.detail}")
    if state not in ("loaded",) and reason:
        problems.append(f"not loaded: {reason}")
    return {
        "id": ext_id,
        "version": manifest.version if manifest else None,
        "state": state,
        "problems": list(dict.fromkeys(problems)),
        "external": external,
        "import": imported,
    }


def _print_examined(found: dict[str, Any]) -> None:
    print(f"{found['id']}  {found['version'] or '-'}  [{found['state']}]")
    for program in found["external"]:
        shown = f" {program['found']}" if program["found"] else ""
        print(f"  external {program['name']}: {program['state']}{shown}")
    for text in found["problems"]:
        print(f"  problem: {text}")
    if not found["problems"]:
        print("  ok")


def _doctor_entries(
    statuses: list[ExtensionStatus], stray: list[InstalledExtension]
) -> list[tuple[str, ExtensionManifest | None, str, str | None]]:
    entries: list[tuple[str, ExtensionManifest | None, str, str | None]] = [
        (s.id, s.manifest, s.state, s.reason) for s in statuses
    ]
    entries += [
        (
            s.id,
            s.manifest,
            "not loadable",
            s.reason or "installed but not discovered by the loader",
        )
        for s in stray
    ]
    return entries


def _doctor(ext_id: str | None, as_json: bool = False, probe: bool = True) -> int:
    statuses, stray = _survey()
    manifests: dict[str, ExtensionManifest] = {
        item.id: item.manifest for item in statuses if item.manifest is not None
    }
    manifests.update(
        {item.id: item.manifest for item in stray if item.manifest is not None}
    )
    entries = _doctor_entries(statuses, stray)
    if ext_id is not None:
        entries = [entry for entry in entries if entry[0] == ext_id]
        if not entries:
            _err(f"{PROG} doctor: no extension {ext_id!r} found")
            return 1
    retired = retired_entry_points() if ext_id is None else []
    findings = [_examine(*entry, manifests, probe) for entry in entries]
    failed = sum(1 for found in findings if found["problems"])
    code = 1 if failed or retired else 0
    if as_json:
        _dump({"extensions": findings, "retired": [old.describe() for old in retired]})
        return code
    if not findings and not retired:
        print("no extensions found")
        return 0
    for found in findings:
        _print_examined(found)
    for old in retired:
        print(f"problem: {old.describe()}")
    if failed:
        print(f"{failed} extension(s) with problems")
    if retired:
        print(f"{len(retired)} retired entry point(s)")
    return code
