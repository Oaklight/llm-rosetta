"""The ``llm-rosetta ext`` subcommands.

``list``, ``install``, ``remove`` and ``doctor`` wrap the installer and the
loader in this package; ``new`` and ``check`` wrap the scaffold in
:mod:`llm_rosetta.devtools.ext_scaffold`.  Every function here returns an exit
code: ``0`` success, ``1`` a failure or a problem found, ``2`` a usage error.
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
from collections.abc import Sequence

from llm_rosetta.contracts.extension import (
    ExtensionManifest,
    compatibility_problems,
)

from ._dirs import ExtensionInstallError, configure_extensions_dir, extensions_dir
from ._install import (
    InstalledExtension,
    check_external,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    missing_dependencies,
    pip_available,
    remove_extension,
)
from ._load import ExtensionStatus, list_extensions, load_extensions

PROG = "llm-rosetta ext"


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _add_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        default=None,
        help="Extensions directory (default: $LLM_ROSETTA_EXTENSIONS_DIR, else "
        "the per-user data directory)",
    )


def _build_parser() -> argparse.ArgumentParser:
    from llm_rosetta.devtools.ext_scaffold import add_ext_arguments

    parser = argparse.ArgumentParser(
        prog=PROG, description="Manage llm-rosetta extensions."
    )
    sub = parser.add_subparsers(dest="action", metavar="<action>")

    p = sub.add_parser("list", help="List the extensions that are installed or found")
    _add_dir(p)

    p = sub.add_parser("install", help="Install an extension wheel")
    p.add_argument("source", nargs="?", help="A wheel file or an https URL")
    p.add_argument("--sha256", default=None, help="Expected SHA-256 of the wheel")
    p.add_argument(
        "--pip",
        metavar="SPEC",
        default=None,
        help="Install with pip instead (only where pip is available)",
    )
    _add_dir(p)

    p = sub.add_parser("remove", help="Remove an installed extension")
    p.add_argument("ext_id", metavar="id")
    _add_dir(p)

    p = sub.add_parser("doctor", help="Check extensions and what they need")
    p.add_argument("ext_id", metavar="id", nargs="?")
    _add_dir(p)

    p = sub.add_parser("new", help="Create the folder of a new extension")
    add_ext_arguments(p, command="new")

    p = sub.add_parser("check", help="Check an extension folder")
    add_ext_arguments(p, command="check")
    return parser


def ext_main(argv: Sequence[str] | None = None) -> int:
    """Run ``llm-rosetta ext``.

    Args:
        argv: The arguments after ``ext``; ``sys.argv[1:]`` by default.

    Returns:
        The exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        return 2
    if args.action == "new":
        from llm_rosetta.devtools.ext_scaffold import run_ext_new

        return run_ext_new(args)
    if args.action == "check":
        from llm_rosetta.devtools.ext_scaffold import run_ext_check

        return run_ext_check(args)
    if args.dir is not None:
        configure_extensions_dir(args.dir)
    try:
        if args.action == "list":
            return _list()
        if args.action == "install":
            return _install(parser, args)
        if args.action == "remove":
            return _remove(args.ext_id)
        return _doctor(args.ext_id)
    except ExtensionInstallError as exc:
        _err(f"{PROG} {args.action}: {exc}")
        return 1
    finally:
        if args.dir is not None:
            configure_extensions_dir(None)


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #


def _survey() -> tuple[list[ExtensionStatus], list[InstalledExtension]]:
    """Reload the extensions and return their statuses and the installed ones."""
    load_extensions()
    statuses = list(list_extensions())
    known = {item.id for item in statuses}
    stray = [item for item in installed_extensions() if item.id not in known]
    return statuses, stray


def _list() -> int:
    statuses, stray = _survey()
    if not statuses and not stray:
        print(f"no extensions found (installation directory: {extensions_dir()})")
        return 0
    rows = [
        (
            item.id,
            item.state,
            item.source,
            item.manifest.version if item.manifest else "-",
            item.reason or "",
        )
        for item in statuses
    ]
    rows += [
        (
            item.id,
            "not loadable",
            "installed",
            item.version or "-",
            item.reason or "installed but not discovered",
        )
        for item in stray
    ]
    headers = ("ID", "STATE", "SOURCE", "VERSION", "REASON")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(4)]
    for row in [headers, *rows]:
        cells = [str(row[i]).ljust(widths[i]) for i in range(4)] + [str(row[4])]
        print("  ".join(cells).rstrip())
    return 0


# --------------------------------------------------------------------------- #
# install / remove
# --------------------------------------------------------------------------- #


def _is_url(text: str) -> bool:
    return urllib.parse.urlsplit(text).scheme.lower() in ("http", "https")


def _install(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.pip is not None:
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
        try:
            installed = install_with_pip(args.pip)
        except ExtensionInstallError as exc:
            _err(f"{PROG} install: {exc}")
            return 1
        print(f"installed {installed.id} {installed.version} -> {installed.path}")
        return 0
    if args.source is None:
        parser.error("give a wheel file, a URL, or --pip SPEC")
    if _is_url(args.source):
        if not args.sha256:
            parser.error("installing from a URL requires --sha256")
        installed = install_wheel_url(args.source, sha256=args.sha256)
    else:
        try:
            installed = install_wheel(args.source, sha256=args.sha256)
        except OSError as exc:
            _err(f"{PROG} install: cannot read {args.source}: {exc}")
            return 1
    print(f"installed {installed.id} {installed.version} -> {installed.path}")
    return 0


def _remove(ext_id: str) -> int:
    if remove_extension(ext_id):
        print(f"removed {ext_id}")
        return 0
    _err(f"{PROG} remove: {ext_id!r} is not installed in {extensions_dir()}")
    return 1


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

_PROBLEM_STATES = ("missing", "too_old", "unknown")


def _check_one(
    ext_id: str,
    manifest: ExtensionManifest | None,
    state: str,
    reason: str | None,
    manifests: dict[str, ExtensionManifest],
) -> list[str]:
    """Print one extension's report; return its problems."""
    problems: list[str] = []
    print(f"{ext_id}  {manifest.version if manifest else '-'}  [{state}]")
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
            found = f" {status.found}" if status.found else ""
            print(f"  external {program.name}: {status.state}{found}")
            if status.state in _PROBLEM_STATES:
                detail = f" ({status.detail})" if status.detail else ""
                hint = f"; {program.install_hint}" if program.install_hint else ""
                problems.append(
                    f"external {program.name} is {status.state}{detail}{hint}"
                )
    if state not in ("loaded",) and reason:
        problems.append(f"not loaded: {reason}")
    for text in dict.fromkeys(problems):
        print(f"  problem: {text}")
    if not problems:
        print("  ok")
    return problems


def _doctor(ext_id: str | None) -> int:
    statuses, stray = _survey()
    manifests: dict[str, ExtensionManifest] = {
        item.id: item.manifest for item in statuses if item.manifest is not None
    }
    manifests.update(
        {item.id: item.manifest for item in stray if item.manifest is not None}
    )
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
    if ext_id is not None:
        entries = [entry for entry in entries if entry[0] == ext_id]
        if not entries:
            _err(f"{PROG} doctor: no extension {ext_id!r} found")
            return 1
    if not entries:
        print("no extensions found")
        return 0
    failed = 0
    for entry in entries:
        if _check_one(*entry, manifests):
            failed += 1
    if failed:
        print(f"{failed} extension(s) with problems")
        return 1
    return 0
