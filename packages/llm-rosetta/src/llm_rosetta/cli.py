"""The ``llm-rosetta`` command: a dispatcher over registered subcommands.

Core owns two names, ``ext`` (manage extensions) and ``help``.  Every other
command is contributed by an installed distribution through
:mod:`llm_rosetta.contracts.command`: the gateway adds ``serve``, ``init`` and
the rest.  Running ``llm-rosetta`` with no arguments prints the list; it never
starts a server.

Only the named command's module is imported, so a command that is not run costs
nothing, and a broken distribution cannot stop another command.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from importlib import util as importlib_util

from llm_rosetta import __version__
from llm_rosetta.contracts.command import (
    CORE_COMMAND_NAMES,
    CORE_OWNER,
    CommandCatalog,
    CommandError,
    command_catalog,
    command_summary,
)

PROG = "llm-rosetta"

_CORE_SUMMARIES = {
    "ext": "Manage extensions: list, install, remove, doctor, new, check",
    "help": "Show this list, or the help of one command",
}

_GATEWAY_HINT = (
    "Gateway commands (serve, init, add, credential, password, db, usage, "
    "migrate) need the llm-rosetta-gateway package: pip install llm-rosetta-gateway"
)


def _gateway_missing(catalog: CommandCatalog) -> bool:
    """Whether the gateway distribution is absent (so its commands are unknown)."""
    return (
        "serve" not in catalog.commands
        and importlib_util.find_spec("llm_rosetta_gateway") is None
    )


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


#: Set it (to anything) to let an unexpected exception raise with its traceback.
DEBUG_ENV = "LLM_ROSETTA_DEBUG"


def _guarded(label: str, call: Callable[[], object]) -> int:
    """Run one command and return its exit code.

    The exit code is ``0`` on success, ``1`` when the command failed and ``2``
    for a usage error; a command that documents another code (``migrate
    --check`` returns ``3`` for pending work) keeps it.  An exception the
    command did not handle is one line, ``llm-rosetta <label>: <message>``, and
    ``1``; set :data:`DEBUG_ENV` to see the traceback instead.

    Args:
        label: What the line names: ``"serve"``, ``"argo tunnel"``.
        call: The command, already bound to its arguments.

    Returns:
        The process exit code.
    """
    try:
        return _exit_code(call())
    except SystemExit as exc:
        return _exit_code(exc.code)
    except Exception as exc:  # noqa: BLE001 - the one place a command's failure ends
        if os.environ.get(DEBUG_ENV):
            raise
        print(f"{PROG} {label}: {str(exc) or type(exc).__name__}", file=sys.stderr)
        return 1


def _rows(catalog: CommandCatalog) -> list[tuple[str, str, str]]:
    rows = [(name, _CORE_SUMMARIES[name], CORE_OWNER) for name in CORE_COMMAND_NAMES]
    areas, _ = _areas()
    for name, area in sorted(areas.items()):
        if name in catalog.commands or name in catalog.conflicts:
            continue
        summary = area.summary or f"Commands: {', '.join(area.verbs)}"
        rows.append((name, summary, f"extension {area.ext_id}"))
    for name in sorted(catalog.commands):
        item = catalog.commands[name]
        rows.append((name, command_summary(item), item.owner))
    return sorted(rows)


def _print_help(file=None) -> None:
    out = file or sys.stdout
    catalog = command_catalog()
    rows = _rows(catalog)
    width = max(len(name) for name, _, _ in rows)
    print(f"usage: {PROG} <command> [arguments]", file=out)
    print(f"       {PROG} help <command>", file=out)
    print(file=out)
    print("commands:", file=out)
    for name, summary, owner in rows:
        print(f"  {name.ljust(width)}  {summary}  [{owner}]", file=out)
    for name, message in sorted(catalog.conflicts.items()):
        print(f"  {name.ljust(width)}  unavailable: {message}", file=out)
    for problem in catalog.problems:
        print(f"note: {problem}", file=out)
    if _gateway_missing(catalog):
        print(file=out)
        print(_GATEWAY_HINT, file=out)


def _unknown(name: str) -> int:
    catalog = command_catalog()
    areas, _ = _areas()
    names = sorted({*CORE_COMMAND_NAMES, *catalog.commands, *areas})
    print(f"{PROG}: unknown command {name!r}", file=sys.stderr)
    print(f"available commands: {', '.join(names)}", file=sys.stderr)
    if _gateway_missing(catalog):
        print(_GATEWAY_HINT, file=sys.stderr)
    return 2


def _areas():
    """The extension areas, with the extensions directory importable first."""
    from llm_rosetta.extensions import activate_extensions_dir, extension_areas

    try:
        activate_extensions_dir()
        return extension_areas()
    except Exception as exc:  # noqa: BLE001 - help must work when discovery fails
        return {}, [f"extensions could not be scanned: {exc}"]


def _run_named(name: str, argv: Sequence[str]) -> int:
    from llm_rosetta.contracts.command import resolve_command

    catalog = command_catalog()
    if name in catalog.conflicts:
        print(f"{PROG}: {catalog.conflicts[name]}", file=sys.stderr)
        return 2
    if name not in catalog.commands:
        areas, _ = _areas()
        if name in areas:
            return _run_area(areas[name], argv)
        return _unknown(name)
    try:
        target = resolve_command(name)
    except CommandError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    return _guarded(name, lambda: target(list(argv)))


def _print_area_help(area, file=None) -> None:
    from llm_rosetta.extensions import area_verbs

    out = file or sys.stdout
    print(f"usage: {PROG} {area.area} <command> [arguments]", file=out)
    print(file=out)
    print(f"commands of the extension {area.ext_id}:", file=out)
    pairs = area_verbs(area)
    width = max((len(verb) for verb, _ in pairs), default=0)
    for verb, summary in pairs:
        print(f"  {verb.ljust(width)}  {summary}".rstrip(), file=out)


def _run_area(area, argv: Sequence[str]) -> int:
    """Run ``llm-rosetta <area> <verb> ...`` for an extension's command."""
    from llm_rosetta.extensions import resolve_extension_command

    if area.problem is not None:
        print(f"{PROG} {area.area}: {area.problem}", file=sys.stderr)
        return 1
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_area_help(area)
        return 0 if argv else 2
    verb, rest = argv[0], list(argv[1:])
    if verb not in area.verbs:
        print(f"{PROG} {area.area}: unknown command {verb!r}", file=sys.stderr)
        print(f"available commands: {', '.join(area.verbs)}", file=sys.stderr)
        return 2
    try:
        target = resolve_extension_command(area, verb)
    except CommandError as exc:
        print(f"{PROG} {area.area}: {exc}", file=sys.stderr)
        return 1
    return _guarded(f"{area.area} {verb}", lambda: target(rest))


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``llm-rosetta``: dispatch a subcommand, or print the command list.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns:
        The process exit code.  ``2`` for an unknown command or a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return 0
    if args[0] in ("-V", "--version"):
        print(f"{PROG} {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name == "help":
        if not rest:
            _print_help()
            return 0
        if rest[0] == "ext":
            return _ext(["--help"])
        catalog = command_catalog()
        if rest[0] not in catalog.commands and rest[0] not in catalog.conflicts:
            areas, _ = _areas()
            if rest[0] in areas:
                _print_area_help(areas[rest[0]])
                return 0
        return _run_named(rest[0], ["--help"])
    if name == "ext":
        return _ext(rest)
    return _run_named(name, rest)


def _ext(argv: Sequence[str]) -> int:
    from llm_rosetta.extensions._cli import ext_main

    return _guarded("ext", lambda: ext_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
