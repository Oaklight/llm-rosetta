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

import sys
from collections.abc import Sequence
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


def _rows(catalog: CommandCatalog) -> list[tuple[str, str, str]]:
    rows = [(name, _CORE_SUMMARIES[name], CORE_OWNER) for name in CORE_COMMAND_NAMES]
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
    names = sorted({*CORE_COMMAND_NAMES, *catalog.commands})
    print(f"{PROG}: unknown command {name!r}", file=sys.stderr)
    print(f"available commands: {', '.join(names)}", file=sys.stderr)
    if _gateway_missing(catalog):
        print(_GATEWAY_HINT, file=sys.stderr)
    return 2


def _run_named(name: str, argv: Sequence[str]) -> int:
    from llm_rosetta.contracts.command import resolve_command

    catalog = command_catalog()
    if name in catalog.conflicts:
        print(f"{PROG}: {catalog.conflicts[name]}", file=sys.stderr)
        return 2
    if name not in catalog.commands:
        return _unknown(name)
    try:
        target = resolve_command(name)
    except CommandError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    try:
        return _exit_code(target(list(argv)))
    except SystemExit as exc:
        return _exit_code(exc.code)


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
        return _run_named(rest[0], ["--help"])
    if name == "ext":
        return _ext(rest)
    return _run_named(name, rest)


def _ext(argv: Sequence[str]) -> int:
    from llm_rosetta.extensions._cli import ext_main

    try:
        return ext_main(argv)
    except SystemExit as exc:
        return _exit_code(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
