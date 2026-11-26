"""The command contract: how a distribution adds a subcommand to ``llm-rosetta``.

``llm-rosetta`` is a dispatcher that owns two names (``ext`` and ``help``).
Every other command comes from a distribution that is installed beside core:
the gateway contributes ``serve``, ``init`` and the rest, and an extension may
contribute its own.  A command is a :class:`CommandSpec`: a name, a one-line
summary and a *target*, ``"package.module:callable"``.  The callable has the
signature ``main(argv: Sequence[str] | None = None) -> int``, takes the
arguments after the command name, and returns the process exit code.

There are two ways to make a command known:

- **Package metadata.**  The distribution declares an entry point in the group
  :data:`COMMAND_ENTRY_POINT_GROUP`; the entry point's name is the command name
  and its value is ``module:callable``.  Its summary is the first line of the
  callable's docstring, read only when the help text is printed.  Dispatching
  a command imports that command's module and no other.
- **Explicit registration.**  :func:`register_command` records a spec in this
  process.  A host that cannot rely on package metadata, such as the Nuitka
  binary, uses it.

A name taken by two owners is an error that names both
(:class:`CommandConflict`); the same owner registering the same target again is
a no-op, so a distribution can declare an entry point *and* register itself.
The core-owned names (:data:`CORE_COMMAND_NAMES`) cannot be taken.

This module uses the standard library only and imports nothing from the rest of
``llm_rosetta``.
"""

from __future__ import annotations

import importlib
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

#: The entry point group a distribution declares its commands in.
COMMAND_ENTRY_POINT_GROUP = "llm_rosetta.commands"

#: Names the dispatcher owns; no distribution can register or declare them.
CORE_COMMAND_NAMES: tuple[str, ...] = ("ext", "help")

#: The names the gateway distribution owns: its ``gateway`` area and the
#: ``serve`` shortcut.  An extension id spelled like one would be shadowed by
#: the command, so its manifest is refused (the gateway may not be installed,
#: and the name must not depend on that).
GATEWAY_COMMAND_NAMES: tuple[str, ...] = ("gateway", "serve")

#: The owner reported for the core-owned names.
CORE_OWNER = "llm-rosetta"

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_TARGET_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)

__all__ = [
    "COMMAND_ENTRY_POINT_GROUP",
    "CORE_COMMAND_NAMES",
    "GATEWAY_COMMAND_NAMES",
    "CORE_OWNER",
    "CommandCatalog",
    "CommandConflict",
    "CommandError",
    "CommandLoadError",
    "CommandMain",
    "CommandNameError",
    "CommandSpec",
    "RegisteredCommand",
    "UnknownCommand",
    "command_catalog",
    "command_summary",
    "register_command",
    "registered_commands",
    "resolve_command",
]

CommandMain = Callable[[Sequence[str] | None], int]


class CommandError(Exception):
    """Base class of the command contract's errors."""


class CommandNameError(CommandError, ValueError):
    """A command name or target is malformed."""


class CommandConflict(CommandError):
    """A command name is already taken by another owner (or is core-owned)."""


class UnknownCommand(CommandError):
    """No command with that name is registered or declared."""


class CommandLoadError(CommandError):
    """A command's target cannot be imported or is not callable."""


@dataclass(frozen=True)
class CommandSpec:
    """One subcommand.

    Attributes:
        name: The command name, ``^[a-z][a-z0-9-]*$``.
        summary: One line for the help text.
        target: ``"package.module:callable"``; the callable is
            ``main(argv: Sequence[str] | None = None) -> int``.
    """

    name: str
    summary: str
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise CommandNameError(
                f"command name {self.name!r} must match {_NAME_RE.pattern}"
            )
        if not isinstance(self.target, str) or not _TARGET_RE.match(self.target):
            raise CommandNameError(
                f"command target {self.target!r} must look like "
                "'package.module:callable'"
            )


@dataclass(frozen=True)
class RegisteredCommand:
    """A command and where it came from.

    Attributes:
        spec: The command.  ``spec.summary`` is empty for a declared entry
            point until :func:`command_summary` reads the docstring.
        owner: The registering owner, or the distribution that declared the
            entry point.
        source: ``"registered"`` or ``"entry point"``.
    """

    spec: CommandSpec
    owner: str
    source: str


@dataclass(frozen=True)
class CommandCatalog:
    """Every command this process knows, and the declarations it refused.

    Attributes:
        commands: The usable commands by name, registered ones first.
        conflicts: A message for each name that two owners claim; such a name
            is not in *commands*.
        problems: One sentence for each entry point that was ignored.
    """

    commands: dict[str, RegisteredCommand]
    conflicts: dict[str, str]
    problems: tuple[str, ...] = ()


_REGISTRY: dict[str, RegisteredCommand] = {}
_LOCK = threading.Lock()


def _default_owner(target: str) -> str:
    return target.partition(":")[0].partition(".")[0]


def _taken_message(name: str, owners: Sequence[tuple[str, str]]) -> str:
    claims = " and ".join(f"{owner} ({target})" for owner, target in owners)
    return f"command {name!r} is claimed by {claims}"


def register_command(spec: CommandSpec, *, owner: str | None = None) -> None:
    """Make a command known to this process.

    Args:
        spec: The command.
        owner: Who registers it; the target's top-level package by default.
            Use the distribution name, so it matches the entry point the same
            distribution declares.

    Raises:
        CommandConflict: *spec.name* is core-owned, or another owner (or the
            same owner with another target) registered it first.
    """
    who = owner or _default_owner(spec.target)
    if spec.name in CORE_COMMAND_NAMES:
        raise CommandConflict(
            f"command {spec.name!r} is owned by {CORE_OWNER} and cannot be "
            f"taken by {who}"
        )
    with _LOCK:
        held = _REGISTRY.get(spec.name)
        if held is not None:
            if held.owner == who and held.spec.target == spec.target:
                return
            raise CommandConflict(
                _taken_message(
                    spec.name, [(held.owner, held.spec.target), (who, spec.target)]
                )
            )
        _REGISTRY[spec.name] = RegisteredCommand(spec, who, "registered")


def registered_commands() -> tuple[RegisteredCommand, ...]:
    """Return the explicitly registered commands, sorted by name."""
    with _LOCK:
        return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def _unregister_all() -> None:
    """Forget every registration (for tests)."""
    with _LOCK:
        _REGISTRY.clear()


def _entry_points() -> list[metadata.EntryPoint]:
    try:
        return list(metadata.entry_points(group=COMMAND_ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001 - broken metadata must not stop the CLI
        return []


def _owner_of(entry: metadata.EntryPoint) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    return name or _default_owner(entry.value)


def command_catalog() -> CommandCatalog:
    """Merge the registered commands with the declared entry points.

    No command module is imported.  An entry point with a bad name or target,
    or one that takes a core-owned name, is reported in ``problems``; a name
    claimed by two owners is reported in ``conflicts`` and is not dispatchable.

    Returns:
        The catalog.
    """
    commands: dict[str, RegisteredCommand] = dict(
        (item.spec.name, item) for item in registered_commands()
    )
    conflicts: dict[str, str] = {}
    problems: list[str] = []
    claims: dict[str, list[tuple[str, str]]] = {}
    for entry in sorted(_entry_points(), key=lambda e: (e.name, e.value)):
        owner = _owner_of(entry)
        target = entry.value.partition("[")[0].strip()
        if entry.name in CORE_COMMAND_NAMES:
            problems.append(
                f"entry point {entry.name!r} from {owner} is ignored: the name "
                f"is owned by {CORE_OWNER}"
            )
            continue
        try:
            spec = CommandSpec(entry.name, "", target)
        except CommandNameError as exc:
            problems.append(f"entry point from {owner} is ignored: {exc}")
            continue
        held = commands.get(spec.name)
        if held is not None and held.owner == owner and held.spec.target == target:
            continue  # the same owner declared and registered it
        if held is None and spec.name not in conflicts:
            commands[spec.name] = RegisteredCommand(spec, owner, "entry point")
            continue
        peers = claims.setdefault(spec.name, [])
        if held is not None and not peers:
            peers.append((held.owner, held.spec.target))
        peers.append((owner, target))
    for name, owners in claims.items():
        # A second claim of the very same owner and target is no conflict.
        distinct = list(dict.fromkeys(owners))
        if len(distinct) < 2:
            continue
        conflicts[name] = _taken_message(name, distinct)
        commands.pop(name, None)
    return CommandCatalog(commands, conflicts, tuple(problems))


def _load(target: str) -> CommandMain:
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - name the command, keep the cause
        raise CommandLoadError(
            f"cannot import {module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    obj: Any = module
    try:
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise CommandLoadError(f"{module_name!r} has no {attr!r}") from exc
    if not callable(obj):
        raise CommandLoadError(f"{target!r} is not callable")
    return obj


def resolve_command(name: str) -> CommandMain:
    """Import a command's target and return its callable.

    Only the named command's module is imported, and only now.

    Args:
        name: The command name.

    Returns:
        ``main(argv) -> int``.

    Raises:
        UnknownCommand: Nothing registers or declares *name*.
        CommandConflict: Two owners claim *name*.
        CommandLoadError: The target cannot be imported or is not callable.
    """
    catalog = command_catalog()
    if name in catalog.conflicts:
        raise CommandConflict(catalog.conflicts[name])
    found = catalog.commands.get(name)
    if found is None:
        raise UnknownCommand(f"no command named {name!r}")
    return _load(found.spec.target)


def command_summary(command: RegisteredCommand) -> str:
    """Return the one-line summary of a command.

    A registered command carries its summary.  A declared entry point has none,
    so the first line of its callable's docstring is read, which imports the
    target; call this only when help is printed.

    Args:
        command: An item of :func:`command_catalog`.

    Returns:
        The summary, or an empty string when none can be read.
    """
    if command.spec.summary:
        return command.spec.summary
    try:
        doc = (_load(command.spec.target).__doc__ or "").strip()
    except CommandLoadError:
        return ""
    return doc.splitlines()[0].strip() if doc else ""
