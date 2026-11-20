"""``llm-rosetta-migrate`` — the one-shot migration for the 0.14.0 removals.

Release 0.14.0 removes the alias tables that translated old provider, shim and
model-type spellings.  A retired name now fails with an error that names its
replacement; this command is what that error points to.  It rewrites a config
file, rewrites the persisted tags in a gateway database, and scans source code
for the old names::

    llm-rosetta-migrate --config config.jsonc            # dry run: diff + report
    llm-rosetta-migrate --config config.jsonc --write    # .bak copy, then rewrite
    llm-rosetta-migrate --db /var/lib/llm-rosetta --write
    llm-rosetta-migrate --scan src/ --json

The names come from :mod:`llm_rosetta.retired_names` and from nothing else, so
the command keeps working after the alias tables are deleted.  It never
guesses: a name it cannot map is reported and left in place, and the exit code
says so.

Exit codes: ``0`` nothing to do, or everything mapped; ``1`` something was
reported that needs a person (an unmapped item, or a ``--scan`` finding);
``2`` the command could not run (unreadable file, parse error, database held by
a running gateway); ``3`` ``--check`` found pending rewrites.

Configs are edited as text (:mod:`.jsonc_spans`): comments, key order and
formatting outside the rewritten names are untouched.  The database step runs
in one transaction after a backup, and refuses while a process holds the file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_rosetta.api import FORMATS_BY_TYPE, retired_names

from .jsonc_spans import (
    Edits,
    Member,
    Node,
    ParseError,
    append_item,
    append_member,
    line_of,
    parse,
)

__all__ = ["Item", "main", "migrate_config_text", "migrate_db", "scan_path"]

#: The report schema version, for ``--json`` readers.
REPORT_VERSION = 1

_DB_FILENAME = "gateway.db"

#: The per-endpoint keys a model type spells into a provider entry
#: (``embedding_format`` ...); mirrors the keys the config loader reads.
_ENDPOINT_SUFFIXES = ("format", "path", "shim", "enabled", "base_url")
#: The pool sections a model type names at the top level of the config.
_SECTION_SUFFIXES = ("providers", "models")
#: ``default_<type>_<suffix>`` top-level keys.
_DEFAULT_SUFFIXES = ("format", "path")


@dataclass
class Item:
    """One line of the report.

    Attributes:
        scope: ``config``, ``db`` or ``scan``.
        action: ``rewrite`` (done, or done on ``--write``), ``add`` (a value
            that must be added alongside a rewrite), or ``unmapped`` (needs a
            person; never changed).
        file: The file the item is about.
        where: A dotted path inside a config, or ``table.column`` in a database.
        line: 1-based line number, for text files.
        old: The retired spelling.
        new: What replaces it.
        count: Rows affected, for a database rewrite.
        detail: Why it was reported, or how to fix it.
    """

    scope: str
    action: str
    file: str
    where: str = ""
    line: int | None = None
    old: str | None = None
    new: str | None = None
    count: int | None = None
    detail: str = ""

    def render(self) -> str:
        """One human-readable line."""
        place = self.file + (f":{self.line}" if self.line else "")
        if self.where:
            place += f" [{self.where}]"
        if self.action == "unmapped":
            return f"{place}: UNMAPPED {self.old!r}: {self.detail}"
        change = f"{self.old!r} -> {self.new!r}" if self.old else f"add {self.new!r}"
        extra = f" ({self.count} rows)" if self.count is not None else ""
        extra += f"  {self.detail}" if self.detail else ""
        return f"{place}: {change}{extra}"


# --------------------------------------------------------------------------
# The vocabulary, read from the retired-names table and nowhere else.
# --------------------------------------------------------------------------


def _model_type_map() -> dict[str, str]:
    """Retired model-type spelling -> replacement."""
    return {
        row.name: row.replacement
        for row in retired_names.RETIRED_NAMES
        if row.kind == retired_names.MODEL_TYPE
    }


def format_of(replacement: str) -> str:
    """The wire format a provider-type replacement stands for.

    A provider-type replacement may be a shim name (``openai--open_responses``
    for ``openai_responses``); a column or key that holds a *format* wants the
    part after ``--``.
    """
    return replacement.rsplit("--", 1)[-1]


def _format_tag_map() -> dict[str, str]:
    """Retired provider-type spelling -> the format it named."""
    return {
        row.name: format_of(row.replacement)
        for row in retired_names.RETIRED_NAMES
        if row.kind == retired_names.PROVIDER_TYPE
    }


def _current_formats() -> frozenset[str]:
    """Every format name the catalog knows, in any model type."""
    return frozenset(name for names in FORMATS_BY_TYPE.values() for name in names)


def _distinctive_retired() -> frozenset[str]:
    """Retired names safe to flag outside a known slot.

    A bare word (``google``, ``anthropic``, ``embedding``) is ordinary text in
    a config; a compound one (``argo--openai_chat``, ``openai_chat``) is not.
    A name that is also a current format (``google_generate``) is excluded.
    """
    current = _current_formats()
    return frozenset(
        row.name
        for row in retired_names.RETIRED_NAMES
        if row.kind != retired_names.MODEL_TYPE
        and ("--" in row.name or "_" in row.name or "-" in row.name)
        and row.name not in current
    )


# --------------------------------------------------------------------------
# Config migration
# --------------------------------------------------------------------------


class _ConfigPlanner:
    """Plans the text edits that migrate one parsed config document."""

    def __init__(self, text: str, file: str) -> None:
        self.text = text
        self.file = file
        self.edits = Edits(text)
        self.items: list[Item] = []
        self.presets: dict[str, str] = {}
        self._handled: set[int] = set()
        self._types = _model_type_map()
        self._type_spellings = set(self._types) | set(self._types.values())
        self._current = _current_formats()

    # -- reporting --------------------------------------------------------

    def _rewrite(self, node: Node, new: str, where: str) -> None:
        self.edits.replace_string(node, new)
        self._handled.add(id(node))
        self.items.append(
            Item(
                "config",
                "rewrite",
                self.file,
                where,
                line_of(self.text, node.start),
                str(node.value),
                new,
            )
        )

    def _unmapped(self, node: Node, where: str, old: str, detail: str) -> None:
        self.items.append(
            Item(
                "config",
                "unmapped",
                self.file,
                where,
                line_of(self.text, node.start),
                old,
                detail=detail,
            )
        )

    # -- value slots ------------------------------------------------------

    def _shim(self, node: Node, where: str) -> None:
        self._handled.add(id(node))
        if node.kind != "str":
            return
        row = retired_names.lookup(node.value, retired_names.SHIM)
        if row is None:
            return
        self._rewrite(node, row.replacement, where)
        if row.preset:
            self.presets.setdefault(row.preset, row.name)

    def _shim_list(self, node: Node, where: str) -> None:
        if node.kind != "arr":
            return
        for index, item in enumerate(node.items):
            self._shim(item, f"{where}[{index}]")

    def _type(self, node: Node, where: str) -> None:
        """A provider ``type:``: a format first, else a shim name."""
        self._handled.add(id(node))
        if node.kind != "str":
            return
        row = retired_names.lookup(node.value, retired_names.PROVIDER_TYPE)
        if row is not None:
            self._rewrite(node, row.replacement, where)
        elif node.value not in self._current:
            self._shim(node, where)

    def _format(self, node: Node, where: str) -> None:
        """A typed ``*_format`` / ``format`` value: a bare format name."""
        self._handled.add(id(node))
        if node.kind != "str":
            return
        row = retired_names.lookup(node.value, retired_names.PROVIDER_TYPE)
        if row is not None:
            self._rewrite(node, format_of(row.replacement), where)

    def _model_type(self, node: Node, where: str) -> None:
        self._handled.add(id(node))
        if node.kind == "str" and node.value in self._types:
            self._rewrite(node, self._types[node.value], where)

    # -- keys -------------------------------------------------------------

    def _rename_key(self, obj: Node, member: Member, new: str, where: str) -> None:
        """Rename a key unless the replacement already exists beside it."""
        if obj.member(new) is not None:
            self._unmapped(
                member.key,
                where,
                str(member.key.value),
                f"both the old key and {new!r} are present; merge them by hand",
            )
            return
        self._rewrite(member.key, new, where)

    def _typed_key(
        self, key: str, suffixes: Iterable[str], default: bool
    ) -> tuple[str, str] | None:
        """Split ``[default_]<type>_<suffix>`` into ``(type, suffix)``."""
        body = key
        if default:
            if not key.startswith("default_"):
                return None
            body = key[len("default_") :]
        for spelling in self._type_spellings:
            for suffix in suffixes:
                if body == f"{spelling}_{suffix}":
                    return spelling, suffix
        return None

    # -- containers -------------------------------------------------------

    def _endpoint_member(
        self, obj: Node, member: Member, where: str, scope: str
    ) -> None:
        """Handle one ``<type>_<suffix>`` key of a provider entry or the root."""
        key = str(member.key.value)
        suffixes = _ENDPOINT_SUFFIXES + (_SECTION_SUFFIXES if scope == "root" else ())
        parts = self._typed_key(key, suffixes, default=False)
        default = False
        if parts is None and scope == "root":
            parts = self._typed_key(key, _DEFAULT_SUFFIXES, default=True)
            default = parts is not None
        if parts is None:
            return
        spelling, suffix = parts
        if spelling in self._types:
            new_type = self._types[spelling]
            new = f"default_{new_type}_{suffix}" if default else f"{new_type}_{suffix}"
            self._rename_key(obj, member, new, where)
        value = member.value
        if suffix == "format":
            self._format(value, where)
        elif suffix == "shim":
            self._shim(value, where)
        elif suffix == "providers" and value.kind == "obj":
            self._entries(value, where, in_section=True)
        elif suffix == "models" and value.kind == "obj":
            self._models(value, where)

    def _face(self, obj: Node, member: Member, where: str) -> None:
        """A nested ``<type>: {shim, format, ...}`` block of a provider entry."""
        key = str(member.key.value)
        if key not in self._type_spellings or member.value.kind != "obj":
            return
        if key in self._types:
            self._rename_key(obj, member, self._types[key], where)
        for inner in member.value.members:
            name = str(inner.key.value)
            if name == "shim":
                self._shim(inner.value, f"{where}.shim")
            elif name == "format":
                self._format(inner.value, f"{where}.format")

    def _entry(self, name: str, entry: Node, where: str, in_section: bool) -> None:
        keys = {str(m.key.value) for m in entry.members}
        identity = ("shim", "type")
        if not in_section and not any(k in keys for k in identity):
            self._bare_key(name, entry, where)
        for member in entry.members:
            key = str(member.key.value)
            here = f"{where}.{key}"
            if key == "shim":
                self._shim(member.value, here)
            elif key == "type":
                self._type(member.value, here)
            elif key == "format" and in_section:
                self._format(member.value, here)
            else:
                self._endpoint_member(entry, member, here, "entry")
                self._face(entry, member, here)

    def _bare_key(self, name: str, entry: Node, where: str) -> None:
        """A provider named after a retired name, with no shim/type of its own.

        The key doubles as the identity, and keys are user identifiers (models
        refer to them), so the tool does not rename it.
        """
        shim = retired_names.lookup(name, retired_names.SHIM)
        ptype = retired_names.lookup(name, retired_names.PROVIDER_TYPE)
        if shim is not None:
            fix = json.dumps(dict(shim=shim.replacement))[1:-1]
        elif ptype is not None:
            fix = json.dumps(dict(type=ptype.replacement))[1:-1]
        else:
            return
        self._unmapped(
            entry,
            where,
            name,
            "the provider key is a retired name and serves as its identity; "
            f"keep the key and add {fix} to the entry",
        )

    def _entries(self, obj: Node, where: str, in_section: bool = False) -> None:
        for member in obj.members:
            if member.value.kind == "obj":
                name = str(member.key.value)
                self._entry(name, member.value, f"{where}.{name}", in_section)

    def _models(self, obj: Node, where: str) -> None:
        for member in obj.members:
            if member.value.kind != "obj":
                continue
            here = f"{where}.{member.key.value}"
            kind = member.value.member("type")
            if kind is not None:
                self._model_type(kind.value, f"{here}.type")
            caps = member.value.member("capabilities")
            if caps is not None and caps.value.kind == "arr":
                for index, item in enumerate(caps.value.items):
                    self._model_type(item, f"{here}.capabilities[{index}]")

    def _shims(self, obj: Node) -> None:
        for key in ("include", "exclude", "disabled"):
            member = obj.member(key)
            if member is not None:
                self._shim_list(member.value, f"shims.{key}")

    # -- the document -----------------------------------------------------

    def plan(self, root: Node) -> None:
        """Plan every edit for the document whose root is *root*."""
        if root.kind != "obj":
            self._unmapped(root, "", "", "the document root is not an object")
            return
        for member in root.members:
            key = str(member.key.value)
            value = member.value
            if key == "providers" and value.kind == "obj":
                self._entries(value, "providers")
            elif key == "models" and value.kind == "obj":
                self._models(value, "models")
            elif key == "shims" and value.kind == "obj":
                self._shims(value)
            else:
                self._endpoint_member(root, member, key, "root")
        self._ensure_presets(root)
        self._sweep(root)

    def _ensure_presets(self, root: Node) -> None:
        """Declare the ``shims.presets`` a rewritten shim now needs."""
        if not self.presets:
            return
        names = sorted(self.presets)
        why = ", ".join(f"{self.presets[n]!r} needs {n!r}" for n in names)
        shims = root.member("shims")
        if shims is None:
            append_member(self.edits, root, "shims", json.dumps(dict(presets=names)))
        elif shims.value.kind != "obj":
            self._unmapped(shims.value, "shims", "shims", f"not an object; {why}")
            return
        else:
            self._merge_presets(shims.value, names, why)
        self._report_presets(names)

    def _merge_presets(self, shims: Node, names: list[str], why: str) -> None:
        """Add *names* to ``shims.presets``; *names* keeps only what was added."""
        member = shims.member("presets")
        if member is None:
            append_member(self.edits, shims, "presets", json.dumps(names))
            return
        if member.value.kind != "arr":
            self._unmapped(member.value, "shims.presets", "presets", why)
            names.clear()
            return
        have = set(i.value for i in member.value.items if i.kind == "str")
        names[:] = [n for n in names if n not in have]
        if names:
            append_item(self.edits, member.value, ", ".join(map(json.dumps, names)))

    def _report_presets(self, names: list[str]) -> None:
        for preset in names:
            self.items.append(
                Item(
                    "config",
                    "add",
                    self.file,
                    "shims.presets",
                    None,
                    None,
                    preset,
                    detail=f"the standalone package for {self.presets[preset]!r}",
                )
            )

    def _sweep(self, root: Node) -> None:
        """Flag a compound retired name in a place no slot above handles."""
        distinctive = _distinctive_retired()
        for child, path in _walk_strings(root, ""):
            if id(child) in self._handled or child.value not in distinctive:
                continue
            self._unmapped(
                child,
                path,
                child.value,
                "retired name in a place this tool does not rewrite; "
                f"its replacement is {_replacement(child.value)!r}",
            )


def _replacement(name: str) -> str | None:
    for kind in (retired_names.SHIM, retired_names.PROVIDER_TYPE):
        row = retired_names.lookup(name, kind)
        if row is not None:
            return row.replacement
    return None


def _walk_strings(node: Node, where: str) -> Iterator[tuple[Node, str]]:
    """Every string *value* under *node* with its dotted path."""
    if node.kind == "str":
        yield node, where
    elif node.kind == "arr":
        for index, item in enumerate(node.items):
            yield from _walk_strings(item, f"{where}[{index}]")
    elif node.kind == "obj":
        for member in node.members:
            sub = f"{where}.{member.key.value}" if where else str(member.key.value)
            yield from _walk_strings(member.value, sub)


@dataclass
class ConfigResult:
    """The outcome of planning one config file."""

    path: str
    old_text: str = ""
    new_text: str = ""
    items: list[Item] = field(default_factory=list)
    error: str | None = None

    @property
    def changed(self) -> bool:
        """Whether the planned text differs from the file."""
        return self.new_text != self.old_text

    def diff(self) -> str:
        """A unified diff of the planned change (empty when unchanged)."""
        return "".join(
            difflib.unified_diff(
                self.old_text.splitlines(keepends=True),
                self.new_text.splitlines(keepends=True),
                fromfile=f"a/{self.path}",
                tofile=f"b/{self.path}",
            )
        )


def migrate_config_text(text: str, file: str = "<config>") -> ConfigResult:
    """Plan the migration of one config document held in *text*.

    Args:
        text: The JSONC document.
        file: The name to put in the report.

    Returns:
        The result: the migrated text, the diff, and the report items.  A
        document that cannot be parsed comes back with ``error`` set and its
        text unchanged.
    """
    result = ConfigResult(path=file, old_text=text, new_text=text)
    try:
        root = parse(text)
    except ParseError as exc:
        result.error = f"{file}:{line_of(text, exc.offset)}: {exc}"
        return result
    planner = _ConfigPlanner(text, file)
    planner.plan(root)
    result.items = planner.items
    result.new_text = planner.edits.apply()
    return result


def _read_text(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    return (raw[3:] if bom else raw).decode("utf-8"), bom


def _backup_name(path: Path) -> Path:
    """``<file>.bak``, or a timestamped name when that one is taken."""
    bak = path.with_name(path.name + ".bak")
    if bak.exists():
        bak = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%dT%H%M%S')}")
    return bak


# --------------------------------------------------------------------------
# Database migration
# --------------------------------------------------------------------------

#: Columns that hold a *format* tag the gateway wrote from the request.
_FORMAT_TAG_COLUMNS = (
    ("request_log", "source_provider"),
    ("request_log", "target_provider"),
    ("error_dumps", "source_provider"),
    ("error_dumps", "target_provider"),
)
#: Columns that hold a model type.
_MODEL_TYPE_COLUMNS = (("request_log", "model_type"),)

_COUNTER_SOURCE = "metrics.counters.by_source_provider"
_COUNTER_TARGET = "metrics.counters.by_target_provider"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute("PRAGMA table_info(" + table + ")").fetchall()
    return set(row[1] for row in rows)


def _plan_column(
    conn: sqlite3.Connection, table: str, column: str, mapping: dict[str, str]
) -> list[tuple[str, str, int]]:
    """``(old, new, rows)`` for every retired value present in a column."""
    if column not in _columns(conn, table):
        return []
    marks = ", ".join("?" * len(mapping))
    sql = (
        f"SELECT {column}, COUNT(*) FROM {table} "
        f"WHERE {column} IN ({marks}) GROUP BY {column}"
    )
    rows = conn.execute(sql, tuple(mapping)).fetchall()
    return [(old, mapping[old], count) for old, count in sorted(rows)]


def _metrics_counters(conn: sqlite3.Connection) -> dict[str, Any] | None:
    sql = "SELECT value FROM metrics WHERE key = 'counters'"
    try:
        row = conn.execute(sql).fetchone()
        data = json.loads(row[0]) if row else None
    except (sqlite3.OperationalError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _fold_counter(
    data: dict[str, Any], key: str, mapping: dict[str, str]
) -> list[tuple[str, str, int]]:
    """Merge the retired keys of one counter dict into their replacements.

    Mutates *data*.  The merge is a sum, which is what the counter does when it
    sees both spellings anyway.
    """
    section = data.get(key)
    if not isinstance(section, dict):
        return []
    found = []
    for old, new in mapping.items():
        if old in section and isinstance(section[old], int):
            found.append((old, new, section[old]))
            section[new] = section.get(new, 0) + section.pop(old)
    return found


def _write_config(path: Path, result: ConfigResult, bom: bool) -> str:
    """Back the file up and replace it with the planned text.

    Takes the same lock the gateway's own config writes take, and refuses if the
    file changed since it was planned.

    Returns:
        The backup path.
    """
    from ..config.io import config_lock

    with config_lock(str(path)):
        current, _ = _read_text(path)
        if current != result.old_text:
            raise OSError(f"{path} changed while it was being migrated; run again")
        backup = _backup_name(path)
        shutil.copy2(path, backup)
        data = (b"\xef\xbb\xbf" if bom else b"") + result.new_text.encode()
        staging = path.with_name(path.name + ".new")
        staging.write_bytes(data)
        shutil.copymode(path, staging)
        os.replace(staging, path)
    return str(backup)


def _plan_db(
    conn: sqlite3.Connection, file: str
) -> tuple[list[Item], dict[str, Any] | None]:
    """Report every retired persisted value.

    Returns:
        The items, and the metrics document to write back (``None`` when the
        counters need no change).
    """
    items: list[Item] = []
    tags, types = _format_tag_map(), _model_type_map()
    plan = [(t, c, tags) for t, c in _FORMAT_TAG_COLUMNS]
    plan += [(t, c, types) for t, c in _MODEL_TYPE_COLUMNS]
    for table, column, mapping in plan:
        for old, new, count in _plan_column(conn, table, column, mapping):
            where = f"{table}.{column}"
            items.append(Item("db", "rewrite", file, where, None, old, new, count))
    counters = _metrics_counters(conn)
    if counters is None:
        return items, None
    folded = _fold_counter(counters, "by_source_provider", tags)
    for old, new, count in folded:
        items.append(
            Item("db", "rewrite", file, _COUNTER_SOURCE, None, old, new, count)
        )
    items.extend(_target_counter_items(counters, tags, file))
    return items, counters if folded else None


def _target_counter_items(
    counters: dict[str, Any], tags: dict[str, str], file: str
) -> list[Item]:
    """Report the ``by_target_provider`` keys that are retired tags.

    The key is a provider name when the request had one and a format tag when
    it had none, so the two cannot be told apart; it is left as it is.
    """
    target = counters.get("by_target_provider")
    if not isinstance(target, dict):
        return []
    detail = (
        "the key is a provider name when the request had one and a format tag "
        "when it had none; the two cannot be told apart, so it is left "
        "(the counters can be rebuilt from request_log)"
    )
    return [
        Item("db", "unmapped", file, _COUNTER_TARGET, None, old, detail=detail)
        for old in sorted(tags)
        if old in target
    ]


def db_holders(path: Path) -> list[int]:
    """Process ids that hold *path* (or its WAL/SHM) open.  Linux only.

    Reads ``/proc/<pid>/fd``; on a platform without ``/proc`` it returns an
    empty list, and the ``BEGIN IMMEDIATE`` probe in :func:`migrate_db` is the
    only guard.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    names = {str(path), str(path.resolve())}
    for base in list(names):
        names.add(base + "-wal")
        names.add(base + "-shm")
    holders = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fds = list(entry.joinpath("fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(fd) in names:
                    holders.append(int(entry.name))
                    break
            except OSError:
                continue
    return holders


@dataclass
class DbResult:
    """The outcome of migrating one database."""

    path: str
    items: list[Item] = field(default_factory=list)
    backup: str | None = None
    error: str | None = None

    @property
    def pending(self) -> int:
        """How many retired-value groups were found."""
        return sum(1 for item in self.items if item.action == "rewrite")


def resolve_db_path(arg: str) -> Path:
    """A ``--db`` argument is the file or the data directory that holds it."""
    path = Path(arg)
    return path / _DB_FILENAME if path.is_dir() else path


def migrate_db(path: Path, write: bool = False, force: bool = False) -> DbResult:
    """Plan (and with *write*, apply) the persisted-tag rewrites in one database.

    Args:
        path: The ``gateway.db`` file.
        write: Back the file up and apply the rewrites in one transaction.
        force: Skip the "a process holds the file" refusal.

    Returns:
        The result.  ``error`` is set when the database could not be read, or
        when ``write`` was refused.
    """
    result = DbResult(path=str(path))
    if not path.is_file():
        result.error = f"{path}: no such database"
        return result
    if write and not force:
        holders = db_holders(path)
        if holders:
            result.error = (
                f"{path} is open in process(es) {holders}; stop the gateway first "
                "(or pass --force)"
            )
            return result
    try:
        return _migrate_open(path, write, result)
    except sqlite3.Error as exc:
        result.error = f"{path}: {exc}"
        return result


def _migrate_open(path: Path, write: bool, result: DbResult) -> DbResult:
    mode = "rw" if write else "ro"
    conn = sqlite3.connect(
        f"file:{path}?mode={mode}", uri=True, timeout=0.2, isolation_level=None
    )
    try:
        if write:
            # A writer elsewhere makes this raise "database is locked".
            conn.execute("BEGIN IMMEDIATE")
        items, counters = _plan_db(conn, str(path))
        result.items = items
        if write and any(i.action == "rewrite" for i in items):
            _backup_db(path, result)
            _apply_db(conn, counters)
        if write:
            conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return result


def _backup_db(path: Path, result: DbResult) -> None:
    """Copy the database with SQLite's own backup API, from a second connection.

    The writer connection holds the write lock by now, and a backup taken from
    inside that transaction never finishes; a second, read-only connection
    still reads the committed state, which is what the backup should hold.
    """
    backup = _backup_name(path)
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    target = sqlite3.connect(str(backup))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    result.backup = str(backup)


def _apply_db(conn: sqlite3.Connection, counters: dict[str, Any] | None) -> None:
    tags, types = _format_tag_map(), _model_type_map()
    plan = [(t, c, tags) for t, c in _FORMAT_TAG_COLUMNS]
    plan += [(t, c, types) for t, c in _MODEL_TYPE_COLUMNS]
    for table, column, mapping in plan:
        if column not in _columns(conn, table):
            continue
        for old, new in mapping.items():
            sql = f"UPDATE {table} SET {column} = ? WHERE {column} = ?"
            conn.execute(sql, (new, old))
    if counters is not None:
        sql = "UPDATE metrics SET value = ? WHERE key = 'counters'"
        conn.execute(sql, (json.dumps(counters, ensure_ascii=False),))


# --------------------------------------------------------------------------
# Source scan
# --------------------------------------------------------------------------

_SCAN_SUFFIXES = (".py", ".json", ".jsonc", ".yaml", ".yml")
#: Directories that hold vendored or generated files.  Hidden directories are
#: skipped by their leading dot.
_SCAN_SKIP_DIRS = ("venv", "node_modules", "__pycache__", "build", "dist")
_SCAN_SKIP_FILES = ("package.json", "package-lock.json")


def _scan_vocabulary() -> dict[str, list[str]]:
    """Retired spelling -> replacement notes, for names worth finding in code.

    The scan has no slot to read, so it keeps only strings that are retired
    wherever they appear:

    * a name that is a current format (``google_generate``,
      ``google_interactions``) is left out, although the table lists it as a
      retired *shim* name: in code it is the format;
    * a bare retired shim name (``openai``, ``google``, ``anthropic``) is left
      out: it is a vendor word in ordinary code.  ``google`` and ``anthropic``
      are still found as retired provider types;
    * the bare model-type words (``embedding``, ``decision``) are left out too:
      they are dictionary keys all over an API client.  Their compound
      config-key spellings (``embedding_providers``) are in.

    ``--config`` has the slots and uses them, so it needs none of this.
    """
    current = _current_formats()
    vocab: dict[str, list[str]] = {}
    for row in retired_names.RETIRED_NAMES:
        if row.kind != retired_names.MODEL_TYPE:
            if row.name in current:
                continue
            bare = not any(mark in row.name for mark in "-_")
            if row.kind == retired_names.SHIM and bare:
                continue
            vocab.setdefault(row.name, []).append(f"{row.replacement} ({row.kind})")
            continue
        for suffix in (*_SECTION_SUFFIXES, *_ENDPOINT_SUFFIXES):
            old, new = f"{row.name}_{suffix}", f"{row.replacement}_{suffix}"
            vocab.setdefault(old, []).append(new)
        for suffix in _DEFAULT_SUFFIXES:
            old = f"default_{row.name}_{suffix}"
            new = f"default_{row.replacement}_{suffix}"
            vocab.setdefault(old, []).append(new)
    return vocab


def _scan_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d not in _SCAN_SKIP_DIRS
        )
        for name in sorted(files):
            path = Path(base, name)
            if path.suffix in _SCAN_SUFFIXES and name not in _SCAN_SKIP_FILES:
                yield path


def scan_path(root: Path) -> list[Item]:
    """Report every occurrence of a retired name under *root*; edit nothing.

    Finds the name as an exact quoted string (``"openai_chat"``) in ``.py``,
    ``.json``/``.jsonc`` and ``.yaml``/``.yml`` files, and unquoted as a YAML
    key or value.  A quoted string is a match, not a proof: ``"anthropic"`` may
    be your own variable, so the report names every replacement the table has.

    Args:
        root: A file or a directory.

    Returns:
        One ``scan`` item per occurrence.
    """
    vocab = _scan_vocabulary()
    alternation = "|".join(re.escape(n) for n in sorted(vocab, key=len, reverse=True))
    quoted = re.compile(r"([\x22\x27])(" + alternation + r")\1")
    bare = re.compile(r"(?:^|[\s:\-])(" + alternation + r")\s*(?::|(?:#.*)?$)")
    items: list[Item] = []
    for path in _scan_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        is_yaml = path.suffix in (".yaml", ".yml")
        for number, line in enumerate(lines, 1):
            names = [m.group(2) for m in quoted.finditer(line)]
            if is_yaml:
                names += [m.group(1) for m in bare.finditer(line)]
            for name in names:
                use = " | ".join(vocab[name])
                items.append(
                    Item(
                        "scan",
                        "unmapped",
                        str(path),
                        line=number,
                        old=name,
                        new=use,
                        detail="use " + use,
                    )
                )
    return items


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-rosetta-migrate",
        description=(
            "Migrate a gateway config, a gateway database and source code off the "
            "names retired in 0.14.0.  Dry run unless --write is given."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        default=[],
        metavar="PATH",
        help="JSONC/JSON config file to migrate (repeatable)",
    )
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        metavar="PATH",
        help="gateway.db, or the data directory that holds it (repeatable)",
    )
    parser.add_argument(
        "--scan",
        action="append",
        default=[],
        metavar="PATH",
        help="report retired names in .py/.json/.yaml files under PATH; edits nothing",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="apply the rewrites (a .bak copy is made first); default is a dry run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="dry run that exits 3 when rewrites are pending (for CI)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write a database even if another process has it open",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    return parser


def _run_configs(args: argparse.Namespace, report: dict[str, Any]) -> None:
    for name in args.config:
        path = Path(name)
        entry: dict[str, Any] = dict(path=name, changed=False, backup=None)
        report["configs"].append(entry)
        try:
            text, bom = _read_text(path)
        except (OSError, UnicodeDecodeError) as exc:
            entry["error"] = f"{name}: {exc}"
            continue
        result = migrate_config_text(text, name)
        report["items"].extend(result.items)
        entry["changed"] = result.changed
        if result.error:
            entry["error"] = result.error
            continue
        if result.changed:
            report["diffs"].append(result.diff())
            if args.write:
                try:
                    entry["backup"] = _write_config(path, result, bom)
                except OSError as exc:
                    entry["error"] = str(exc)


def _run_dbs(args: argparse.Namespace, report: dict[str, Any]) -> None:
    for name in args.db:
        result = migrate_db(resolve_db_path(name), args.write, args.force)
        report["items"].extend(result.items)
        entry: dict[str, Any] = dict(
            path=result.path, pending=result.pending, backup=result.backup
        )
        if result.error:
            entry["error"] = result.error
        report["databases"].append(entry)


def _exit_code(args: argparse.Namespace, report: dict[str, Any]) -> int:
    files = report["configs"] + report["databases"]
    if any("error" in f for f in files):
        return 2
    if any(item.action == "unmapped" for item in report["items"]):
        return 1
    pending = any(item.action == "rewrite" for item in report["items"])
    if args.check and not args.write and pending:
        return 3
    return 0


def _print_text(args: argparse.Namespace, report: dict[str, Any]) -> None:
    mode = "WRITE" if args.write else "dry run (use --write to apply)"
    print(f"llm-rosetta-migrate: {mode}")
    for diff in report["diffs"]:
        print(diff, end="" if diff.endswith("\n") else "\n")
    for item in report["items"]:
        print(item.render())
    for entry in report["configs"] + report["databases"]:
        if "error" in entry:
            print(f"error: {entry['error']}", file=sys.stderr)
        if entry.get("backup"):
            print(f"backup: {entry['backup']}")
    rewrites = sum(1 for i in report["items"] if i.action == "rewrite")
    unmapped = sum(1 for i in report["items"] if i.action == "unmapped")
    print(f"{rewrites} rewrite(s), {unmapped} unmapped; exit {report['exit_code']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns:
        The exit code (see the module docstring).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not (args.config or args.db or args.scan):
        from ..config import discover_config

        found = discover_config()
        if found is None:
            parser.error("nothing to do: pass --config, --db or --scan")
        args.config = [found]
    report: dict[str, Any] = dict(configs=[], databases=[], items=[], diffs=[])
    _run_configs(args, report)
    _run_dbs(args, report)
    for name in args.scan:
        report["items"].extend(scan_path(Path(name)))
    report["exit_code"] = _exit_code(args, report)
    if args.json:
        out = dict(
            version=REPORT_VERSION,
            write=args.write,
            exit_code=report["exit_code"],
            configs=report["configs"],
            databases=report["databases"],
            items=[asdict(item) for item in report["items"]],
        )
        print(json.dumps(out, indent=2))
    else:
        _print_text(args, report)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
