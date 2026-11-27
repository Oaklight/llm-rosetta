"""``llm-rosetta gateway migrate`` — the one-shot migration for the 0.14.0 removals.

Release 0.14.0 removes the alias tables that translated old provider, shim and
model-type spellings.  A retired name now fails with an error that names its
replacement; this command is what that error points to.  It rewrites a config
file, rewrites the persisted tags in a gateway database, and scans source code
for the old names::

    llm-rosetta gateway migrate --config config.jsonc            # dry run: diff + report
    llm-rosetta gateway migrate --config config.jsonc --write    # .bak copy, then rewrite
    llm-rosetta gateway migrate --db /var/lib/llm-rosetta --write
    llm-rosetta gateway migrate --scan src/ --json

The names come from :mod:`llm_rosetta.retired_names` and from nothing else, so
the command keeps working after the alias tables are deleted.  It never
guesses: a name it cannot map is reported and left in place, and the exit code
says so.

Exit codes: ``0`` nothing to do, or everything mapped (a ``note``, an item the
tool leaves on purpose, does not count); ``1`` something was reported that needs
a person (an unmapped item, or a ``--scan`` mention or import); ``2`` the command could
not run (unreadable file, parse error, database held by a running gateway);
``3`` ``--check`` found pending rewrites.

A shim whose vendor moved out of core to a shim source repository keeps its
name (:data:`llm_rosetta.retired_names.MOVED_VENDORS`), so nothing is rewritten
for it: ``--config`` and ``--scan`` report it as a ``note`` (the gateway
installs the vendor by itself), which never counts in the exit code.

``shims.presets`` keys are spelled as the ``shims.install`` entries they map
to (#1660): a preset becomes its selector (``free`` → ``contrib:tag:tier:free``),
an ``include`` becomes one ``<source>:<vendor>`` per kept vendor, an
``exclude`` becomes ``!<source>:<vendor>`` per dropped vendor, and a shim a
selection drops from a kept vendor goes to ``shims.disabled``.  A legacy shim
name that needs a package gets the install entry, not a preset.  A key that
cannot be converted (an unknown key) stays.

A pip-installed package that still declares an entry point in the retired
``llm_rosetta.shim_providers`` group is not migrated (it is code, not config).
The text report ends with one ``NOTE`` per such entry point, pointing to
``llm-rosetta ext new``; the exit code and the dry-run semantics are unchanged.

Configs are edited as text (:mod:`.jsonc_spans`): comments, key order and
formatting outside the rewritten names are untouched.  The database step runs
in one transaction after a backup, and refuses while a process holds the file.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
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

from llm_rosetta.api import FORMATS_BY_TYPE, retired_entry_points, retired_names

from ...shims.catalog import STANDALONE_VENDORS

from .jsonc_spans import (
    Edits,
    Member,
    Node,
    ParseError,
    append_item,
    append_member,
    line_of,
    parse,
    remove_member,
    rename_member,
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
            that must be added alongside a rewrite), ``unmapped`` (needs a
            person; never changed), ``pending`` (a load-time migration only
            the gateway can run; counts for ``--check``), ``mention`` (a ``--scan`` hit: advisory,
            a textual mention that may or may not be a use), ``import`` (a
            ``--scan`` hit: a Python import of a module path that was removed
            and does not resolve in the installed core) or ``note`` (left
            on purpose; informational, never counted in the exit code).
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
        if self.action == "mention":
            return f"{place}: mentions {self.old!r} (advisory): {self.detail}"
        if self.action == "import":
            return f"{place}: imports removed module {self.old!r}: {self.detail}"
        if self.action == "note":
            return f"{place}: NOTE {self.old!r} left as it is: {self.detail}"
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


def _db_tag_map() -> dict[str, str]:
    """Retired value -> replacement, for a column that holds a provider tag.

    The gateway wrote the *format* of the request into ``source_provider`` and
    ``target_provider``; for a typed (non-chat) request that was the model type,
    so the retired model-type spellings (``embedding``) appear there as well as
    the retired format names.  Both come from the retired-names table.
    """
    return {**_model_type_map(), **_format_tag_map()}


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

    def _note_moved(self, node: Node, where: str, name: str, row: Any) -> None:
        """Report a moved vendor's shim or probe; the name stays as written."""
        self.items.append(
            Item(
                "config",
                "note",
                self.file,
                where,
                line_of(self.text, node.start),
                name,
                detail=_moved_detail(row),
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
        moved = retired_names.moved_vendor(node.value)
        if row is None:
            if moved is not None:
                self._note_moved(node, where, str(node.value), moved)
            return
        self._rewrite(node, row.replacement, where)
        if row.install:
            vendor = next(
                (v for v in STANDALONE_VENDORS if v.install == row.install), None
            )
            if vendor is not None:
                self.presets.setdefault(vendor.key, row.name)
        if moved is not None:
            self._note_moved(node, where, row.replacement, moved)

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
            elif key == "quota" and member.value.kind == "obj":
                self._quota(member.value, here)
            else:
                self._endpoint_member(entry, member, here, "entry")
                self._face(entry, member, here)

    def _quota(self, quota: Node, where: str) -> None:
        """A ``quota.probe`` that names a probe which moved with its vendor."""
        probe = quota.member("probe")
        if probe is None or probe.value.kind != "str":
            return
        moved = retired_names.moved_probe_vendor(probe.value.value)
        if moved is not None:
            self._note_moved(
                probe.value, f"{where}.probe", str(probe.value.value), moved
            )

    def _bare_key(self, name: str, entry: Node, where: str) -> None:
        """A provider named after a retired name, with no shim/type of its own.

        The key doubles as the identity, and keys are user identifiers (models
        refer to them), so the tool does not rename it.
        """
        shim = retired_names.lookup(name, retired_names.SHIM)
        ptype = retired_names.lookup(name, retired_names.PROVIDER_TYPE)
        moved = retired_names.moved_vendor(name)
        if moved is not None and shim is None and ptype is None:
            self._note_moved(entry, where, name, moved)
            return
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
        # ``include`` / ``exclude`` belong to the presets: they are rewritten
        # with them (``_convert_presets``), or name-by-name when no preset
        # can be converted.
        member = obj.member("disabled")
        if member is not None:
            self._shim_list(member.value, "shims.disabled")

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
        self._convert_presets(root)
        self._sweep(root)
        self._version(root)

    def _version(self, root: Node) -> None:
        """Keep the gateway's first load from rewriting the file again.

        An old ``config_version`` makes the gateway run its load-time steps and
        rewrite the whole file (comments dropped) on the first start, so a
        config this tool left would not stay as it was written.  When those
        steps change nothing but the version stamp and empty sections (the flat ``embeddings_*``
        keys this tool writes stay readable, so folding them into blocks
        changes no behaviour), the stamp is written here.  When they change
        the structure, the file is reported as ``pending``: only the gateway
        can run those steps, and ``--check`` exits 3 until it has.
        """
        import copy

        from ..io import _loads
        from .migrations import CURRENT_VERSION, migrate

        try:
            doc = _loads(self.edits.apply())
        except ValueError:
            return
        if not isinstance(doc, dict):
            return
        declared = root.member("config_version")
        if declared is None:
            # A document with no recorded version predates the key; what the
            # gateway makes of it is not this tool's to predict.
            return
        migrated, changed = migrate(copy.deepcopy(doc))
        if not changed:
            return
        old = doc.get("config_version")
        rest = {k: v for k, v in doc.items() if k != "config_version"}
        after = {k: v for k, v in migrated.items() if k != "config_version"}
        where = "config_version"
        # Only the version stamp and empty sections the steps would add (an
        # absent ``models`` becomes ``{}``) differ: nothing a person wrote moves.
        stamp_only = all(
            (k in rest and rest[k] == after[k]) or (k not in rest and not after[k])
            for k in after
        ) and all(k in after for k in rest)
        if stamp_only or (isinstance(old, int) and old == CURRENT_VERSION - 1):
            stamp = str(CURRENT_VERSION)
            self.edits.replace_span(declared.value, stamp)
            self.items.append(
                Item(
                    "config",
                    "rewrite",
                    self.file,
                    where,
                    line_of(self.text, declared.value.start),
                    str(old),
                    stamp,
                    detail="so the gateway's first load does not rewrite this file",
                )
            )
            return
        self.items.append(
            Item(
                "config",
                "pending",
                self.file,
                where,
                line_of(self.text, declared.value.start),
                str(old),
                str(CURRENT_VERSION),
                detail=(
                    "the gateway runs load-time migrations on its first start and "
                    "rewrites this file (comments are not kept)"
                ),
            )
        )

    def _canonical(self, names: Any) -> list[str]:
        """Shim-name patterns with any retired spelling replaced."""
        out = []
        for name in names if isinstance(names, list) else []:
            row = retired_names.lookup(name, retired_names.SHIM)
            out.append(row.replacement if row is not None else name)
        return out

    def _selection(self, shims: Node | None, key: str) -> dict[str, list[str]]:
        member = shims.member(key) if shims is not None else None
        if member is None or member.value.kind != "obj":
            return {}
        return {
            str(m.key.value): self._canonical(_plain(m.value))
            for m in member.value.members
        }

    def _convert_presets(self, root: Node) -> None:
        """Spell ``shims.presets`` as explicit ``shims.install`` entries (#1660).

        The declared presets, and the ones a rewritten legacy shim name needs,
        are replaced by the entries
        :func:`~llm_rosetta_gateway.config.migrate.migrate_presets.presets_as_install` gives:
        the mapped selector, one ``<source>:<vendor>`` per vendor an
        ``include`` keeps, ``!<source>:<vendor>`` per vendor an ``exclude``
        drops, and ``shims.disabled`` for a shim dropped from a kept vendor.
        A key that cannot be converted stays a preset and is reported.
        """
        from .migrate_presets import presets_as_install

        shims = root.member("shims")
        obj = shims.value if shims is not None and shims.value.kind == "obj" else None
        if shims is not None and obj is None:
            if self.presets:
                why = ", ".join(
                    f"{self.presets[n]!r} needs {n!r}" for n in self.presets
                )
                self._unmapped(shims.value, "shims", "shims", f"not an object; {why}")
            return
        remotes_member = obj.member("remotes") if obj is not None else None
        if remotes_member is not None:
            self._unmapped(
                remotes_member.value,
                "shims.remotes",
                "remotes",
                "removed in 0.15.0, and an explicit archive URL has no manifest "
                "equivalent: publish an extension index and declare it in "
                "shims.sources, or install the extension folder into the "
                "extensions directory; the gateway refuses the key until it "
                "is removed",
            )
        presets_member = obj.member("presets") if obj is not None else None
        declared: list[str] = []
        if presets_member is not None:
            if presets_member.value.kind != "arr":
                self._unmapped(
                    presets_member.value, "shims.presets", "presets", "not an array"
                )
                return
            declared = [i.value for i in presets_member.value.items if i.kind == "str"]
        keys = list(dict.fromkeys([*declared, *sorted(self.presets)]))
        include = self._selection(obj, "include")
        exclude = self._selection(obj, "exclude")
        rewrite = presets_as_install(keys, include, exclude)
        if not rewrite.converted or self._blocked(obj):
            self._keep_presets(obj, keys, rewrite)
            return
        self._report_conversion(presets_member, keys, include, exclude, rewrite)
        if obj is None:
            append_member(
                self.edits, root, "shims", json.dumps({"install": rewrite.install})
            )
            return
        self._write_conversion(obj, keys, include, exclude, rewrite)

    def _blocked(self, obj: Node | None) -> bool:
        """Whether a key the rewrite writes into is not an array.

        The conversion then neither happens nor half-happens: the presets stay,
        and each offending key is reported for a person to fix.
        """
        blocked = False
        for key in ("install", "disabled", "sources"):
            member = obj.member(key) if obj is not None else None
            if member is not None and member.value.kind != "arr":
                self._unmapped(
                    member.value,
                    f"shims.{key}",
                    key,
                    "not an array, so shims.presets cannot be rewritten into "
                    "shims.install entries; fix its type and run again",
                )
                blocked = True
        return blocked

    def _keep_presets(self, obj: Node | None, keys: list[str], rewrite: Any) -> None:
        """Nothing converts: rewrite selection names in place, as before."""
        for key in ("include", "exclude"):
            member = obj.member(key) if obj is not None else None
            if member is not None and member.value.kind == "obj":
                for inner in member.value.members:
                    self._shim_list(inner.value, f"shims.{key}.{inner.key.value}")
        self._report_unconvertible(obj, rewrite)

    def _report_unconvertible(self, obj: Node | None, rewrite: Any) -> None:
        anchor = obj.member("presets") if obj is not None else None
        for key, why in rewrite.unconvertible.items():
            node = anchor.value if anchor is not None else obj
            self.items.append(
                Item(
                    "config",
                    "unmapped",
                    self.file,
                    "shims.presets",
                    line_of(self.text, node.start) if node is not None else None,
                    key,
                    detail=f"left as a preset: {why}",
                )
            )

    def _report_conversion(
        self,
        presets_member: Member | None,
        keys: list[str],
        include: dict[str, list[str]],
        exclude: dict[str, list[str]],
        rewrite: Any,
    ) -> None:
        from .migrate_presets import presets_as_install

        line = (
            line_of(self.text, presets_member.value.start) if presets_member else None
        )
        for key in rewrite.converted:
            one = presets_as_install([key], include, exclude)
            detail = "legacy preset spelled as shims.install entries"
            if key in self.presets:
                detail = f"needed by {self.presets[key]!r}; " + detail
            if one.disabled:
                detail += "; shims.disabled += " + ", ".join(one.disabled)
            self.items.append(
                Item(
                    "config",
                    "rewrite",
                    self.file,
                    "shims.presets",
                    line,
                    key,
                    ", ".join(one.install),
                    detail=detail,
                )
            )
        self._report_unconvertible(None, rewrite)

    def _write_conversion(
        self,
        obj: Node,
        keys: list[str],
        include: dict[str, list[str]],
        exclude: dict[str, list[str]],
        rewrite: Any,
    ) -> None:
        """Edit the ``shims`` object: drop what converted, add the entries."""
        gone = set(rewrite.converted)
        slots: list[Member] = []
        remaining = [k for k in keys if k not in gone and k in rewrite.unconvertible]
        self._settle(obj, "presets", remaining, slots)
        for key, sel in (("include", include), ("exclude", exclude)):
            self._settle(
                obj, key, {k: v for k, v in sel.items() if k not in gone}, slots
            )
        changes = self._extend(obj, "install", rewrite.install)
        changes += self._extend(obj, "disabled", rewrite.disabled)
        self._declare_sources(obj, rewrite)
        # A new key takes the slot of a dropped one, so no insertion lands
        # inside a span that is being removed.
        for key, value_json in changes:
            if slots:
                rename_member(self.edits, slots.pop(0), key, value_json)
            else:
                append_member(self.edits, obj, key, value_json)
        self._remove_slots(obj, slots)

    def _remove_slots(self, obj: Node, slots: list[Member]) -> None:
        """Drop the freed members; ``presets``' comment travels to ``install``."""
        gone = {id(m) for m in slots}
        members = obj.members
        for index, member in enumerate(members):
            if id(member) not in gone:
                continue
            tail = all(id(m) in gone for m in members[index:])
            first_of_tail = tail and (index == 0 or id(members[index - 1]) not in gone)
            previous = members[index - 1] if first_of_tail and index else None
            comment = remove_member(self.edits, member, previous)
            install = obj.member("install")
            if comment and member.key.value == "presets" and install is not None:
                end = self.text.find("\n", install.value.end)
                self.edits.insert(len(self.text) if end < 0 else end, "  " + comment)

    def _declare_sources(self, obj: Node, rewrite: Any) -> None:
        """Add the sources the entries need, unless one is already declared.

        A source counts as declared by its id or by its catalogue manifest URL;
        an unknown URL may be a mirror of it, so the key is then reported rather
        than added (a second declaration would fetch the source twice).
        """
        from .migrate_presets import sources_to_declare

        member = obj.member("sources")
        if member is None or member.value.kind != "arr":
            return
        add, unsure = sources_to_declare(_plain(member.value), rewrite.sources)
        for key in add:
            append_item(self.edits, member.value, json.dumps(key))
        for key in unsure:
            self.items.append(
                Item(
                    "config",
                    "note",
                    self.file,
                    "shims.sources",
                    line_of(self.text, member.value.start),
                    key,
                    detail=(
                        "not added: shims.sources declares a manifest URL this tool "
                        f"does not know, which may already be {key!r}; add it if not"
                    ),
                )
            )

    def _settle(self, obj: Node, key: str, remaining: Any, slots: list[Member]) -> None:
        """Keep what is left of *key*, or free its slot when nothing is."""
        member = obj.member(key)
        if member is None:
            return
        if remaining:
            self.edits.replace_span(member.value, json.dumps(remaining))
        else:
            slots.append(member)

    def _extend(self, obj: Node, key: str, values: list[str]) -> list[tuple[str, str]]:
        """Append *values* to an array member; a missing member is returned."""
        member = obj.member(key)
        if member is None:
            return [(key, json.dumps(values))] if values else []
        if member.value.kind == "arr":
            have = {i.value for i in member.value.items if i.kind == "str"}
            for value in values:
                if value not in have:
                    append_item(self.edits, member.value, json.dumps(value))
        return []

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


def _plain(node: Node) -> Any:
    """The JSON value of *node* (strings, lists and objects; literals as text)."""
    if node.kind == "str":
        return node.value
    if node.kind == "arr":
        return [_plain(i) for i in node.items]
    if node.kind == "obj":
        return {str(m.key.value): _plain(m.value) for m in node.members}
    return node.value


def _moved_detail(row: Any) -> str:
    """The note for a name whose vendor moved out of core."""
    return (
        f"now in {row.package} (vendor {row.vendor!r}, moved in {row.moved_in}); "
        "the gateway installs it automatically, no edit needed"
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
    if not bak.exists():
        return bak
    stamp = time.strftime("%Y%m%dT%H%M%S")
    bak = path.with_name(f"{path.name}.bak-{stamp}")
    n = 1
    while bak.exists():  # two runs inside one second
        bak = path.with_name(f"{path.name}.bak-{stamp}-{n}")
        n += 1
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
    from ..io import config_lock

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
    tags, types = _db_tag_map(), _model_type_map()
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
    it had none, so the two cannot be told apart; it is left as it is, and
    reported as a ``note`` that does not count in the exit code.
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
        Item("db", "note", file, _COUNTER_TARGET, None, old, detail=detail)
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
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc):
            result.error = (
                f"{path}: the database is locked by another writer (sqlite: {exc}); "
                "stop the process that writes it and run again (--force skips only "
                "the open-file check, not a SQLite lock)"
            )
        else:
            result.error = f"{path}: {exc}"
        return result
    except sqlite3.Error as exc:
        result.error = f"{path}: {exc}"
        return result
    except OSError as exc:
        # Creating the backup file: a full disk, a read-only directory.
        result.error = f"{path}: could not write the backup: {exc}"
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
    # The copy holds the same rows as the database, so it gets exactly the
    # database's permission bits: sqlite would create it with the umask's
    # (often 0644), which leaves a world-readable copy beside a 0600 original.
    # ``open`` filters its mode through the umask, hence the ``fchmod``.
    # ``O_EXCL`` keeps the never-overwrite rule even when two runs race.
    mode = path.stat().st_mode & 0o777
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(fd, mode)
    finally:
        os.close(fd)
    done = False
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(str(backup))
        try:
            source.backup(target)
        finally:
            target.close()
        done = True
    finally:
        source.close()
        if not done:
            # A copy that did not complete is not a backup; leave nothing behind.
            backup.unlink(missing_ok=True)
    result.backup = str(backup)


def _apply_db(conn: sqlite3.Connection, counters: dict[str, Any] | None) -> None:
    tags, types = _db_tag_map(), _model_type_map()
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


def _resolves(path: str) -> bool:
    """Whether module *path* can be imported from the installed packages."""
    try:
        return importlib.util.find_spec(path) is not None
    except (ImportError, ValueError):
        return False


def _imported_modules(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(line, dotted path)`` for every absolute import in *tree*.

    ``from pkg import name`` yields ``pkg`` and ``pkg.name``: *name* may be a
    submodule, and a removed submodule of a package that still exists is only
    visible that way (``from llm_rosetta.converters.base.tools import batch``).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.lineno, node.module
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{node.module}.{alias.name}"


def _scan_imports(path: Path, text: str, gone: dict[str, bool]) -> list[Item]:
    """One ``import`` item per import of a removed module path in *text*.

    A path is reported only when its row's path does not resolve in the
    installed core (*gone* caches the answer per row), so a path that a
    later change will remove, but that still exists, is never reported.
    """
    try:
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        return []
    items: list[Item] = []
    seen: set[tuple[int, str]] = set()
    for line, module in _imported_modules(tree):
        row = retired_names.retired_module(module)
        if row is None or (line, row.path) in seen:
            continue
        if row.path not in gone:
            gone[row.path] = not _resolves(row.path)
        if not gone[row.path]:
            continue
        seen.add((line, row.path))
        items.append(
            Item(
                "scan",
                "import",
                str(path),
                line=line,
                old=module,
                new=row.replacement,
                detail=retired_names.retired_module_message(module) or "",
            )
        )
    return items


class _MovedScan:
    """Finds the names of vendors that moved to contrib, for ``--scan`` notes.

    Three kinds of name, matched with different care:

    * ``<vendor>--<format>`` shim names and quota probe ids (``openai_codex``,
      ``opencode_go``) are distinctive, so any quoted string equal to one is
      reported;
    * a bare moved shim name (``xai``, ``qwen``, ``zhipu``, ``moonshot``) is an
      ordinary word in code, so it is reported only where it is a whole string
      value in a structured position: the value of ``type``, ``shim`` or a
      ``<type>_shim`` key, a mapping key (a provider named after the shim), or
      an item of an ``install`` array (``"xai"``, ``"contrib:xai"``), in JSON,
      JSONC or YAML.
    """

    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}
        bare: list[str] = []
        for row in retired_names.MOVED_VENDORS:
            for name in row.shims:
                self.rows[name] = row
                if "--" not in name:
                    bare.append(name)
            for probe in row.probes:
                self.rows[probe] = row
        distinctive = sorted(
            (n for n in self.rows if n not in bare), key=len, reverse=True
        )
        words = "|".join(re.escape(n) for n in sorted(bare, key=len, reverse=True))
        self.quoted = re.compile(
            r"([\x22\x27])(" + "|".join(map(re.escape, distinctive)) + r")\1"
        )
        slot = r"(?:type|shim|[a-z_]+_shim)"
        self.json_slot = re.compile(
            r"\x22" + slot + r"\x22\s*:\s*\x22(" + words + r")\x22"
        )
        self.json_key = re.compile(r"\x22(" + words + r")\x22\s*:\s*\{")
        self.yaml_slot = re.compile(
            r"^\s*" + slot + r"\s*:\s*([\x22\x27]?)(" + words + r")\1\s*(?:#.*)?$"
        )
        self.yaml_key = re.compile(r"^\s*(" + words + r")\s*:\s*(?:#.*)?$")
        self.install = re.compile(r"\x22install\x22\s*:\s*\[([^\]]*)\]")
        self.install_item = re.compile(r"\x22(?:[a-z0-9_-]+:)?(" + words + r")\x22")

    def items(self, path: Path, text: str) -> list[Item]:
        """One ``note`` per moved name in *text*."""
        found: list[tuple[int, str]] = []
        is_yaml = path.suffix in (".yaml", ".yml")
        structured = is_yaml or path.suffix in (".json", ".jsonc")
        for number, line in enumerate(text.splitlines(), 1):
            found += [(number, m.group(2)) for m in self.quoted.finditer(line)]
            if not structured:
                continue
            if is_yaml:
                found += [(number, m.group(2)) for m in self.yaml_slot.finditer(line)]
                found += [(number, m.group(1)) for m in self.yaml_key.finditer(line)]
            else:
                found += [(number, m.group(1)) for m in self.json_slot.finditer(line)]
                found += [(number, m.group(1)) for m in self.json_key.finditer(line)]
        if structured and not is_yaml:
            for block in self.install.finditer(text):
                for item in self.install_item.finditer(block.group(1)):
                    number = text.count("\n", 0, block.start(1) + item.start()) + 1
                    found.append((number, item.group(1)))
        return [
            Item(
                "scan",
                "note",
                str(path),
                line=number,
                old=name,
                detail=_moved_detail(self.rows[name]),
            )
            for number, name in sorted(set(found))
        ]


def scan_path(root: Path, skip: Iterable[Path] = ()) -> list[Item]:
    """Report every occurrence of a retired name under *root*; edit nothing.

    Finds the name as an exact quoted string (``"openai_chat"``) in ``.py``,
    ``.json``/``.jsonc`` and ``.yaml``/``.yml`` files, and unquoted as a YAML
    key or value.  A quoted string is a match, not a proof: ``"anthropic"`` may
    be your own variable, so the report names every replacement the table has.
    Each such finding is an advisory textual mention, not a proof of a use.

    A quoted shim name of a vendor that moved out of core
    (``"opencode--chat_completions"``) is a ``note``: the name is still
    valid, so it is reported for information and never counts.

    In ``.py`` files it also reads the import statements: an import of a
    module path listed in ``retired_names.RETIRED_MODULES`` is an ``import``
    item that names what to import instead.  It is reported only when the
    path does not resolve in the installed core.

    Args:
        root: A file or a directory.
        skip: Files to leave out, such as the configs ``--config`` rewrites in
            the same run.

    Returns:
        One ``scan`` item per occurrence.
    """
    skipped = {Path(p).resolve() for p in skip}
    vocab = _scan_vocabulary()
    alternation = "|".join(re.escape(n) for n in sorted(vocab, key=len, reverse=True))
    quoted = re.compile(r"([\x22\x27])(" + alternation + r")\1")
    bare = re.compile(r"(?:^|[\s:\-])(" + alternation + r")\s*(?::|(?:#.*)?$)")
    moved = _MovedScan()
    items: list[Item] = []
    gone: dict[str, bool] = {}
    for path in _scan_files(root):
        if path.resolve() in skipped:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if path.suffix == ".py":
            items.extend(_scan_imports(path, text, gone))
        items.extend(moved.items(path, text))
        lines = text.splitlines()
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
                        "mention",
                        str(path),
                        line=number,
                        old=name,
                        new=use,
                        detail="if it names a provider, format or shim, use " + use,
                    )
                )
    return items


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-rosetta gateway migrate",
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
        help=(
            "report retired names in .py/.json/.yaml files, and imports of removed "
            "module paths in .py files, under PATH; edits nothing"
        ),
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
        _file_items(report, result.items)
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


def _file_items(report: dict[str, Any], items: Iterable[Item]) -> None:
    """Put each item in ``notes`` or ``items``; a note never counts."""
    for item in items:
        report["notes" if item.action == "note" else "items"].append(item)


def _run_dbs(args: argparse.Namespace, report: dict[str, Any]) -> None:
    for name in args.db:
        result = migrate_db(resolve_db_path(name), args.write, args.force)
        for item in result.items:
            report["notes" if item.action == "note" else "items"].append(item)
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
    if any(
        item.action in ("unmapped", "mention", "import") for item in report["items"]
    ):
        return 1
    pending = any(item.action in ("rewrite", "pending") for item in report["items"])
    if args.check and not args.write and pending:
        return 3
    return 0


def _print_text(args: argparse.Namespace, report: dict[str, Any]) -> None:
    mode = "WRITE" if args.write else "dry run (use --write to apply)"
    print(f"llm-rosetta gateway migrate: {mode}")
    for diff in report["diffs"]:
        print(diff, end="" if diff.endswith("\n") else "\n")
    for item in report["items"] + report["notes"]:
        print(item.render())
    for entry in report["configs"] + report["databases"]:
        if "error" in entry:
            print(f"error: {entry['error']}", file=sys.stderr)
        if entry.get("backup"):
            print(f"backup: {entry['backup']}")
    rewrites = sum(1 for i in report["items"] if i.action == "rewrite")
    unmapped = sum(1 for i in report["items"] if i.action == "unmapped")
    mentions = sum(1 for i in report["items"] if i.action == "mention")
    imports = sum(1 for i in report["items"] if i.action == "import")
    notes = len(report["notes"])
    print(
        f"{rewrites} rewrite(s), {unmapped} unmapped, {mentions} mention(s), "
        f"{imports} removed import(s), {notes} note(s); exit {report['exit_code']}"
    )
    for old in retired_entry_points():
        print(
            f"NOTE: {old.describe()}. This command does not convert installed "
            "packages: run `llm-rosetta ext new` to scaffold the extension."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Migrate a config and database off the names 0.14.0 retires.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns:
        The exit code (see the module docstring).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not (args.config or args.db or args.scan):
        from .. import discover_config

        found = discover_config()
        if found is None:
            parser.error("nothing to do: pass --config, --db or --scan")
        args.config = [found]
    report: dict[str, Any] = dict(
        configs=[], databases=[], items=[], notes=[], diffs=[]
    )
    _run_configs(args, report)
    _run_dbs(args, report)
    for name in args.scan:
        _file_items(report, scan_path(Path(name), skip=map(Path, args.config)))
    report["exit_code"] = _exit_code(args, report)
    if args.json:
        out = dict(
            version=REPORT_VERSION,
            write=args.write,
            exit_code=report["exit_code"],
            configs=report["configs"],
            databases=report["databases"],
            items=[asdict(item) for item in report["items"]],
            notes=[asdict(item) for item in report["notes"]],
        )
        print(json.dumps(out, indent=2))
    else:
        _print_text(args, report)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
