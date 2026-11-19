#!/usr/bin/env python3
"""Check that a refactor is a pure move of code.

Usage::

    python scripts/dev/check_move.py <base-ref> [<head-ref>=HEAD]
        [--allow FILE] [--markdown] [--repo DIR]

For every ``.py`` file that differs between the two refs (``_vendor/`` is
skipped) the tool collects each function, method, class and class-level (or
module-level) assignment, keyed by qualified name (``Class.method``).  A def
is compared as a whole through ``ast.dump``: decorators, arguments,
annotations, body and docstring all count.  Comments and formatting do not,
because the AST does not carry them.

The base side is the merge base of the two refs, so commits that landed on the
base branch after the head branched do not show up as removals.

Items are matched in this order:

1. same qualified name and identical dump (same file first),
2. identical dump under the same bare name, so a method that moved into a mixin
   class (the class name changed) still matches,
3. same qualified name, or the only remaining pair with that bare name, with a
   different dump -> ``moved-with-diff``.

Reported categories:

* ``unchanged``       identical item that stayed in its file
* ``moved-identical`` identical item in another file or class
* ``moved-with-diff`` the pair exists but the dump differs (FAILS)
* ``missing``         present in base, absent in head (FAILS)
* ``added``           present in head only; listed, never failing, because new
  glue code is normal in a split
* ``type-checking``   added item inside ``if TYPE_CHECKING:`` (ignorable)
* ``import-hoisted``  the only difference is a function-local import that is now
  a module-level import of the same name in the head file (ignorable)
* ``logger-name``     the only difference is ``getLogger(__name__)`` becoming
  ``getLogger("<literal>")`` where the literal is the base module's dotted name
  (ignorable)
* ``undefined-global`` (FAILS) a matched function reads a module-level name that
  its head module does not define, though its base module did (a lost
  ``logger`` or constant); found with ``symtable``
* ``decorator-census`` (FAILS) the number of ``@staticmethod``, ``@classmethod``
  and ``@property`` decorators differs between base and head over the defs that
  were matched between the two sides.  Added defs are not counted (a split may
  add a decorated helper), nor are defs under ``if TYPE_CHECKING:``; a dropped
  decorator on a moved def is counted

A class is compared by its decorators, keywords and docstring only; its bases
are not compared, because a split into mixins changes them.

An ``--allow`` file lists accepted pairs, one ``path::qualified.name  reason``
per line (``#`` starts a comment), where ``path`` is the repo-relative path of
the file holding the def.  A moved-with-diff entry matches the head path or the
base path of the pair, so either end of a def that moved between files works;
a missing def is keyed on its base path.  An undefined-global item is allowed
as ``undefined-global:<head path>::<qualified.name>``.  A census mismatch is
allowed as ``decorator-census:<decorator>`` (``decorator-census:property``; the
report prints the exact entry, and the ``@``-prefixed spelling
``decorator-census:@property`` is accepted too); the census is a global count,
so that entry is not tied to a file.  A bare ``qualified.name`` (the older format)
is rejected with an error, because it would excuse a same-named def in every
file.  An allowed item is reported but does not fail the run.

Exit status: 0 clean, 1 a ``missing`` / ``moved-with-diff`` item is not
allowed, 2 usage or parse error.  Standard library only, Python >= 3.10.
"""

from __future__ import annotations

import argparse
import ast
import copy
import builtins
import difflib
import subprocess
import symtable
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DIFF_LINE_CAP = 40

FUNC = "function"
CLASS = "class"
ASSIGN = "assign"

_MEMBER_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)


@dataclass
class Item:
    """One collected definition."""

    kind: str
    qual: str
    bare: str
    file: str
    dump: str
    src: str
    node: ast.AST = field(repr=False)
    typecheck: bool = False
    undefined: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        return f"{self.file}:{self.qual}"

    @property
    def key(self) -> str:
        """The allow-file key: ``path::qualified.name``."""
        return f"{self.file}::{self.qual}"


# --------------------------------------------------------------------------
# Git access
# --------------------------------------------------------------------------


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def changed_files(repo: str, base: str, head: str) -> list[str]:
    """Return the ``.py`` paths touched between ``base`` and ``head``."""
    out = _git(repo, "diff", "--name-only", "--no-renames", base, head, "--", "*.py")
    return sorted(p for p in out.splitlines() if p and "/_vendor/" not in f"/{p}")


def read_at(repo: str, ref: str, path: str) -> str | None:
    """Return the file text at ``ref``, or ``None`` if it does not exist."""
    proc = subprocess.run(
        ["git", "-C", repo, "show", f"{ref}:{path}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8")


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _segment(lines: list[str], node: ast.stmt) -> str:
    start = node.lineno
    for deco in getattr(node, "decorator_list", []):
        start = min(start, deco.lineno)
    end = node.end_lineno or start
    return textwrap.dedent("\n".join(lines[start - 1 : end]))


def _assign_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [ast.unparse(t) for t in targets]


def _class_shell(node: ast.ClassDef) -> ast.ClassDef:
    """Return the class without its members and bases."""
    shell = copy.copy(node)
    shell.bases = []
    shell.body = [s for s in node.body if not isinstance(s, _MEMBER_NODES)] or [
        ast.Pass()
    ]
    return shell


def collect(text: str, path: str) -> list[Item]:
    """Collect the comparable items of one module."""
    tree = ast.parse(text, filename=path)
    lines = text.splitlines()
    items: list[Item] = []
    undefined = undefined_globals(text, path, tree)

    def add(kind: str, qual: str, node: ast.stmt, dump_node: ast.AST, tc: bool):
        items.append(
            Item(
                kind=kind,
                qual=qual,
                bare=qual.rsplit(".", 1)[-1],
                file=path,
                dump=ast.dump(dump_node),
                src=_segment(lines, node),
                node=node,
                typecheck=tc,
                undefined=undefined.get(qual, frozenset())
                if kind == FUNC
                else frozenset(),
            )
        )

    def walk(body: list[ast.stmt], prefix: str, typecheck: bool) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(FUNC, prefix + stmt.name, stmt, stmt, typecheck)
            elif isinstance(stmt, ast.ClassDef):
                add(CLASS, prefix + stmt.name, stmt, _class_shell(stmt), typecheck)
                walk(stmt.body, prefix + stmt.name + ".", typecheck)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                for name in _assign_names(stmt):
                    if name != "__all__":
                        add(ASSIGN, prefix + name, stmt, stmt, typecheck)
            elif isinstance(stmt, ast.If):
                walk(stmt.body, prefix, typecheck or _is_type_checking(stmt.test))
                walk(stmt.orelse, prefix, typecheck)
            elif isinstance(stmt, ast.Try):
                for block in (stmt.body, stmt.orelse, stmt.finalbody):
                    walk(block, prefix, typecheck)
                for handler in stmt.handlers:
                    walk(handler.body, prefix, typecheck)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                walk(stmt.body, prefix, typecheck)

    walk(tree.body, "", False)
    return items


def _bound(node: ast.Import | ast.ImportFrom) -> list[str]:
    out = []
    for alias in node.names:
        if alias.asname:
            out.append(alias.asname)
        elif isinstance(node, ast.Import):
            out.append(alias.name.split(".")[0])
        else:
            out.append(alias.name)
    return out


def module_imports(text: str) -> set[str]:
    """Return the names bound by imports outside any def or class body."""
    names: set[str] = set()

    def walk(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                names.update(_bound(stmt))
            elif isinstance(stmt, ast.If):
                walk(stmt.body)
                walk(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                for block in (stmt.body, stmt.orelse, stmt.finalbody):
                    walk(block)
                for handler in stmt.handlers:
                    walk(handler.body)

    walk(ast.parse(text).body)
    return names


class _StripImports(ast.NodeTransformer):
    """Drop function-local imports whose names the head imports at module level."""

    def __init__(self, hoisted: set[str]):
        self.hoisted = hoisted
        self.removed = 0

    def _filter(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        body = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) and all(
                n in self.hoisted for n in _bound(stmt)
            ):
                self.removed += 1
                continue
            body.append(stmt)
        node.body = body or [ast.Pass()]
        self.generic_visit(node)
        return node

    visit_FunctionDef = _filter
    visit_AsyncFunctionDef = _filter


def import_hoisted(base: Item, head: Item, head_imports: set[str]) -> bool:
    """Tell whether ``base`` differs from ``head`` only by hoisted imports."""
    if base.kind != FUNC or head.kind != FUNC:
        return False
    stripper = _StripImports(head_imports)
    stripped = stripper.visit(copy.deepcopy(base.node))
    return stripper.removed > 0 and ast.dump(stripped) == head.dump


CENSUS_DECORATORS = ("staticmethod", "classmethod", "property")

_MODULE_DUNDERS = frozenset(
    {"__name__", "__file__", "__doc__", "__spec__", "__package__", "__loader__"}
)


def _decorator_names(item: Item) -> list[str]:
    """Return the census decorators on the def of ``item``."""
    if item.kind != FUNC:
        return []
    names = []
    for deco in getattr(item.node, "decorator_list", []):
        if isinstance(deco, ast.Name):
            name = deco.id
        elif isinstance(deco, ast.Attribute):
            name = deco.attr
        else:
            continue
        if name in CENSUS_DECORATORS:
            names.append(name)
    return names


def decorator_census(pairs: list[tuple[Item, Item]]) -> dict[str, tuple[int, int]]:
    """Count the census decorators over the matched pairs.

    Only a def present on both sides counts, so an added helper never fails the
    census while a decorator dropped from a moved def does.  A pair with a
    side under ``if TYPE_CHECKING:`` is skipped: such stubs carry the decorators
    of the real def they describe.
    """
    base: Counter[str] = Counter()
    head: Counter[str] = Counter()
    for b, h in pairs:
        if b.typecheck or h.typecheck:
            continue
        base.update(_decorator_names(b))
        head.update(_decorator_names(h))
    return {name: (base[name], head[name]) for name in CENSUS_DECORATORS}


def _walk_tables(table: symtable.SymbolTable):
    yield table
    for child in table.get_children():
        yield from _walk_tables(child)


def _globals_read(table: symtable.SymbolTable) -> set[str]:
    out: set[str] = set()
    for sub in _walk_tables(table):
        out.update(
            s.get_name()
            for s in sub.get_symbols()
            if s.is_global() and s.is_referenced()
        )
    return out


def undefined_globals(
    text: str, path: str, tree: ast.Module | None = None
) -> dict[str, frozenset[str]]:
    """Map a def's qualified name to the module-level names it reads but lacks."""
    tree = tree or ast.parse(text)
    star = any(
        isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
        for n in ast.walk(tree)
    )
    if star:
        return {}
    module = symtable.symtable(text, path, "exec")
    defined = {
        s.get_name()
        for s in module.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_namespace()
    }
    for sub in _walk_tables(module):
        defined.update(
            s.get_name()
            for s in sub.get_symbols()
            if s.is_declared_global() and s.is_assigned()
        )
    known = defined | set(dir(builtins)) | _MODULE_DUNDERS
    result: dict[str, set[str]] = {}

    def visit(table: symtable.SymbolTable, prefix: str) -> None:
        for child in table.get_children():
            kind = child.get_type()
            if kind == "class":
                visit(child, f"{prefix}{child.get_name()}.")
            elif kind == "function":
                lost = _globals_read(child) - known
                if lost:
                    result.setdefault(prefix + child.get_name(), set()).update(lost)

    visit(module, "")
    return {k: frozenset(v) for k, v in result.items()}


class _NormaliseLoggers(ast.NodeTransformer):
    """Replace the argument of ``getLogger(...)`` with a fixed placeholder."""

    def __init__(self) -> None:
        self.dunder_name = 0
        self.literals: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "getLogger" and len(node.args) == 1 and not node.keywords:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == "__name__":
                self.dunder_name += 1
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self.literals.append(arg.value)
            else:
                return node
            node.args = [ast.Constant("<logger>")]
        return node


def _module_parts(path: str) -> list[str]:
    parts = Path(path).with_suffix("").parts
    return [p for p in parts if p != "__init__"]


def logger_renamed(base: Item, head: Item) -> bool:
    """Tell whether only ``getLogger(__name__)`` became ``getLogger("<literal>")``.

    The literal must be the dotted name of the base module, so the logger keeps
    the name it had before the move.
    """
    if base.kind == CLASS or head.kind == CLASS:
        return False
    nb, nh = _NormaliseLoggers(), _NormaliseLoggers()
    db = ast.dump(nb.visit(copy.deepcopy(base.node)))
    dh = ast.dump(nh.visit(copy.deepcopy(head.node)))
    if db != dh or not nb.dunder_name or not nh.literals or nh.dunder_name:
        return False
    parts = _module_parts(base.file)
    for literal in nh.literals:
        want = literal.split(".")
        if parts[-len(want) :] != want:
            return False
    return True


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


@dataclass
class Report:
    """The result of one comparison."""

    unchanged: list[tuple[Item, Item]] = field(default_factory=list)
    moved_identical: list[tuple[Item, Item]] = field(default_factory=list)
    moved_with_diff: list[tuple[Item, Item]] = field(default_factory=list)
    import_hoisted: list[tuple[Item, Item]] = field(default_factory=list)
    missing: list[Item] = field(default_factory=list)
    added: list[Item] = field(default_factory=list)
    type_checking: list[Item] = field(default_factory=list)
    logger_name: list[tuple[Item, Item]] = field(default_factory=list)
    undefined_global: list[tuple[Item, Item, list[str]]] = field(default_factory=list)
    census: dict[str, tuple[int, int]] = field(default_factory=dict)
    allowed: dict[str, str] = field(default_factory=dict)
    unused_allow: list[str] = field(default_factory=list)

    def matched_pairs(self) -> list[tuple[Item, Item]]:
        return (
            self.unchanged
            + self.moved_identical
            + self.moved_with_diff
            + self.import_hoisted
            + self.logger_name
        )

    def is_allowed(self, *names: str) -> str | None:
        for n in names:
            if n in self.allowed:
                return n
        return None


def _pop_match(base: Item, pool: list[Item], pred) -> Item | None:
    """Pop the best candidate in ``pool`` for ``base`` (same file first)."""
    cands = [h for h in pool if h.kind == base.kind and pred(base, h)]
    if not cands:
        return None
    cands.sort(key=lambda h: h.file != base.file)
    pool.remove(cands[0])
    return cands[0]


def match(
    base_items: list[Item],
    head_items: list[Item],
    head_imports: dict[str, set[str]],
) -> Report:
    """Pair base items with head items and classify the pairs."""
    report = Report()
    base_rem = list(base_items)
    head_rem = list(head_items)

    def run(pred, record) -> None:
        for b in list(base_rem):
            h = _pop_match(b, head_rem, pred)
            if h is not None:
                base_rem.remove(b)
                record(b, h)

    def record_identical(b: Item, h: Item) -> None:
        same = b.file == h.file and b.qual == h.qual
        (report.unchanged if same else report.moved_identical).append((b, h))

    def record_diff(b: Item, h: Item) -> None:
        if logger_renamed(b, h):
            report.logger_name.append((b, h))
        elif import_hoisted(b, h, head_imports.get(h.file, set())):
            report.import_hoisted.append((b, h))
        else:
            report.moved_with_diff.append((b, h))

    run(lambda b, h: b.qual == h.qual and b.dump == h.dump, record_identical)
    run(lambda b, h: b.bare == h.bare and b.dump == h.dump, record_identical)
    run(lambda b, h: b.qual == h.qual, record_diff)

    # Bare-name fallback, only when exactly one pair is left under that name.
    for b in list(base_rem):
        same_b = [x for x in base_rem if x.kind == b.kind and x.bare == b.bare]
        same_h = [x for x in head_rem if x.kind == b.kind and x.bare == b.bare]
        if len(same_b) == 1 and len(same_h) == 1:
            base_rem.remove(b)
            head_rem.remove(same_h[0])
            record_diff(b, same_h[0])

    report.missing = base_rem
    for b, h in report.matched_pairs():
        lost = sorted(h.undefined - b.undefined)
        if h.kind == FUNC and lost:
            report.undefined_global.append((b, h, lost))
    report.census = decorator_census(report.matched_pairs())
    for h in head_rem:
        (report.type_checking if h.typecheck else report.added).append(h)
    return report


def _failing_keys(report: Report) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = [(b.key, h.key) for b, h in report.moved_with_diff]
    keys += [(m.key,) for m in report.missing]
    keys += [(f"undefined-global:{h.key}",) for _, h, _ in report.undefined_global]
    keys += [
        (f"decorator-census:{name}",)
        for name, (nb, nh) in report.census.items()
        if nb != nh
    ]
    return keys


def apply_allow(report: Report, allow: dict[str, str]) -> None:
    """Record which allow-file entries cover a failing item."""
    report.allowed = {}
    used: set[str] = set()
    for names in _failing_keys(report):
        for name in names:
            if name in allow:
                report.allowed[name] = allow[name]
                used.add(name)
    report.unused_allow = sorted(set(allow) - used)


def failing_diffs(report: Report) -> list[tuple[Item, Item]]:
    return [
        (b, h) for b, h in report.moved_with_diff if not report.is_allowed(b.key, h.key)
    ]


def failing_missing(report: Report) -> list[Item]:
    return [m for m in report.missing if not report.is_allowed(m.key)]


def failing_undefined(report: Report) -> list[tuple[Item, Item, list[str]]]:
    return [
        t
        for t in report.undefined_global
        if not report.is_allowed(f"undefined-global:{t[1].key}")
    ]


def failing_census(report: Report) -> list[str]:
    return [
        name
        for name, (nb, nh) in report.census.items()
        if nb != nh and not report.is_allowed(f"decorator-census:{name}")
    ]


def has_failures(report: Report) -> bool:
    """Tell whether any un-allowed failing item remains."""
    return bool(
        failing_diffs(report)
        or failing_missing(report)
        or failing_undefined(report)
        or failing_census(report)
    )


def load_allow(path: str) -> dict[str, str]:
    """Parse an allow-file into ``{key: reason}``.

    A key is ``path::qualified.name``, ``undefined-global:path::qualified.name``
    or ``decorator-census:<decorator>``.  A leading ``@`` on the decorator is
    stripped, so ``decorator-census:@property`` is the key
    ``decorator-census:property``.  A bare ``qualified.name`` raises
    ``ValueError``: it would match a same-named def in every file.
    """
    allow: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition(" ")
        if "::" not in name and not name.startswith("decorator-census:"):
            raise ValueError(
                f"allow entry `{name}` has no file path; write it as "
                f"`<path>::{name.removeprefix('undefined-global:')}` "
                "(an entry without a path would excuse the name in every file)"
            )
        if name.startswith("decorator-census:@"):
            name = "decorator-census:" + name.removeprefix("decorator-census:@")
        allow[name] = reason.strip() or "(no reason given)"
    return allow


# --------------------------------------------------------------------------
# Driver and rendering
# --------------------------------------------------------------------------


def run_check(repo: str, base: str, head: str, allow: dict[str, str]) -> Report:
    """Compare ``base`` and ``head`` and return the classified report."""
    merge_base = _git(repo, "merge-base", base, head).strip() or base
    base_items: list[Item] = []
    head_items: list[Item] = []
    head_imports: dict[str, set[str]] = {}
    for path in changed_files(repo, merge_base, head):
        old = read_at(repo, merge_base, path)
        new = read_at(repo, head, path)
        try:
            if old is not None:
                base_items += collect(old, path)
            if new is not None:
                head_items += collect(new, path)
                head_imports[path] = module_imports(new)
        except SyntaxError as exc:
            raise RuntimeError(f"cannot parse {path}: {exc}") from exc
    report = match(base_items, head_items, head_imports)
    apply_allow(report, allow)
    return report


def compact_diff(b: Item, h: Item) -> list[str]:
    """Return a short unified diff of the two source segments."""
    diff = list(
        difflib.unified_diff(
            b.src.splitlines(),
            h.src.splitlines(),
            fromfile=b.label,
            tofile=h.label,
            n=1,
            lineterm="",
        )
    )
    if len(diff) > DIFF_LINE_CAP:
        extra = len(diff) - DIFF_LINE_CAP
        diff = diff[:DIFF_LINE_CAP] + [f"... ({extra} more diff lines)"]
    return diff


def _counts(report: Report) -> list[tuple[str, int]]:
    return [
        ("unchanged", len(report.unchanged)),
        ("moved-identical", len(report.moved_identical)),
        ("moved-with-diff", len(report.moved_with_diff)),
        ("missing", len(report.missing)),
        ("added", len(report.added)),
        ("type-checking (ignored)", len(report.type_checking)),
        ("import-hoisted (ignored)", len(report.import_hoisted)),
        ("logger-name (ignored)", len(report.logger_name)),
        ("undefined-global", len(report.undefined_global)),
        ("decorator-census mismatches", sum(a != b for a, b in report.census.values())),
    ]


def _allow_note(report: Report, *names: str) -> str:
    tag = report.is_allowed(*names)
    return f" (allowed by {tag}: {report.allowed[tag]})" if tag else ""


def render_text(report: Report) -> str:
    """Render the report as plain text."""
    out = ["move check: " + ", ".join(f"{k}={v}" for k, v in _counts(report))]
    for b, h in report.moved_with_diff:
        note = _allow_note(report, b.key, h.key)
        out.append(f"\nMOVED-WITH-DIFF{note}: {b.label} -> {h.label}")
        out += ["  " + line for line in compact_diff(b, h)]
    for m in report.missing:
        out.append(f"\nMISSING{_allow_note(report, m.key)}: {m.label}")
    for _, h, names in report.undefined_global:
        note = _allow_note(report, f"undefined-global:{h.key}")
        out.append(f"\nUNDEFINED-GLOBAL{note}: {h.label} reads {', '.join(names)}")
    for name, (nb, nh) in report.census.items():
        if nb != nh:
            note = _allow_note(report, f"decorator-census:{name}")
            out.append(
                f"\nDECORATOR-CENSUS{note}: @{name} base={nb} head={nh}"
                f" (allow entry: decorator-census:{name})"
            )
    for b, h in report.import_hoisted:
        out.append(f"\nimport-hoisted (ignored): {b.label} -> {h.label}")
    for b, h in report.logger_name:
        out.append(f"\nlogger-name (ignored): {b.label} -> {h.label}")
    for h in report.type_checking:
        out.append(f"\ntype-checking addition (ignored): {h.label}")
    for h in report.added:
        out.append(f"\nadded: {h.label}")
    for name in report.unused_allow:
        out.append(f"\nwarning: allow entry matches nothing failing: {name}")
    out.append("\nRESULT: " + ("FAIL" if has_failures(report) else "OK"))
    return "\n".join(out)


def render_markdown(report: Report) -> str:
    """Render the report as a markdown summary for a PR body."""
    out = [
        f"### Pure-move check: {'FAIL' if has_failures(report) else 'OK'}",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    out += [f"| {k} | {v} |" for k, v in _counts(report)]
    for b, h in report.moved_with_diff:
        note = _allow_note(report, b.key, h.key)
        out += ["", f"**moved-with-diff**{note}: `{b.label}` -> `{h.label}`"]
        out += ["", "```diff", *compact_diff(b, h), "```"]
    if report.missing:
        out += ["", "**missing**"]
        out += [f"- `{m.label}`{_allow_note(report, m.key)}" for m in report.missing]
    if report.undefined_global:
        out += ["", "**undefined-global**"]
        for _, h, names in report.undefined_global:
            note = _allow_note(report, f"undefined-global:{h.key}")
            lost = ", ".join(f"`{n}`" for n in names)
            out.append(f"- `{h.label}` reads {lost}{note}")
    mismatches = [(n, c) for n, c in report.census.items() if c[0] != c[1]]
    if mismatches:
        out += ["", "**decorator-census**"]
        for name, (nb, nh) in mismatches:
            note = _allow_note(report, f"decorator-census:{name}")
            out.append(
                f"- `@{name}`: base {nb}, head {nh}{note}"
                f" (allow entry: `decorator-census:{name}`)"
            )
    sections = (
        (
            "logger-name (ignored)",
            [f"`{b.label}` -> `{h.label}`" for b, h in report.logger_name],
        ),
        (
            "import-hoisted (ignored)",
            [f"`{b.label}` -> `{h.label}`" for b, h in report.import_hoisted],
        ),
        (
            "type-checking additions (ignored)",
            [f"`{h.label}`" for h in report.type_checking],
        ),
        ("added", [f"`{h.label}`" for h in report.added]),
    )
    for title, rows in sections:
        if rows:
            out += ["", f"<details><summary>{title}: {len(rows)}</summary>", ""]
            out += [f"- {r}" for r in rows]
            out += ["", "</details>"]
    if report.unused_allow:
        names = ", ".join(f"`{n}`" for n in report.unused_allow)
        out += ["", f"Unused allow entries: {names}"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the check; return the process exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("base", help="base ref (its merge base with head is used)")
    parser.add_argument("head", nargs="?", default="HEAD", help="head ref")
    parser.add_argument("--allow", help="file of `path::qualified.name  reason` lines")
    parser.add_argument("--markdown", action="store_true", help="print markdown")
    parser.add_argument("--repo", default=".", help="repository directory")
    args = parser.parse_args(argv)
    try:
        allow = load_allow(args.allow) if args.allow else {}
        report = run_check(args.repo, args.base, args.head, allow)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(report) if args.markdown else render_text(report))
    return 1 if has_failures(report) else 0


if __name__ == "__main__":
    sys.exit(main())
