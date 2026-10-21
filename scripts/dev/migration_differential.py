#!/usr/bin/env python3
"""Show that two source trees open a store the same way, or show where they differ.

A refactor of the migrations is only worth anything if it can be *shown* that
nothing changed, and "nothing changed" is a claim about what a reader of the
store can observe: the schema objects, the version rows, the columns and their
defaults, the indexes, and the log lines the open writes.  This runs exactly
that observation against two trees — each a working copy or a git ref — over a
store per pre-state, and diffs the results.

    python scripts/dev/migration_differential.py --old origin/master

The states are the shapes a store can be found in, and they are built from the
tree's own ``PersistenceManager._MIGRATIONS`` rather than from a hand-written
history:

- ``fresh`` — no store at all, so the open creates one;
- ``at_0`` … ``at_6`` — a store the tree's own first *n* migrations leave
  behind, *n* from 0 (the base tables alone) to every migration applied;
- ``pre_registry`` — every column present and no version row, which is what the
  old unversioned ``_migrate_add_columns`` left and what the pre-registry
  inference has to recognise.

Because a store at v3 really is missing v4's column, the ``ALTER TABLE``
statements run instead of hitting the duplicate-column tolerance, which a store
whose version rows were merely deleted would not exercise.  A tree whose
numbering differs from the other's shows up as a missing or extra state rather
than as a crash.

The store root is a temporary directory per tree and is masked out of the
comparison, along with timestamps and the backup file's size, so two runs can be
compared though they ran at different times in different places.  Exit status is
0 only when every state matches, which makes it usable in front of a refactor
PR:

    python scripts/dev/migration_differential.py --old origin/master --new HEAD

**This compares success.** A store that opens cleanly is byte-identical between
the two trees; the failure path deliberately is not — a failing step now logs a
WARNING naming the step and how long it ran, and the traceback carries the
registry's frames — so a run that matches has not said anything about what a
deployment with a pinned log stream sees when a migration fails.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: A timestamp, or the stub of one, in either form: the run's clock rather than
#: the store's shape.  The second one is how a backup file is named.
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_COMPACT_STAMP = re.compile(r"\d{8}[-_]\d{6}")

#: A backup's size in the log line that reports it.  It is the SQLite build's
#: page size showing through, not the migrations' doing.
_BACKUP_SIZE = re.compile(r"\(\d+(?:\.\d+)? MB\)")


# --------------------------------------------------------------------------- #
# The observation: what opening a store does, in whichever tree is importable.
# --------------------------------------------------------------------------- #


#: The observability loggers, in the spellings the package actually uses:
#: ``llm-rosetta.observability`` (hyphens) is the store's own logger.
_OBSERVABILITY_LOGGERS = ("llm-rosetta.observability", "llm_rosetta.observability")

#: The packages whose level has to be raised for their INFO records to exist.
_LOG_PACKAGES = ("llm-rosetta", "llm_rosetta")


class _Collect(logging.Handler):
    """The open's own log lines: an observable effect, so part of the diff."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_OBSERVABILITY_LOGGERS):
            self.lines.append(f"{record.levelname} {record.getMessage()}")


@contextlib.contextmanager
def _collecting() -> Iterator[_Collect]:
    """Capture the opens' own log lines, whatever the ambient level is.

    A handler cannot capture a record that was never emitted: an INFO line is
    dropped at the originating logger when its effective level is higher, and in
    a test process the root sits at WARNING.  That is how a recorded snapshot
    came to hold empty log lists — keys that compare equal to nothing — while a
    differential run, whose child sets the root to INFO, counted thirteen lines
    in the same state.  The levels are raised for the duration of the capture
    and put back afterwards, so observing a store does not change any caller's
    logging.
    """
    collector = _Collect()
    root = logging.getLogger()
    root.addHandler(collector)
    raised: list[tuple[logging.Logger, int]] = []
    for name in _LOG_PACKAGES:
        package = logging.getLogger(name)
        if package.level == logging.NOTSET or package.level > logging.INFO:
            raised.append((package, package.level))
            package.setLevel(logging.INFO)
    try:
        yield collector
    finally:
        for package, level in raised:
            package.setLevel(level)
        root.removeHandler(collector)


def _schema_of(db: Path) -> dict[str, Any]:
    """Everything about the store a reader could see, and nothing about the run."""
    conn = sqlite3.connect(db)
    try:
        objects = conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        versions = conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        columns = conn.execute("PRAGMA table_info(request_log)").fetchall()
        indexes = conn.execute("PRAGMA index_list(request_log)").fetchall()
    finally:
        conn.close()
    return {
        "objects": [
            [kind, name, (sql or "").replace("\n", " ").strip()]
            for kind, name, sql in objects
        ],
        "versions": [version for (version,) in versions],
        "columns": [[row[1], row[2], row[4]] for row in columns],
        "indexes": sorted(row[1] for row in indexes),
    }


async def _nothing(_manager_class: Any, _store: Path) -> None:
    """The fresh case: there is no store yet, so the open creates one."""


def _left_at(version: int) -> Any:
    """A store the tree's own first ``version`` migrations leave behind.

    Truncating the migration table rather than deleting version rows is what
    makes the *statements* run: a store at v3 is genuinely missing v4's column,
    so ``ALTER TABLE`` adds it instead of hitting the duplicate-column
    tolerance.  Version 0 is the base tables alone.
    """

    async def _prepare(manager_class: Any, store: Path) -> None:
        saved = manager_class._MIGRATIONS
        manager_class._MIGRATIONS = saved[:version]
        try:
            manager = await manager_class.create(str(store))
            await manager.close()
        finally:
            manager_class._MIGRATIONS = saved

    return _prepare


async def _pre_registry(manager_class: Any, store: Path) -> None:
    """Every column present, and no record of any of them.

    This is the one state no truncation can build, and the one the pre-registry
    inference exists for: if the adoption step does not recognise it, every
    migration after v1 tries to add a column that is already there.
    """
    manager = await manager_class.create(str(store))
    await manager.close()
    conn = sqlite3.connect(store / "gateway.db")
    try:
        conn.execute("DELETE FROM schema_version")
        conn.commit()
    finally:
        conn.close()


async def _open_and_observe(
    manager_class: Any, store: Path, prepare: Any
) -> dict[str, Any]:
    """Put the store into one pre-state, then open it twice and report both.

    The state's own construction is not observed — for ``at_n`` it is an older
    release doing the work — and both of the opens that follow are, because
    which one has something to do depends on the state: a fresh store does all
    of it on the first open, a back-level store on the first as well, and a
    current one on neither.
    """
    if store.exists():
        shutil.rmtree(store)
    await prepare(manager_class, store)

    with _collecting() as first_log:
        first = await manager_class.create(str(store))
        await first.close()

    with _collecting() as second_log:
        again = await manager_class.create(str(store))
        try:
            return {
                "schema": _schema_of(store / "gateway.db"),
                "schema_version": await again.schema_version(),
                "log_first_open": sorted(set(first_log.lines)),
                "log_second_open": sorted(set(second_log.lines)),
            }
        finally:
            await again.close()


def _hash_tree(root: Path) -> str:
    """A hash of the Python under ``root``: the contents, never where they live.

    Nothing about the directory is hashed, only the paths inside it, so two
    copies of one tree in different places hash alike.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint() -> str:
    """A hash of the code that opens a store, so a same-tree run is caught.

    Two byte-identical trees would report ``IDENTICAL`` and prove nothing, and
    the paths cannot tell us: two exports of the same ref live in different
    directories.  The bytes can.
    """
    import llm_rosetta.observability as package

    return _hash_tree(Path(package.__file__).parent)


async def _probe(store_root: Path) -> dict[str, Any]:
    """Observe every state, using whichever tree ``PYTHONPATH`` selects."""
    from llm_rosetta import __file__ as package_file
    from llm_rosetta.observability import PersistenceManager

    current = max(
        version for version, _description, _sql in PersistenceManager._MIGRATIONS
    )
    states: dict[str, Any] = {"fresh": _nothing, "pre_registry": _pre_registry}
    states.update(
        {f"at_{version}": _left_at(version) for version in range(current + 1)}
    )

    observed: dict[str, Any] = {}
    for name, prepare in states.items():
        observed[name] = await _open_and_observe(
            PersistenceManager, store_root / name, prepare
        )
    return {
        "package": package_file,
        "fingerprint": _fingerprint(),
        "states": observed,
    }


def observe_states(store_root: Path) -> dict[str, Any]:
    """Every state, normalised, in the tree that is importable right now.

    This is what the differential does to a child's dump, in one process, and
    what the snapshot test uses to record and to re-derive the frozen store:
    the fixture is this function's output, so a later slice that changes what an
    open leaves behind fails against it.
    """
    report = asyncio.run(_probe(store_root))
    return {
        state: _normalise(value, store_root)
        for state, value in report["states"].items()
    }


# --------------------------------------------------------------------------- #
# The comparison.
# --------------------------------------------------------------------------- #


def _normalise(value: Any, root: Path) -> Any:
    """Mask what belongs to this run rather than to the store."""
    if isinstance(value, str):
        masked = value.replace(str(root), "<store-root>")
        masked = _TIMESTAMP.sub("<ts>", masked)
        masked = _COMPACT_STAMP.sub("<ts>", masked)
        return _BACKUP_SIZE.sub("(<size> MB)", masked)
    if isinstance(value, list):
        return [_normalise(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item, root) for key, item in value.items()}
    return value


def _summarise(state: dict[str, Any]) -> str:
    """The numbers that make a row of this table readable at a glance."""
    schema = state["schema"]
    return (
        f"v={schema['versions']} cols={len(schema['columns'])} "
        f"idx={len(schema['indexes'])} "
        f"log={len(state['log_first_open'])}/{len(state['log_second_open'])}"
    )


def _diff(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """The two states as text, diffed, so the report says *what* changed."""
    return list(
        difflib.unified_diff(
            json.dumps(old, indent=1, sort_keys=True).splitlines(),
            json.dumps(new, indent=1, sort_keys=True).splitlines(),
            fromfile=f"{name} (old)",
            tofile=f"{name} (new)",
            lineterm="",
        )
    )


def _compare(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Print the report; return whether every state matched."""
    old_states: dict[str, Any] = old["states"]
    new_states: dict[str, Any] = new["states"]
    names = sorted(set(old_states) | set(new_states))

    width = max(len(name) for name in names)
    differing: list[str] = []
    diff_text: list[str] = []
    for name in names:
        if name not in old_states or name not in new_states:
            # The tree that has it is the other one.
            side = "new" if name not in old_states else "old"
            differing.append(name)
            diff_text.append(f"{name}: only the {side} tree has this state")
            print(f"{name:<{width}}  MISSING ({side} tree only)")
            continue
        matched = old_states[name] == new_states[name]
        print(
            f"{name:<{width}}  {_summarise(old_states[name])} "
            f"{'==' if matched else '!='} {_summarise(new_states[name])}"
        )
        if not matched:
            differing.append(name)
            diff_text.extend(_diff(name, old_states[name], new_states[name]))

    print()
    if differing:
        print("\n".join(diff_text))
        print()
        print(
            f"DIFFERENT: {', '.join(differing)} ({len(differing)} of {len(names)} states)"
        )
        return False
    print(
        f"IDENTICAL: {len(names)} states, same schema, same version rows, "
        f"same log lines"
    )
    return True


# --------------------------------------------------------------------------- #
# Picking the two trees, and running the probe in each.
# --------------------------------------------------------------------------- #


def _source_root(root: Path) -> Path | None:
    """Where ``llm_rosetta`` is importable from, inside a checkout or an export."""
    for candidate in (
        root / "packages/llm-rosetta/src",
        root / "src",
        root,
    ):
        if (candidate / "llm_rosetta").is_dir():
            return candidate
    return None


def _export_ref(ref: str, dest: Path) -> None:
    """A git ref as a directory, without touching the working tree."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", ref, "src"], cwd=REPO, capture_output=True
    )
    if archive.returncode != 0:
        raise SystemExit(
            f"cannot export {ref!r}: {archive.stderr.decode(errors='replace').strip()}"
        )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        try:
            bundle.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12 has no filter
            bundle.extractall(dest)


def _tree(value: str, workdir: Path, name: str) -> tuple[str, Path]:
    """Resolve ``--old``/``--new`` to a label and a directory to put on PYTHONPATH."""
    path = Path(value).expanduser()
    if path.is_dir():
        root = _source_root(path.resolve())
        if root is None:
            raise SystemExit(f"{value!r} holds no importable llm_rosetta package")
        return f"{value} (working copy)", root

    dest = workdir / f"tree-{name}"
    _export_ref(value, dest)
    root = _source_root(dest)
    if root is None:  # pragma: no cover - a ref that exports no src/ is a bad ref
        raise SystemExit(f"{value!r} exports no importable llm_rosetta package")
    return f"{value} (git ref)", root


def _observe(label: str, src: Path, workdir: Path, name: str) -> dict[str, Any]:
    """Run the probe in a child process, so it imports the tree we selected."""
    out = workdir / f"{name}.json"
    store_root = workdir / f"stores-{name}"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    probe = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dump",
            str(out),
            "--store-root",
            str(store_root),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not out.exists():
        raise SystemExit(f"the probe failed for {label}:\n{probe.stdout}{probe.stderr}")
    report = json.loads(out.read_text())
    return {
        # Left absolute on purpose: it is how the operator sees that the two
        # children really did import different trees.  A PYTHONPATH that an
        # editable install overrides shows up here before it can be mistaken
        # for a pass, and the fingerprint below is what catches the rest.
        "package": report["package"],
        "fingerprint": report["fingerprint"],
        "states": {
            state: _normalise(value, store_root)
            for state, value in report["states"].items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diff what opening a store does, between two source trees."
    )
    parser.add_argument(
        "--old",
        default="origin/master",
        help="a git ref or a working copy holding the tree to compare against "
        "(default: origin/master)",
    )
    parser.add_argument(
        "--new",
        default=".",
        help="a git ref or a working copy holding the tree to check "
        "(default: the working copy this script runs from)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary trees and stores, and say where they are",
    )
    parser.add_argument("--dump", help=argparse.SUPPRESS)
    parser.add_argument("--store-root", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # The child: one tree, observed, written where the parent can read it.
    if args.dump:
        logging.basicConfig(level=logging.INFO)
        store_root = Path(args.store_root)
        store_root.mkdir(parents=True, exist_ok=True)
        report = asyncio.run(_probe(store_root))
        Path(args.dump).write_text(json.dumps(report, indent=1, sort_keys=True))
        return 0

    workdir = Path(tempfile.mkdtemp(prefix="migration-differential-"))
    try:
        old_label, old_src = _tree(args.old, workdir, "old")
        new_label, new_src = _tree(args.new, workdir, "new")
        old = _observe(old_label, old_src, workdir, "old")
        new = _observe(new_label, new_src, workdir, "new")

        print(f"old  {old_label}\n     {old['package']}")
        print(f"new  {new_label}\n     {new['package']}")
        if old["fingerprint"] == new["fingerprint"]:
            raise SystemExit(
                "both trees hold the same observability code, so every state "
                "would match by construction and the check would prove nothing: "
                "point --old at another tree or another ref"
            )
        print()

        return 0 if _compare(old, new) else 1
    finally:
        if args.keep:
            print(f"\nkept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
