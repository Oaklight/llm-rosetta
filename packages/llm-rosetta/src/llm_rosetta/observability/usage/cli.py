"""``llm-rosetta-usage`` — the usage report, read from a shell.

The gateway renders this report in its admin panel and a library gets it from
:meth:`~llm_rosetta.observability.usage.report.UsageReport.build`; this is the
third reader: an operator at a terminal asking the same question of the same
store, without starting the gateway.  It builds the report with the same code
and prints it as a table or as JSON, so ``--json`` is the report itself rather
than a second rendering of it that could disagree with the panel.

    llm-rosetta-usage --db /var/lib/llm-rosetta --window 24h
    llm-rosetta-usage --db ./data --by subject,type,model --json

The store is opened through
:class:`~llm_rosetta.observability.persistence.PersistenceManager` — the
gateway's own reader — but **without writing to it**: no backup, no table
creation, no migrations.  A lookup should not leave a copy of the database in
the data directory, and it should not take a write lock on a store another
process is serving.  When the store's schema is older than this build's, the
command refuses and says so, naming the two ways out: start the gateway once
(to migrate it where the migration belongs), or pass ``--migrate``, which opens
the store exactly as the gateway would — backup first, then the migrations.

Attribution needs the credentials that still exist, and they live in the
gateway's keys database, which this module reads directly — two columns of one
table, through the standard library — rather than through the gateway's
keystore: ``usage`` may not import the gateway, and attribution is the only
thing it needs from there.  Without that database every token is reported as
unattributed, which is a true answer to a smaller question: the totals are the
store's, only the names are missing.  ``--keys-db`` points at it when it does
not sit beside the store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_rosetta import __version__
from llm_rosetta.observability.persistence import PersistenceManager
from llm_rosetta.observability.usage.report import (
    INTERNAL_LABEL,
    INTERNAL_SUBJECT,
    NO_SUBJECT_LABEL,
    USAGE_WINDOW_LABELS,
    UsageReport,
)

__all__ = ["main"]

#: The store's filename inside a data directory, and the keys database's.
_DB_FILENAME = "gateway.db"
_KEYS_DB_FILENAME = "keys.db"

#: How a counter is printed: ledger column, then the heading a reader knows.
#: The order is the one a table reads well in — the whole last, after the parts
#: it is summed from — which is deliberately not the ledger's own order.
_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("request_count", "requests"),
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("cache_read_tokens", "cache read"),
    ("cache_creation_tokens", "cache write"),
    ("reasoning_tokens", "reasoning"),
    ("total_tokens", "total"),
)

_COUNTER_KEYS: tuple[str, ...] = tuple(key for key, _heading in _TABLE_COLUMNS)
_COUNTER_HEADINGS: tuple[str, ...] = tuple(heading for _key, heading in _TABLE_COLUMNS)

#: The cuts the caller can ask for.
_CUTS = ("subject", "type", "model")

#: A name column is the only unbounded one; a long one would push the numbers
#: off the screen, and it is the numbers the reader came for.
_NAME_WIDTH = 34

#: The window a tie is broken by: the widest one, so rows with no traffic in
#: the window being shown still come out in a meaningful order.
_WIDEST_WINDOW = USAGE_WINDOW_LABELS[-1]


class _StaleStore(Exception):
    """The store's schema is older than this build's, and we may not migrate."""


class _UnreadableSchema(Exception):
    """The store's schema could not be read, so it is not a stale store.

    A store whose write-ahead log cannot be attached reads as *empty* — SQLite
    reports no tables rather than an error the caller can see in a version
    number — and a store that is not a database at all reads the same way.
    Sending that reader to ``--migrate`` would point at a migration, when what
    failed is the read itself (review on #1031).
    """


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-rosetta-usage",
        description=(
            "Report token usage from an LLM-Rosetta store: the figures the "
            "gateway's admin panel shows, over the same rolling windows."
        ),
        epilog=(
            "Windows are rolling, not calendar: 24h means the last 24 hours. "
            "Each table is ordered by the window it shows, the widest window "
            "breaking ties, so a row's position is explained by the numbers "
            "beside it.  Every table's rows add up to the store's total, "
            "because traffic whose credential is gone is reported as "
            "unattributed rather than dropped.  The store is read without "
            "being written to; see --migrate."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="the data directory holding gateway.db, or that file itself",
    )
    parser.add_argument(
        "--window",
        choices=USAGE_WINDOW_LABELS,
        default="24h",
        help="the window the tables show (default: %(default)s)",
    )
    parser.add_argument(
        "--by",
        default="subject",
        metavar="CUT[,CUT]",
        help=(
            "what to cut the report by, comma-separated, from "
            f"{', '.join(_CUTS)} (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--keys-db",
        metavar="PATH",
        help=(
            "the gateway's keys database, for attribution "
            f"(default: {_KEYS_DB_FILENAME} beside the store)"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help=(
            "keep this many rows in each table, 0 for all; a table's total "
            "still sums every row of its cut, and --json is never truncated "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help=(
            "open the store exactly as the gateway would — take a backup, "
            "create missing tables and apply this build's migrations.  Off by "
            "default: a report is refused rather than migrated when the "
            "store's schema is older than this build's"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of tables",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _resolve_store(target: str) -> Path | None:
    """The data directory *target* names, or ``None`` if it holds no store.

    A store that does not exist is reported as missing rather than created: a
    mistyped path would otherwise leave an empty database behind and then read
    as a store that saw no traffic.
    """
    path = Path(target).expanduser()
    if path.is_file():
        return path.parent if path.name == _DB_FILENAME else None
    if path.is_dir() and (path / _DB_FILENAME).is_file():
        return path
    return None


def _read_keys(keys_db: Path) -> tuple[list[dict[str, str]], str | None]:
    """The live credentials as ``{id, label, usage_bucket}``, and any reason not.

    The keys database is opened read-only, so reporting cannot change a
    credential.  A database that is missing or unreadable yields nothing to
    attribute *and says which it was*: the caller tells the reader, because an
    all-zero per-subject table would otherwise read as "nobody spent this" when
    what happened is that nobody could be named.
    """
    if not keys_db.is_file():
        return [], f"no keys database at {keys_db}"
    try:
        with sqlite3.connect(f"{keys_db.resolve().as_uri()}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT id, label, usage_bucket FROM api_keys "
                "WHERE usage_bucket IS NOT NULL"
            ).fetchall()
    except sqlite3.Error as exc:
        return [], f"could not read the keys database at {keys_db}: {exc}"
    return (
        [
            {"id": str(key_id), "label": str(label or ""), "usage_bucket": str(bucket)}
            for key_id, label, bucket in rows
        ],
        None,
    )


def _counter_rows(
    labels: Sequence[str], windows: Iterable[Mapping[str, int]]
) -> list[list[str]]:
    """Pair each row's label with its counters, in the table's column order."""
    return [
        [label, *(f"{window.get(key, 0):,}" for key in _COUNTER_KEYS)]
        for label, window in zip(labels, windows)
    ]


def _rank(
    entries: Sequence[tuple[str, Mapping[str, Mapping[str, int]]]], window: str
) -> list[tuple[str, Mapping[str, Mapping[str, int]]]]:
    """Order ``(label, windows)`` rows by the window the table shows.

    One rule for every cut, so a row's position is explained by the numbers
    printed beside it: the model cut used to inherit the store's widest-window
    ordering while displaying a narrower window, which put a row of zeros at
    the top of a 24h table and, under ``--top``, hid the row that had the
    traffic (review on #1031).

    The widest window breaks ties, because a row with no traffic in the shown
    window has no figure there to order it by, and "which of these mattered
    most overall" is the honest second question.  The label breaks the
    remainder, so two models with nothing at all still come out in one order.
    """

    def key(entry: tuple[str, Mapping[str, Mapping[str, int]]]) -> tuple[int, int, str]:
        label, windows = entry
        return (
            -_window_totals(windows, window),
            -_window_totals(windows, _WIDEST_WINDOW),
            label,
        )

    return sorted(entries, key=key)


def _window_totals(windows: Mapping[str, Mapping[str, int]], window: str) -> int:
    """One window's canonical total for a row, zero when the row has none."""
    return (windows.get(window) or {}).get("total_tokens", 0)


def _counters(
    windows: Mapping[str, Mapping[str, int]], window: str
) -> Mapping[str, int]:
    """One window's counter dict, empty when the row has none (an old store)."""
    return windows.get(window) or {}


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Lay *rows* out under *headers*, right-aligned from the second column on.

    The first column is a name and the rest are counts, so every number in a
    column lines up on its last digit and can be compared down the page.
    """
    widths = [len(head) for head in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [
            cell.rjust(widths[index]) if index else cell.ljust(widths[index])
            for index, cell in enumerate(cells)
        ]
        return "  ".join(padded).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return [line(headers), rule, *map(line, rows)]


def _shorten(text: str, width: int = _NAME_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _sum_windows(windows: Iterable[dict[str, int]]) -> dict[str, int]:
    """Add up window dicts, for a total over rows that are already folded."""
    total: dict[str, int] = {}
    for window in windows:
        for key, value in window.items():
            total[key] = total.get(key, 0) + value
    return total


def _truncate(lines: list[str], shown: int, hidden: int) -> list[str]:
    """Note the rows a table is not showing, under the last one it shows.

    Each table's own total still sums every row of its cut, so bounding the
    body is safe: what is cut is rows, never arithmetic.  ``lines`` opens with
    a heading and a rule, so ``shown`` data rows sit at 2..2+shown and the note
    takes the place of the ones it hides.
    """
    if hidden > 0:
        lines.insert(
            2 + shown, f"… and {hidden} more; --top 0 for all, --json for the lot"
        )
    return lines


def _subject_table(report: UsageReport, window: str, top: int) -> list[str]:
    """The report's classes, largest window total first, then the whole."""
    entries: list[tuple[str, Mapping[str, Mapping[str, int]]]] = [
        (
            f"{_shorten(usage.label or NO_SUBJECT_LABEL)} (shared)"
            if usage.shared_label
            else _shorten(usage.label or NO_SUBJECT_LABEL),
            usage.windows,
        )
        for usage in report.by_subject.values()
    ]
    if report.internal is not None:
        entries.append((f"{INTERNAL_LABEL} (this gateway)", report.internal))
    if report.unattributed is not None:
        entries.append(("(unattributed)", report.unattributed))

    ranked = _rank(entries, window)
    shown = top if top > 0 else len(ranked)
    ranked = [*ranked[:shown], ("TOTAL", {window: report.total(window)})]
    lines = _format_table(
        ["subject", *_COUNTER_HEADINGS],
        _counter_rows(
            [label for label, _windows in ranked],
            [_counters(windows, window) for _label, windows in ranked],
        ),
    )
    # Truncating the subject table is the case that matters: it grows with
    # every credential the gateway has ever seen.
    return _truncate(lines, shown, len(entries) - shown)


def _type_table(report: UsageReport, window: str, top: int) -> list[str]:
    """One row per model type that saw traffic, then the whole.

    The types partition every row, so their sum is the total the subject table
    ends on: the two cuts agree down to the last token.
    """
    ranked = _rank(list(report.by_model_type.items()), window)
    shown = top if top > 0 else len(ranked)
    ranked = [*ranked[:shown], ("TOTAL", {window: report.total(window)})]
    lines = _format_table(
        ["model type", *_COUNTER_HEADINGS],
        _counter_rows(
            [name for name, _windows in ranked],
            [_counters(windows, window) for _name, windows in ranked],
        ),
    )
    return _truncate(lines, shown, len(report.by_model_type) - shown)


def _model_table(
    by_model: Sequence[dict[str, Any]], window: str, top: int
) -> list[str]:
    """The busiest models in the window shown, then the whole.

    The store returns these ordered by the widest window — a read has no
    window to order by — so the table re-ranks them under the one rule every
    cut uses (:func:`_rank`).  The total is over every model, not over the rows
    shown: truncation is a property of the screen, and a total that shrank with
    it would be a lie.
    """
    entries = [
        (
            f"{_shorten(str(group.get('model') or '(unnamed)'))} "
            f"[{group.get('model_type')}]",
            group["windows"],
        )
        for group in by_model
    ]
    ranked = _rank(entries, window)

    total = _sum_windows(group["windows"][window] for group in by_model)
    shown = top if top > 0 else len(ranked)
    lines = _format_table(
        ["model", *_COUNTER_HEADINGS],
        _counter_rows(
            [label for label, _windows in ranked[:shown]] + ["TOTAL (all models)"],
            [_counters(windows, window) for _label, windows in ranked[:shown]]
            + [total],
        ),
    )
    return _truncate(lines, shown, len(ranked) - shown)


def _render(
    store: Path,
    window: str,
    cuts: Sequence[str],
    report: UsageReport,
    by_model: Sequence[dict[str, Any]],
    top: int,
) -> str:
    """The report as text: one table per requested cut, under one heading."""
    lines = [f"store: {store}", f"window: {window} (rolling)", ""]
    for cut in cuts:
        if cut == "subject":
            lines.extend(_subject_table(report, window, top))
        elif cut == "type":
            lines.extend(_type_table(report, window, top))
        else:
            lines.extend(_model_table(by_model, window, top))
        lines.append("")
    return "\n".join(lines).rstrip()


def _payload(
    store: Path,
    window: str,
    cuts: Sequence[str],
    report: UsageReport,
    names: dict[str, str],
    by_model: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The report as JSON: the panel's payload, plus the other two cuts.

    The subject cut adds exactly the keys the admin API returns, so a script
    reading this and one reading the API see one shape.  ``window`` is the one
    the tables would show; ``windows``, from the report, is the rolling set.
    """
    payload: dict[str, Any] = {
        "store": str(store),
        "window": window,
        "windows": list(report.windows),
    }
    if "subject" in cuts:
        payload.update(report.to_payload(names))
    if "type" in cuts:
        payload["model_types"] = {
            name: dict(windows) for name, windows in report.by_model_type.items()
        }
    if "model" in cuts:
        payload["models"] = [dict(group) for group in by_model]
    return payload


async def _collect(
    store: Path, keys_db: Path, *, migrate: bool, by_model: bool
) -> tuple[UsageReport, dict[str, str], list[dict[str, Any]], str | None]:
    """Read the store once; every cut renders from what comes back.

    Without ``--migrate`` the store is opened for reading only and its schema
    is checked first: a store older than this build would be read with the
    wrong shape, and migrating it here would take a write lock on a file
    another process is serving, so that case refuses instead.

    ``by_model`` says whether the model cut was asked for: its aggregate is a
    second pass over the log, and a run that only wants subjects should not pay
    for it.

    Raises:
        _StaleStore: The store's schema is behind this build's and the caller
            did not ask for the migration.
        _UnreadableSchema: The store's schema could not be read at all, which
            is a different answer from "it is old".
    """
    persistence = await PersistenceManager.create(
        str(store), backup=migrate, migrate=migrate
    )
    try:
        if not migrate:
            current = await persistence.read_schema_version()
            expected = PersistenceManager.expected_schema_version()
            if current is None:
                raise _UnreadableSchema(
                    f"could not read the schema of the store at {store}: it "
                    "reads as a database with no tables, which is what a "
                    "write-ahead log this process cannot attach looks like.  "
                    "The store is not stale — its -wal and -shm files, or the "
                    "directory that would hold them, have to be writable, or "
                    "the store checkpointed first"
                )
            if current < expected:
                raise _StaleStore(
                    f"the store at {store} is at schema version {current} and "
                    f"this build expects {expected}"
                )
        groups = (await persistence.query_usage_groups())["groups"]
        models = (
            (await persistence.query_usage_by_model())["groups"] if by_model else []
        )
    finally:
        await persistence.close()

    keys, unreadable = _read_keys(keys_db)
    subjects = [
        (key["usage_bucket"], key["label"]) for key in keys if key["usage_bucket"]
    ]
    names = {key["usage_bucket"]: key["id"] for key in keys if key["usage_bucket"]}
    report = UsageReport.build(
        groups=groups,
        subjects=subjects,
        internal_subject=INTERNAL_SUBJECT,
        internal_label=INTERNAL_LABEL,
    )
    return report, names, models, unreadable


def _resolve_cuts(raw: str) -> list[str] | None:
    cuts = [part.strip() for part in raw.split(",") if part.strip()]
    if not cuts or any(cut not in _CUTS for cut in cuts):
        return None
    return cuts


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``llm-rosetta-usage`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    cuts = _resolve_cuts(args.by)
    if cuts is None:
        parser.error(f"--by takes some of {', '.join(_CUTS)}, not {args.by!r}")
    if args.top < 0:
        parser.error(f"--top cannot be negative, got {args.top}")

    store = _resolve_store(args.db)
    if store is None:
        print(
            f"no store at {args.db}: expected a data directory holding "
            f"{_DB_FILENAME}, or that file",
            file=sys.stderr,
        )
        return 1

    keys_db = (
        Path(args.keys_db).expanduser() if args.keys_db else store / _KEYS_DB_FILENAME
    )
    try:
        report, names, by_model, unreadable = asyncio.run(
            _collect(
                store,
                keys_db,
                migrate=args.migrate,
                by_model="model" in cuts,
            )
        )
    except _StaleStore as exc:
        print(
            f"{exc}.  Start the gateway once to migrate it there, or pass "
            "--migrate to do it here (it takes a backup first).",
            file=sys.stderr,
        )
        return 1
    except _UnreadableSchema as exc:
        # No "--migrate" hint: the store is not old, it could not be read.
        print(f"{exc}.", file=sys.stderr)
        return 1
    except (sqlite3.Error, OSError) as exc:
        # OSError covers the store the operator cannot write: a mounted
        # volume, a root-owned file, a snapshot copy.  Reading a current one
        # works; reading a stale one is refused above, naming the schema
        # rather than the permission, and only --migrate gets as far as the
        # write that fails.
        attempted = "migrate" if args.migrate else "read"
        print(
            f"could not {attempted} the store at {store}: {exc}",
            file=sys.stderr,
        )
        return 1

    if "subject" in cuts and unreadable is not None:
        print(
            f"note: {unreadable}, so no token could be attributed to a "
            "credential; every one is reported as unattributed (pass --keys-db "
            "to point at it)",
            file=sys.stderr,
        )

    if args.json:
        payload = _payload(store, args.window, cuts, report, names, by_model)
        print(json.dumps(payload, indent=2))
    else:
        print(_render(store, args.window, cuts, report, by_model, args.top))
    return 0


if __name__ == "__main__":  # python -m llm_rosetta.observability.usage.cli
    raise SystemExit(main())
