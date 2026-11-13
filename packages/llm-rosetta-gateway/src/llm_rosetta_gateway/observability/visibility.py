"""Time ranges the operator cleared: hidden from reads, not yet deleted.

A clear or a cleanup does not delete on the request path.  It records a range
here, at O(1) cost, so every read stops returning those rows immediately while a
background job deletes them in batches.  A read appends the predicate
``NOT ((timestamp >= ? AND timestamp < ?) OR …)`` — literal bounds, so the
planner keeps using the timestamp indexes — and a filter with no applicable
range returns no SQL at all, which is the normal state.

The ranges are held in memory and kept in step with the ``hidden_ranges`` table
by :class:`~llm_rosetta_gateway.observability.persistence.PersistenceManager`;
a read never queries the table.

Timestamp-space only: the gateway stamps a row when it is written, so a range
never catches a later row.  A store written by a second process concurrently
would need an insert-time column and a ``< hidden_at`` condition; that is out of
scope until the gateway runs more than one process.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Stand-ins for an open bound, so a range can be unbounded at either end
#: without a nullable column.  The empty string sorts below every ISO timestamp
#: and this one sorts above every timestamp this decade.
MIN_TS = ""
MAX_TS = "9999-12-31T23:59:59.999999+00:00"

#: The tables whose rows can be hidden.  All three carry a ``timestamp``
#: column, which is what a range is expressed in.
HIDDEN_TABLES: tuple[str, ...] = ("request_log", "error_dumps", "ops_log")


class VisibilityFilter:
    """The active hidden ranges, keyed by table."""

    __slots__ = ("_ranges",)

    def __init__(self) -> None:
        self._ranges: dict[str, list[tuple[str, str]]] = {
            table: [] for table in HIDDEN_TABLES
        }

    def load(self, rows: Iterable[Sequence[str]]) -> None:
        """Replace every cached range from ``(table, start, end)`` rows."""
        self._ranges = {table: [] for table in HIDDEN_TABLES}
        for table, start, end in rows:
            self._ranges.setdefault(table, []).append((start or MIN_TS, end or MAX_TS))

    def add(self, table: str, start: str, end: str) -> None:
        """Record a range whose row is already in the store."""
        self._ranges.setdefault(table, []).append((start or MIN_TS, end or MAX_TS))

    def remove(self, table: str, start: str, end: str) -> None:
        """Forget a range whose rows are physically gone."""
        try:
            self._ranges.get(table, []).remove((start or MIN_TS, end or MAX_TS))
        except ValueError:
            pass

    def ranges(self, table: str) -> list[tuple[str, str]]:
        """The active ranges for *table*, oldest first."""
        return list(self._ranges.get(table, ()))

    def is_empty(self) -> bool:
        """True when nothing is hidden anywhere — the read path's fast case."""
        return not any(self._ranges.values())

    def hidden_clause(
        self,
        table: str,
        *,
        alias: str = "",
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[str, list[str]]:
        """The rows a read must exclude, and their parameters.

        *since* and *until* describe the window the caller is already reading:
        a range wholly outside it is irrelevant to this query and is skipped, so
        a 24-hour query pays nothing for a range that ended months ago.
        """
        applicable = self._applicable(table, since=since, until=until)
        if not applicable:
            return "", []
        column = f"{alias}timestamp" if alias else "timestamp"
        parts = [f"({column} >= ? AND {column} < ?)" for _ in applicable]
        params = [bound for pair in applicable for bound in pair]
        return "(" + " OR ".join(parts) + ")", params

    def clause(
        self,
        table: str,
        *,
        alias: str = "",
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[str, list[str]]:
        """The predicate a read appends, and its parameters.

        Returns ``("", [])`` when nothing in the query's window is hidden.
        """
        hidden, params = self.hidden_clause(
            table, alias=alias, since=since, until=until
        )
        if not hidden:
            return "", []
        return f"NOT {hidden}", params

    def _applicable(
        self,
        table: str,
        *,
        since: str | None,
        until: str | None,
    ) -> list[tuple[str, str]]:
        ranges = self._ranges.get(table)
        if not ranges:
            return []
        return [
            (start, end)
            for start, end in ranges
            if (until is None or start < until) and (since is None or end > since)
        ]
