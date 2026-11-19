"""Visibility filter, raw connection access and write transactions."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from ..._vendor import aiosqlite
from ..visibility import MAX_TS, MIN_TS, VisibilityFilter
from ._state import _StoreState


class _ConnectionMixin(_StoreState):
    """Visibility filter, raw connection access and write transactions."""

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    @property
    def visibility(self) -> VisibilityFilter:
        """The ranges the operator cleared; empty unless a clear is pending."""
        return self._visibility

    @property
    def connection(self) -> aiosqlite.Connection:
        """The open connection, for a subsystem that owns its own tables.

        The job store keeps its rows in this database and needs the same
        connection, lock and pragmas; this property keeps that dependency
        explicit instead of reaching into a private attribute.
        """
        return self._conn

    def _visible(
        self,
        table: str,
        *,
        alias: str = "",
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[str, list[str]]:
        """The predicate that hides *table*'s cleared rows, and its parameters.

        Returns ``("", [])`` when nothing in the query's window is hidden.
        """
        return self._visibility.clause(table, alias=alias, since=since, until=until)

    async def reload_visibility(self) -> None:
        """Re-read the hidden ranges after a subsystem edited them directly."""
        await self._load_visibility()

    async def _load_visibility(self) -> None:
        """Load the hidden ranges into the cache every read consults."""
        try:
            rows = await self._conn.execute_fetchall(
                "SELECT table_name, start_ts, end_ts FROM hidden_ranges"
            )
        except Exception:
            # A read-only open of a store that predates the table: nothing is
            # hidden, which is exactly true for that store.
            rows = []
        self._visibility.load(rows)

    async def add_hidden_range(
        self, table: str, start: str, end: str, job_id: str = ""
    ) -> str:
        """Record *table*'s rows in ``[start, end)`` as hidden; return the id.

        Nothing is written to the table itself.  The caller adds the job that
        will delete the rows in the same transaction, so a crash cannot leave
        rows hidden with nothing coming to remove them.
        """
        start = start or MIN_TS
        end = end or MAX_TS
        range_id = uuid.uuid4().hex
        await self._conn.execute(
            "INSERT INTO hidden_ranges "
            "(id, table_name, start_ts, end_ts, job_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                range_id,
                table,
                start,
                end,
                job_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._conn.commit()
        self._visibility.add(table, start, end)
        return range_id

    async def remove_hidden_ranges_for_job(self, job_id: str) -> int:
        """Forget every range *job_id* owns, because its rows are gone."""
        rows = await self._conn.execute_fetchall(
            "SELECT table_name, start_ts, end_ts FROM hidden_ranges WHERE job_id = ?",
            (job_id,),
        )
        cur = await self._conn.execute(
            "DELETE FROM hidden_ranges WHERE job_id = ?", (job_id,)
        )
        await self._conn.commit()
        for table, start, end in rows:
            self._visibility.remove(table, start, end)
        return cur.rowcount

    async def count_hidden_entries(self, table: str) -> int:
        """Rows physically present but hidden from reads for *table*."""
        hidden_sql, params = self._visibility.hidden_clause(table)
        if not hidden_sql:
            return 0
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM {table} WHERE {hidden_sql}",  # noqa: S608
            params,
        )
        return row[0] if row else 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run several writes as one unit, with no coroutine interleaving.

        The lock is what makes the unit: sqlite3 already keeps consecutive DML
        in one implicit transaction until ``commit``, so holding it for the
        block stops another handler's statement landing in the middle.  A
        caller inside the block must not call a method that commits on its own.
        """
        async with self._txn_lock:
            try:
                yield self._conn
            except BaseException:
                await self._conn.rollback()
                raise
            else:
                await self._conn.commit()
