"""The job store: one SQLite row per job, claimed under a lease.

Every write goes through :meth:`PersistenceManager.transaction`, so the
"record the hidden range and the job that will delete it" pair cannot be torn:
either both rows exist or neither does.  The claim is a select-then-update
inside that same lock; the update repeats its ``status = 'queued'`` predicate,
so a second process that read the same candidate loses the race instead of
running the job twice.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from ..observability.visibility import HIDDEN_TABLES, MAX_TS, MIN_TS
from .model import (
    DEFAULT_LEASE_SECONDS,
    HIDING_KINDS,
    PRIORITY_DEFAULT,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    PAYLOAD_VERSION,
    Job,
    iso_after,
    now_iso,
)

_JOB_COLUMNS = (
    "id, kind, payload, lane, status, priority, created_at, started_at, "
    "finished_at, attempts, max_attempts, progress_done, progress_total, "
    "progress_unit, result, error, hidden_range_ids, dedupe_key, "
    "cancel_requested"
)

_NO_WINDOW = "this job has no recorded hide window to restore; run the operation again"


def _stored_hide_spec(payload: str | None) -> list[tuple[str, str, str]] | None:
    """The window an enqueue recorded beside a job, or ``None`` if absent.

    A missing key is different from an empty list: the empty list is a job that
    never hid anything (trim, VACUUM, rebuild), while a missing key means a job
    written before the window was recorded, or one whose range rows were lost.
    Requeueing the latter would delete rows that no read is hiding, so callers
    refuse it instead of guessing.
    """
    try:
        raw = json.loads(payload) if payload else {}
    except ValueError as exc:
        raise ValueError(_NO_WINDOW) from exc
    if "_hide" not in raw:
        return None
    spec = raw["_hide"]
    if not isinstance(spec, list):
        raise ValueError(_NO_WINDOW)
    out: list[tuple[str, str, str]] = []
    for item in spec:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(_NO_WINDOW)
        table, start, end = item
        if table not in HIDDEN_TABLES:
            raise ValueError(_NO_WINDOW)
        out.append((str(table), str(start), str(end)))
    return out


class JobStore:
    """Reads and writes the ``jobs`` table for one persistence manager."""

    def __init__(self, persistence: Any) -> None:
        self._p = persistence

    @property
    def db_path(self) -> str | None:
        """The database file behind this store, when it has one.

        The worker places its instance lock beside it, so the path belongs on
        the store rather than on the private persistence handle.
        """
        return getattr(self._p, "db_path", None)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        lane: str = "sqlite",
        priority: int = PRIORITY_DEFAULT,
        dedupe_key: str | None = None,
        hidden_ranges: Sequence[tuple[str, str, str]] = (),
        max_attempts: int = 3,
    ) -> Job:
        """Record a job, plus the ranges it will delete, and return it.

        *hidden_ranges* is a sequence of ``(table, start, end)``; the rows in
        those ranges stop being returned by every read as soon as this
        commits.  An empty bound means "open at that end".

        A *dedupe_key* makes the call idempotent while a job with the same key
        is queued or running: the existing job is returned and nothing new is
        inserted.
        """
        if dedupe_key:
            existing = await self._active_for_key(dedupe_key)
            if existing is not None:
                return existing

        job_id = uuid.uuid4().hex
        created = now_iso()
        normalized = [
            (table, start or MIN_TS, end or MAX_TS)
            for table, start, end in hidden_ranges
        ]
        range_ids: list[str] = []
        async with self._p.transaction() as conn:
            for table, start, end in normalized:
                range_id = uuid.uuid4().hex
                range_ids.append(range_id)
                await conn.execute(
                    "INSERT INTO hidden_ranges "
                    "(id, table_name, start_ts, end_ts, job_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (range_id, table, start, end, job_id, created),
                )
            stored = {"v": PAYLOAD_VERSION, **(payload or {})}
            # Keep the resolved window beside the job, including an empty one.
            # A clean-undo cancel deletes the range rows, and resume has to hide
            # the same rows again before it deletes them; the empty list is the
            # marker that says this kind of job never hid anything, so a
            # requeue can tell it apart from a job whose window was lost.
            stored["_hide"] = [list(item) for item in normalized]
            await conn.execute(
                "INSERT INTO jobs (id, kind, payload, lane, status, priority, "
                "created_at, attempts, max_attempts, progress_done, "
                "hidden_range_ids, dedupe_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(stored),
                    lane,
                    JOB_QUEUED,
                    priority,
                    created,
                    max_attempts,
                    json.dumps(range_ids),
                    dedupe_key,
                ),
            )
        for table, start, end in normalized:
            self._p.visibility.add(table, start, end)
        job = await self.get(job_id)
        assert job is not None  # just inserted
        return job

    async def _active_for_key(self, dedupe_key: str) -> Job | None:
        row = await self._p.connection.execute_fetchone(
            "SELECT id FROM jobs WHERE dedupe_key = ? "
            "AND status IN ('queued', 'running') LIMIT 1",
            (dedupe_key,),
        )
        return await self.get(row[0]) if row else None

    # ------------------------------------------------------------------
    # Claim and run
    # ------------------------------------------------------------------

    async def claim(
        self,
        lane: str,
        owner: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Job | None:
        """Take the next queued job in *lane*, or return ``None``."""
        now = now_iso()
        expires = iso_after(lease_seconds)
        async with self._p.transaction() as conn:
            row = await conn.execute_fetchone(
                "SELECT id FROM jobs WHERE status = ? AND lane = ? "
                "AND cancel_requested = 0 "
                "ORDER BY priority ASC, created_at ASC LIMIT 1",
                (JOB_QUEUED, lane),
            )
            if row is None:
                return None
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, lease_owner = ?, "
                "lease_expires_at = ?, heartbeat_at = ?, "
                "started_at = COALESCE(started_at, ?), attempts = attempts + 1 "
                "WHERE id = ? AND status = ?",
                (JOB_RUNNING, owner, expires, now, now, row[0], JOB_QUEUED),
            )
            if cur.rowcount == 0:
                return None
        return await self.get(row[0])

    async def heartbeat(
        self,
        job_id: str,
        owner: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Extend a running job's lease; False if it is no longer ours."""
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ? "
                "WHERE id = ? AND lease_owner = ? AND status = ?",
                (now_iso(), iso_after(lease_seconds), job_id, owner, JOB_RUNNING),
            )
        return cur.rowcount > 0

    async def set_progress(
        self,
        job_id: str,
        done: int,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> None:
        """Record how far a running job has got."""
        async with self._p.transaction() as conn:
            await conn.execute(
                "UPDATE jobs SET progress_done = ?, progress_total = ?, "
                "progress_unit = ? WHERE id = ?",
                (done, total, unit, job_id),
            )

    async def release(self, job_id: str, owner: str) -> bool:
        """Put a running job back in the queue (a graceful shutdown).

        The attempt the claim consumed is given back: an orderly restart is not
        a failure, and burning the retry budget on it would eventually fail a
        job that never ran.
        """
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, attempts = MAX(attempts - 1, 0) "
                "WHERE id = ? AND lease_owner = ? AND status = ?",
                (JOB_QUEUED, job_id, owner, JOB_RUNNING),
            )
        return cur.rowcount > 0

    async def finish(
        self,
        job_id: str,
        owner: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Close a running job.

        A terminal job releases its hidden ranges.  Success has deleted the
        rows, so the reservation is spent.  A failure or a partial cancel
        leaves the rows on disk, but keeping them invisible would need a
        second operator action to bring them back; retry and resume already
        re-hide the same window from the payload before they delete, so the
        reservation can end with the job.
        """
        if status not in (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED):
            raise ValueError(f"{status!r} is not a terminal job state")
        dropped: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            # Success is a claim about the rows, so check the rows.  A handler
            # that returns normally while its own range still covers data (a
            # predicate that matched nothing, a batch that silently skipped)
            # would otherwise end with the rows visible again and the job
            # reporting success — the one promise this subsystem makes.
            if status == JOB_SUCCEEDED:
                ranges = await conn.execute_fetchall(
                    "SELECT table_name, start_ts, end_ts FROM hidden_ranges "
                    "WHERE job_id = ?",
                    (job_id,),
                )
                remaining = await self._rows_in_ranges(conn, ranges)
                if remaining:
                    status = JOB_FAILED
                    error = error or (
                        f"{remaining} hidden rows remain after the handler "
                        "reported success"
                    )
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, result = ?, "
                "error = ?, lease_owner = NULL, lease_expires_at = NULL, "
                "hidden_range_ids = '[]' "
                "WHERE id = ? AND lease_owner = ? AND status = ?",
                (
                    status,
                    now_iso(),
                    json.dumps(result) if result else None,
                    error,
                    job_id,
                    owner,
                    JOB_RUNNING,
                ),
            )
            if cur.rowcount == 0:
                return False
            dropped = await self._drop_ranges(conn, job_id)
        for table, start, end in dropped:
            self._p.visibility.remove(table, start, end)
        return True

    async def _rows_in_ranges(self, conn: Any, ranges: Sequence[Sequence[str]]) -> int:
        """How many rows the job's own ranges still cover.

        The table name is the one value that reaches the query text, so it is
        checked against the tables this subsystem hides rather than trusted.
        """
        total = 0
        for table, start, end in ranges:
            if table not in HIDDEN_TABLES:
                continue
            row = await conn.execute_fetchone(
                f"SELECT COUNT(*) FROM {table} "  # noqa: S608
                "WHERE timestamp >= ? AND timestamp < ?",
                (start, end),
            )
            total += row[0] if row else 0
        return total

    async def _drop_ranges(self, conn: Any, job_id: str) -> list[tuple[str, str, str]]:
        rows = await conn.execute_fetchall(
            "SELECT table_name, start_ts, end_ts FROM hidden_ranges WHERE job_id = ?",
            (job_id,),
        )
        await conn.execute("DELETE FROM hidden_ranges WHERE job_id = ?", (job_id,))
        return [(row[0], row[1], row[2]) for row in rows]

    # ------------------------------------------------------------------
    # Cancel, retry, resume
    # ------------------------------------------------------------------

    async def request_cancel(self, job_id: str) -> tuple[bool, str | None]:
        """Ask a job to stop; returns ``(found, status)``.

        A queued job is cancelled outright.  A running one is flagged, and its
        worker stops at the next checkpoint, because a delete already handed to
        SQLite cannot be recalled.
        """
        job = await self.get(job_id)
        if job is None:
            return False, None
        if job.status == JOB_QUEUED:
            dropped: list[tuple[str, str, str]] = []
            async with self._p.transaction() as conn:
                await conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, "
                    "cancel_requested = 1, hidden_range_ids = '[]' "
                    "WHERE id = ? AND status = ?",
                    (JOB_CANCELLED, now_iso(), job_id, JOB_QUEUED),
                )
                dropped = await self._drop_ranges(conn, job_id)
            for table, start, end in dropped:
                self._p.visibility.remove(table, start, end)
            return True, JOB_CANCELLED
        if job.status == JOB_RUNNING:
            async with self._p.transaction() as conn:
                await conn.execute(
                    "UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,)
                )
            return True, JOB_RUNNING
        return True, job.status

    async def retry(self, job_id: str) -> bool:
        """Put a failed or cancelled job back in the queue, fresh."""
        return await self._requeue(job_id, reset_attempts=True)

    async def resume(self, job_id: str) -> bool:
        """Continue a cancelled job, keeping the rows it has not deleted yet."""
        return await self._requeue(job_id, reset_attempts=False)

    async def _requeue(self, job_id: str, *, reset_attempts: bool) -> bool:
        """Queue a terminal job again, re-hiding its rows if it lost them.

        A cancel before anything was deleted is a clean undo and drops the
        range rows, so resuming that job would otherwise delete rows that were
        never hidden: they would vanish on the first batch instead of leaving
        the listings first.  The window is the one recorded at enqueue, not a
        recomputed one — for a clear, recomputing would sweep in traffic that
        arrived after the operator asked for it.
        """
        restored: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            # The UPDATE is the fence: a job that changed state under us must
            # not get its ranges back on the way out.
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, finished_at = NULL, error = NULL, "
                "cancel_requested = 0, lease_owner = NULL, lease_expires_at = NULL"
                + (", attempts = 0" if reset_attempts else "")
                + " WHERE id = ? AND status IN ('failed', 'cancelled')",
                (JOB_QUEUED, job_id),
            )
            if cur.rowcount == 0:
                return False
            row = await conn.execute_fetchone(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            owned = await conn.execute_fetchone(
                "SELECT COUNT(*) FROM hidden_ranges WHERE job_id = ?", (job_id,)
            )
            if not (owned and owned[0]):
                spec = _stored_hide_spec(row[0] if row else None)
                if spec is None:
                    raise ValueError(_NO_WINDOW)
                range_ids: list[str] = []
                created = now_iso()
                for table, start, end in spec:
                    range_id = uuid.uuid4().hex
                    range_ids.append(range_id)
                    await conn.execute(
                        "INSERT INTO hidden_ranges "
                        "(id, table_name, start_ts, end_ts, job_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (range_id, table, start, end, job_id, created),
                    )
                    restored.append((table, start, end))
                if range_ids:
                    await conn.execute(
                        "UPDATE jobs SET hidden_range_ids = ? WHERE id = ?",
                        (json.dumps(range_ids), job_id),
                    )
        for table, start, end in restored:
            self._p.visibility.add(table, start, end)
        return True

    # ------------------------------------------------------------------
    # Recovery and reads
    # ------------------------------------------------------------------

    async def _backfill_hide(
        self, conn: Any, job_id: str, kind: str, payload: str | None, ranges: Any
    ) -> bool:
        """Record a failed or cancelled job's released window in its payload."""
        try:
            data = json.loads(payload) if payload else {}
        except ValueError:
            return False
        if "_hide" in data:
            return False
        if ranges:
            data["_hide"] = [list(row) for row in ranges]
        elif kind not in HIDING_KINDS:
            data["_hide"] = []
        else:
            return False
        await conn.execute(
            "UPDATE jobs SET payload = ? WHERE id = ?", (json.dumps(data), job_id)
        )
        return True

    async def _repair_terminal_job(
        self,
        conn: Any,
        job_id: str,
        kind: str,
        status: str,
        payload: str | None,
        range_ids: str | None,
    ) -> tuple[bool, list[tuple[str, str, str]]]:
        """Release one terminal job's ranges, backfilling its retry window."""
        ranges = await conn.execute_fetchall(
            "SELECT table_name, start_ts, end_ts FROM hidden_ranges WHERE job_id = ?",
            (job_id,),
        )
        touched = False
        if status in (JOB_FAILED, JOB_CANCELLED):
            touched = await self._backfill_hide(conn, job_id, kind, payload, ranges)
        if ranges:
            await conn.execute("DELETE FROM hidden_ranges WHERE job_id = ?", (job_id,))
            touched = True
        if range_ids and range_ids != "[]":
            await conn.execute(
                "UPDATE jobs SET hidden_range_ids = '[]' WHERE id = ?", (job_id,)
            )
            touched = True
        return touched, [(row[0], row[1], row[2]) for row in ranges]

    async def release_terminal_ranges(self) -> int:
        """Release ranges a terminal job still owned before this release.

        Earlier versions kept a failed job's reservation and hid a job's rows
        until something deleted them; both are gone now, so a store written by
        one of those versions would otherwise keep its windows invisible with
        no route back.  For a failed or cancelled job the released window is
        written into ``payload["_hide"]`` first, which is what Retry and Resume
        re-hide; a job that never hid anything gets an empty window so it stays
        requeueable.  A job whose window was already lost keeps no ``_hide``:
        requeueing it is refused rather than deleting visible rows silently.

        Idempotent, and called once by :meth:`JobManager.start` before serving.
        """
        dropped: list[tuple[str, str, str]] = []
        changed = 0
        async with self._p.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT DISTINCT j.id, j.kind, j.status, j.payload, "
                "j.hidden_range_ids "
                "FROM jobs AS j "
                "LEFT JOIN hidden_ranges AS h ON h.job_id = j.id "
                "WHERE j.status IN (?, ?) OR (j.status = ? AND "
                "(j.hidden_range_ids <> '[]' OR h.job_id IS NOT NULL))",
                (JOB_FAILED, JOB_CANCELLED, JOB_SUCCEEDED),
            )
            for row in rows:
                touched, gone = await self._repair_terminal_job(conn, *row)
                dropped.extend(gone)
                changed += 1 if touched else 0
        if dropped:
            await self._p.reload_visibility()
        return changed

    async def sweep_orphan_ranges(self) -> int:
        """Drop ranges whose job row no longer exists, and reload the cache.

        A range outlives its job only if the job was deleted by something other
        than its own terminal transition — a hand-edited database, or a job row
        pruned while it still owned the range.  Left alone it would hide rows
        that nothing will ever delete, so startup repairs it.
        """
        async with self._p.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM hidden_ranges WHERE job_id NOT IN (SELECT id FROM jobs)"
            )
            if rows:
                await conn.execute(
                    "DELETE FROM hidden_ranges "
                    "WHERE job_id NOT IN (SELECT id FROM jobs)"
                )
        if rows:
            await self._p.reload_visibility()
        return len(rows)

    async def recover(
        self, *, owner: str | None = None, skip: Sequence[str] = ()
    ) -> int:
        """Return interrupted jobs to the queue.

        A lease that expired is an interrupted run.  Passing *owner* widens
        the sweep to rows leased by anyone else: a process that has just
        started cannot own a row it never claimed, so at startup those are
        interrupted too.  Without that, a crash-restart inside the 60-second
        lease leaves the job `running` — rows hidden, no worker deleting them,
        and nothing but a second restart to free it.

        *skip* leaves rows alone: the caller passes the jobs it is running
        right now, which are not interrupted even if their lease lapsed.

        A job that has already used its attempts becomes failed instead: it
        would otherwise be claimed, killed and requeued forever.
        """
        now = now_iso()
        foreign = " OR lease_owner IS NOT ?" if owner is not None else ""
        untouched = ""
        if skip:
            untouched = " AND id NOT IN (" + ", ".join("?" for _ in skip) + ")"
        expired = "status = ? AND (lease_expires_at IS NULL OR lease_expires_at < ?"
        where = expired + foreign + ")" + untouched
        select_params: list[Any] = [JOB_RUNNING, now]
        if owner is not None:
            select_params.append(owner)
        select_params.extend(skip)
        params: list[Any] = [
            JOB_CANCELLED,
            JOB_FAILED,
            JOB_QUEUED,
            now,
            now,
            *select_params,
        ]
        dropped: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, cancel_requested, attempts, max_attempts FROM jobs "
                f"WHERE {where}",
                select_params,
            )
            if not rows:
                return 0
            cur = await conn.execute(
                "UPDATE jobs SET "
                "status = CASE "
                "WHEN cancel_requested = 1 THEN ? "
                "WHEN attempts >= max_attempts THEN ? ELSE ? END, "
                "lease_owner = NULL, lease_expires_at = NULL, "
                "error = CASE "
                "WHEN cancel_requested = 1 "
                "THEN COALESCE(error, 'cancelled while running') "
                "WHEN attempts >= max_attempts "
                "THEN COALESCE(error, 'interrupted by a restart') "
                "ELSE error END, "
                "finished_at = CASE "
                "WHEN cancel_requested = 1 THEN ? "
                "WHEN attempts >= max_attempts THEN ? ELSE NULL END "
                "WHERE " + where,
                params,
            )
            if cur.rowcount == 0:
                return 0
            # Recovery settles a cancelled or exhausted job instead of
            # requeueing it.  Like `finish`, that releases its reservation:
            # retry and resume re-hide the payload's window before deleting.
            terminal = [row[0] for row in rows if row[1] or row[2] >= row[3]]
            if terminal:
                marks = ", ".join("?" for _ in terminal)
                dropped = [
                    (row[0], row[1], row[2])
                    for row in await conn.execute_fetchall(
                        "SELECT table_name, start_ts, end_ts FROM hidden_ranges "
                        f"WHERE job_id IN ({marks})",
                        terminal,
                    )
                ]
                await conn.execute(
                    f"DELETE FROM hidden_ranges WHERE job_id IN ({marks})",
                    terminal,
                )
                await conn.execute(
                    f"UPDATE jobs SET hidden_range_ids = '[]' WHERE id IN ({marks})",
                    terminal,
                )
        for table, start, end in dropped:
            self._p.visibility.remove(table, start, end)
        return cur.rowcount

    async def get(self, job_id: str) -> Job | None:
        """Return one job, or ``None``."""
        row = await self._p.connection.execute_fetchone(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
        )
        return _row_to_job(row) if row else None

    @staticmethod
    def _category_clause(category: str | None) -> tuple[str, list[Any]]:
        """The SQL predicate for a job category, or ``("", [])`` for all.

        One home for the ``batch.*`` namespace: the listing and the active
        count must answer the same question, or the header disagrees with the
        table it sits above.
        """
        if category == "requests":
            return "kind GLOB 'batch.*'", []
        if category == "ops":
            return "kind NOT GLOB 'batch.*'", []
        if category is not None:
            raise ValueError(f"unknown job category {category!r}")
        return "", []

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Job], int]:
        """Return jobs newest first, and the total matching the filters.

        *category* is ``requests`` for the reserved ``batch.*`` namespace and
        ``ops`` for everything else.  The predicate is in SQL rather than the
        caller so one busy half of the board cannot push the other half out of
        the page.
        """
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        category_clause, category_params = self._category_clause(category)
        if category_clause:
            where.append(category_clause)
            params.extend(category_params)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        conn = self._p.connection
        row = await conn.execute_fetchone(f"SELECT COUNT(*) FROM jobs{clause}", params)
        total = row[0] if row else 0
        rows = await conn.execute_fetchall(
            f"SELECT {_JOB_COLUMNS} FROM jobs{clause} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_row_to_job(r) for r in rows], total

    async def active_count(self, *, category: str | None = None) -> int:
        """How many jobs are queued or running, optionally in one category."""
        clause, params = self._category_clause(category)
        where = "status IN ('queued', 'running')"
        if clause:
            where += " AND " + clause
        row = await self._p.connection.execute_fetchone(
            "SELECT COUNT(*) FROM jobs WHERE " + where, params
        )
        return row[0] if row else 0

    async def hidden_end(self, job_id: str, table: str) -> str | None:
        """The end of the window *job_id* hid for *table*, if it still owns one.

        A job written before the payload carried a cutoff still has its range
        row, so the handler can delete by the same bound the reads hide by
        instead of falling back to "everything".
        """
        row = await self._p.connection.execute_fetchone(
            "SELECT MAX(end_ts) FROM hidden_ranges WHERE job_id = ? AND table_name = ?",
            (job_id, table),
        )
        if not row or not row[0] or row[0] == MAX_TS:
            return None
        return str(row[0])

    async def extend_clear_window(self, job_id: str, *, table: str, end: str) -> bool:
        """Move a still-queued clear's cutoff forward to *end*.

        A second Clear click is the same request with a later cutoff.  Nothing
        has been deleted while the job is queued, so extending its range hides
        and deletes the rows that arrived in between.  A running job has
        already read its cutoff and is left alone.
        """
        old: list[tuple[str, str, str]] = []
        new: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            row = await conn.execute_fetchone(
                "SELECT payload FROM jobs WHERE id = ? AND status = ? AND kind = ?",
                (job_id, JOB_QUEUED, "data.clear"),
            )
            if row is None:
                return False
            data = json.loads(row[0]) if row[0] else {}
            if data.get("table") != table:
                return False
            current_end = data.get("end")
            if isinstance(current_end, str) and current_end >= end:
                return False
            ranges = await conn.execute_fetchall(
                "SELECT table_name, start_ts, end_ts FROM hidden_ranges "
                "WHERE job_id = ?",
                (job_id,),
            )
            old = [(r[0], r[1], r[2]) for r in ranges]
            if not any(t == table for t, _, _ in old):
                return False
            for name, start, previous in old:
                if name == table:
                    await conn.execute(
                        "UPDATE hidden_ranges SET end_ts = ? "
                        "WHERE job_id = ? AND table_name = ? AND start_ts = ?",
                        (end, job_id, table, start),
                    )
                    new.append((name, start, end))
                else:
                    new.append((name, start, previous))
            data["end"] = end
            if isinstance(data.get("_hide"), list):
                data["_hide"] = [list(item) for item in new]
            await conn.execute(
                "UPDATE jobs SET payload = ? WHERE id = ?",
                (json.dumps(data), job_id),
            )
        for name, start, previous in old:
            self._p.visibility.remove(name, start, previous)
        for name, start, previous in new:
            self._p.visibility.add(name, start, previous)
        return True

    async def last_job_at(
        self,
        kind: str,
        *,
        status: str | None = None,
        source: str | None = None,
    ) -> str | None:
        """The newest matching job's completion time, or ``None``.

        A periodic task uses this to survive restarts: the worker and its
        in-memory timer do not, but the jobs table does.  The timestamp rather
        than a yes/no answer is what lets the caller sleep the remainder of the
        interval, so a wake a few milliseconds early cannot double the gap.
        *status* and *source* narrow the question — a failed or cancelled pass,
        or a one-hour manual delete, must not count as the weekly sweep.

        A finished job reports ``finished_at``; an unfinished one falls back to
        ``created_at``.  Anchoring on completion means a cleanup that takes
        hours does not pull the next weekly pass earlier by its own runtime.
        """
        where = "kind = ?"
        params: list[Any] = [kind]
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        rows = await self._p.connection.execute_fetchall(
            f"SELECT COALESCE(finished_at, created_at), payload "
            f"FROM jobs WHERE {where}",
            params,
        )
        newest: str | None = None
        for at, payload in rows:
            if source is not None:
                try:
                    data = json.loads(payload) if payload else {}
                except ValueError:
                    continue
                if data.get("source") != source:
                    continue
            if newest is None or at > newest:
                newest = at
        return newest

    async def prune(self, keep: int = 200) -> int:
        """Delete old terminal jobs, never one that still owns a range."""
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM jobs WHERE status IN "
                "('succeeded', 'failed', 'cancelled') "
                "AND id NOT IN (SELECT job_id FROM hidden_ranges) "
                "AND id NOT IN ("
                "  SELECT id FROM jobs WHERE status IN "
                "  ('succeeded', 'failed', 'cancelled') "
                "  ORDER BY created_at DESC LIMIT ?"
                ")",
                (keep,),
            )
        return cur.rowcount


def _row_to_job(row: Sequence[Any]) -> Job:
    payload = json.loads(row[2]) if row[2] else {}
    payload.pop("v", None)
    # The hide window is the store's, not the handler's: only `_requeue` reads
    # it, and it must not reach a handler or the admin API as a payload field.
    payload.pop("_hide", None)
    return Job(
        id=row[0],
        kind=row[1],
        payload=payload,
        lane=row[3],
        status=row[4],
        priority=row[5],
        created_at=row[6],
        started_at=row[7],
        finished_at=row[8],
        attempts=row[9],
        max_attempts=row[10],
        progress_done=row[11],
        progress_total=row[12],
        progress_unit=row[13],
        result=json.loads(row[14]) if row[14] else None,
        error=row[15],
        hidden_range_ids=json.loads(row[16]) if row[16] else [],
        dedupe_key=row[17],
        cancel_requested=bool(row[18]),
    )
